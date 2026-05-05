"""
Tier 3 filter: SigLIP zero-shot scene classifier.

Uses SigLIP SO400M (google/siglip-so400m-patch14-384) to score each image
against the target caption and a list of distractors (highway, night,
tunnel, rural, truck-blocked, blurry, indoor, sign close-up). An image
passes Tier 3 iff the urban-daytime class is top-1 AND its softmax
probability exceeds TIER3_URBAN_MIN_SCORE.

Input:  data/processed/tier2_osm.csv   (survivors: tier2_pass == True)
Output: data/processed/tier3_siglip.csv
        (adds per-class scores + tier3_pass)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import config


def _load_siglip(device: str):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(config.TIER3_SIGLIP_MODEL).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(config.TIER3_SIGLIP_MODEL)
    return model, processor


def _build_prompts() -> list[str]:
    """Class 0 is the target (urban-daytime), the rest are distractors."""
    return [config.TIER3_URBAN_PROMPT] + list(config.TIER3_DISTRACTOR_PROMPTS)


@torch.no_grad()
def score_images(image_paths: list[Path], device: str,
                 batch_size: int = 16) -> list[dict]:
    """Return per-image class probs and decisions."""
    model, processor = _load_siglip(device)
    prompts = _build_prompts()

    # Encode text once
    text_inputs = processor(text=prompts, padding="max_length", return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_feats = model.get_text_features(**text_inputs)
    text_feats = F.normalize(text_feats, dim=-1)

    # Temperature/bias per SigLIP
    logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 1.0
    logit_bias = model.logit_bias.item() if hasattr(model, "logit_bias") else 0.0

    rows: list[dict] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="siglip"):
        batch_paths = image_paths[i:i + batch_size]
        try:
            pil = [Image.open(p).convert("RGB") for p in batch_paths]
        except Exception:
            for p in batch_paths:
                rows.append({"image_path": str(p), "siglip_error": True})
            continue
        img_inputs = processor(images=pil, return_tensors="pt").to(device)
        img_feats = model.get_image_features(**img_inputs)
        img_feats = F.normalize(img_feats, dim=-1)

        logits = img_feats @ text_feats.T * logit_scale + logit_bias
        # SigLIP is sigmoid per class; convert to class-wise soft probs via softmax
        # across classes (what we want is top-1 + margin), so use softmax here.
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        for path, row_probs in zip(batch_paths, probs):
            top_cls = int(row_probs.argmax())
            urban_p = float(row_probs[0])
            passed = (top_cls == 0) and (urban_p >= config.TIER3_URBAN_MIN_SCORE)
            rec = {
                "image_path": str(path.relative_to(config.ROOT).as_posix()),
                "siglip_urban_p": round(urban_p, 4),
                "siglip_top_class": top_cls,
                "siglip_top_prompt": _build_prompts()[top_cls],
                "tier3_pass": bool(passed),
            }
            for j, prompt in enumerate(_build_prompts()):
                rec[f"siglip_p_{j:02d}"] = round(float(row_probs[j]), 4)
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(config.PROCESSED_DIR / "tier2_osm.csv"))
    ap.add_argument("--out_csv", default=str(config.PROCESSED_DIR / "tier3_siglip.csv"))
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--skip_failed_tiers", action="store_true",
                    help="Only score images that passed all prior tiers")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.skip_failed_tiers and "tier2_pass" in df.columns:
        df_active = df[df["tier2_pass"] == True].copy()
    else:
        df_active = df.copy()
    print(f"[tier3] scoring {len(df_active)}/{len(df)} images")

    # Resolve absolute paths
    paths = [config.ROOT / p for p in df_active["image_path"].tolist()]
    scores = score_images(paths, args.device, args.batch_size)

    score_df = pd.DataFrame(scores)
    merged = df.merge(score_df, on="image_path", how="left")
    merged["tier3_pass"] = merged["tier3_pass"].fillna(False).astype(bool)
    merged.to_csv(args.out_csv, index=False)

    n_pass = int(merged["tier3_pass"].sum())
    scored = int(merged["tier3_pass"].notna().sum())
    print(f"[tier3] pass {n_pass}/{scored} "
          f"({100.0 * n_pass / max(1, scored):.1f}%)")


if __name__ == "__main__":
    main()
