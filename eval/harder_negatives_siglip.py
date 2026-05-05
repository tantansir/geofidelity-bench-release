"""
Metric-independent hard negatives using SigLIP image features.

Round 2 reviewer noted that DINOv2-based visual-nearest negatives
could be circular (the metric under test is itself DINOv2-based).
Here we define the hardest-wrong-city negative using SigLIP-SO400M
image features, which are trained independently of DINOv2 and are
already used in our Tier-3 curation filter.

Output: outputs/validity_v2/metric_validity_siglip_harder_summary.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

import config
from metrics.geo_attribute import GeoAttributeAgreementScore
from metrics.panel_retrieval import PanelRetriever
from metrics.set_fidelity import diversity_calibrated_set_fidelity


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places = {p["place_id"]: p for p in bench["places"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP on {device}...")
    sl_model = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    # Use AutoImageProcessor only; avoids the SentencePiece/protobuf
    # dependencies that SiglipTokenizer pulls in via AutoProcessor.
    sl_proc = AutoImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384")
    sl_model.eval()

    @torch.no_grad()
    def siglip_mean_emb(paths):
        feats = []
        for fp in paths:
            try:
                img = Image.open(fp).convert("RGB")
                ins = sl_proc(images=img, return_tensors="pt").to(device)
                if device == "cuda":
                    ins = {k: v.half() if v.dtype == torch.float32 else v
                           for k, v in ins.items()}
                out = sl_model.get_image_features(**ins)
                # Handle both tensor and BaseModelOutputWithPooling returns
                if hasattr(out, "pooler_output"):
                    t = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    t = out.last_hidden_state.mean(dim=1)
                else:
                    t = out
                feats.append(t.cpu().float().numpy()[0])
            except Exception as e:
                print(f"  siglip err on {fp.name}: {e}", flush=True)
        if not feats:
            return None
        m = np.stack(feats).mean(axis=0)
        return m

    print("Computing SigLIP mean embeddings per place...")
    sl_emb = {}
    for pid, p in tqdm(places.items()):
        paths = [config.ROOT / rp for rp in p["image_paths"]
                 if (config.ROOT / rp).exists()]
        if len(paths) >= 2:
            m = siglip_mean_emb(paths)
            if m is not None:
                sl_emb[pid] = m

    # Find visual-nearest wrong-city negative in SigLIP space
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    print("\nFinding SigLIP-nearest wrong-city negatives...")
    siglip_neg = {}
    for pid, e in tqdm(sl_emb.items()):
        p = places[pid]
        best, best_sim = None, -1e9
        for qid, qe in sl_emb.items():
            if qid == pid or places[qid]["city"] == p["city"]:
                continue
            s = cos(e, qe)
            if s > best_sim:
                best_sim = s
                best = qid
        siglip_neg[pid] = best

    # Now evaluate the four DINOv2/Mask2Former metrics against this
    # SigLIP-chosen negative. The metric under test is independent of
    # the negative-selection feature space.
    print("\nEvaluating DINOv2/Mask2Former metrics against SigLIP-chosen negatives...")
    retriever = PanelRetriever()
    gaas = GeoAttributeAgreementScore()
    rng = np.random.default_rng(42)

    def load_imgs(paths):
        return [Image.open(config.ROOT / rp).convert("RGB")
                for rp in paths if (config.ROOT / rp).exists()]

    rows = []
    for pid, p in tqdm(places.items()):
        img_paths = p.get("image_paths", [])
        if len(img_paths) < 4:
            continue
        idx = rng.permutation(len(img_paths))
        probe = [img_paths[i] for i in idx[:2]]
        gallery = [img_paths[i] for i in idx[2:]]

        pimgs = load_imgs(probe)
        gimgs = load_imgs(gallery)
        pe = retriever.encode_batch(pimgs)
        ge = retriever.encode_batch(gimgs)
        pm = pe.mean(axis=0); gm = ge.mean(axis=0)

        row = {"place_id": pid, "city": p["city"],
               "same_place_cos_sim": cos(pm, gm)}
        d = diversity_calibrated_set_fidelity(pe, ge)
        row["same_place_dcsf"] = d["dcsf"]
        row["same_place_mmd"] = d["mmd"]
        try:
            row["same_place_gaas"] = gaas.evaluate_place(
                [config.ROOT / x for x in probe],
                [config.ROOT / x for x in gallery])["overall"]
        except Exception:
            row["same_place_gaas"] = float("nan")

        neg_id = siglip_neg.get(pid)
        if not neg_id or neg_id not in places:
            rows.append(row); continue
        nimgs = load_imgs(places[neg_id]["image_paths"])
        if len(nimgs) < 2:
            rows.append(row); continue
        ne = retriever.encode_batch(nimgs)
        nm = ne.mean(axis=0)
        row["siglip_nearest_cos_sim"] = cos(pm, nm)
        d = diversity_calibrated_set_fidelity(pe, ne)
        row["siglip_nearest_dcsf"] = d["dcsf"]
        row["siglip_nearest_mmd"] = d["mmd"]
        try:
            row["siglip_nearest_gaas"] = gaas.evaluate_place(
                [config.ROOT / x for x in probe],
                [config.ROOT / x for x in places[neg_id]["image_paths"]]
            )["overall"]
        except Exception:
            row["siglip_nearest_gaas"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = config.OUTPUT_DIR / "validity_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metric_validity_siglip_harder_raw.csv", index=False)

    # Summary
    summary = []
    METRICS = ["cos_sim", "dcsf", "mmd", "gaas"]
    print("\n=== SIGLIP-INDEPENDENT HARD NEGATIVES ===")
    for m in METRICS:
        print(f"\n--- {m.upper()} ---")
        for c in ("same_place", "siglip_nearest"):
            col = f"{c}_{m}"
            if col not in df.columns:
                continue
            vals = df[col].dropna().to_numpy()
            if len(vals) == 0:
                continue
            boot = np.random.default_rng(0).choice(
                vals, size=(1000, len(vals)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {c:20s}: {vals.mean():.4f}  (95% CI [{lo:.4f}, {hi:.4f}], n={len(vals)})")
            summary.append({"metric": m, "condition": c,
                            "mean": vals.mean(), "ci_lo": lo, "ci_hi": hi,
                            "n": len(vals)})
    pd.DataFrame(summary).to_csv(
        out_dir / "metric_validity_siglip_harder_summary.csv", index=False)
    with open(out_dir / "siglip_harder_negatives.json", "w") as f:
        json.dump(siglip_neg, f, indent=2)
    print(f"\n[siglip_harder] saved to {out_dir}")


if __name__ == "__main__":
    main()
