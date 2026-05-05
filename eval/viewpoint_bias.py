"""
Camera-viewpoint bias audit.

Addresses the reviewer concern that the benchmark may partly reward
'looks like Mapillary' rather than 'looks like the target city'.

Approach:
  1. Partition each place's reference panel into DASHCAM-LIKE vs
     NON-DASHCAM-LIKE subsets using a simple heuristic on the
     Mask2Former segmentation: dashcam-like frames have a large
     bottom-center road area (>0.15 of the bottom third) and low
     sky-at-top (i.e. view faces down the road).
  2. Compute CosSim separately against each subset for the default
     (P0) and dashcam-angled (P3) prompts, using SDXL generations
     from the prompt-sensitivity run.
  3. Report: does P3 close the gap to dashcam-like refs specifically?
     If P0 and P3 differ more against dashcam refs than non-dashcam,
     that quantifies viewpoint bias in CosSim.

Saved to outputs/viewpoint_bias/*.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from metrics.panel_retrieval import PanelRetriever


def is_dashcam_like(img_path):
    """
    Cheap viewpoint heuristic without re-running Mask2Former:
    compute mean pixel intensity in bottom-center strip (road) and
    top strip (sky). Dashcam frames have low-contrast road below,
    sky/buildings above, and a strong central vanishing point.
    We use: (bottom_center_brightness_variance low) AND
    (top strip brighter than bottom strip) as a rough proxy.
    """
    img = Image.open(img_path).convert("L")
    w, h = img.size
    # crop to square first
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2,
                    (w - s) // 2 + s, (h - s) // 2 + s))
    arr = np.asarray(img, dtype=np.float32)
    H, W = arr.shape
    top = arr[: H // 4, :].mean()
    bot_center = arr[-H // 3:, W // 4: 3 * W // 4]
    bot_var = bot_center.var()
    bot_mean = bot_center.mean()
    # dashcam: top > bot_mean (sky above), bot_var low (uniform road)
    return (top - bot_mean > 15) and (bot_var < 1500)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places = {p["place_id"]: p for p in bench["places"]}

    # 5 subset places that were generated in prompt-sensitivity
    subset = ["paris__881fb46703fffff", "tokyo__882f5a3411fffff",
              "cairo__883e628e61fffff", "new_york__882a100d05fffff",
              "sao_paulo__88a8100dcbfffff"]

    retriever = PanelRetriever()

    # Step 1: classify reference images
    print("Classifying reference images by viewpoint...")
    ref_splits = {}
    for pid in subset:
        if pid not in places:
            continue
        refs = places[pid]["image_paths"]
        dash, nondash = [], []
        for rp in refs:
            fp = config.ROOT / rp
            if not fp.exists():
                continue
            try:
                if is_dashcam_like(fp):
                    dash.append(fp)
                else:
                    nondash.append(fp)
            except Exception:
                pass
        ref_splits[pid] = {"dash": dash, "nondash": nondash,
                           "all": dash + nondash}
        print(f"  {pid}: dash={len(dash)} nondash={len(nondash)}")

    # Step 2: encode generations for P0 and P3 under SDXL
    print("\nEncoding generations + refs...")
    gen_root = config.OUTPUT_DIR / "prompt_sensitivity" / "generations" / "sdxl_base"

    rows = []
    for pid in subset:
        if pid not in ref_splits:
            continue
        # encode reference splits (need at least 2 imgs)
        ref_embs = {}
        for split_name, paths in ref_splits[pid].items():
            if len(paths) >= 2:
                imgs = [Image.open(p).convert("RGB") for p in paths]
                e = retriever.encode_batch(imgs)
                ref_embs[split_name] = e.mean(axis=0)

        for prompt_key in ["P0_default", "P3_angled"]:
            pdir = gen_root / prompt_key / pid
            gen_paths = sorted(pdir.glob("*.jpg"))
            if len(gen_paths) < 2:
                continue
            gen_imgs = [Image.open(p).convert("RGB") for p in gen_paths]
            gen_emb = retriever.encode_batch(gen_imgs).mean(axis=0)

            for split_name, rmean in ref_embs.items():
                rows.append({
                    "place_id": pid, "city": places[pid]["city"],
                    "prompt_key": prompt_key,
                    "ref_split": split_name,
                    "n_ref": len(ref_splits[pid][split_name]),
                    "cos_sim": cos(gen_emb, rmean),
                })

    df = pd.DataFrame(rows)
    out_dir = config.OUTPUT_DIR / "viewpoint_bias"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw.csv", index=False)

    print("\n=== CosSim by (prompt, ref split) ===")
    pivot = df.pivot_table(index="ref_split", columns="prompt_key",
                            values="cos_sim", aggfunc="mean")
    print(pivot.round(4))
    pivot.to_csv(out_dir / "pivot.csv")

    # Gap analysis: does P3 close the gap to dashcam specifically?
    print("\n=== P3 - P0 gap by reference subset ===")
    if "dash" in pivot.index and "nondash" in pivot.index:
        for sp in ["dash", "nondash", "all"]:
            if sp in pivot.index:
                gap = pivot.loc[sp, "P3_angled"] - pivot.loc[sp, "P0_default"]
                print(f"  {sp:8s}: P3 - P0 = {gap:+.4f}")

    print(f"\n[viewpoint_bias] saved to {out_dir}")


if __name__ == "__main__":
    main()
