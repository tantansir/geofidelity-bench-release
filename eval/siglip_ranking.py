"""
Cross-encoder ranking validation: reproduce the main model ranking
using SigLIP-SO400M image features instead of DINOv2.

Addresses reviewer concern that CosSim headline ranking may be
encoder-specific. If the SigLIP-computed CosSim ranking agrees with
the DINOv2 ranking (high Spearman ρ), the benchmark's main claim
is robust to the choice of image encoder.
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
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

import config


MODELS = ["sdxl_base", "sd35_large", "flux_dev", "flux_schnell",
          "pixart_sigma", "hunyuan_dit"]


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places = bench["places"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP on {device}...")
    sl_model = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    sl_proc = AutoImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384")

    @torch.no_grad()
    def embed(paths):
        feats = []
        for fp in paths:
            try:
                img = Image.open(fp).convert("RGB")
                ins = sl_proc(images=img, return_tensors="pt").to(device)
                if device == "cuda":
                    ins = {k: v.half() if v.dtype == torch.float32 else v
                           for k, v in ins.items()}
                out = sl_model.get_image_features(**ins)
                if hasattr(out, "pooler_output"):
                    t = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    t = out.last_hidden_state.mean(dim=1)
                else:
                    t = out
                feats.append(t.cpu().float().numpy()[0])
            except Exception as e:
                print(f"  err on {fp.name}: {e}")
        return feats

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # Per-place real embedding (mean over reference images)
    print("Encoding real reference panels...")
    real_mean = {}
    for p in tqdm(places):
        ref_paths = [config.ROOT / rp for rp in p["image_paths"]
                     if (config.ROOT / rp).exists()]
        if len(ref_paths) < 2:
            continue
        feats = embed(ref_paths)
        if feats:
            real_mean[p["place_id"]] = np.stack(feats).mean(axis=0)
    print(f"  encoded {len(real_mean)} real panels")

    # Per-(model, place) generated embedding + cossim to real
    print("Encoding generated panels and computing CosSim...")
    rows = []
    for model in MODELS:
        print(f"  {model}")
        for p in tqdm(places, leave=False):
            pid = p["place_id"]
            if pid not in real_mean:
                continue
            pdir = config.GEN_DIR / model / pid
            if not pdir.exists():
                continue
            gen_paths = sorted(pdir.glob("*.jpg"))
            if len(gen_paths) < 2:
                continue
            feats = embed(gen_paths)
            if not feats:
                continue
            gen_mean = np.stack(feats).mean(axis=0)
            c = cos(gen_mean, real_mean[pid])
            rows.append({"method": model, "place_id": pid,
                         "city": p["city"], "siglip_cos_sim": c})

    # Also compute Oracle-NN (real held-out vs same real mean)
    # and Random-Global (real from any city vs real mean)
    import random
    rng = random.Random(42)
    all_pids = list(real_mean.keys())
    # Oracle-NN: use half of refs as "generation" vs remaining mean
    # Simplified proxy: each place already has mean - we skip per-split,
    # use the real_mean itself (self-similarity ~1.0), which is a ceiling.
    # For a meaningful oracle, sample leave-k-out, but compute lightly:
    for p in places:
        pid = p["place_id"]
        if pid not in real_mean:
            continue
        ref_paths = [config.ROOT / rp for rp in p["image_paths"]
                     if (config.ROOT / rp).exists()]
        if len(ref_paths) < 6:
            continue
        rng.shuffle(ref_paths)
        holdout = ref_paths[:2]
        gallery = ref_paths[2:]
        hf = embed(holdout)
        gf = embed(gallery)
        if hf and gf:
            h_mean = np.stack(hf).mean(axis=0)
            g_mean = np.stack(gf).mean(axis=0)
            rows.append({"method": "oracle_nn", "place_id": pid,
                         "city": p["city"],
                         "siglip_cos_sim": cos(h_mean, g_mean)})

    # Random-Global: real from another random city vs place's real mean
    for p in places:
        pid = p["place_id"]
        if pid not in real_mean:
            continue
        while True:
            other = rng.choice(all_pids)
            if other != pid and places[all_pids.index(other)]["city"] != p["city"]:
                break
        rows.append({"method": "random_global", "place_id": pid,
                     "city": p["city"],
                     "siglip_cos_sim": cos(real_mean[other], real_mean[pid])})

    df = pd.DataFrame(rows)
    out_dir = config.OUTPUT_DIR / "siglip_ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "siglip_per_place.csv", index=False)

    print("\n=== SigLIP CosSim by method ===")
    grp = df.groupby("method")["siglip_cos_sim"].agg(["mean", "std", "count"])
    print(grp.round(4).to_string())
    grp.to_csv(out_dir / "siglip_summary.csv")

    # Compare SigLIP-based vs DINOv2-based generator ranking
    print("\n=== Generator ranking agreement ===")
    dino = pd.read_csv(config.OUTPUT_DIR / "eval_v2" / "raw_results.csv")
    dino_means = dino[dino["method"].isin(MODELS)].groupby("method")["cos_sim"].mean()
    siglip_means = df[df["method"].isin(MODELS)].groupby("method")["siglip_cos_sim"].mean()

    common = sorted(set(dino_means.index) & set(siglip_means.index))
    print(f"{'model':<16} {'DINOv2':>10} {'SigLIP':>10}")
    for m in common:
        print(f"{m:<16} {dino_means[m]:>10.4f} {siglip_means[m]:>10.4f}")

    dino_ranks = dino_means.loc[common].rank(ascending=False)
    siglip_ranks = siglip_means.loc[common].rank(ascending=False)
    rho, p = spearmanr(dino_ranks, siglip_ranks)
    print(f"\nSpearman rho (DINOv2 vs SigLIP ranking) = {rho:.3f}  p = {p:.3f}")

    pd.DataFrame({"model": common,
                  "dino_cossim": [dino_means[m] for m in common],
                  "siglip_cossim": [siglip_means[m] for m in common],
                  "dino_rank": [int(dino_ranks[m]) for m in common],
                  "siglip_rank": [int(siglip_ranks[m]) for m in common]}).to_csv(
        out_dir / "ranking_comparison.csv", index=False)


if __name__ == "__main__":
    main()
