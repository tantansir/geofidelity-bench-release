"""
Analyze prompt sensitivity on whatever has been generated so far
(partial run). Prints per-prompt mean CosSim/DCSF/MMD/GAAS and the
per-city rank stability across prompts.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

import config
from metrics.geo_attribute import GeoAttributeAgreementScore
from metrics.panel_retrieval import PanelRetriever
from metrics.set_fidelity import diversity_calibrated_set_fidelity


PROMPTS = ["P0_default", "P1_minimal", "P2_rich", "P3_angled"]
MODELS = ["sdxl_base"]  # FLUX-schnell skipped: gated HF repo access error


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places = {p["place_id"]: p for p in bench["places"]}

    gen_root = config.OUTPUT_DIR / "prompt_sensitivity" / "generations"
    out_dir = config.OUTPUT_DIR / "prompt_sensitivity"

    retriever = PanelRetriever()
    gaas = GeoAttributeAgreementScore()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    rows = []
    for model in MODELS:
        for pk in PROMPTS:
            for place_dir in (gen_root / model / pk).glob("*"):
                if not place_dir.is_dir():
                    continue
                pid = place_dir.name
                if pid not in places:
                    continue
                gen_paths = sorted(place_dir.glob("*.jpg"))
                if len(gen_paths) < 2:
                    continue
                gen_imgs = [Image.open(p).convert("RGB") for p in gen_paths]
                ref_paths = [config.ROOT / rp for rp in places[pid]["image_paths"]]
                ref_imgs = [Image.open(p).convert("RGB") for p in ref_paths if p.exists()]
                if len(ref_imgs) < 2:
                    continue
                g_e = retriever.encode_batch(gen_imgs)
                r_e = retriever.encode_batch(ref_imgs)
                c = cos(g_e.mean(0), r_e.mean(0))
                d = diversity_calibrated_set_fidelity(g_e, r_e)
                try:
                    g_gaas = gaas.evaluate_place(gen_paths, ref_paths)["overall"]
                except Exception:
                    g_gaas = float("nan")
                rows.append({"model": model, "prompt_key": pk,
                             "city": places[pid]["city"], "place_id": pid,
                             "n_gen": len(gen_imgs),
                             "cos_sim": c, "dcsf": d["dcsf"], "mmd": d["mmd"],
                             "gaas": g_gaas})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "prompt_sensitivity_partial_raw.csv", index=False)
    print(f"\nCollected {len(df)} per-(model,prompt,city) rows")

    print("\n=== Per-prompt mean (averaged over cities with full k=4) ===")
    full = df[df["n_gen"] >= 4]
    summary = full.groupby(["model", "prompt_key"])[
        ["cos_sim", "dcsf", "mmd", "gaas"]].mean().round(4)
    print(summary)
    summary.to_csv(out_dir / "prompt_summary_partial.csv")

    print("\n=== Per-city CosSim under each prompt (SDXL only) ===")
    sdxl = full[full["model"] == "sdxl_base"]
    if len(sdxl):
        pivot = sdxl.pivot_table(index="city", columns="prompt_key",
                                  values="cos_sim", aggfunc="mean")
        print(pivot.round(4))

    print("\n=== Rank stability (city ordering) across prompts ===")
    rank_rows = []
    for model in MODELS:
        sub = full[full["model"] == model]
        if len(sub) == 0:
            continue
        ranks = {}
        for pk in PROMPTS:
            s = sub[sub["prompt_key"] == pk].groupby("city")["cos_sim"].mean()
            if len(s) >= 3:
                ranks[pk] = s.sort_values(ascending=False).index.tolist()
        print(f"\n  {model} ({len(ranks)} prompts have enough data):")
        prompts = list(ranks.keys())
        for i in range(len(prompts)):
            for j in range(i+1, len(prompts)):
                pa, pb = prompts[i], prompts[j]
                ra = {c: k for k, c in enumerate(ranks[pa])}
                rb = {c: k for k, c in enumerate(ranks[pb])}
                common = set(ra) & set(rb)
                if len(common) < 3:
                    continue
                rho, pv = spearmanr([ra[c] for c in common],
                                     [rb[c] for c in common])
                print(f"    {pa:15s} vs {pb:15s}: ρ={rho:+.3f} (n={len(common)})")
                rank_rows.append({"model": model,
                                  "prompt_a": pa, "prompt_b": pb,
                                  "spearman_rho": rho, "n": len(common)})
    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(
            out_dir / "prompt_rank_stability_partial.csv", index=False)


if __name__ == "__main__":
    main()
