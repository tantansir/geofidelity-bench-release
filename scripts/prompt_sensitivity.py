"""
Prompt sensitivity ablation.

Research question: does our benchmark's conclusion depend on the
specific prompt wording we used ("A street in {city}, {country}.")?
This is the single most common reviewer concern for benchmarks that
evaluate T2I with a fixed prompt template.

Design: pick the top model (SDXL) + one diagnostic model (FLUX.1-schnell,
which ranked differently on CosSim vs DCSF), and a subset of 5 cities
spanning continents. Generate k=4 images for each of 4 prompt
variants covering:
  P0 (default)  : the template used in the main paper
  P1 (minimal)  : bare "A street in {city}."
  P2 (rich)     : longer description emphasising locale
  P3 (angled)   : explicit viewpoint/composition language

Then evaluate CosSim/GAAS against the same real references and show
that the *ranking* of cities (easy→hard) is preserved across prompts,
even if absolute values shift.

This addresses reviewer concern #2 from Round 1: whether the
benchmark conclusions generalise beyond a single prompt template.
"""
import sys
import time
import hashlib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import config
from generation.registry import load_generator
from metrics.panel_retrieval import PanelRetriever
from metrics.geo_attribute import GeoAttributeAgreementScore
from metrics.set_fidelity import diversity_calibrated_set_fidelity


# -------------- Prompt variants --------------
PROMPTS = {
    "P0_default": (
        "A street-level photograph taken in {city}, {country}. "
        "The image shows a typical street scene with buildings, roads, and "
        "urban environment characteristic of this location. "
        "Photorealistic, daytime, clear weather."
    ),
    "P1_minimal": "A street in {city}, {country}.",
    "P2_rich": (
        "An authentic, ground-level photograph of a neighborhood street "
        "in {city}, {country}, capturing the local architectural character, "
        "signage, vehicles, road surface, and vegetation that distinguish "
        "this specific city from others. Shot on a clear day from "
        "pedestrian eye level. Documentary style."
    ),
    "P3_angled": (
        "Street view in {city}, {country}, perspective from the middle of "
        "the road looking down the street, dashcam-style composition, "
        "natural daylight, no filter, capturing local buildings on both "
        "sides and the road surface ahead."
    ),
}

MODELS_TO_RUN = ["sdxl_base", "flux_schnell"]  # top + diverse

# 5 cities spanning continents, chosen for high-image-count stability
SUBSET_CITIES = ["paris", "tokyo", "cairo", "new_york", "sao_paulo"]
K = 4


def _seed(model, place_id, k, prompt_key):
    h = hashlib.sha256(f"{model}/{prompt_key}/{place_id}/{k}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    # pick one place per subset city (the one with max images)
    places_by_city = {}
    for p in bench["places"]:
        places_by_city.setdefault(p["city"], []).append(p)
    subset = []
    for c in SUBSET_CITIES:
        if c not in places_by_city:
            continue
        place = max(places_by_city[c], key=lambda p: len(p["image_paths"]))
        subset.append(place)
    print(f"Subset: {[p['place_id'] for p in subset]}")

    out_dir = config.OUTPUT_DIR / "prompt_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_root = out_dir / "generations"
    gen_root.mkdir(exist_ok=True)

    # ---------- Step 1: generate (cached) ----------
    manifest_path = out_dir / "manifest.csv"
    rows = []
    for model_name in MODELS_TO_RUN:
        print(f"\n=== Generating with {model_name} ===")
        gen = None
        try:
            for prompt_key, tmpl in PROMPTS.items():
                for place in subset:
                    # check if all k outputs already exist
                    pdir = gen_root / model_name / prompt_key / place["place_id"]
                    pdir.mkdir(parents=True, exist_ok=True)
                    all_exist = all((pdir / f"{k:02d}.jpg").exists()
                                    for k in range(K))
                    if all_exist:
                        for k in range(K):
                            rows.append({"model": model_name,
                                         "prompt_key": prompt_key,
                                         "place_id": place["place_id"],
                                         "city": place["city"],
                                         "k": k,
                                         "path": str((pdir / f"{k:02d}.jpg")
                                                     .relative_to(config.ROOT)
                                                     .as_posix())})
                        continue

                    if gen is None:
                        gen = load_generator(model_name, device="cuda")

                    prompt = tmpl.format(
                        city=place["city"].replace("_", " ").title(),
                        country=place["country"])
                    print(f"  {prompt_key} @ {place['place_id']}: ", end="", flush=True)
                    t0 = time.time()
                    for k in range(K):
                        path = pdir / f"{k:02d}.jpg"
                        if path.exists():
                            rows.append({"model": model_name,
                                         "prompt_key": prompt_key,
                                         "place_id": place["place_id"],
                                         "city": place["city"], "k": k,
                                         "path": str(path.relative_to(config.ROOT).as_posix())})
                            continue
                        seed = _seed(model_name, place["place_id"], k, prompt_key)
                        img = gen.generate(prompt=prompt, seed=seed)
                        img.save(str(path), quality=95)
                        rows.append({"model": model_name,
                                     "prompt_key": prompt_key,
                                     "place_id": place["place_id"],
                                     "city": place["city"], "k": k,
                                     "path": str(path.relative_to(config.ROOT).as_posix())})
                    print(f"{time.time()-t0:.1f}s")
        finally:
            if gen is not None:
                gen.unload()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    pd.DataFrame(rows).drop_duplicates(
        ["model", "prompt_key", "place_id", "k"]).to_csv(
        manifest_path, index=False)

    # ---------- Step 2: evaluate ----------
    print("\n=== Evaluating ===")
    retriever = PanelRetriever()
    gaas = GeoAttributeAgreementScore()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    results = []
    for model_name in MODELS_TO_RUN:
        for prompt_key in PROMPTS:
            for place in subset:
                pdir = gen_root / model_name / prompt_key / place["place_id"]
                gen_paths = sorted(pdir.glob("*.jpg"))
                if len(gen_paths) < 2:
                    continue
                gen_imgs = [Image.open(p).convert("RGB") for p in gen_paths]
                ref_paths = [config.ROOT / rp for rp in place["image_paths"]]
                ref_imgs = [Image.open(p).convert("RGB") for p in ref_paths
                            if p.exists()]
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

                results.append({
                    "model": model_name, "prompt_key": prompt_key,
                    "city": place["city"], "place_id": place["place_id"],
                    "cos_sim": c, "dcsf": d["dcsf"], "mmd": d["mmd"],
                    "gaas": g_gaas,
                })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "prompt_sensitivity_raw.csv", index=False)

    # ---------- Step 3: summarise ----------
    print("\n=== Per-prompt mean (across 5 subset cities) ===")
    summary = df.groupby(["model", "prompt_key"])[
        ["cos_sim", "dcsf", "mmd", "gaas"]].mean().round(4)
    print(summary)
    summary.to_csv(out_dir / "summary_by_prompt.csv")

    # Rank stability: does the city order (easy->hard by CosSim)
    # stay consistent across prompts for each model?
    from scipy.stats import spearmanr
    print("\n=== Rank stability across prompts (per model) ===")
    rank_rows = []
    for model_name in MODELS_TO_RUN:
        sub = df[df["model"] == model_name]
        ranks = {}
        for pk in PROMPTS:
            city_cos = sub[sub["prompt_key"] == pk].groupby("city")[
                "cos_sim"].mean().sort_values(ascending=False)
            if len(city_cos) < 3:
                continue
            ranks[pk] = list(city_cos.index)
        prompts = list(ranks.keys())
        print(f"\n  {model_name}:")
        for i, pa in enumerate(prompts):
            for pb in prompts[i+1:]:
                ra = {c: i for i, c in enumerate(ranks[pa])}
                rb = {c: i for i, c in enumerate(ranks[pb])}
                common = set(ra) & set(rb)
                if len(common) < 3:
                    continue
                rho, p = spearmanr([ra[c] for c in common],
                                    [rb[c] for c in common])
                print(f"    {pa:15s} vs {pb:15s}: ρ={rho:+.3f} (n={len(common)})")
                rank_rows.append({"model": model_name,
                                  "prompt_a": pa, "prompt_b": pb,
                                  "spearman_rho": rho, "n": len(common)})
    pd.DataFrame(rank_rows).to_csv(
        out_dir / "rank_stability.csv", index=False)
    print(f"\n[prompt_sensitivity] saved to {out_dir}")


if __name__ == "__main__":
    main()
