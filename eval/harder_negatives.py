"""
Stronger hard-negative design for metric validity.

Motivation: Round 1 reviewer noted that same-climate-wrong-city
negatives scored nearly identically to random-city. This suggests
the existing negatives separate mostly coarse style, not fine
geographic cues. We design two stronger matched-negative classes:

  1. visual_nearest: for each place P, find the place from a
     *different* city whose real reference panel has the highest
     DINOv2 mean-cosine similarity to P. This is the visually
     hardest wrong-city case.

  2. same_country_visual: same as (1) but restricted to other
     places in the same country (when available). This isolates
     country-level priors from sub-country fidelity.

Output: outputs/validity_v2/metric_validity_harder_summary.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

import config
from metrics.geo_attribute import GeoAttributeAgreementScore
from metrics.panel_retrieval import PanelRetriever
from metrics.set_fidelity import diversity_calibrated_set_fidelity


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places = {p["place_id"]: p for p in bench["places"]}
    by_city = defaultdict(list)
    by_country = defaultdict(list)
    for p in bench["places"]:
        by_city[p["city"]].append(p["place_id"])
        by_country[p["country"]].append(p["place_id"])

    retriever = PanelRetriever()
    gaas = GeoAttributeAgreementScore(
        tier4_csv=config.PROCESSED_DIR / "tier6_review.csv"
        if (config.PROCESSED_DIR / "tier6_review.csv").exists() else None)

    # Step 1: compute DINOv2 mean embedding per place
    print("Computing place embeddings...")
    emb = {}
    for pid, p in tqdm(places.items()):
        paths = [config.ROOT / rp for rp in p["image_paths"]]
        imgs = [Image.open(fp).convert("RGB") for fp in paths if fp.exists()]
        if len(imgs) >= 2:
            feats = retriever.encode_batch(imgs)
            emb[pid] = feats.mean(axis=0)

    # Step 2: find nearest-different-city and nearest-same-country
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    visual_neg = {}
    country_neg = {}
    for pid, e in tqdm(emb.items(), desc="Find hard negatives"):
        p = places[pid]
        best_v, best_v_sim = None, -1e9
        best_c, best_c_sim = None, -1e9
        for qid, qe in emb.items():
            if qid == pid:
                continue
            q = places[qid]
            if q["city"] == p["city"]:
                continue
            s = cos(e, qe)
            if s > best_v_sim:
                best_v_sim = s
                best_v = qid
            if q["country"] == p["country"] and s > best_c_sim:
                best_c_sim = s
                best_c = qid
        visual_neg[pid] = best_v
        country_neg[pid] = best_c

    # Step 3: evaluate probe/gallery with stronger negatives
    print("\nEvaluating metric validity on harder negatives...")
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

        row = {"place_id": pid, "city": p["city"]}
        pimgs = load_imgs(probe)
        pe = retriever.encode_batch(pimgs)
        pm = pe.mean(axis=0)

        # same place
        gimgs = load_imgs(gallery)
        ge = retriever.encode_batch(gimgs)
        gm = ge.mean(axis=0)
        row["same_place_cos_sim"] = cos(pm, gm)
        d = diversity_calibrated_set_fidelity(pe, ge)
        row["same_place_dcsf"] = d["dcsf"]
        row["same_place_mmd"] = d["mmd"]
        try:
            row["same_place_gaas"] = gaas.evaluate_place(
                [config.ROOT / x for x in probe],
                [config.ROOT / x for x in gallery])["overall"]
        except Exception:
            row["same_place_gaas"] = float("nan")

        # visual nearest wrong-city negative
        for label, neg_id in [
            ("visual_nearest_wrong_city", visual_neg.get(pid)),
            ("same_country_visual", country_neg.get(pid)),
        ]:
            if not neg_id or neg_id not in places:
                continue
            nimgs = load_imgs(places[neg_id]["image_paths"])
            if len(nimgs) < 2:
                continue
            ne = retriever.encode_batch(nimgs)
            nm = ne.mean(axis=0)
            row[f"{label}_cos_sim"] = cos(pm, nm)
            d = diversity_calibrated_set_fidelity(pe, ne)
            row[f"{label}_dcsf"] = d["dcsf"]
            row[f"{label}_mmd"] = d["mmd"]
            try:
                row[f"{label}_gaas"] = gaas.evaluate_place(
                    [config.ROOT / x for x in probe],
                    [config.ROOT / x for x in places[neg_id]["image_paths"]]
                )["overall"]
            except Exception:
                row[f"{label}_gaas"] = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = config.OUTPUT_DIR / "validity_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metric_validity_harder_raw.csv", index=False)

    # Summarize
    CONDITIONS = ["same_place", "visual_nearest_wrong_city",
                  "same_country_visual"]
    METRICS = ["cos_sim", "dcsf", "mmd", "gaas"]
    summary = []
    print("\n=== HARDER NEGATIVES SUMMARY ===")
    for m in METRICS:
        print(f"\n--- {m.upper()} ---")
        for c in CONDITIONS:
            col = f"{c}_{m}"
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            boot = np.random.default_rng(0).choice(
                vals, size=(1000, len(vals)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {c:28s}: {vals.mean():.4f} (95% CI [{lo:.4f}, {hi:.4f}], n={len(vals)})")
            summary.append({"metric": m, "condition": c,
                            "mean": vals.mean(), "ci_lo": lo, "ci_hi": hi,
                            "n": len(vals)})
    pd.DataFrame(summary).to_csv(
        out_dir / "metric_validity_harder_summary.csv", index=False)

    # Save negative mapping for reproducibility
    with open(out_dir / "harder_negatives.json", "w") as f:
        json.dump({"visual_nearest": visual_neg,
                   "same_country_visual": country_neg}, f, indent=2)

    print(f"\n[harder_negatives] saved to {out_dir}")


if __name__ == "__main__":
    main()
