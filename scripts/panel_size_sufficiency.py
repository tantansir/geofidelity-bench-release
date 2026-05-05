"""
Panel-size sufficiency analysis.

Reviewer concern: some cities have only 1 tile or as few as 6
reference images; how reliable are per-city/per-tile conclusions?

Approach: drop panels with fewer than k references (for k = 6, 8,
10, 12), recompute the headline CosSim ranking, and check whether
top-tier/bottom-tier assignments change.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config


MODELS = ["sdxl_base", "sd35_large", "flux_dev", "flux_schnell",
          "pixart_sigma", "hunyuan_dit"]


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    n_by_place = {p["place_id"]: len(p["image_paths"]) for p in bench["places"]}

    df = pd.read_csv(config.OUTPUT_DIR / "eval_v2" / "raw_results.csv")
    df["n_ref_actual"] = df["place_id"].map(n_by_place)
    df = df[df["method"].isin(MODELS)]

    # Baseline ranking (all 86 tiles)
    base = df.groupby("method")["cos_sim"].mean().sort_values(ascending=False)
    base_ranks = base.rank(ascending=False).astype(int)

    print(f"Full (n={df['place_id'].nunique()} tiles) ranking:")
    print(base.round(4).to_string(), "\n")

    # Try thresholds
    results = []
    for k in [6, 8, 10, 12, 14]:
        sub = df[df["n_ref_actual"] >= k]
        n_tiles = sub["place_id"].nunique()
        n_cities = sub["city"].nunique()
        if len(sub) == 0:
            continue
        ranking = sub.groupby("method")["cos_sim"].mean().sort_values(
            ascending=False)
        r_ranks = ranking.rank(ascending=False).astype(int)
        rho, p = spearmanr(base_ranks.loc[MODELS],
                            r_ranks.loc[MODELS])
        top_same = ranking.idxmax() == base.idxmax()
        bot_same = ranking.idxmin() == base.idxmin()
        results.append({
            "min_panel_size": k,
            "n_tiles_kept": n_tiles,
            "n_cities_kept": n_cities,
            "spearman_vs_full": rho,
            "top_model_same": top_same,
            "bottom_model_same": bot_same,
            "top_model": ranking.idxmax(),
            "bot_model": ranking.idxmin(),
        })
        print(f"min panel size {k}:  kept {n_tiles} tiles / {n_cities} cities  "
              f"ρ={rho:.3f}  top={ranking.idxmax()}  bot={ranking.idxmin()}")

    out_dir = config.OUTPUT_DIR / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "panel_size_sufficiency.csv",
                                  index=False)


if __name__ == "__main__":
    main()
