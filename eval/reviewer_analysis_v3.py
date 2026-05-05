"""
Review-response analyses for GeoFidelity-Bench v3.

This script turns the main reviewer requests into reproducible numbers:
  * paired prompt-level deltas with bootstrap CIs and Wilcoxon tests
  * city-balanced deltas to check coverage sensitivity
  * same-block vs same-neighborhood hierarchy gaps
  * city-bootstrap ranking stability at L1
  * human-study archive summary from the released CSVs
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config


MODELS = [
    "sdxl_base",
    "sd35_large",
    "flux_dev",
    "flux_schnell",
    "pixart_sigma",
    "hunyuan_dit",
]
LEVELS = ["L0", "L1", "L2"]


def _normalize_levels(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                       n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = draws.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _load_ratings(ratings_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(ratings_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"trial_id", "choice"}.issubset(df.columns):
            continue
        if "rater_id" not in df.columns:
            df["rater_id"] = path.stem
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["trial_id", "choice"]).copy()
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    df["choice"] = pd.to_numeric(df["choice"], errors="coerce")
    df = df.dropna(subset=["trial_id", "choice"]).copy()
    df["trial_id"] = df["trial_id"].astype(int)
    df = df.sort_values(["trial_id", "rater_id", "source_file"])
    return df.drop_duplicates(subset=["trial_id", "rater_id"], keep="last")


def prompt_delta_stats(df: pd.DataFrame, out_dir: Path,
                       rng: np.random.Generator) -> pd.DataFrame:
    pivot = df.pivot_table(index=["method", "block_id", "city"],
                           columns="level", values="cos_sim")
    rows = []
    for low, high in [("L0", "L1"), ("L1", "L2"), ("L0", "L2")]:
        delta = (pivot[high] - pivot[low]).dropna().to_numpy()
        lo, hi = _bootstrap_mean_ci(delta, rng)
        try:
            p_val = float(wilcoxon(delta).pvalue)
        except ValueError:
            p_val = float("nan")
        rows.append({
            "comparison": f"{high}-{low}",
            "unit": "matched_model_block",
            "n_pairs": int(delta.size),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
            "wilcoxon_p": p_val,
            "positive_fraction": float((delta > 0).mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "prompt_delta_stats.csv", index=False)
    return out


def city_balanced_stats(df: pd.DataFrame, out_dir: Path,
                        rng: np.random.Generator) -> pd.DataFrame:
    city_level = df.groupby(["city", "level"])["cos_sim"].mean().unstack("level")
    rows = []
    for low, high in [("L0", "L1"), ("L1", "L2")]:
        delta = (city_level[high] - city_level[low]).dropna().to_numpy()
        lo, hi = _bootstrap_mean_ci(delta, rng)
        rows.append({
            "comparison": f"{high}-{low}",
            "unit": "city",
            "n_cities": int(delta.size),
            "mean_delta": float(delta.mean()),
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
            "positive_count": int((delta > 0).sum()),
            "negative_or_zero_count": int((delta <= 0).sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "city_balanced_deltas.csv", index=False)
    return out


def hierarchy_stats(df: pd.DataFrame, out_dir: Path,
                    rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    hierarchy_rows = []
    for query_name, sub in [
        ("oracle_nn_l1", df[(df["method"] == "oracle_nn") & (df["level"] == "L1")]),
        ("six_generator_mean_l1", df[(df["method"].isin(MODELS)) & (df["level"] == "L1")]),
    ]:
        hierarchy_rows.append({
            "query_set": query_name,
            "same_block": float(sub["sim_same_block"].mean()),
            "same_neighborhood": float(sub["sim_neg_same_neighborhood_diff_block"].mean()),
            "same_city": float(sub["sim_neg_same_city_diff_neighborhood"].mean()),
            "same_driving_side": float(sub["sim_neg_same_driving_side_diff_city"].mean()),
            "random_city": float(sub["sim_neg_random_city"].mean()),
        })
    hierarchy_df = pd.DataFrame(hierarchy_rows)
    hierarchy_df.to_csv(out_dir / "reference_hierarchy_means.csv", index=False)

    gap_rows = []
    gen_df = df[df["method"].isin(MODELS)].copy()
    for level in LEVELS:
        sub = gen_df[gen_df["level"] == level]
        gap = (sub["sim_same_block"] -
               sub["sim_neg_same_neighborhood_diff_block"]).to_numpy()
        lo, hi = _bootstrap_mean_ci(gap, rng)
        gap_rows.append({
            "level": level,
            "n_rows": int(gap.size),
            "mean_gap": float(gap.mean()),
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
            "same_block_mean": float(sub["sim_same_block"].mean()),
            "same_neighborhood_mean": float(sub["sim_neg_same_neighborhood_diff_block"].mean()),
        })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(out_dir / "same_block_gap_stats.csv", index=False)
    return hierarchy_df, gap_df


def coverage_stats(benchmark_path: Path, out_dir: Path) -> pd.DataFrame:
    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    rows = []
    for place in bench["blocks"]:
        rows.append({
            "city": place["city"],
            "block_id": place["block_id"],
            "n_images": len(place["images"]),
        })
    cov = pd.DataFrame(rows).groupby("city", as_index=False).agg(
        n_blocks=("block_id", "nunique"),
        n_images=("n_images", "sum"),
    )
    cov["image_share"] = cov["n_images"] / cov["n_images"].sum()
    cov["block_share"] = cov["n_blocks"] / cov["n_blocks"].sum()
    cov = cov.sort_values(["n_images", "n_blocks"], ascending=False)
    cov.to_csv(out_dir / "coverage_summary.csv", index=False)
    return cov


def rank_stability(df: pd.DataFrame, out_dir: Path,
                   rng: np.random.Generator,
                   n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_method = (
        df[df["level"] == "L1"]
        .groupby(["city", "method"])["cos_sim"]
        .mean()
        .unstack("method")
        .reindex(columns=MODELS)
    )
    cities = city_method.index.to_numpy()
    rank_records = []
    winner_records = []
    model_ranks = {m: [] for m in MODELS}
    rank1_counts = {m: 0 for m in MODELS}

    for _ in range(n_boot):
        sampled = rng.choice(cities, size=len(cities), replace=True)
        means = city_method.loc[sampled].mean(axis=0)
        ranking = means.rank(ascending=False, method="min")
        top_model = means.idxmax()
        rank1_counts[top_model] += 1
        for model in MODELS:
            model_ranks[model].append(float(ranking[model]))

    for model in MODELS:
        ranks = np.asarray(model_ranks[model], dtype=float)
        lo, hi = np.percentile(ranks, [2.5, 97.5])
        rank_records.append({
            "model": model,
            "mean_rank": float(ranks.mean()),
            "rank_ci_lo": float(lo),
            "rank_ci_hi": float(hi),
            "rank1_probability": float(rank1_counts[model] / n_boot),
        })

    winners = city_method.idxmax(axis=1).value_counts()
    for model in MODELS:
        winner_records.append({
            "model": model,
            "n_city_wins": int(winners.get(model, 0)),
        })

    rank_df = pd.DataFrame(rank_records).sort_values(["mean_rank", "model"])
    win_df = pd.DataFrame(winner_records).sort_values(
        ["n_city_wins", "model"], ascending=[False, True])
    rank_df.to_csv(out_dir / "l1_city_bootstrap_ranks.csv", index=False)
    win_df.to_csv(out_dir / "l1_city_winners.csv", index=False)
    return rank_df, win_df


def human_archive_summary(trials_path: Path, ratings_dir: Path,
                          out_dir: Path) -> pd.DataFrame:
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    ratings = _load_ratings(ratings_dir)
    trial_types = pd.Series([t["type"] for t in trials]).value_counts()
    answered_types = pd.Series(dtype="int64")
    if not ratings.empty:
        tmap = {int(t["trial_id"]): t["type"] for t in trials}
        answered_types = ratings["trial_id"].map(tmap).value_counts()

    rows = [
        {"key": "n_trials_defined", "value": int(len(trials))},
        {"key": "n_unique_raters", "value": int(ratings["rater_id"].nunique()) if not ratings.empty else 0},
        {"key": "n_rating_rows", "value": int(len(ratings))},
        {"key": "n_answered_trials", "value": int(ratings["trial_id"].nunique()) if not ratings.empty else 0},
    ]
    for name in ["within_geo", "model_pair", "real_vs_gen"]:
        rows.append({
            "key": f"defined_{name}",
            "value": int(trial_types.get(name, 0)),
        })
        rows.append({
            "key": f"answered_{name}",
            "value": int(answered_types.get(name, 0)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "human_archive_summary.csv", index=False)
    return out


def main():
    rng = np.random.default_rng(42)
    out_dir = config.OUTPUT_DIR / "eval_v3" / "reviewer_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.OUTPUT_DIR / "eval_v3" / "raw_results.csv")
    df = _normalize_levels(df)
    df = df[df["method"].isin(MODELS + ["oracle_nn"]) & df["level"].isin(LEVELS)].copy()

    prompt_df = prompt_delta_stats(df[df["method"].isin(MODELS)], out_dir, rng)
    city_df = city_balanced_stats(df[df["method"].isin(MODELS)], out_dir, rng)
    hierarchy_df, gap_df = hierarchy_stats(df, out_dir, rng)
    coverage_df = coverage_stats(config.V3_BENCHMARK_JSON, out_dir)
    ranks_df, winners_df = rank_stability(df[df["method"].isin(MODELS)], out_dir, rng)
    human_df = human_archive_summary(
        config.OUTPUT_DIR / "human_eval" / "trials.json",
        config.OUTPUT_DIR / "human_eval" / "ratings",
        out_dir,
    )

    print("\n[prompt deltas]")
    print(prompt_df.round(4).to_string(index=False))
    print("\n[city-balanced deltas]")
    print(city_df.round(4).to_string(index=False))
    print("\n[hierarchy means]")
    print(hierarchy_df.round(4).to_string(index=False))
    print("\n[same-block gap]")
    print(gap_df.round(4).to_string(index=False))
    print("\n[top coverage cities]")
    print(coverage_df.head(5).round(4).to_string(index=False))
    print("\n[L1 rank stability]")
    print(ranks_df.round(4).to_string(index=False))
    print("\n[per-city winners]")
    print(winners_df.to_string(index=False))
    print("\n[human archive]")
    print(human_df.to_string(index=False))
    print(f"\n[reviewer_analysis_v3] saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
