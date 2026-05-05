"""
Metric-human correlation analysis for GeoFidelity-Bench v3.

This script correlates automatic metric differences with human choices
on `model_pair` trials. It is robust to multiple CSV exports in the
ratings directory, de-duplicates repeated exports from the same rater,
and aggregates judgments at the trial level before computing Spearman
correlations.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config


METRICS_HIGHER_BETTER = ("cos_sim", "retrieval_acc", "mrr")
METRICS_LOWER_BETTER = ("dcsf", "mmd", "gaas")


def _load_eval(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep = ["method", "level", "city"] + [
        c for c in (*METRICS_HIGHER_BETTER, *METRICS_LOWER_BETTER)
        if c in df.columns
    ]
    df = df[keep].copy()
    return df.groupby(["method", "level", "city"]).mean().reset_index()


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


def _load_trials(trials_path: Path) -> dict[int, dict]:
    with open(trials_path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    return {t["trial_id"]: t for t in trials}


def _model_pair_votes(ratings: pd.DataFrame,
                      trials: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for _, row in ratings.iterrows():
        trial = trials.get(int(row["trial_id"]))
        if not trial or trial["type"] != "model_pair":
            continue
        rows.append({
            "trial_id": int(row["trial_id"]),
            "choice_mean": float(row["choice"]),
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.groupby("trial_id", as_index=False).agg(
        choice_mean=("choice_mean", "mean"),
        n_ratings=("choice_mean", "size"),
    )


def correlate(eval_df: pd.DataFrame, votes: pd.DataFrame,
              trials: dict[int, dict]) -> dict[str, tuple[float, float, int]]:
    """Return {metric: (rho, p, n_trials)}."""
    by_ml_city = eval_df.set_index(["method", "level", "city"])
    by_m_city = eval_df.groupby(["method", "city"]).mean(numeric_only=True)
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for _, vote in votes.iterrows():
        tid = int(vote["trial_id"])
        meta = trials[tid]["meta"]
        m_a, m_b = meta["model_A"], meta["model_B"]
        city = trials[tid]["target_city"]
        level = meta.get("level")

        def _lookup(method: str, metric: str) -> float:
            if level and (method, level, city) in by_ml_city.index:
                return float(by_ml_city.loc[(method, level, city), metric])
            if (method, city) in by_m_city.index:
                return float(by_m_city.loc[(method, city), metric])
            return float("nan")

        for metric in (*METRICS_HIGHER_BETTER, *METRICS_LOWER_BETTER):
            if metric not in eval_df.columns:
                continue
            v_a = _lookup(m_a, metric)
            v_b = _lookup(m_b, metric)
            if np.isnan(v_a) or np.isnan(v_b):
                continue
            diff = v_b - v_a
            if metric in METRICS_LOWER_BETTER:
                diff = -diff
            out[metric].append((float(vote["choice_mean"]), diff))

    result: dict[str, tuple[float, float, int]] = {}
    for metric, pairs in out.items():
        if len(pairs) < 10:
            result[metric] = (float("nan"), float("nan"), len(pairs))
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        rho, p = spearmanr(x, y)
        result[metric] = (float(rho), float(p), len(pairs))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_csv",
        default=str(config.OUTPUT_DIR / "eval_v3" / "raw_results.csv"),
    )
    ap.add_argument(
        "--trials",
        default=str(config.OUTPUT_DIR / "human_eval" / "trials.json"),
    )
    ap.add_argument(
        "--ratings_dir",
        default=str(config.OUTPUT_DIR / "human_eval" / "ratings"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(config.OUTPUT_DIR / "eval_v3"),
    )
    args = ap.parse_args()

    eval_path = Path(args.eval_csv)
    trials_path = Path(args.trials)
    ratings_dir = Path(args.ratings_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, label in [(eval_path, "eval CSV"), (trials_path, "trials.json")]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    eval_df = _load_eval(eval_path)
    ratings = _load_ratings(ratings_dir)
    if ratings.empty:
        print(
            "No ratings found yet; generate trials with eval/human_eval.py "
            f"then collect rater CSVs into {ratings_dir}"
        )
        return
    trials = _load_trials(trials_path)
    votes = _model_pair_votes(ratings, trials)
    if votes.empty:
        print("No answered model_pair trials found in the ratings archive.")
        return

    corr = correlate(eval_df, votes, trials)
    n_unique_raters = int(ratings["rater_id"].nunique())
    n_ratings = int(
        ratings[ratings["trial_id"].isin(votes["trial_id"])]["choice"].shape[0]
    )
    mean_raters = float(votes["n_ratings"].mean())

    rows = []
    print("\nMetric vs human model-pair preference (Spearman rho):")
    print(f"{'metric':15s} {'rho':>8s} {'p':>8s} {'n':>6s}")
    for metric, (rho, p, n) in sorted(corr.items()):
        print(f"{metric:15s} {rho:8.3f} {p:8.3g} {n:6d}")
        rows.append({
            "metric": metric,
            "spearman_rho": rho,
            "p_value": p,
            "n_trials": n,
            "n_ratings": n_ratings,
            "n_unique_raters": n_unique_raters,
            "mean_raters_per_trial": mean_raters,
        })
    pd.DataFrame(rows).to_csv(out_dir / "metric_human_correlation.csv", index=False)
    print(
        f"\narchive summary: {n_unique_raters} unique raters, "
        f"{n_ratings} model-pair ratings, {len(votes)} answered model-pair trials"
    )
    print(f"\nsaved: {out_dir / 'metric_human_correlation.csv'}")


if __name__ == "__main__":
    main()
