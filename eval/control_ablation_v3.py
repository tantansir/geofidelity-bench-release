"""
Analyze v3 prompt-specificity controls against the main L1 condition.

Expected control levels:
  * C_WRONG_STREET
  * C_SHUFFLED_NEIGHBORHOOD
  * C_WRONG_STREET_NEIGHBORHOOD

The script reports both raw paired L1-minus-control deltas and
direction-normalized benefit deltas, where positive means L1 is better
than the control for that metric.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config


CONTROL_LEVELS = [
    "C_WRONG_STREET",
    "C_SHUFFLED_NEIGHBORHOOD",
    "C_WRONG_STREET_NEIGHBORHOOD",
]
MODEL_SET = [
    "sdxl_base",
    "sd35_large",
    "flux_dev",
    "flux_schnell",
    "pixart_sigma",
    "hunyuan_dit",
]
HIGHER_BETTER = {"cos_sim", "retrieval_acc", "mrr"}


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000,
                       seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = draws.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _paired_delta(df: pd.DataFrame, metric: str, control_level: str) -> pd.DataFrame:
    sub = df[df["level"].isin(["L1", control_level])].copy()
    pivot = sub.pivot_table(
        index=["method", "block_id", "city"],
        columns="level",
        values=metric,
    )
    if "L1" not in pivot.columns or control_level not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["L1", control_level])
    if pivot.empty:
        return pd.DataFrame()

    raw_delta = pivot["L1"] - pivot[control_level]
    benefit_delta = raw_delta if metric in HIGHER_BETTER else -raw_delta
    out = pivot.reset_index()[["method", "block_id", "city"]].copy()
    out["raw_delta_l1_minus_control"] = raw_delta.to_numpy()
    out["benefit_delta"] = benefit_delta.to_numpy()
    return out


def summarize(df: pd.DataFrame, metrics: list[str], out_dir: Path) -> None:
    overall_rows = []
    model_rows = []

    for control_level in CONTROL_LEVELS:
        for metric in metrics:
            paired = _paired_delta(df, metric, control_level)
            if paired.empty:
                continue

            raw_values = paired["raw_delta_l1_minus_control"].to_numpy()
            values = paired["benefit_delta"].to_numpy()
            ci_lo, ci_hi = _bootstrap_mean_ci(values)
            try:
                p_val = float(wilcoxon(values).pvalue)
            except ValueError:
                p_val = float("nan")
            overall_rows.append({
                "control_level": control_level,
                "metric": metric,
                "higher_is_better": metric in HIGHER_BETTER,
                "n_pairs": int(values.size),
                "mean_l1_minus_control_raw": float(raw_values.mean()),
                "median_l1_minus_control_raw": float(np.median(raw_values)),
                "mean_l1_benefit": float(values.mean()),
                "median_l1_benefit": float(np.median(values)),
                "benefit_bootstrap_ci_lo": ci_lo,
                "benefit_bootstrap_ci_hi": ci_hi,
                "wilcoxon_p": p_val,
                "positive_benefit_fraction": float((values > 0).mean()),
            })

            for method, sub in paired.groupby("method"):
                raw_vals = sub["raw_delta_l1_minus_control"].to_numpy()
                vals = sub["benefit_delta"].to_numpy()
                ci_lo, ci_hi = _bootstrap_mean_ci(vals)
                try:
                    p_val = float(wilcoxon(vals).pvalue)
                except ValueError:
                    p_val = float("nan")
                model_rows.append({
                    "control_level": control_level,
                    "metric": metric,
                    "method": method,
                    "higher_is_better": metric in HIGHER_BETTER,
                    "n_pairs": int(vals.size),
                    "mean_l1_minus_control_raw": float(raw_vals.mean()),
                    "mean_l1_benefit": float(vals.mean()),
                    "benefit_bootstrap_ci_lo": ci_lo,
                    "benefit_bootstrap_ci_hi": ci_hi,
                    "wilcoxon_p": p_val,
                })

    overall_df = pd.DataFrame(overall_rows)
    model_df = pd.DataFrame(model_rows)
    overall_df.to_csv(out_dir / "control_summary_overall.csv", index=False)
    model_df.to_csv(out_dir / "control_summary_by_model.csv", index=False)

    if overall_df.empty:
        print("[control_ablation_v3] no control levels found in raw_results.csv")
        return

    print("\n[control_ablation_v3 overall]")
    print(overall_df.round(4).to_string(index=False))
    print("\n[control_ablation_v3 by model]")
    print(model_df.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_results",
        default=str(config.OUTPUT_DIR / "eval_v3" / "raw_results.csv"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(config.OUTPUT_DIR / "eval_v3" / "control_ablation"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw_results)
    df = df[df["method"].isin(MODEL_SET)].copy()
    metrics = [
        metric for metric in
        ["cos_sim", "retrieval_acc", "mrr", "dcsf", "mmd", "gaas"]
        if metric in df.columns
    ]
    summarize(df, metrics, out_dir)


if __name__ == "__main__":
    main()
