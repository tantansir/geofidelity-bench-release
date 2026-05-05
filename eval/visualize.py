"""
Visualization for GeoFidelity-Bench results.
Generates paper-ready figures and analysis plots.
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path

import config


def set_paper_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    sns.set_palette("Set2")


def plot_method_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing methods across all metrics."""
    set_paper_style()

    metrics = ["gaas", "retrieval_acc", "mrr", "dcsf"]
    metric_labels = {
        "gaas": "GAAS (↓)",
        "retrieval_acc": "Retrieval Acc (↑)",
        "mrr": "MRR (↑)",
        "dcsf": "DCSF (↓)",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, metric in zip(axes, metrics):
        method_means = df.groupby("method")[metric].mean().sort_values()
        colors = ["#e74c3c" if "random" in m or "medoid" in m else
                  "#2ecc71" if "oracle" in m else "#3498db"
                  for m in method_means.index]

        method_means.plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel(metric_labels.get(metric, metric))
        ax.set_ylabel("")
        ax.set_title(metric_labels.get(metric, metric))

    plt.tight_layout()
    fig.savefig(str(output_dir / "method_comparison.pdf"))
    fig.savefig(str(output_dir / "method_comparison.png"))
    plt.close()
    print(f"Saved method comparison to {output_dir}")


def plot_city_heatmap(df: pd.DataFrame, metric: str, output_dir: Path):
    """Heatmap showing per-city, per-method performance."""
    set_paper_style()

    pivot = df.pivot_table(index="city", columns="method", values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                     max(6, len(pivot.index) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r" if "gaas" in metric or "dcsf" in metric else "RdYlGn",
                ax=ax, linewidths=0.5)
    ax.set_title(f"{metric} by City and Method")
    ax.set_ylabel("City")
    ax.set_xlabel("Method")

    fig.savefig(str(output_dir / f"heatmap_{metric}.pdf"))
    fig.savefig(str(output_dir / f"heatmap_{metric}.png"))
    plt.close()


def plot_metric_correlation(df: pd.DataFrame, output_dir: Path):
    """Correlation matrix between all metrics."""
    set_paper_style()

    metrics = ["gaas", "retrieval_acc", "mrr", "sim_gap", "dcsf", "mmd"]
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]

    if len(available) < 2:
        print("Not enough metrics for correlation plot.")
        return

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, vmin=-1, vmax=1)
    ax.set_title("Inter-Metric Correlation")

    fig.savefig(str(output_dir / "metric_correlation.pdf"))
    fig.savefig(str(output_dir / "metric_correlation.png"))
    plt.close()


def plot_negative_degradation(results_with_negatives: dict, output_dir: Path):
    """Show that metrics degrade correctly across negative types.

    Expected order: same_place > same_city_wrong_nbhd > same_climate_wrong_city > random
    """
    set_paper_style()

    if not results_with_negatives:
        print("No negative degradation data available.")
        return

    categories = ["Same Place\n(Oracle)", "Same City\nWrong Nbhd",
                   "Same Climate\nWrong City", "Random\nCity"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (metric, label) in zip(axes, [
        ("retrieval_sim", "Mean Cosine Similarity (↑)"),
        ("gaas", "GAAS Score (↓)"),
        ("dcsf", "DCSF Score (↓)"),
    ]):
        if metric in results_with_negatives:
            values = results_with_negatives[metric]
            ax.bar(categories[:len(values)], values,
                   color=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"])
            ax.set_ylabel(label)
            ax.set_title(f"Metric Validity: {metric}")

    plt.tight_layout()
    fig.savefig(str(output_dir / "negative_degradation.pdf"))
    fig.savefig(str(output_dir / "negative_degradation.png"))
    plt.close()


def plot_per_city_radar(df: pd.DataFrame, method: str, output_dir: Path):
    """Radar chart showing per-attribute-group GAAS for a method."""
    set_paper_style()

    gaas_cols = [c for c in df.columns if c.startswith("gaas_") and c != "gaas"]
    if not gaas_cols:
        return

    method_df = df[df["method"] == method]
    if method_df.empty:
        return

    means = method_df[gaas_cols].mean()
    labels = [c.replace("gaas_", "").replace("_", " ").title() for c in gaas_cols]

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = means.values.tolist()

    # Close the polygon
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"GAAS Breakdown: {method}")

    fig.savefig(str(output_dir / f"radar_{method}.pdf"))
    fig.savefig(str(output_dir / f"radar_{method}.png"))
    plt.close()


def generate_all_figures(results_csv: Path, output_dir: Path):
    """Generate all paper figures from results CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(str(results_csv))

    print("Generating figures...")
    plot_method_comparison(df, output_dir)
    plot_metric_correlation(df, output_dir)

    for metric in ["gaas", "retrieval_acc", "dcsf"]:
        if metric in df.columns:
            plot_city_heatmap(df, metric, output_dir)

    for method in df["method"].unique():
        plot_per_city_radar(df, method, output_dir)

    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(config.OUTPUT_DIR / "figures"))
    args = parser.parse_args()

    generate_all_figures(Path(args.results), Path(args.output))
