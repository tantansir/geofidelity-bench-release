"""
Publication figures v2:

A. Per-city box plots of CosSim per model.  One figure per metric
   (CosSim, GAAS), showing per-city distribution across place units
   so reviewers see *where* each model is strong/weak.

B. Pairwise similarity heatmaps (Jang-et-al. Fig. 5 style): one
   25x25 matrix per generator, showing how the generator's outputs
   for city i look relative to city j. If the model captures place
   identity, diagonal (i=j) > off-diagonal.

Saved to paper/figures/*.pdf and *.png.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import torch
from tqdm import tqdm

import config


MODELS = ["sdxl_base", "sd35_large", "flux_dev", "flux_schnell",
          "pixart_sigma", "hunyuan_dit"]
MODEL_LABEL = {
    "sdxl_base": "SDXL",
    "sd35_large": "SD 3.5 L",
    "flux_dev": "FLUX-dev",
    "flux_schnell": "FLUX-sch",
    "pixart_sigma": r"PixArt-$\Sigma$",
    "hunyuan_dit": "HunyuanDiT",
}

# Academic palette: burgundy, teal, gold, slate, olive, navy
COLORS = ["#9B2C2C", "#1F4E4A", "#8A6818", "#3B4252", "#556B2F", "#2B3A55"]


# ============================================================
# A. BOX PLOTS
# ============================================================
def box_plots():
    df = pd.read_csv(config.OUTPUT_DIR / "eval_v2" / "raw_results.csv")
    df = df[df["method"].isin(MODELS)].copy()

    # Order cities by mean CosSim descending (so hard cities are on right)
    order = df.groupby("city")["cos_sim"].mean().sort_values(
        ascending=False).index.tolist()

    out_dir = config.ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, higher_better in [
        ("cos_sim", "CosSim  (↑ better)", True),
        ("gaas",    "GAAS  (↓ better)",   False),
    ]:
        fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
        positions = np.arange(len(order))
        width = 0.12
        n_models = len(MODELS)

        for mi, model in enumerate(MODELS):
            data = []
            for c in order:
                sub = df[(df["city"] == c) & (df["method"] == model)][metric]
                data.append(sub.dropna().values)
            off = (mi - (n_models - 1) / 2) * width
            bp = ax.boxplot(
                data, positions=positions + off, widths=width * 0.85,
                patch_artist=True, showfliers=False, medianprops=dict(
                    color="black", linewidth=0.8),
                whiskerprops=dict(color=COLORS[mi], linewidth=0.6),
                capprops=dict(color=COLORS[mi], linewidth=0.6),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS[mi])
                patch.set_alpha(0.55)
                patch.set_edgecolor(COLORS[mi])
                patch.set_linewidth(0.8)

        ax.set_xticks(positions)
        city_labels = [c.replace("_", " ").title() for c in order]
        ax.set_xticklabels(city_labels, rotation=55, ha="right",
                           fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#CCCCCC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="both", length=3, width=0.6)

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i],
                                  alpha=0.55, edgecolor=COLORS[i])
                   for i in range(n_models)]
        ax.legend(handles, [MODEL_LABEL[m] for m in MODELS],
                  loc="upper center", bbox_to_anchor=(0.5, 1.14),
                  ncol=n_models, frameon=False, fontsize=9,
                  columnspacing=1.2, handlelength=1.2, handleheight=0.9)

        ax.margins(x=0.01)
        plt.tight_layout()

        suffix = "cos_sim" if metric == "cos_sim" else "gaas"
        pdf = out_dir / f"fig5_boxplot_{suffix}.pdf"
        png = out_dir / f"fig5_boxplot_{suffix}.png"
        fig.savefig(str(pdf), dpi=300, bbox_inches="tight")
        fig.savefig(str(png), dpi=200, bbox_inches="tight")
        print(f"Saved {pdf.name}")
        plt.close(fig)


# ============================================================
# B. PAIRWISE HEATMAPS (Jang Fig. 5 style)
# ============================================================
def pairwise_heatmaps():
    # Need DINOv2 embeddings per (model, city). We compute them from
    # the generated images.
    from metrics.panel_retrieval import PanelRetriever
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places_by_city = defaultdict(list)
    for p in bench["places"]:
        places_by_city[p["city"]].append(p)
    cities = sorted(places_by_city.keys())

    retriever = PanelRetriever()

    out_dir = config.ROOT / "paper" / "figures"
    cache_path = config.OUTPUT_DIR / "pairwise_model_embeddings.npz"
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path.name}")
        data = np.load(cache_path, allow_pickle=True)
        model_city_emb = {k: data[k].item() for k in data.files}
    else:
        model_city_emb = {}
        for model in MODELS:
            print(f"\nEncoding generations for {model}...")
            city_emb = {}
            for city in tqdm(cities):
                imgs = []
                for p in places_by_city[city]:
                    pdir = config.GEN_DIR / model / p["place_id"]
                    if not pdir.exists():
                        continue
                    for fp in sorted(pdir.glob("*.jpg")):
                        try:
                            imgs.append(Image.open(fp).convert("RGB"))
                        except Exception:
                            pass
                if len(imgs) >= 2:
                    feats = retriever.encode_batch(imgs)
                    city_emb[city] = feats.mean(axis=0)
            model_city_emb[model] = city_emb
        # cache
        np.savez(str(cache_path),
                 **{m: np.array(model_city_emb[m], dtype=object)
                    for m in MODELS})
        print(f"Saved cache {cache_path}")

    # Also compute real-reference pairwise for overlay comparison
    real_city_emb = {}
    print("\nEncoding real references per city...")
    real_cache = config.OUTPUT_DIR / "pairwise_real_embeddings.npz"
    if real_cache.exists():
        real_city_emb = np.load(real_cache, allow_pickle=True)["real"].item()
    else:
        for city in tqdm(cities):
            imgs = []
            for p in places_by_city[city]:
                for rp in p["image_paths"]:
                    fp = config.ROOT / rp
                    if fp.exists():
                        try:
                            imgs.append(Image.open(fp).convert("RGB"))
                        except Exception:
                            pass
            if len(imgs) >= 2:
                feats = retriever.encode_batch(imgs)
                real_city_emb[city] = feats.mean(axis=0)
        np.savez(str(real_cache), real=np.array(real_city_emb, dtype=object))

    def cos_matrix(emb_dict, cities):
        n = len(cities)
        M = np.full((n, n), np.nan)
        for i, ci in enumerate(cities):
            if ci not in emb_dict:
                continue
            for j, cj in enumerate(cities):
                if cj not in emb_dict:
                    continue
                a = emb_dict[ci]; b = emb_dict[cj]
                M[i, j] = float(a @ b / (np.linalg.norm(a) *
                                          np.linalg.norm(b) + 1e-8))
        return M

    cmap = LinearSegmentedColormap.from_list(
        "academic",
        [
            "#1F4E4A",  # deep teal (low)
            "#E6E2D8",  # paper
            "#9B2C2C",  # burgundy (high)
        ], N=256)

    # 6 models + 1 real panel = 7 panels. 2x4 grid (last blank),
    # but Jang Fig. 5 uses one big matrix. We do a 2x3 grid of the
    # 6 model heatmaps, plus a separate single real-reference matrix.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), dpi=200)
    axes = axes.flatten()

    # Normalize so shared colorbar range reflects similarity
    all_M = []
    mats = {}
    for model in MODELS:
        M = cos_matrix(model_city_emb[model], cities)
        mats[model] = M
        all_M.extend(M[np.isfinite(M)].tolist())
    lo = float(np.nanpercentile(all_M, 5))
    hi = float(np.nanpercentile(all_M, 95))

    ticklabels = [c.replace("_", " ").title() for c in cities]

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        M = mats[model]
        im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="equal",
                       interpolation="nearest")
        ax.set_title(MODEL_LABEL[model], fontsize=11, pad=6,
                     color="#13161C", fontweight="bold")

        ax.set_xticks(range(len(cities)))
        ax.set_yticks(range(len(cities)))
        ax.set_xticklabels(ticklabels, rotation=90, fontsize=5.5)
        ax.set_yticklabels(ticklabels, fontsize=5.5)
        ax.tick_params(axis="both", length=0, pad=1)

        # Diagonal emphasis
        for i in range(len(cities)):
            ax.plot(i, i, marker="s", markersize=3.2,
                    markerfacecolor="none", markeredgecolor="#13161C",
                    markeredgewidth=0.7)

        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color("#666666")

        # Diagonal - off-diagonal numeric summary in corner
        diag_mean = float(np.nanmean(np.diag(M)))
        off_mask = ~np.eye(len(cities), dtype=bool) & np.isfinite(M)
        off_mean = float(np.nanmean(M[off_mask]))
        gap = diag_mean - off_mean
        ax.text(0.015, 0.98,
                f"diag = {diag_mean:.3f}\noff = {off_mean:.3f}\nΔ = {gap:+.3f}",
                transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                family="monospace", color="#13161C",
                bbox=dict(facecolor="#FBF7F0", edgecolor="#B8AD8A",
                          boxstyle="round,pad=0.25", linewidth=0.5))

    # Shared colorbar on the right
    fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.08,
                        wspace=0.32, hspace=0.55)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("DINOv2 cosine similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Per-generator pairwise city similarity "
        "(25×25, DINOv2 ViT-B/14 mean embeddings)",
        fontsize=12, y=0.985, fontweight="medium", color="#13161C")

    pdf = config.ROOT / "paper" / "figures" / "fig6_pairwise_heatmaps.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(str(pdf), dpi=300, bbox_inches="tight")
    fig.savefig(str(png), dpi=200, bbox_inches="tight")
    print(f"\nSaved {pdf.name}")
    plt.close(fig)

    # Also save a real-reference-only heatmap for appendix comparison
    fig2, ax2 = plt.subplots(figsize=(6.5, 6), dpi=200)
    Mreal = cos_matrix(real_city_emb, cities)
    im2 = ax2.imshow(Mreal, cmap=cmap, vmin=lo, vmax=1.0, aspect="equal",
                     interpolation="nearest")
    ax2.set_title("Real Mapillary references (ground truth)",
                  fontsize=11, pad=8, fontweight="bold")
    ax2.set_xticks(range(len(cities))); ax2.set_yticks(range(len(cities)))
    ax2.set_xticklabels(ticklabels, rotation=90, fontsize=7)
    ax2.set_yticklabels(ticklabels, fontsize=7)
    ax2.tick_params(axis="both", length=0, pad=1)
    for i in range(len(cities)):
        ax2.plot(i, i, marker="s", markersize=4,
                 markerfacecolor="none", markeredgecolor="#13161C",
                 markeredgewidth=0.7)
    for s in ax2.spines.values():
        s.set_linewidth(0.5)
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("DINOv2 cosine", fontsize=9)
    pdf2 = config.ROOT / "paper" / "figures" / "fig6b_pairwise_real.pdf"
    fig2.tight_layout()
    fig2.savefig(str(pdf2), dpi=300, bbox_inches="tight")
    fig2.savefig(str(pdf2.with_suffix(".png")), dpi=200, bbox_inches="tight")
    print(f"Saved {pdf2.name}")
    plt.close(fig2)


if __name__ == "__main__":
    print("=== A. Box plots ===")
    box_plots()
    print("\n=== B. Pairwise heatmaps ===")
    pairwise_heatmaps()
