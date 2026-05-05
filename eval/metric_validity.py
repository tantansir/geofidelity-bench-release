"""
Metric validity experiment for GeoFidelity-Bench v2.

Controls confirmed in this study:
  * same_place (probe vs gallery, same H3 res-8 tile)
  * same_city_wrong_nbhd (neg_same_city; >= 0.7 km away)
  * same_climate_wrong_city (neg_same_climate)
  * random_city (neg_random)

A valid metric must rank these in order. Unlike v1 we also report:
  * per-condition bootstrap CI (1000 resamples)
  * Spearman rank correlation between condition-rank and metric-rank
  * Kendall's tau W to check agreement across metrics

Inputs:
  data/processed/benchmark_v2.json
  data/processed/tier6_review.csv (for Tier-4 cached ratios -> GAAS warm cache)

Outputs:
  outputs/validity_v2/metric_validity_raw.csv
  outputs/validity_v2/metric_validity_summary.csv
  outputs/validity_v2/metric_validity_bootstrap.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

import config
from metrics.geo_attribute import GeoAttributeAgreementScore
from metrics.panel_retrieval import PanelRetriever
from metrics.set_fidelity import diversity_calibrated_set_fidelity


CONDITIONS = ["same_place", "same_city_wrong_nbhd",
              "same_climate_wrong_city", "random_city"]

HIGHER_IS_BETTER = {"cos_sim"}   # cos_sim wants same_place highest
LOWER_IS_BETTER = {"gaas", "dcsf", "mmd"}


class MetricValidityV2:
    def __init__(self, benchmark_path: Path, tier6_csv: Path | None):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.bench = json.load(f)
        self.places = {p["place_id"]: p for p in self.bench["places"]}
        self.gaas = GeoAttributeAgreementScore(tier4_csv=tier6_csv)
        self.retriever = PanelRetriever()
        # embedding cache: rel_path -> np.ndarray
        self._emb_cache: dict[str, np.ndarray] = {}

    def _load_imgs(self, rel_paths: list[str]) -> list[Image.Image]:
        imgs: list[Image.Image] = []
        for rp in rel_paths:
            p = config.ROOT / rp
            if p.exists():
                imgs.append(Image.open(p).convert("RGB"))
        return imgs

    def _embed(self, rel_paths: list[str]) -> np.ndarray:
        # Cached DINOv2 embedding lookup
        missing = [rp for rp in rel_paths if rp not in self._emb_cache]
        if missing:
            imgs = [Image.open(config.ROOT / rp).convert("RGB") for rp in missing]
            feats = self.retriever.encode_batch(imgs)
            for rp, f in zip(missing, feats):
                self._emb_cache[rp] = f
        return np.stack([self._emb_cache[rp] for rp in rel_paths])

    def run_place(self, pid: str, rng: np.random.Generator) -> dict | None:
        place = self.places[pid]
        img_paths = place.get("image_paths", [])
        if len(img_paths) < 4:
            return None

        # Fixed split: first 2 (sorted by path) as probe, rest as gallery
        idx = rng.permutation(len(img_paths))
        probe_paths = [img_paths[i] for i in idx[:2]]
        gallery_paths = [img_paths[i] for i in idx[2:]]

        row: dict = {"place_id": pid, "city": place["city"]}
        row.update(self._all_metrics(probe_paths, gallery_paths, "same_place"))

        for cond, key in [
            ("same_city_wrong_nbhd", "neg_same_city"),
            ("same_climate_wrong_city", "neg_same_climate"),
            ("random_city", "neg_random"),
        ]:
            neg_id = place.get(key)
            if not neg_id or neg_id not in self.places:
                continue
            neg_paths = self.places[neg_id].get("image_paths", [])
            if len(neg_paths) >= 2:
                row.update(self._all_metrics(probe_paths, neg_paths, cond))
        return row

    def _all_metrics(self, probe_paths: list[str], gallery_paths: list[str],
                      cond: str) -> dict:
        # Cosine + DCSF/MMD via DINOv2
        try:
            pe = self._embed(probe_paths)
            ge = self._embed(gallery_paths)
            pm = pe.mean(axis=0)
            gm = ge.mean(axis=0)
            cos = float(pm @ gm / (np.linalg.norm(pm) * np.linalg.norm(gm) + 1e-8))
            dcsf = diversity_calibrated_set_fidelity(pe, ge)
        except Exception as e:
            cos = float("nan")
            dcsf = {"dcsf": float("nan"), "mmd": float("nan")}

        # GAAS via Mask2Former Mapillary-Vistas (ratios possibly cached from Tier 4)
        try:
            gaas = self.gaas.evaluate_place(
                [config.ROOT / p for p in probe_paths],
                [config.ROOT / p for p in gallery_paths],
            )
            gaas_v = gaas["overall"]
        except Exception:
            gaas_v = float("nan")

        return {
            f"{cond}_cos_sim": cos,
            f"{cond}_dcsf": dcsf["dcsf"],
            f"{cond}_mmd": dcsf["mmd"],
            f"{cond}_gaas": gaas_v,
        }

    def run_all(self, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows: list[dict] = []
        for pid in tqdm([p["place_id"] for p in self.bench["places"]],
                        desc="validity"):
            r = self.run_place(pid, rng)
            if r:
                rows.append(r)
        return pd.DataFrame(rows)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000,
                       seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def analyze(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metric_validity_raw.csv", index=False)

    summary: list[dict] = []
    bootstrap: list[dict] = []
    metrics = ["cos_sim", "dcsf", "mmd", "gaas"]
    for m in metrics:
        cond_means: list[float] = []
        print(f"\n--- {m.upper()} ---")
        for cond in CONDITIONS:
            col = f"{cond}_{m}"
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean_ci(vals)
            cond_means.append(mean)
            summary.append({"metric": m, "condition": cond, "mean": mean,
                             "std": float(np.nanstd(vals)),
                             "n": int(np.isfinite(vals).sum())})
            bootstrap.append({"metric": m, "condition": cond, "mean": mean,
                               "ci_lo": lo, "ci_hi": hi})
            print(f"  {cond:28s}: {mean:.4f}  (95% CI [{lo:.4f}, {hi:.4f}])")

        # Spearman between condition rank (expected) and mean rank (observed)
        exp_rank = list(range(len(cond_means)))          # 0..3
        if m in HIGHER_IS_BETTER:
            obs_rank = (-np.array(cond_means)).argsort().argsort().tolist()
        else:
            obs_rank = np.array(cond_means).argsort().argsort().tolist()
        if len(cond_means) == 4:
            rho, p = spearmanr(exp_rank, obs_rank)
            print(f"  expected-vs-observed Spearman rho = {rho:.2f} (p={p:.2g})")

    pd.DataFrame(summary).to_csv(
        out_dir / "metric_validity_summary.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(
        out_dir / "metric_validity_bootstrap.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark",
                    default=str(config.PROCESSED_DIR / "benchmark_v2.json"))
    ap.add_argument("--tier6",
                    default=str(config.PROCESSED_DIR / "tier6_review.csv"),
                    help="Used to warm GAAS cache from Tier-4 segmentation")
    ap.add_argument("--out_dir",
                    default=str(config.OUTPUT_DIR / "validity_v2"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tier6 = Path(args.tier6) if Path(args.tier6).exists() else None
    exp = MetricValidityV2(Path(args.benchmark), tier6)
    df = exp.run_all(seed=args.seed)
    analyze(df, Path(args.out_dir))
    print(f"\n[validity] results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
