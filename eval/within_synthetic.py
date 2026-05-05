"""
Within-synthetic controlled comparison (v2 P4 experiment).

Purpose: defeat the v1 reviewer objection that "DALL-E 3 worse than random
real" is just a real-vs-synthetic distributional gap unrelated to geography.

For every generator M, every target city A, and every swap city B != A:
    matched   = metric( M's gens for A,   real refs for A )   [same M, same A]
    mismatched= metric( M's gens for B,   real refs for A )   [same M, diff A]

If a metric measures geographic fidelity rather than real-vs-synthetic
gap, `matched` should be better than `mismatched` within the same
synthetic population. We report:
  * per-model geographic-sensitivity Delta = mean(mismatched - matched)
    (dist metrics; sign flipped for similarity metrics)
  * paired Wilcoxon p-value over (A, B) pairs
  * bootstrap 95% CI on Delta

Inputs:
  data/processed/benchmark_v2.json
  generations/{model}/{place_id}/*.jpg       (from generation/run_generation.py)

Output:
  outputs/within_synthetic/within_synthetic_{model}.csv
  outputs/within_synthetic/within_synthetic_summary.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from tqdm import tqdm

import config
from metrics.geo_attribute import GeoAttributeAgreementScore
from metrics.panel_retrieval import PanelRetriever
from metrics.set_fidelity import diversity_calibrated_set_fidelity


HIGHER_IS_BETTER = {"cos_sim"}
METRICS = ["cos_sim", "dcsf", "mmd", "gaas"]


class WithinSyntheticExperiment:
    def __init__(self, benchmark_path: Path, tier6_csv: Path | None):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.bench = json.load(f)
        self.places_by_city: dict[str, list[dict]] = defaultdict(list)
        for p in self.bench["places"]:
            self.places_by_city[p["city"]].append(p)
        self.cities = sorted(self.places_by_city.keys())

        self.gaas = GeoAttributeAgreementScore(tier4_csv=tier6_csv)
        self.retriever = PanelRetriever()
        self._emb_cache: dict[str, np.ndarray] = {}

    # ------------------------------- IO ---------------------------------
    def _city_ref_paths(self, city: str, max_per_tile: int = 6) -> list[Path]:
        out: list[Path] = []
        for p in self.places_by_city[city]:
            for rp in p["image_paths"][:max_per_tile]:
                out.append(config.ROOT / rp)
        return out

    def _city_gen_paths(self, city: str, model: str) -> list[Path]:
        out: list[Path] = []
        gen_root = config.GEN_DIR / model
        for p in self.places_by_city[city]:
            pdir = gen_root / p["place_id"]
            if pdir.exists():
                out.extend(sorted(pdir.glob("*.jpg")))
        return out

    # ----------------------------- Embedding ----------------------------
    def _embed(self, paths: list[Path]) -> np.ndarray:
        keys = [str(p) for p in paths]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._emb_cache]
        if missing_idx:
            imgs = [Image.open(paths[i]).convert("RGB") for i in missing_idx]
            feats = self.retriever.encode_batch(imgs)
            for i, f in zip(missing_idx, feats):
                self._emb_cache[keys[i]] = f
        return np.stack([self._emb_cache[k] for k in keys])

    # --------------------------- Per-pair metric ------------------------
    def score_pair(self, gen_paths: list[Path], ref_paths: list[Path]) -> dict:
        if not gen_paths or not ref_paths:
            return {m: float("nan") for m in METRICS}
        ge = self._embed(gen_paths)
        re = self._embed(ref_paths)
        gm = ge.mean(axis=0); rm = re.mean(axis=0)
        cos = float(gm @ rm / (np.linalg.norm(gm) * np.linalg.norm(rm) + 1e-8))
        dcsf = diversity_calibrated_set_fidelity(ge, re)
        try:
            gaas = self.gaas.evaluate_place(gen_paths, ref_paths)["overall"]
        except Exception:
            gaas = float("nan")
        return {"cos_sim": cos, "dcsf": dcsf["dcsf"],
                "mmd": dcsf["mmd"], "gaas": gaas}

    # --------------------------- Main loop ------------------------------
    def run_model(self, model: str, out_dir: Path) -> pd.DataFrame:
        rows: list[dict] = []
        cities_with_gens = [c for c in self.cities
                             if self._city_gen_paths(c, model)]
        if not cities_with_gens:
            print(f"[{model}] no generations found under {config.GEN_DIR / model}")
            return pd.DataFrame()

        for a in tqdm(cities_with_gens, desc=f"{model}: target"):
            ref_a = self._city_ref_paths(a)
            if len(ref_a) < 4:
                continue
            gen_a = self._city_gen_paths(a, model)
            m_match = self.score_pair(gen_a, ref_a)

            for b in cities_with_gens:
                if b == a:
                    continue
                gen_b = self._city_gen_paths(b, model)
                if len(gen_b) < 4:
                    continue
                m_mis = self.score_pair(gen_b, ref_a)
                row = {"model": model, "ref_city": a, "gen_city": b}
                for k in METRICS:
                    row[f"matched_{k}"] = m_match[k]
                    row[f"mismatched_{k}"] = m_mis[k]
                    if k in HIGHER_IS_BETTER:
                        row[f"delta_{k}"] = m_match[k] - m_mis[k]
                    else:
                        row[f"delta_{k}"] = m_mis[k] - m_match[k]
                rows.append(row)

        df = pd.DataFrame(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"within_synthetic_{model}.csv", index=False)
        return df


def summarize(model_dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    summary: list[dict] = []
    for model, df in model_dfs.items():
        if df.empty:
            continue
        for m in METRICS:
            d = df[f"delta_{m}"].dropna().to_numpy()
            if len(d) < 3:
                continue
            # Wilcoxon (paired, one-sided: delta > 0 = metric detects geography)
            try:
                stat, p = wilcoxon(d, alternative="greater")
            except ValueError:
                stat, p = float("nan"), float("nan")
            rng = np.random.default_rng(0)
            boot = rng.choice(d, size=(1000, len(d)), replace=True).mean(axis=1)
            summary.append({
                "model": model,
                "metric": m,
                "mean_delta": float(d.mean()),
                "median_delta": float(np.median(d)),
                "ci95_lo": float(np.percentile(boot, 2.5)),
                "ci95_hi": float(np.percentile(boot, 97.5)),
                "wilcoxon_p": float(p),
                "n_pairs": int(len(d)),
            })
    s = pd.DataFrame(summary)
    out = out_dir / "within_synthetic_summary.csv"
    s.to_csv(out, index=False)
    print("\nGeographic sensitivity (mean delta, positive = metric is geo-aware):")
    if not s.empty:
        print(s.pivot(index="model", columns="metric", values="mean_delta")
               .round(4).to_string())
        print("\nWilcoxon p-values (one-sided delta > 0):")
        print(s.pivot(index="model", columns="metric", values="wilcoxon_p")
               .round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark",
                    default=str(config.PROCESSED_DIR / "benchmark_v2.json"))
    ap.add_argument("--tier6",
                    default=str(config.PROCESSED_DIR / "tier6_review.csv"))
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of models to evaluate; default all found")
    ap.add_argument("--out_dir",
                    default=str(config.OUTPUT_DIR / "within_synthetic"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tier6 = Path(args.tier6) if Path(args.tier6).exists() else None
    exp = WithinSyntheticExperiment(Path(args.benchmark), tier6)

    models = args.models
    if not models:
        models = [d.name for d in config.GEN_DIR.iterdir() if d.is_dir()]
    print(f"[within-synth] models: {models}")

    model_dfs: dict[str, pd.DataFrame] = {}
    for m in models:
        model_dfs[m] = exp.run_model(m, out_dir)
    summarize(model_dfs, out_dir)


if __name__ == "__main__":
    main()
