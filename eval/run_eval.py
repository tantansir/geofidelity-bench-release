"""
GeoFidelity-Bench v2 main evaluation.

Scores a method's per-tile generation set against the curated Mapillary
reference panel using all three benchmark metrics (CosSim, DCSF, GAAS) and
the panel-retrieval accuracy against hard negatives.

Methods supported:
  * Open-source T2I models in generations/{name}/ (seven in the default roster)
  * Retrieval baselines computed on the fly:
      - oracle_nn:    use the tile's own real refs as "generations"
                       (empirical ceiling; holdout-2 split to avoid leakage)
      - random_global: random real images from any city
      - random_same_country: random real images sampled within the same
                              ISO country (a weak non-trivial baseline)

Resumes automatically by skipping (method, place) rows already present in
the output CSV.

Inputs:
  data/processed/benchmark_v2.json
  generations/{method}/{place_id}/*.jpg
  data/processed/tier6_review.csv  (warms the GAAS ratio cache)

Outputs:
  outputs/eval_v2/raw_results.csv            per (method, place) row
  outputs/eval_v2/summary_by_method.csv
  outputs/eval_v2/summary_by_city.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from metrics.geo_attribute import GeoAttributeAgreementScore
from metrics.panel_retrieval import PanelRetriever
from metrics.set_fidelity import diversity_calibrated_set_fidelity


# --------------------------------- Helpers ---------------------------------

def _load_bench(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _city_index(bench: dict) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = defaultdict(list)
    for p in bench["places"]:
        d[p["city"]].append(p)
    return d


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class Evaluator:
    def __init__(self, bench_path: Path, tier6_csv: Path | None,
                 device: str = config.DEVICE, ref_holdout: int = 2):
        self.bench = _load_bench(bench_path)
        self.places = {p["place_id"]: p for p in self.bench["places"]}
        self.cities = _city_index(self.bench)
        self.ref_holdout = ref_holdout    # used for oracle_nn split

        self.gaas = GeoAttributeAgreementScore(tier4_csv=tier6_csv, device=device)
        self.retriever = PanelRetriever(device=device)
        self._emb_cache: dict[str, np.ndarray] = {}

    # ------------------------- paths / images -------------------------------
    def _abs(self, rel: str) -> Path:
        return config.ROOT / rel

    def _ref_split(self, place: dict, seed: int = 0
                   ) -> tuple[list[str], list[str]]:
        """Split reference paths into (oracle, gallery). Seeded on place_id."""
        paths = list(place["image_paths"])
        rng = random.Random(_stable_seed(seed, place["place_id"]))
        rng.shuffle(paths)
        k = max(1, min(self.ref_holdout, len(paths) // 2))
        return paths[:k], paths[k:]

    def _gen_paths(self, method: str, place_id: str) -> list[str]:
        pdir = config.GEN_DIR / method / place_id
        if not pdir.exists():
            return []
        rel = []
        for p in sorted(pdir.glob("*.jpg")):
            rel.append(str(p.relative_to(config.ROOT).as_posix()))
        return rel

    # ------------------------- retrieval baselines --------------------------
    def _all_ref_paths(self) -> list[str]:
        out = []
        for p in self.bench["places"]:
            out.extend(p["image_paths"])
        return out

    def _make_baseline_paths(self, method: str, place: dict,
                              k: int = None, seed: int = 0) -> list[str]:
        k = k or config.GEN_IMAGES_PER_TILE
        rng = random.Random(_stable_seed(seed, place["place_id"], method))

        if method == "oracle_nn":
            # Oracle: use held-out probe of same tile's own refs
            oracle, _ = self._ref_split(place, seed=seed)
            return (oracle * ((k // len(oracle)) + 1))[:k]

        if method == "random_global":
            pool = self._all_ref_paths()
            return rng.sample(pool, k=min(k, len(pool)))

        if method == "random_same_country":
            country = place["country"]
            pool = [rp for p in self.bench["places"]
                    if p["country"] == country and p["place_id"] != place["place_id"]
                    for rp in p["image_paths"]]
            if not pool:
                pool = self._all_ref_paths()
            return rng.sample(pool, k=min(k, len(pool)))

        raise ValueError(f"unknown retrieval baseline: {method}")

    # ----------------------------- embedding --------------------------------
    def _embed(self, rel_paths: list[str]) -> np.ndarray:
        missing = [rp for rp in rel_paths if rp not in self._emb_cache]
        if missing:
            imgs = [Image.open(self._abs(rp)).convert("RGB") for rp in missing]
            feats = self.retriever.encode_batch(imgs)
            for rp, f in zip(missing, feats):
                self._emb_cache[rp] = f
        return np.stack([self._emb_cache[rp] for rp in rel_paths])

    # ----------------------------- core metric ------------------------------
    def score(self, gen_rel: list[str], ref_rel: list[str],
              neg_ref_rels: list[list[str]]) -> dict:
        if not gen_rel or not ref_rel:
            return {}

        gen_e = self._embed(gen_rel)
        ref_e = self._embed(ref_rel)

        # CosSim / DCSF / MMD
        gm = gen_e.mean(axis=0); rm = ref_e.mean(axis=0)
        cos = float(gm @ rm / (np.linalg.norm(gm) * np.linalg.norm(rm) + 1e-8))
        dcsf = diversity_calibrated_set_fidelity(gen_e, ref_e)

        # Panel retrieval acc over {target + negatives}
        panels = [ref_e] + [self._embed(n) for n in neg_ref_rels if n]
        panel_means = np.stack([p.mean(axis=0) for p in panels])
        panel_norm = panel_means / (np.linalg.norm(panel_means, axis=1, keepdims=True) + 1e-8)
        gen_norm = gen_e / (np.linalg.norm(gen_e, axis=1, keepdims=True) + 1e-8)
        sims = gen_norm @ panel_norm.T                 # [k, 1 + Nneg]
        ranks = (-sims).argsort(axis=1)
        target_ranks = np.argmin((ranks != 0).cumsum(axis=1), axis=1)
        retrieval_acc = float((target_ranks == 0).mean())
        mrr = float((1.0 / (target_ranks + 1)).mean())

        # GAAS
        try:
            gaas = self.gaas.evaluate_place(
                [self._abs(p) for p in gen_rel],
                [self._abs(p) for p in ref_rel])["overall"]
        except Exception:
            gaas = float("nan")

        return {
            "cos_sim": cos,
            "dcsf": dcsf["dcsf"], "mmd": dcsf["mmd"],
            "diversity_gap": dcsf["diversity_gap"],
            "gaas": gaas,
            "retrieval_acc": retrieval_acc,
            "mrr": mrr,
        }

    # --------------------- per-place / per-method orchestrator --------------
    def evaluate_place(self, method: str, place: dict) -> dict:
        # Reference: always use tile's real refs
        _, gallery = self._ref_split(place)
        # Build generation set
        if method in {"oracle_nn", "random_global", "random_same_country"}:
            gen_rel = self._make_baseline_paths(method, place)
        else:
            gen_rel = self._gen_paths(method, place["place_id"])
        # Build negatives
        neg_rels = []
        for key in ("neg_same_city", "neg_same_climate", "neg_random"):
            nid = place.get(key)
            if nid and nid in self.places:
                _, nrel = self._ref_split(self.places[nid])
                neg_rels.append(nrel)
        out = {
            "method": method,
            "place_id": place["place_id"],
            "city": place["city"],
            "country": place["country"],
            "n_gen": len(gen_rel),
            "n_ref": len(gallery),
        }
        out.update(self.score(gen_rel, gallery, neg_rels))
        return out

    def evaluate_method(self, method: str, places: list[dict],
                         resume: pd.DataFrame | None = None
                         ) -> list[dict]:
        done = set()
        if resume is not None and len(resume):
            done = set(zip(resume["method"], resume["place_id"]))
        rows: list[dict] = []
        for p in tqdm(places, desc=method):
            if (method, p["place_id"]) in done:
                continue
            try:
                rows.append(self.evaluate_place(method, p))
            except Exception as e:
                print(f"  {method} {p['place_id']}: {e}")
        return rows


# --------------------------------- Reporting --------------------------------

def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    num = df.select_dtypes(include=[np.number]).columns
    g = df.groupby("method")[num].agg(["mean", "std"]).round(4)
    g.to_csv(out_dir / "summary_by_method.csv")
    print("\n", g.to_string())
    city = df.groupby(["city", "method"])[num].mean().round(4)
    city.to_csv(out_dir / "summary_by_city.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark",
                    default=str(config.PROCESSED_DIR / "benchmark_v2.json"))
    ap.add_argument("--tier6",
                    default=str(config.PROCESSED_DIR / "tier6_review.csv"))
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Methods to run; default = all auto-discovered gens "
                         "plus oracle_nn/random_global/random_same_country")
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--out_dir", default=str(config.OUTPUT_DIR / "eval_v2"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_results.csv"

    tier6 = Path(args.tier6) if Path(args.tier6).exists() else None
    ev = Evaluator(Path(args.benchmark), tier6)

    places = list(ev.places.values())
    if args.cities:
        want = set(args.cities); places = [p for p in places if p["city"] in want]

    methods = args.methods
    if not methods:
        methods = ["oracle_nn", "random_global", "random_same_country"]
        if config.GEN_DIR.exists():
            methods.extend(sorted(d.name for d in config.GEN_DIR.iterdir()
                                   if d.is_dir()))
    print(f"[eval] methods: {methods}  places: {len(places)}")

    resume_df = pd.read_csv(csv_path) if (args.resume and csv_path.exists()) else None
    all_rows: list[dict] = []
    for m in methods:
        rows = ev.evaluate_method(m, places, resume=resume_df)
        all_rows.extend(rows)
        # Checkpoint after each method
        if all_rows:
            if resume_df is not None:
                merged = pd.concat([resume_df, pd.DataFrame(all_rows)], ignore_index=True)
            else:
                merged = pd.DataFrame(all_rows)
            merged.to_csv(csv_path, index=False)

    final = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    summarize(final, out_dir)
    print(f"\n[eval] saved to {out_dir}")


if __name__ == "__main__":
    main()
