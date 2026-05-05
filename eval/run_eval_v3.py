"""
GeoFidelity-Bench v3 evaluator.

Key differences from v2's `run_eval.py`:
  * Block units (not place units): one entry per OSM-way block.
  * 5-way panel retrieval over {same_block,
    same_neighborhood_diff_block, same_city_diff_neighborhood,
    same_driving_side_diff_city, random_city}. v2's 4-way version
    saturated at 1.00 for every method (incl. Random-Global); widening
    the same-city layer into "same hood" vs "diff hood" breaks that
    saturation.
  * Prompt-level breakdown: each block carries L0/L1/L2 generations
    separately; the evaluator reports metrics per level and the
    paired difference (L1-L0, L2-L1) as the new
    "does block-level conditioning help?" headline result.

Inputs:
  data/processed/v3/benchmark_v3.json
  generations_v3/{method}/{level}/{block_id}/*.jpg
  data/processed/v3/tier5_quality.csv  (feeds the GAAS cache if present)

Outputs:
  outputs/eval_v3/raw_results.csv          per (method, block, level)
  outputs/eval_v3/summary_by_method.csv
  outputs/eval_v3/summary_by_level.csv
  outputs/eval_v3/summary_by_city.csv
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


def _image_paths_of(block: dict) -> list[str]:
    return [img["image_path"] for img in block["images"]]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class BlockEvaluator:
    def __init__(
        self,
        bench_path: Path,
        tier_csv: Path | None,
        device: str = config.DEVICE,
        ref_holdout: int = 4,
    ):
        self.bench = _load_bench(bench_path)
        self.blocks = {b["block_id"]: b for b in self.bench["blocks"]}
        self.cities: dict[str, list[dict]] = defaultdict(list)
        for b in self.bench["blocks"]:
            self.cities[b["city"]].append(b)
        self.ref_holdout = ref_holdout
        self.gaas = GeoAttributeAgreementScore(tier4_csv=tier_csv, device=device)
        self.retriever = PanelRetriever(device=device)
        self._emb_cache: dict[str, np.ndarray] = {}

    # ------------------------- paths / images -------------------------------
    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else config.ROOT / rel

    def _ref_split(
        self,
        block: dict,
        seed: int = 0,
    ) -> tuple[list[str], list[str]]:
        paths = list(_image_paths_of(block))
        rng = random.Random(_stable_seed(seed, block["block_id"]))
        rng.shuffle(paths)
        k = max(1, min(self.ref_holdout, len(paths) // 2))
        return paths[:k], paths[k:]

    def _gen_paths(self, method: str, level: str, block_id: str) -> list[str]:
        pdir = config.V3_GEN_DIR / method / level / block_id
        if not pdir.exists():
            return []
        return [str(p.relative_to(config.ROOT).as_posix()) for p in sorted(pdir.glob("*.jpg"))]

    # ------------------------- retrieval baselines --------------------------
    def _all_ref_paths(self) -> list[str]:
        out = []
        for b in self.bench["blocks"]:
            out.extend(_image_paths_of(b))
        return out

    def _make_baseline_paths(
        self,
        method: str,
        block: dict,
        k: int = None,
        seed: int = 0,
    ) -> list[str]:
        k = k or config.V3_GEN_IMAGES_PER_BLOCK
        rng = random.Random(_stable_seed(seed, block["block_id"], method))

        if method == "oracle_nn":
            oracle, _ = self._ref_split(block, seed=seed)
            return (oracle * ((k // len(oracle)) + 1))[:k]

        if method == "random_global":
            pool = self._all_ref_paths()
            return rng.sample(pool, k=min(k, len(pool)))

        if method == "random_same_country":
            country = block["country"]
            pool = [
                rp
                for b in self.bench["blocks"]
                if b["country"] == country and b["block_id"] != block["block_id"]
                for rp in _image_paths_of(b)
            ]
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
    NEG_KEYS = (
        "neg_same_neighborhood_diff_block",
        "neg_same_city_diff_neighborhood",
        "neg_same_driving_side_diff_city",
        "neg_random_city",
    )

    def score(self, gen_rel: list[str], ref_rel: list[str], neg_ref_rels: list[list[str]]) -> dict:
        if not gen_rel or not ref_rel:
            return {}
        gen_e = self._embed(gen_rel)
        ref_e = self._embed(ref_rel)

        gm = gen_e.mean(axis=0)
        rm = ref_e.mean(axis=0)
        cos = float(gm @ rm / (np.linalg.norm(gm) * np.linalg.norm(rm) + 1e-8))
        dcsf = diversity_calibrated_set_fidelity(gen_e, ref_e)

        panels = [ref_e] + [self._embed(n) for n in neg_ref_rels if n]
        panel_means = np.stack([p.mean(axis=0) for p in panels])
        panel_norm = panel_means / (
            np.linalg.norm(panel_means, axis=1, keepdims=True) + 1e-8
        )
        gen_norm = gen_e / (np.linalg.norm(gen_e, axis=1, keepdims=True) + 1e-8)
        sims = gen_norm @ panel_norm.T  # [k, 1+Nneg]
        ranks = (-sims).argsort(axis=1)
        target_ranks = (ranks == 0).argmax(axis=1)
        retrieval_acc = float((target_ranks == 0).mean())
        mrr = float((1.0 / (target_ranks + 1)).mean())

        try:
            gaas = self.gaas.evaluate_place(
                [self._abs(p) for p in gen_rel],
                [self._abs(p) for p in ref_rel],
            )["overall"]
        except Exception:
            gaas = float("nan")

        out = {
            "cos_sim": cos,
            "dcsf": dcsf["dcsf"],
            "mmd": dcsf["mmd"],
            "diversity_gap": dcsf["diversity_gap"],
            "gaas": gaas,
            "retrieval_acc": retrieval_acc,
            "mrr": mrr,
        }

        mean_sims = sims.mean(axis=0)
        for i, key in enumerate(["same_block"] + list(self.NEG_KEYS)):
            if i < mean_sims.shape[0]:
                out[f"sim_{key}"] = float(mean_sims[i])
        return out

    # --------------------- per-block / per-method orchestrator --------------
    def evaluate_block(self, method: str, level: str, block: dict) -> dict:
        _, gallery = self._ref_split(block)
        if method in {"oracle_nn", "random_global", "random_same_country"}:
            gen_rel = self._make_baseline_paths(method, block)
        else:
            gen_rel = self._gen_paths(method, level, block["block_id"])

        neg_rels = []
        for key in self.NEG_KEYS:
            nid = block.get(key)
            if nid and nid in self.blocks:
                _, nrel = self._ref_split(self.blocks[nid])
                neg_rels.append(nrel)

        out = {
            "method": method,
            "level": level,
            "block_id": block["block_id"],
            "city": block["city"],
            "country": block["country"],
            "stratum": block["stratum"],
            "neighborhood": block["neighborhood"],
            "street_name": block["street_name"],
            "n_gen": len(gen_rel),
            "n_ref": len(gallery),
        }
        out.update(self.score(gen_rel, gallery, neg_rels))
        return out

    def evaluate_method(
        self,
        method: str,
        blocks: list[dict],
        levels: list[str],
        resume: pd.DataFrame | None = None,
    ) -> list[dict]:
        done = set()
        if resume is not None and len(resume):
            done = set(zip(resume["method"], resume["level"], resume["block_id"]))
        rows: list[dict] = []
        pairs = [(lvl, b) for b in blocks for lvl in levels]
        for lvl, b in tqdm(pairs, desc=method):
            if (method, lvl, b["block_id"]) in done:
                continue
            # For retrieval baselines, level is a no-op: evaluate once only.
            if method in {"oracle_nn", "random_global", "random_same_country"} and lvl != levels[0]:
                continue
            try:
                rows.append(self.evaluate_block(method, lvl, b))
            except Exception as e:
                block_id = str(b["block_id"]).encode(
                    "ascii",
                    errors="backslashreplace",
                ).decode("ascii")
                print(f"  {method}/{lvl}/{block_id}: {e}")
        return rows


# --------------------------------- Reporting -------------------------------


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    num = df.select_dtypes(include=[np.number]).columns
    by_method = df.groupby("method")[num].agg(["mean", "std"]).round(4)
    by_method.to_csv(out_dir / "summary_by_method.csv")

    by_ml = df.groupby(["method", "level"])[num].mean().round(4)
    by_ml.to_csv(out_dir / "summary_by_method_level.csv")

    by_city = df.groupby(["city", "method"])[num].mean().round(4)
    by_city.to_csv(out_dir / "summary_by_city.csv")

    by_lvl = df.groupby("level")[num].mean().round(4)
    by_lvl.to_csv(out_dir / "summary_by_level.csv")

    print("\n[by method]")
    print(by_method.to_string())
    print("\n[by level]")
    print(by_lvl.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(config.V3_BENCHMARK_JSON))
    ap.add_argument(
        "--tier_csv",
        default=str(config.V3_PROCESSED_DIR / "tier5_quality.csv"),
    )
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", default=config.V3_PROMPT_LEVELS_MAIN)
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--out_dir", default=str(config.OUTPUT_DIR / "eval_v3"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_results.csv"
    tier_csv = Path(args.tier_csv) if Path(args.tier_csv).exists() else None

    ev = BlockEvaluator(Path(args.benchmark), tier_csv)
    blocks = list(ev.blocks.values())
    if args.cities:
        want = set(args.cities)
        blocks = [b for b in blocks if b["city"] in want]

    methods = args.methods
    if not methods:
        methods = ["oracle_nn", "random_global", "random_same_country"]
        methods.extend(config.OPEN_SOURCE_MODELS)
    print(f"[eval_v3] methods: {methods}  levels: {args.levels}  blocks: {len(blocks)}")

    resume_df = pd.read_csv(csv_path) if (args.resume and csv_path.exists()) else None
    all_rows: list[dict] = []
    for m in methods:
        rows = ev.evaluate_method(m, blocks, args.levels, resume=resume_df)
        all_rows.extend(rows)
        if all_rows:
            merged = (
                pd.concat([resume_df, pd.DataFrame(all_rows)], ignore_index=True)
                if resume_df is not None
                else pd.DataFrame(all_rows)
            )
            merged.to_csv(csv_path, index=False)

    final = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    summarize(final, out_dir)
    print(f"\n[eval_v3] saved to {out_dir}")


if __name__ == "__main__":
    main()
