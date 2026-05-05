"""
Reuse v2 generations as v3 L0 generations.

The v3 L0 prompt template is identical to v2's `config.PROMPT_TEMPLATE`
(both carry only city + country). That means every v2 generation for
a given city is, by construction, a valid L0 sample for every v3 block
in the same city.

For each (model, v3 block) we copy k random v2 gens from the block's
city into `generations_v3/{model}/L0/{block_id}/{k:02d}.jpg` and append
rows to `generations_v3/{model}/manifest.csv` so the v3 evaluator
treats them as real L0 generations. L1 and L2 levels must still be
produced by `run_generation_v3.py --levels L1 L2`.

This saves the full L0 generation pass (~1/3 of the v3 gen budget) and
also ties v2 and v3 evaluations to literally the same pixels under
city-only conditioning — a useful cross-benchmark calibration point.

Usage:
    python generation/seed_v3_L0_from_v2.py                    # all models
    python generation/seed_v3_L0_from_v2.py --models sdxl_base
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import config


def _seed_for(model: str, block_id: str, level: str, k: int) -> int:
    h = hashlib.sha256(f"{model}/{block_id}/{level}/{k}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _v2_prompt(city: str, country: str) -> str:
    return config.PROMPT_TEMPLATE.format(
        city=city.replace("_", " ").title(), country=country)


def _v2_pool_for_city(city: str, model: str) -> list[Path]:
    model_dir = config.GEN_DIR / model
    if not model_dir.exists():
        return []
    out: list[Path] = []
    # v2 place_ids start with the city name: "{city}__{h3tile}"
    for place_dir in model_dir.iterdir():
        if not place_dir.is_dir():
            continue
        if not place_dir.name.startswith(f"{city}__"):
            continue
        out.extend(sorted(place_dir.glob("*.jpg")))
    return out


def _load_v3_blocks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["blocks"]


def _load_manifest(model: str) -> pd.DataFrame:
    p = config.V3_GEN_DIR / model / "manifest.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["model", "block_id", "level", "k",
                                  "path", "seed", "prompt"])


def _append_manifest(model: str, rows: list[dict]) -> None:
    if not rows:
        return
    p = config.V3_GEN_DIR / model / "manifest.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(
            subset=["model", "block_id", "level", "k"], keep="last")
    new.to_csv(p, index=False)


def seed_one_model(model: str, blocks: list[dict],
                    k_per_block: int, overwrite: bool) -> int:
    pools: dict[str, list[Path]] = {}
    rows: list[dict] = []
    n_copied = 0
    existing = _load_manifest(model)
    done = set() if not len(existing) else set(
        zip(existing["block_id"], existing["level"], existing["k"]))

    for block in tqdm(blocks, desc=f"{model}/L0"):
        city = block["city"]
        if city not in pools:
            pools[city] = _v2_pool_for_city(city, model)
        pool = pools[city]
        if not pool:
            continue
        # Deterministic seed so re-runs pick identical pixels
        rng = random.Random(hashlib.sha256(
            f"{model}/{block['block_id']}/L0".encode()).digest())
        sel = rng.sample(pool, k=min(k_per_block, len(pool))) \
              if len(pool) >= k_per_block \
              else [rng.choice(pool) for _ in range(k_per_block)]

        out_dir = config.V3_GEN_DIR / model / "L0" / block["block_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, src in enumerate(sel):
            dst = out_dir / f"{k:02d}.jpg"
            if (block["block_id"], "L0", k) in done and not overwrite:
                if dst.exists():
                    continue
            shutil.copy2(src, dst)
            n_copied += 1
            rows.append({
                "model": model,
                "block_id": block["block_id"],
                "level": "L0",
                "k": k,
                "path": str(dst.relative_to(config.ROOT).as_posix()),
                "seed": _seed_for(model, block["block_id"], "L0", k),
                "prompt": _v2_prompt(city, block["country"]),
                "source_v2": str(src.relative_to(config.ROOT).as_posix()),
            })
    _append_manifest(model, rows)
    return n_copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(config.V3_BENCHMARK_JSON))
    ap.add_argument("--models", nargs="*", default=None,
                    help="v2 model names to reuse; defaults to every "
                         "subdir under generations/")
    ap.add_argument("--k", type=int, default=config.V3_GEN_IMAGES_PER_BLOCK)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    models = args.models
    if not models:
        models = [d.name for d in config.GEN_DIR.iterdir()
                  if d.is_dir()] if config.GEN_DIR.exists() else []
    if not models:
        print("[seed] no v2 generations found; run `run_generation.py` first")
        return
    blocks = _load_v3_blocks(Path(args.benchmark))
    print(f"[seed] {len(blocks)} v3 blocks x {len(models)} models x k={args.k}")

    total = 0
    for m in models:
        n = seed_one_model(m, blocks, args.k, args.overwrite)
        print(f"  {m}: copied {n} images")
        total += n
    print(f"[seed] total L0 files copied: {total}")


if __name__ == "__main__":
    main()
