"""
Batch-generate GeoFidelity-Bench v2 images with 7 open-source T2I models.

Usage:
    # run one model over the whole benchmark
    python generation/run_generation.py --model sdxl_base

    # run a subset of models / cities
    python generation/run_generation.py --model flux_schnell --cities tokyo paris

    # resume; skip finished (model, place, k) tuples
    python generation/run_generation.py --model flux_dev --resume

Outputs:
    generations/{model_name}/{place_id}/{k:02d}.jpg
    generations/{model_name}/manifest.csv        (model, place_id, k, path, seed)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import time
from dataclasses import asdict

import pandas as pd
import torch
from tqdm import tqdm

import config
from generation.registry import MODEL_REGISTRY, load_generator


def _seed_for(model: str, place_id: str, k: int) -> int:
    """Deterministic 64-bit seed per (model, place, k)."""
    h = hashlib.sha256(f"{model}/{place_id}/{k}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _prompt_for(city: str, country: str) -> str:
    return config.PROMPT_TEMPLATE.format(
        city=city.replace("_", " ").title(),
        country=country,
    )


def _load_places(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["places"]


def _manifest_path(model: str) -> Path:
    return config.GEN_DIR / model / "manifest.csv"


def _load_manifest(model: str) -> pd.DataFrame:
    p = _manifest_path(model)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["model", "place_id", "k", "path", "seed",
                                  "prompt"])


def _append_manifest(model: str, rows: list[dict]) -> None:
    if not rows:
        return
    p = _manifest_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(
            subset=["model", "place_id", "k"], keep="last")
    new.to_csv(p, index=False)


def run_one_model(model: str, places: list[dict], k_per_place: int,
                  device: str, resume: bool) -> None:
    existing = _load_manifest(model)
    done = set(zip(existing["place_id"], existing["k"])) if len(existing) else set()
    tasks = []
    for place in places:
        for k in range(k_per_place):
            if resume and (place["place_id"], k) in done:
                continue
            tasks.append((place, k))
    if not tasks:
        print(f"[{model}] nothing to do (all done)")
        return

    print(f"[{model}] {len(tasks)} generations pending")
    gen = load_generator(model, device=device)
    out_root = config.GEN_DIR / model
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    try:
        pbar = tqdm(tasks, desc=model)
        for place, k in pbar:
            tile_dir = out_root / place["place_id"]
            tile_dir.mkdir(parents=True, exist_ok=True)
            path = tile_dir / f"{k:02d}.jpg"
            seed = _seed_for(model, place["place_id"], k)
            prompt = _prompt_for(place["city"], place["country"])
            t0 = time.time()
            img = gen.generate(prompt=prompt, seed=seed)
            img.save(path, quality=95)
            pbar.set_postfix({"s/img": f"{time.time()-t0:.1f}"})
            rows.append({
                "model": model,
                "place_id": place["place_id"],
                "k": k,
                "path": str(path.relative_to(config.ROOT).as_posix()),
                "seed": seed,
                "prompt": prompt,
            })
            if len(rows) % 50 == 0:
                _append_manifest(model, rows)
                rows = []
        _append_manifest(model, rows)
    finally:
        gen.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark",
                    default=str(config.PROCESSED_DIR / "benchmark_v2.json"))
    ap.add_argument("--model", action="append", dest="models",
                    help=f"Model name. Repeat to run several. "
                         f"Defaults to all open-source: {list(MODEL_REGISTRY.keys())}")
    ap.add_argument("--cities", nargs="*", default=None,
                    help="Optional subset of target cities")
    ap.add_argument("--k", type=int, default=config.GEN_IMAGES_PER_TILE)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--resume", action="store_true",
                    help="Skip entries already in each model's manifest.csv")
    args = ap.parse_args()

    models = args.models or list(MODEL_REGISTRY.keys())
    places = _load_places(Path(args.benchmark))
    if args.cities:
        want = set(args.cities)
        places = [p for p in places if p["city"] in want]
    print(f"[run_gen] {len(places)} places x {args.k} gens x "
          f"{len(models)} models = {len(places)*args.k*len(models)} images")

    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"!! unknown model {m}, skipping")
            continue
        run_one_model(m, places, args.k, args.device, args.resume)

    print("[run_gen] done")


if __name__ == "__main__":
    main()
