"""
Block-level generation for GeoFidelity-Bench v3.

Supports the main prompt levels (L0/L1/L2) and prompt-specificity
controls such as wrong-street and shuffled-neighborhood prompts.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import time

import pandas as pd
import torch
from tqdm import tqdm

import config
from generation.registry import MODEL_REGISTRY, load_generator


CONTROL_PROMPT_TEMPLATES = {
    "C_WRONG_STREET": (
        "A street-level photograph taken on {street_name} in the "
        "{neighborhood} district of {city}, {country}. "
        "The image shows a typical street scene with buildings, roads, "
        "and urban environment characteristic of this block. "
        "Photorealistic, daytime, clear weather."
    ),
    "C_SHUFFLED_NEIGHBORHOOD": (
        "A street-level photograph taken on {street_name} in the "
        "{neighborhood} district of {city}, {country}. "
        "The image shows a typical street scene with buildings, roads, "
        "and urban environment characteristic of this block. "
        "Photorealistic, daytime, clear weather."
    ),
    "C_WRONG_STREET_NEIGHBORHOOD": (
        "A street-level photograph taken on {street_name} in the "
        "{neighborhood} district of {city}, {country}. "
        "The image shows a typical street scene with buildings, roads, "
        "and urban environment characteristic of this block. "
        "Photorealistic, daytime, clear weather."
    ),
}

ALL_PROMPT_TEMPLATES = {
    **config.V3_PROMPT_TEMPLATES,
    **CONTROL_PROMPT_TEMPLATES,
}


def _seed_for(model: str, block_id: str, level: str, k: int) -> int:
    h = hashlib.sha256(f"{model}/{block_id}/{level}/{k}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _load_blocks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["blocks"]


def _load_control_specs(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("blocks", {})


def _prompt_payload(block: dict, level: str,
                    control_specs: dict[str, dict]) -> tuple[str, dict]:
    tmpl = ALL_PROMPT_TEMPLATES[level]
    city = block["city"].replace("_", " ").title()
    street_name = block["street_name"]
    neighborhood = block["neighborhood"]
    prompt_meta = {"level": level}

    if level in CONTROL_PROMPT_TEMPLATES:
        spec = control_specs.get(block["block_id"], {}).get(level)
        if not spec:
            raise KeyError(f"missing control assignment for {block['block_id']} @ {level}")
        street_name = spec.get("street_name", street_name)
        neighborhood = spec.get("neighborhood", neighborhood)
        prompt_meta["control"] = spec

    fmt = {
        "city": city,
        "country": block["country"],
        "street_name": street_name,
        "neighborhood": neighborhood,
        "lat": block["centroid"][0],
        "lon": block["centroid"][1],
    }
    return tmpl.format(**fmt), prompt_meta


def _manifest_path(model: str) -> Path:
    return config.V3_GEN_DIR / model / "manifest.csv"


def _load_manifest(model: str) -> pd.DataFrame:
    path = _manifest_path(model)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=["model", "block_id", "level", "k", "path",
                 "seed", "prompt", "prompt_meta"]
    )


def _append_manifest(model: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = _manifest_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(
            subset=["model", "block_id", "level", "k"], keep="last"
        )
    new.to_csv(path, index=False)


def run_one_model(model: str, blocks: list[dict], levels: list[str],
                  k_per_level: int, device: str, resume: bool,
                  control_specs: dict[str, dict]) -> None:
    existing = _load_manifest(model)
    done = set() if existing.empty else set(
        zip(existing["block_id"], existing["level"], existing["k"])
    )

    tasks: list[tuple[dict, str, int]] = []
    for block in blocks:
        for level in levels:
            if level in CONTROL_PROMPT_TEMPLATES and \
               level not in control_specs.get(block["block_id"], {}):
                continue
            for k in range(k_per_level):
                if resume and (block["block_id"], level, k) in done:
                    continue
                tasks.append((block, level, k))
    if not tasks:
        print(f"[{model}] nothing to do (all done)")
        return

    print(
        f"[{model}] {len(tasks)} generations pending "
        f"({len(blocks)} blocks x {len(levels)} levels x {k_per_level} k)"
    )
    gen = load_generator(model, device=device)

    rows: list[dict] = []
    try:
        pbar = tqdm(tasks, desc=model)
        for block, level, k in pbar:
            out_dir = config.V3_GEN_DIR / model / level / block["block_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{k:02d}.jpg"
            seed = _seed_for(model, block["block_id"], level, k)
            prompt, prompt_meta = _prompt_payload(block, level, control_specs)

            if path.exists() and resume:
                rows.append({
                    "model": model,
                    "block_id": block["block_id"],
                    "level": level,
                    "k": k,
                    "path": str(path.relative_to(config.ROOT).as_posix()),
                    "seed": seed,
                    "prompt": prompt,
                    "prompt_meta": json.dumps(prompt_meta, ensure_ascii=False),
                })
                continue

            t0 = time.time()
            img = gen.generate(prompt=prompt, seed=seed)
            img.save(path, quality=95)
            pbar.set_postfix({"s/img": f"{time.time() - t0:.1f}"})
            rows.append({
                "model": model,
                "block_id": block["block_id"],
                "level": level,
                "k": k,
                "path": str(path.relative_to(config.ROOT).as_posix()),
                "seed": seed,
                "prompt": prompt,
                "prompt_meta": json.dumps(prompt_meta, ensure_ascii=False),
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
    ap.add_argument(
        "--benchmark",
        default=str(config.V3_BENCHMARK_JSON),
        help="v3 benchmark JSON (with blocks)",
    )
    ap.add_argument(
        "--model",
        action="append",
        dest="models",
        help=f"Repeat for several. Defaults to all open-source: {list(MODEL_REGISTRY.keys())}",
    )
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument(
        "--levels",
        nargs="*",
        default=config.V3_PROMPT_LEVELS_MAIN,
        help="Prompt levels, including optional controls",
    )
    ap.add_argument(
        "--control_spec",
        default=str(config.V3_PROCESSED_DIR / "prompt_controls_v3.json"),
        help="JSON built by data/build_prompt_controls_v3.py",
    )
    ap.add_argument("--k", type=int, default=config.V3_GEN_IMAGES_PER_BLOCK)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    models = args.models or list(MODEL_REGISTRY.keys())
    blocks = _load_blocks(Path(args.benchmark))
    if args.cities:
        wanted = set(args.cities)
        blocks = [b for b in blocks if b["city"] in wanted]
    control_specs = _load_control_specs(Path(args.control_spec))

    for level in args.levels:
        if level not in ALL_PROMPT_TEMPLATES:
            raise SystemExit(f"unknown prompt level: {level}")
        if level in CONTROL_PROMPT_TEMPLATES and not control_specs:
            raise SystemExit(
                "control levels requested but no control spec found; "
                "run `python data/build_prompt_controls_v3.py` first"
            )

    print(
        f"[run_gen_v3] {len(blocks)} blocks x {len(args.levels)} levels x "
        f"{args.k} k x {len(models)} models = "
        f"{len(blocks) * len(args.levels) * args.k * len(models)} images"
    )

    for model in models:
        if model not in MODEL_REGISTRY:
            print(f"!! unknown model {model}, skipping")
            continue
        try:
            run_one_model(
                model,
                blocks,
                args.levels,
                args.k,
                args.device,
                args.resume,
                control_specs,
            )
        except Exception as exc:
            print(f"[run_gen_v3] MODEL {model} FAILED: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    print("[run_gen_v3] done")


if __name__ == "__main__":
    main()
