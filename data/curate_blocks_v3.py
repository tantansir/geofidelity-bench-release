"""
Build GeoFidelity-Bench v3 benchmark JSON from the final curation CSV.

A v3 BlockUnit = one OSM named way (carved by `carve_blocks.py`), its
reference images (with full GPS metadata), and four hard negatives
covering the location-hierarchy:
  * neg_same_neighborhood_diff_block  — block within V3_NEIGHBORHOOD_RADIUS_M
  * neg_same_city_diff_neighborhood   — another block in the same city
                                        but beyond the neighborhood radius
  * neg_same_driving_side_diff_city   — any block in a different city with
                                        the same left/right convention
  * neg_random_city                   — any block in a different city

With the block itself as the positive (same_block), this gives the 5-way
retrieval taxonomy specified in `V3_NEG_LEVELS` — a deliberate upgrade
from v2's 4-way set, whose panel-retrieval score saturated at 1.00 for
all methods (incl. random) on the v2 data.

Input:  tier5 quality CSV (v3), falls back to tier1_candidates.csv if
        curation pipeline has not finished yet.
Output: data/processed/v3/benchmark_v3.json
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import pandas as pd

import config


@dataclass
class ImageRecord:
    image_id: str
    image_path: str
    lat: float
    lon: float
    captured_at_utc_ms: int
    compass_angle: Optional[float]
    sun_elev_deg: float
    h3_r9_cell: str
    camera_type: str
    mapillary_quality: Optional[float]


@dataclass
class BlockUnit:
    block_id: str
    city: str
    country: str
    driving_side: str
    stratum: str
    way_id: str
    street_name: str
    neighborhood: str
    highway_tag: str
    length_m: float
    centroid: list[float]
    bbox: list[float]
    polyline: list[list[float]]
    h3_r9_cells: list[str]
    images: list[ImageRecord]
    # 4-way hard negatives (block_id strings; optional while mining fails)
    neg_same_neighborhood_diff_block: Optional[str] = None
    neg_same_city_diff_neighborhood: Optional[str] = None
    neg_same_driving_side_diff_city: Optional[str] = None
    neg_random_city: Optional[str] = None
    source: str = "mapillary_v3"


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _load_blocks_spec(path: Path) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {b["block_id"]: b for b in data["blocks"]}


def _load_curation(in_csv: Path, pass_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    if pass_col and pass_col in df.columns:
        df = df[df[pass_col].fillna(False).astype(bool)].copy()
    return df


def _rank_images(group: pd.DataFrame) -> pd.DataFrame:
    """Prefer highest urbanness, then mapillary_quality, then sun elevation."""
    sort_cols = []
    for col in ("urbanness", "mapillary_quality", "sun_elev_deg"):
        if col in group.columns:
            sort_cols.append(col)
    if sort_cols:
        group = group.sort_values(sort_cols, ascending=False)
    return group


def build_block_units(spec: dict[str, dict], df: pd.DataFrame,
                       target_n: int, min_n: int) -> list[BlockUnit]:
    blocks: list[BlockUnit] = []
    for block_id, group in df.groupby("block_id"):
        if block_id not in spec:
            continue
        ranked = _rank_images(group).head(target_n)
        if len(ranked) < min_n:
            continue
        meta = spec[block_id]
        images = [
            ImageRecord(
                image_id=str(r["image_id"]),
                image_path=r["image_path"],
                lat=float(r["latitude"]),
                lon=float(r["longitude"]),
                captured_at_utc_ms=int(r["captured_at_utc_ms"]),
                compass_angle=(None if pd.isna(r.get("compass_angle"))
                               else float(r["compass_angle"])),
                sun_elev_deg=float(r["sun_elev_deg"]),
                h3_r9_cell=str(r.get("h3_r9_cell", "")),
                camera_type=str(r.get("camera_type", "perspective")),
                mapillary_quality=(None if pd.isna(r.get("mapillary_quality"))
                                    else float(r["mapillary_quality"])),
            )
            for _, r in ranked.iterrows()
        ]
        blocks.append(BlockUnit(
            block_id=block_id,
            city=meta["city"], country=meta["country"],
            driving_side=meta["driving_side"], stratum=meta["stratum"],
            way_id=meta["way_id"], street_name=meta["street_name"],
            neighborhood=meta["neighborhood"], highway_tag=meta["highway_tag"],
            length_m=meta["length_m"], centroid=meta["centroid"],
            bbox=meta["bbox"], polyline=meta["polyline"],
            h3_r9_cells=meta["h3_r9_cells"], images=images,
        ))
    return blocks


def mine_hard_negatives(blocks: list[BlockUnit], seed: int = 42
                         ) -> list[BlockUnit]:
    rng = random.Random(seed)
    by_city: dict[str, list[BlockUnit]] = {}
    by_drive: dict[str, list[BlockUnit]] = {}
    for b in blocks:
        by_city.setdefault(b.city, []).append(b)
        by_drive.setdefault(b.driving_side, []).append(b)

    for b in blocks:
        # same_neighborhood_diff_block: any block in same city with centroid
        # distance <= V3_NEIGHBORHOOD_RADIUS_M, excluding self
        near = []
        far = []
        for q in by_city[b.city]:
            if q.block_id == b.block_id:
                continue
            d = _haversine_m(b.centroid[0], b.centroid[1],
                             q.centroid[0], q.centroid[1])
            if d <= config.V3_NEIGHBORHOOD_RADIUS_M:
                near.append(q)
            else:
                far.append(q)
        # If neighborhood-name metadata agrees, prefer those; else use distance
        named_near = [q for q in near if q.neighborhood == b.neighborhood
                      and b.neighborhood != "downtown"]
        pool_near = named_near or near
        if pool_near:
            b.neg_same_neighborhood_diff_block = rng.choice(pool_near).block_id
        elif far:
            # fallback: any same-city block, still informative
            b.neg_same_neighborhood_diff_block = rng.choice(far).block_id

        if far:
            b.neg_same_city_diff_neighborhood = rng.choice(far).block_id
        elif by_city[b.city]:
            others = [q for q in by_city[b.city] if q.block_id != b.block_id]
            if others:
                b.neg_same_city_diff_neighborhood = rng.choice(others).block_id

        same_drive = [q for q in by_drive[b.driving_side] if q.city != b.city]
        if same_drive:
            b.neg_same_driving_side_diff_city = rng.choice(same_drive).block_id

        others = [q for q in blocks if q.city != b.city]
        if others:
            b.neg_random_city = rng.choice(others).block_id

    full = sum(1 for b in blocks
               if all((b.neg_same_neighborhood_diff_block,
                       b.neg_same_city_diff_neighborhood,
                       b.neg_same_driving_side_diff_city,
                       b.neg_random_city)))
    print(f"[curate] hard negatives: {full}/{len(blocks)} fully covered")
    return blocks


def save(blocks: list[BlockUnit], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "GeoFidelity-Bench",
        "version": "3.0.0",
        "h3_resolution_cells": config.V3_BLOCK_H3_RES,
        "num_cities": len({b.city for b in blocks}),
        "num_blocks": len(blocks),
        "num_images": sum(len(b.images) for b in blocks),
        "neg_levels": config.V3_NEG_LEVELS,
        "prompt_levels": list(config.V3_PROMPT_TEMPLATES.keys()),
        "blocks": [asdict(b) for b in blocks],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summary(blocks: list[BlockUnit]) -> None:
    print("\n" + "=" * 60)
    print("GeoFidelity-Bench v3 summary")
    print("=" * 60)
    total_imgs = sum(len(b.images) for b in blocks)
    print(f"  blocks:      {len(blocks)}")
    print(f"  cities:      {len({b.city for b in blocks})}")
    print(f"  images:      {total_imgs}")
    if blocks:
        print(f"  avg / block: {total_imgs / len(blocks):.1f}")
    by_city: dict[str, list[BlockUnit]] = {}
    for b in blocks:
        by_city.setdefault(b.city, []).append(b)
    print("\n  per-city:")
    for city in sorted(by_city):
        bs = by_city[city]
        nimg = sum(len(b.images) for b in bs)
        print(f"    {city:18s}: {len(bs):2d} blocks  {nimg:5d} imgs "
              f"(avg {nimg/max(1,len(bs)):.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks_spec", default=str(config.V3_BLOCKS_JSON))
    ap.add_argument("--in_csv",
                    default=str(config.V3_PROCESSED_DIR / "tier5_quality.csv"))
    ap.add_argument("--pass_col", default="tier5_pass")
    ap.add_argument("--fallback_csv",
                    default=str(config.V3_TIER1_CSV),
                    help="Used if main CSV is missing (e.g. curation not run)")
    ap.add_argument("--out", default=str(config.V3_BENCHMARK_JSON))
    ap.add_argument("--target_per_block", type=int,
                    default=config.V3_TARGET_IMAGES_PER_BLOCK)
    ap.add_argument("--min_per_block", type=int,
                    default=config.V3_MIN_IMAGES_PER_BLOCK)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = _load_blocks_spec(Path(args.blocks_spec))

    csv_path = Path(args.in_csv)
    pass_col: str | None = args.pass_col
    if not csv_path.exists():
        csv_path = Path(args.fallback_csv)
        pass_col = None
        print(f"[curate] {args.in_csv} missing -> "
              f"falling back to tier1_candidates ({csv_path.name}), "
              f"skipping curation gates")

    df = _load_curation(csv_path, pass_col)
    print(f"[curate] {len(df)} rows after filter (pass_col={pass_col})")

    blocks = build_block_units(spec, df, args.target_per_block,
                                args.min_per_block)
    print(f"[curate] {len(blocks)} blocks retained "
          f"(min_per_block={args.min_per_block})")

    blocks = mine_hard_negatives(blocks, seed=args.seed)
    save(blocks, Path(args.out))
    summary(blocks)
    print(f"\n[curate] wrote {args.out}")


if __name__ == "__main__":
    main()
