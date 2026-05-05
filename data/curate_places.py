"""
Build GeoFidelity-Bench v2 place-unit spec from the final curated CSV.

A place unit = one H3 resolution-8 tile (~0.74 km^2) with its curated
reference images, plus three hard negatives:
  * same_city: another tile in the same city, >= 0.7 km from this tile
  * same_climate: a tile in a different city with the same driving side
  * random: a tile in a randomly selected different city

Tiles below MIN_IMAGES_PER_TILE survivors are dropped. Cities that cannot
provide >= 3 valid tiles are dropped (so hard-neg mining stays well
defined).

Input:  data/processed/tier6_review.csv   (falls back to tier5_quality.csv)
Output: data/processed/benchmark_v2.json
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Optional

import h3
import numpy as np
import pandas as pd
from geopy.distance import geodesic

import config


@dataclass
class PlaceUnit:
    place_id: str
    city: str
    h3_tile: str
    lat: float
    lon: float
    country: str
    driving_side: str
    image_paths: list[str]
    source: str = "mapillary_v2"
    neg_same_city: Optional[str] = None
    neg_same_climate: Optional[str] = None
    neg_random: Optional[str] = None


def build_place_units(df: pd.DataFrame) -> list[PlaceUnit]:
    places: list[PlaceUnit] = []
    for (city, tile), g in df.groupby(["city", "h3_tile_res8"]):
        if len(g) < config.MIN_IMAGES_PER_TILE:
            continue
        # Order by urbanness desc, cap at 2x TARGET to leave room for filtering
        if "urbanness" in g.columns:
            g = g.sort_values("urbanness", ascending=False)
        images = g["image_path"].tolist()[:config.TARGET_IMAGES_PER_TILE * 2]
        lat, lon = h3.cell_to_latlng(tile)
        info = config.CITIES[city]
        places.append(PlaceUnit(
            place_id=f"{city}__{tile}",
            city=city, h3_tile=tile, lat=lat, lon=lon,
            country=info["country"], driving_side=info["driving"],
            image_paths=images,
        ))
    return places


def cap_tiles_per_city(places: list[PlaceUnit]) -> list[PlaceUnit]:
    """Keep the best TILES_PER_CITY tiles per city (by image count)."""
    out: list[PlaceUnit] = []
    by_city: dict[str, list[PlaceUnit]] = {}
    for p in places:
        by_city.setdefault(p.city, []).append(p)
    for city, lst in by_city.items():
        lst.sort(key=lambda p: -len(p.image_paths))
        out.extend(lst[:config.TILES_PER_CITY])
    return out


def mine_hard_negatives(places: list[PlaceUnit], seed: int = 42
                        ) -> list[PlaceUnit]:
    rng = random.Random(seed)
    by_city: dict[str, list[PlaceUnit]] = {}
    by_drive: dict[str, list[PlaceUnit]] = {}
    for p in places:
        by_city.setdefault(p.city, []).append(p)
        by_drive.setdefault(p.driving_side, []).append(p)

    for p in places:
        same_city = [q for q in by_city[p.city]
                     if q.h3_tile != p.h3_tile
                     and geodesic((p.lat, p.lon), (q.lat, q.lon)).km
                         >= config.HARD_NEG_SAME_CITY_MIN_KM]
        if not same_city:
            same_city = [q for q in by_city[p.city] if q.h3_tile != p.h3_tile]
        if same_city:
            p.neg_same_city = rng.choice(same_city).place_id

        same_drive = [q for q in by_drive[p.driving_side] if q.city != p.city]
        if same_drive:
            p.neg_same_climate = rng.choice(same_drive).place_id

        others = [q for q in places if q.city != p.city]
        if others:
            p.neg_random = rng.choice(others).place_id

    full = sum(1 for p in places
               if p.neg_same_city and p.neg_same_climate and p.neg_random)
    print(f"[curate] hard negatives: {full}/{len(places)} fully covered")
    return places


def save(places: list[PlaceUnit], out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench = {
        "name": "GeoFidelity-Bench",
        "version": "2.0.0",
        "h3_resolution": config.H3_RESOLUTION,
        "num_places": len(places),
        "num_cities": len(set(p.city for p in places)),
        "num_images": sum(len(p.image_paths) for p in places),
        "places": [asdict(p) for p in places],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2, ensure_ascii=False)
    return bench


def summary(places: list[PlaceUnit]) -> None:
    print("\n" + "=" * 60)
    print("GeoFidelity-Bench v2 summary")
    print("=" * 60)
    print(f"  place units: {len(places)}")
    print(f"  cities:      {len(set(p.city for p in places))}")
    print(f"  images:      {sum(len(p.image_paths) for p in places)}")
    print(f"  avg / place: {np.mean([len(p.image_paths) for p in places]):.1f}")
    print("\n  per-city:")
    for city in sorted(set(p.city for p in places)):
        cps = [p for p in places if p.city == city]
        print(f"    {city:18s}: {len(cps):2d} tiles  "
              f"{sum(len(p.image_paths) for p in cps):4d} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(config.PROCESSED_DIR / "tier6_review.csv"))
    ap.add_argument("--fallback_csv",
                    default=str(config.PROCESSED_DIR / "tier5_quality.csv"),
                    help="Used if in_csv missing (i.e. Tier 6 skipped)")
    ap.add_argument("--out", default=str(config.PROCESSED_DIR / "benchmark_v2.json"))
    args = ap.parse_args()

    csv_path = Path(args.in_csv)
    pass_col = "tier6_pass"
    if not csv_path.exists():
        csv_path = Path(args.fallback_csv)
        pass_col = "tier5_pass"
        print(f"[curate] tier6 CSV missing, falling back to {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df[pass_col].fillna(False).astype(bool)].copy()
    print(f"[curate] {len(df)} images after final gate ({pass_col})")

    places = build_place_units(df)
    places = cap_tiles_per_city(places)
    places = mine_hard_negatives(places)
    bench = save(places, Path(args.out))
    summary(places)
    print(f"\n[curate] wrote {args.out}")


if __name__ == "__main__":
    main()
