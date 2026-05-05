"""
Curate real OSV-5M data for GeoFidelity-Bench.
Uses downloaded real images to build benchmark place units.
Handles sparse coverage by selecting cities with enough data.
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import h3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from geopy.distance import geodesic

import config
from data.curate_places import PlaceUnit, mine_hard_negatives, save_benchmark, print_summary


def curate_from_real_images(data_dir: Path, min_images_per_tile: int = 3,
                             min_tiles_per_city: int = 2) -> list[PlaceUnit]:
    """Build place units from downloaded real images.

    Adapts to sparse coverage by:
    1. Counting images per tile
    2. Keeping tiles with >= min_images_per_tile
    3. Keeping cities with >= min_tiles_per_city valid tiles
    """
    real_dir = data_dir / "osv5m_real"
    if not real_dir.exists():
        print(f"ERROR: {real_dir} not found. Download images first.")
        return []

    places = []

    for city_dir in sorted(real_dir.iterdir()):
        if not city_dir.is_dir():
            continue

        city_name = city_dir.name
        city_info = config.CITIES.get(city_name)
        if not city_info:
            continue

        valid_tiles = []
        for tile_dir in sorted(city_dir.iterdir()):
            if not tile_dir.is_dir():
                continue

            images = list(tile_dir.glob("*.jpg"))
            if len(images) >= min_images_per_tile:
                valid_tiles.append((tile_dir.name, images))

        if len(valid_tiles) < min_tiles_per_city:
            print(f"  SKIP {city_name}: only {len(valid_tiles)} valid tiles "
                  f"(need {min_tiles_per_city})")
            continue

        # Select up to TILES_PER_CITY tiles
        for tile_id, images in valid_tiles[:config.TILES_PER_CITY]:
            lat, lon = h3.cell_to_latlng(tile_id)
            place = PlaceUnit(
                place_id=f"{city_name}__{tile_id}",
                city=city_name,
                h3_tile=tile_id,
                lat=lat,
                lon=lon,
                country=city_info["country"],
                driving_side=city_info["driving"],
                image_ids=[p.stem for p in images],
                source="osv5m_real",
            )
            places.append(place)

        print(f"  {city_name}: {len(valid_tiles)} valid tiles, "
              f"kept {min(len(valid_tiles), config.TILES_PER_CITY)}")

    print(f"\nTotal: {len(places)} place units from real data")
    return places


def merge_with_synthetic(real_places: list[PlaceUnit],
                          synthetic_benchmark_path: Path) -> list[PlaceUnit]:
    """Merge real places with synthetic ones for cities without real coverage."""
    with open(str(synthetic_benchmark_path)) as f:
        syn_benchmark = json.load(f)

    real_cities = {p.city for p in real_places}

    # Add synthetic places for uncovered cities
    merged = list(real_places)
    for syn_place_data in syn_benchmark["places"]:
        if syn_place_data["city"] not in real_cities:
            place = PlaceUnit(
                place_id=syn_place_data["place_id"],
                city=syn_place_data["city"],
                h3_tile=syn_place_data["h3_tile"],
                lat=syn_place_data["lat"],
                lon=syn_place_data["lon"],
                country=syn_place_data["country"],
                driving_side=syn_place_data["driving_side"],
                image_ids=syn_place_data["image_ids"],
                source="synthetic",
            )
            merged.append(place)

    print(f"Merged: {len(real_places)} real + "
          f"{len(merged) - len(real_places)} synthetic = {len(merged)} total")
    return merged


def main():
    print("=" * 60)
    print("GeoFidelity-Bench: Real Data Curation")
    print("=" * 60)

    # Build from real images (lower threshold for sparse data)
    places = curate_from_real_images(
        config.DATA_DIR,
        min_images_per_tile=2,  # lowered for sparse OSV-5M test set
        min_tiles_per_city=2,
    )

    if not places:
        print("No real data available. Run download_real_data.py first.")
        return

    # Mine hard negatives
    places = mine_hard_negatives(places)

    # Save as separate benchmark
    save_benchmark(places, config.PROCESSED_DIR / "benchmark_real.json")
    print_summary(places)


if __name__ == "__main__":
    main()
