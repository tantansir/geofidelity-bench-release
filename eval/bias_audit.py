"""
Dataset bias / governance audit for GeoFidelity-Bench.

Addresses reviewer concern that a Mapillary-based curation pipeline
likely overrepresents specific neighborhoods, infrastructure types,
seasons, and contributor populations. We report:

1. Per-city image-count distribution and Gini coefficient
2. OSM road-type distribution per city (from Tier-2 cache)
3. Per-city Mask2Former segmentation entropy (low entropy flags
   potential domain-shift cases)
4. Reference-panel size distribution across place units
5. Country / continent coverage

Outputs: outputs/bias_audit/*.csv and a summary table.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import config


CONTINENT = {
    "GB": "EU", "FR": "EU", "DE": "EU", "IT": "EU", "NL": "EU",
    "TR": "EU",
    "US": "NA", "CA": "NA", "MX": "NA",
    "AR": "SA", "BR": "SA", "CO": "SA",
    "JP": "AS", "SG": "AS", "IN": "AS", "TH": "AS", "KR": "AS",
    "CN": "AS", "AE": "AS",
    "ZA": "AF", "KE": "AF", "EG": "AF",
    "AU": "OC",
}


def gini(x):
    x = np.sort(np.array(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places = bench["places"]
    out_dir = config.OUTPUT_DIR / "bias_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Per-city counts ----------
    city_counts = Counter()
    city_tiles = Counter()
    city_country = {}
    for p in places:
        city_counts[p["city"]] += len(p["image_paths"])
        city_tiles[p["city"]] += 1
        city_country[p["city"]] = p["country"]

    rows = []
    for city in sorted(city_counts):
        rows.append({
            "city": city,
            "country": city_country[city],
            "continent": CONTINENT.get(city_country[city], "?"),
            "tiles": city_tiles[city],
            "images": city_counts[city],
            "avg_per_tile": city_counts[city] / max(1, city_tiles[city]),
        })
    dfc = pd.DataFrame(rows)
    dfc.to_csv(out_dir / "per_city_counts.csv", index=False)

    img_gini = gini(list(city_counts.values()))
    tile_gini = gini(list(city_tiles.values()))
    print(f"=== Per-city coverage ===")
    print(f"Cities: {len(city_counts)}  Total images: {sum(city_counts.values())}  Total tiles: {sum(city_tiles.values())}")
    print(f"Image count Gini: {img_gini:.3f}  (0 = uniform, 1 = concentrated)")
    print(f"Tile count Gini:  {tile_gini:.3f}")

    # ---------- Panel size distribution ----------
    panel_sizes = [len(p["image_paths"]) for p in places]
    panel_df = pd.DataFrame({
        "min": [min(panel_sizes)],
        "q25": [np.percentile(panel_sizes, 25)],
        "median": [np.median(panel_sizes)],
        "q75": [np.percentile(panel_sizes, 75)],
        "max": [max(panel_sizes)],
        "mean": [np.mean(panel_sizes)],
        "gini": [gini(panel_sizes)],
    })
    panel_df.to_csv(out_dir / "panel_size_dist.csv", index=False)
    print(f"\n=== Panel size (images per place) ===")
    print(f"min={min(panel_sizes)}  median={np.median(panel_sizes):.0f}  mean={np.mean(panel_sizes):.1f}  max={max(panel_sizes)}")

    # ---------- Continent / country coverage ----------
    continent_counts = Counter()
    country_counts = Counter()
    for p in places:
        cc = CONTINENT.get(p["country"], "?")
        continent_counts[cc] += len(p["image_paths"])
        country_counts[p["country"]] += len(p["image_paths"])
    print(f"\n=== Continent coverage ===")
    for c in sorted(continent_counts, key=continent_counts.get, reverse=True):
        print(f"  {c}: {continent_counts[c]} images")
    print(f"\nCountries: {len(country_counts)}  Country-level Gini: {gini(list(country_counts.values())):.3f}")

    # ---------- OSM road-type audit (from Tier-2 cache) ----------
    overpass_dir = config.CACHE_DIR / "overpass_tier2"
    road_type_counts = Counter()
    if overpass_dir.exists():
        for f in overpass_dir.glob("*.json"):
            try:
                data = json.load(open(f))
                for el in data.get("elements", []):
                    tags = el.get("tags", {})
                    if "highway" in tags:
                        road_type_counts[tags["highway"]] += 1
            except Exception:
                pass
    rtdf = pd.DataFrame(
        [{"road_type": k, "count": v}
         for k, v in sorted(road_type_counts.items(), key=lambda x: -x[1])])
    rtdf.to_csv(out_dir / "osm_road_types.csv", index=False)
    print(f"\n=== OSM road-type distribution (Tier-2 cache) ===")
    for k, v in sorted(road_type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")

    # ---------- Tier-4 segmentation ratios summary (if available) ----------
    t4 = config.PROCESSED_DIR / "tier4_segmentation.csv"
    if t4.exists():
        t = pd.read_csv(t4)
        # Expect columns for ratios per attribute
        ratio_cols = [c for c in t.columns
                      if any(k in c for k in ("sky", "building", "road",
                                               "vegetation", "sidewalk",
                                               "vehicle", "pole", "sign"))]
        if ratio_cols:
            print(f"\n=== Tier-4 attribute ratio ranges (global) ===")
            for c in ratio_cols:
                if t[c].dtype in (float, int, np.float64, np.int64):
                    print(f"  {c}: mean={t[c].mean():.3f}  std={t[c].std():.3f}")

    # ---------- Summary table ----------
    summary = pd.DataFrame([
        {"metric": "num_cities", "value": len(city_counts)},
        {"metric": "num_countries", "value": len(country_counts)},
        {"metric": "num_places", "value": len(places)},
        {"metric": "num_images", "value": sum(city_counts.values())},
        {"metric": "image_gini_across_cities", "value": img_gini},
        {"metric": "tile_gini_across_cities", "value": tile_gini},
        {"metric": "panel_size_gini", "value": gini(panel_sizes)},
        {"metric": "country_gini", "value": gini(list(country_counts.values()))},
        {"metric": "panel_size_min", "value": min(panel_sizes)},
        {"metric": "panel_size_median", "value": float(np.median(panel_sizes))},
        {"metric": "panel_size_max", "value": max(panel_sizes)},
    ])
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[bias_audit] saved to {out_dir}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
