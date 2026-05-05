"""
Tier 2 filter: OSM road-type classification.

For every Tier 1 candidate image, snap its (lat, lon) to the nearest OSM
highway and inspect the `highway=*` tag. Drop images on motorway/trunk/
service/track (kaizhen's "big highway" complaint); keep residential /
primary / secondary / tertiary / pedestrian / unclassified.

Overpass is queried once per tile (tile bbox + 50 m padding), results are
cached locally so re-runs are free. Closest-way matching is done locally
with shapely.

Input:  data/processed/tier1_candidates.csv
Output: data/processed/tier2_osm.csv
        (adds columns: osm_highway, osm_distance_m, tier2_pass)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import time

import h3
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform
from tqdm import tqdm

import config


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]
OVERPASS_TIMEOUT_S = 90
CACHE_DIR = config.CACHE_DIR / "overpass_tier2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------- Overpass --------------------------------

def _cache_path(cell: str) -> Path:
    return CACHE_DIR / f"{cell}.json"


def fetch_highways_for_tile(cell: str, pad_m: float = 50.0) -> dict:
    """Fetch all OSM ways with highway=* inside a tile bbox (with padding)."""
    cache_path = _cache_path(cell)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    boundary = h3.cell_to_boundary(cell)
    lats = [p[0] for p in boundary]
    lons = [p[1] for p in boundary]
    lat_pad = pad_m / 111000.0
    lon_pad = pad_m / (111000.0 * max(0.01, math.cos(math.radians(sum(lats) / len(lats)))))
    s, n = min(lats) - lat_pad, max(lats) + lat_pad
    w, e = min(lons) - lon_pad, max(lons) + lon_pad

    q = f"""[out:json][timeout:{OVERPASS_TIMEOUT_S}];
(way["highway"]({s},{w},{n},{e}););
out geom tags;"""

    last_exc: Exception | None = None
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": q}, timeout=OVERPASS_TIMEOUT_S + 10)
            if r.status_code == 200:
                data = r.json()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            last_exc = RuntimeError(f"http {r.status_code}")
        except Exception as exc:
            last_exc = exc
        time.sleep(2.0)

    raise RuntimeError(f"Overpass unavailable ({last_exc})") from last_exc


# --------------------------- Local geo distance -----------------------------

def _meters_per_deg(lat_deg: float) -> tuple[float, float]:
    """Local equirectangular scale."""
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * max(0.01, math.cos(math.radians(lat_deg)))
    return m_per_lat, m_per_lon


def _way_to_linestring_m(way_geom: list[dict], ref_lat: float) -> LineString:
    m_lat, m_lon = _meters_per_deg(ref_lat)
    pts = [((p["lon"]) * m_lon, (p["lat"]) * m_lat) for p in way_geom]
    return LineString(pts) if len(pts) >= 2 else LineString([pts[0], pts[0]])


def closest_highway(img_lat: float, img_lon: float,
                    ways: list[dict]) -> tuple[str | None, float]:
    """Return (highway_tag, distance_m) of the closest OSM highway way."""
    if not ways:
        return None, float("inf")
    m_lat, m_lon = _meters_per_deg(img_lat)
    p = Point(img_lon * m_lon, img_lat * m_lat)
    best: tuple[str | None, float] = (None, float("inf"))
    for w in ways:
        geom = w.get("geometry")
        if not geom or len(geom) < 2:
            continue
        line = _way_to_linestring_m(geom, img_lat)
        d = p.distance(line)
        if d < best[1]:
            tag = w.get("tags", {}).get("highway", "?")
            best = (tag, d)
    return best


# ----------------------------- Main pipeline --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(config.PROCESSED_DIR / "tier1_candidates.csv"))
    ap.add_argument("--out_csv", default=str(config.PROCESSED_DIR / "tier2_osm.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    print(f"[tier2] {len(df)} tier1 rows across "
          f"{df['h3_tile_res8'].nunique()} tiles")

    out_rows: list[dict] = []
    tiles = list(df.groupby("h3_tile_res8"))
    include = config.TIER2_INCLUDE_HIGHWAY_TAGS
    exclude = config.TIER2_EXCLUDE_HIGHWAY_TAGS

    for cell, group in tqdm(tiles, desc="tiles"):
        try:
            data = fetch_highways_for_tile(cell)
            ways = data.get("elements", [])
        except Exception as e:
            print(f"  tile {cell[:12]}: overpass failed ({e}), keeping all")
            ways = []
        for _, row in group.iterrows():
            tag, dist_m = closest_highway(row["latitude"], row["longitude"], ways)
            if tag is None:
                passed = None
            elif tag in exclude:
                passed = False
            elif tag in include:
                passed = dist_m <= config.TIER2_SNAP_RADIUS_M
            else:
                passed = dist_m <= config.TIER2_SNAP_RADIUS_M
            out = row.to_dict()
            out["osm_highway"] = tag
            out["osm_distance_m"] = round(dist_m, 2) if math.isfinite(dist_m) else None
            out["tier2_pass"] = bool(passed) if passed is not None else False
            out_rows.append(out)
        # Checkpoint every 10 tiles
        if len(out_rows) % 300 == 0 and out_rows:
            pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)

    out = pd.DataFrame(out_rows)
    out.to_csv(args.out_csv, index=False)
    n_pass = int(out["tier2_pass"].sum())
    print(f"[tier2] pass {n_pass}/{len(out)} "
          f"({100.0 * n_pass / max(1, len(out)):.1f}%)")
    if "osm_highway" in out:
        print("[tier2] top highway tags:")
        print(out["osm_highway"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
