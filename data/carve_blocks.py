"""
Carve OSM-way-based blocks for GeoFidelity-Bench v3.

For each target city we pull every named `highway=*` way within
`V3_CITY_SEARCH_RADIUS_M` of the city centroid from OSM (Overpass), then
sample V3_BLOCKS_PER_CITY blocks stratified across four highway classes:
  * major       -> primary / secondary (+link variants)
  * residential -> residential / living_street / unclassified
  * pedestrian  -> pedestrian / footway / cycleway
  * tertiary    -> tertiary (+link)

A "block" is one named way (truncated to V3_MAX_WAY_LENGTH_M), plus its
neighbourhood label (nearest OSM `place=suburb|neighbourhood|quarter`
node), the polyline, bbox, centroid, driving side, and the set of H3
resolution-9 cells its `V3_WAY_BUFFER_M` buffer covers.

Output: data/processed/v3/blocks_v3.json

The script is idempotent and Overpass responses are cached under
data/cache/overpass_blocks/ so re-runs cost nothing network-wise.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict, field

import h3
import requests
from shapely.geometry import LineString, Point, box
from shapely.ops import substring
from tqdm import tqdm

import config


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]


# --------------------------------------------------------------------------
# Overpass helpers
# --------------------------------------------------------------------------

def _city_cache_path(city: str, kind: str) -> Path:
    return config.V3_OVERPASS_CACHE / f"{city}__{kind}.json"


def _overpass_query(q: str, timeout_s: int = None) -> dict:
    timeout_s = timeout_s or config.V3_OVERPASS_TIMEOUT_S
    last_exc: Exception | None = None
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": q}, timeout=timeout_s + 10)
            if r.status_code == 200:
                return r.json()
            last_exc = RuntimeError(f"http {r.status_code}: {r.text[:200]}")
        except Exception as exc:
            last_exc = exc
        time.sleep(3.0)
    raise RuntimeError(f"Overpass unavailable ({last_exc})") from last_exc


def fetch_city_ways(city: str, lat: float, lon: float,
                    radius_m: int) -> list[dict]:
    """All named highway ways within radius_m of (lat, lon)."""
    cache = _city_cache_path(city, "ways")
    if cache.exists():
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f).get("elements", [])
    q = f"""[out:json][timeout:{config.V3_OVERPASS_TIMEOUT_S}];
(
  way(around:{radius_m},{lat},{lon})["highway"]["name"];
);
out geom tags;"""
    data = _overpass_query(q)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data.get("elements", [])


def fetch_city_places(city: str, lat: float, lon: float,
                      radius_m: int) -> list[dict]:
    """Neighbourhood-ish place nodes (named) near the city centre."""
    cache = _city_cache_path(city, "places")
    if cache.exists():
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f).get("elements", [])
    q = f"""[out:json][timeout:{config.V3_OVERPASS_TIMEOUT_S}];
(
  node(around:{radius_m},{lat},{lon})["place"~"suburb|neighbourhood|quarter|district|town"]["name"];
);
out;"""
    data = _overpass_query(q)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data.get("elements", [])


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _m_per_deg(lat_deg: float) -> tuple[float, float]:
    return 111320.0, 111320.0 * max(0.01, math.cos(math.radians(lat_deg)))


def way_polyline_m(geom: list[dict], ref_lat: float) -> LineString:
    m_lat, m_lon = _m_per_deg(ref_lat)
    pts = [(p["lon"] * m_lon, p["lat"] * m_lat) for p in geom
           if "lat" in p and "lon" in p]
    return LineString(pts) if len(pts) >= 2 else LineString([(0, 0), (0, 0)])


def polyline_latlon(geom: list[dict]) -> list[tuple[float, float]]:
    return [(p["lat"], p["lon"]) for p in geom if "lat" in p and "lon" in p]


def geodesic_length_m(latlon: list[tuple[float, float]]) -> float:
    if len(latlon) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(latlon)):
        total += _haversine_m(latlon[i - 1][0], latlon[i - 1][1],
                              latlon[i][0], latlon[i][1])
    return total


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def truncate_polyline_m(latlon: list[tuple[float, float]],
                         max_m: float) -> list[tuple[float, float]]:
    """Return prefix of polyline with cumulative length <= max_m."""
    if not latlon:
        return []
    out = [latlon[0]]
    acc = 0.0
    for i in range(1, len(latlon)):
        seg = _haversine_m(*latlon[i - 1], *latlon[i])
        if acc + seg > max_m:
            frac = (max_m - acc) / seg
            lat = latlon[i - 1][0] + frac * (latlon[i][0] - latlon[i - 1][0])
            lon = latlon[i - 1][1] + frac * (latlon[i][1] - latlon[i - 1][1])
            out.append((lat, lon))
            break
        acc += seg
        out.append(latlon[i])
    return out


def polyline_bbox(latlon: list[tuple[float, float]]
                  ) -> tuple[float, float, float, float]:
    lats = [p[0] for p in latlon]
    lons = [p[1] for p in latlon]
    return min(lats), min(lons), max(lats), max(lons)


def polyline_centroid(latlon: list[tuple[float, float]]) -> tuple[float, float]:
    """Midpoint of polyline (arc-length halfway)."""
    total = geodesic_length_m(latlon)
    if total <= 0:
        return latlon[0]
    target = total / 2
    acc = 0.0
    for i in range(1, len(latlon)):
        seg = _haversine_m(*latlon[i - 1], *latlon[i])
        if acc + seg >= target:
            frac = (target - acc) / seg
            lat = latlon[i - 1][0] + frac * (latlon[i][0] - latlon[i - 1][0])
            lon = latlon[i - 1][1] + frac * (latlon[i][1] - latlon[i - 1][1])
            return lat, lon
        acc += seg
    return latlon[-1]


def h3_cells_along_polyline(latlon: list[tuple[float, float]],
                             buffer_m: float,
                             h3_res: int) -> list[str]:
    """All H3 cells whose centre lies within buffer_m of the polyline."""
    cells = set()
    # Sample polyline densely, add each sample's h3 cell; also add neighbours
    # up to the ring radius that covers buffer_m at this resolution.
    if not latlon:
        return []
    # ring radius: edge length at res-9 ~170 m → 2-ring covers ~500 m
    ring_r = max(1, int(math.ceil(buffer_m / 170.0)))
    step = 15.0  # metres between samples along the polyline
    accum = [latlon[0]]
    acc_m = 0.0
    for i in range(1, len(latlon)):
        seg_m = _haversine_m(*latlon[i - 1], *latlon[i])
        if seg_m == 0:
            continue
        n_step = max(1, int(seg_m / step))
        for s in range(1, n_step + 1):
            f = s / n_step
            lat = latlon[i - 1][0] + f * (latlon[i][0] - latlon[i - 1][0])
            lon = latlon[i - 1][1] + f * (latlon[i][1] - latlon[i - 1][1])
            accum.append((lat, lon))
    for lat, lon in accum:
        cell = h3.latlng_to_cell(lat, lon, h3_res)
        cells.add(cell)
        for nbr in h3.grid_disk(cell, ring_r):
            cells.add(nbr)
    return sorted(cells)


def nearest_neighborhood(mid_lat: float, mid_lon: float,
                          places: list[dict]) -> tuple[str, float]:
    """Return (neighborhood_name, distance_m) of nearest named place node.

    Falls back to ('downtown', inf) if no place is within 2 km.
    """
    best = ("downtown", float("inf"))
    for p in places:
        lat, lon = p.get("lat"), p.get("lon")
        if lat is None or lon is None:
            continue
        name = p.get("tags", {}).get("name")
        if not name:
            continue
        d = _haversine_m(mid_lat, mid_lon, lat, lon)
        if d < best[1]:
            best = (name, d)
    if best[1] > 2000:
        return "downtown", best[1]
    return best


# --------------------------------------------------------------------------
# Block selection
# --------------------------------------------------------------------------

@dataclass
class Block:
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
    polyline: list[list[float]]
    bbox: list[float]                 # [lat_min, lon_min, lat_max, lon_max]
    centroid: list[float]             # [lat, lon]
    h3_r9_cells: list[str]


def _classify_way(highway_tag: str) -> str | None:
    for label, tags, _ in config.V3_BLOCK_STRATA:
        if highway_tag in tags:
            return label
    return None


def _dedup_by_name(ways: list[dict]) -> list[dict]:
    """Merge ways sharing the same name into the single longest segment."""
    by_name: dict[str, dict] = {}
    for w in ways:
        name = w.get("tags", {}).get("name")
        if not name:
            continue
        pl = polyline_latlon(w.get("geometry", []))
        length = geodesic_length_m(pl)
        if length <= 0:
            continue
        w["_polyline"] = pl
        w["_length_m"] = length
        prev = by_name.get(name)
        if prev is None or length > prev["_length_m"]:
            by_name[name] = w
    return list(by_name.values())


def select_blocks_for_city(city: str, info: dict,
                            rng: random.Random) -> list[Block]:
    """Pick V3_BLOCKS_PER_CITY blocks for one city."""
    lat, lon = info["lat"], info["lon"]
    ways_raw = fetch_city_ways(city, lat, lon, config.V3_CITY_SEARCH_RADIUS_M)
    places = fetch_city_places(city, lat, lon, config.V3_CITY_SEARCH_RADIUS_M)

    # Classify, filter on length
    ways = _dedup_by_name(ways_raw)
    buckets: dict[str, list[dict]] = {s: [] for s, _, _ in config.V3_BLOCK_STRATA}
    for w in ways:
        tag = w.get("tags", {}).get("highway")
        if not tag:
            continue
        stratum = _classify_way(tag)
        if stratum is None:
            continue
        if w["_length_m"] < config.V3_MIN_WAY_LENGTH_M:
            continue
        buckets[stratum].append(w)

    # Greedy pick from each stratum with spatial separation
    chosen: list[Block] = []
    for stratum, _tags, target_n in config.V3_BLOCK_STRATA:
        picks = _spread_pick(buckets[stratum], target_n, chosen)
        for w in picks:
            chosen.append(_make_block(city, info, stratum, w, places))

    # Top-up if any stratum was starved: fill from longest remaining ways
    if len(chosen) < config.V3_BLOCKS_PER_CITY:
        pool = []
        for stratum, _tags, _n in config.V3_BLOCK_STRATA:
            for w in buckets[stratum]:
                if not any(str(w["id"]) == b.way_id for b in chosen):
                    pool.append((stratum, w))
        pool.sort(key=lambda x: -x[1]["_length_m"])
        for stratum, w in pool:
            if len(chosen) >= config.V3_BLOCKS_PER_CITY:
                break
            # enforce separation against already chosen
            mid = polyline_centroid(w["_polyline"])
            ok = all(
                _haversine_m(mid[0], mid[1], b.centroid[0], b.centroid[1])
                >= config.V3_MIN_SEPARATION_M for b in chosen
            )
            if ok:
                chosen.append(_make_block(city, info, stratum, w, places))
    return chosen


def _spread_pick(cands: list[dict], n: int,
                  already: list[Block]) -> list[dict]:
    """Pick up to n candidates, longest first, spatial-spread ≥ V3_MIN_SEPARATION_M."""
    cands_sorted = sorted(cands, key=lambda w: -w["_length_m"])
    picked: list[dict] = []
    centres: list[tuple[float, float]] = [(b.centroid[0], b.centroid[1])
                                           for b in already]
    for w in cands_sorted:
        if len(picked) >= n:
            break
        mid = polyline_centroid(w["_polyline"])
        if all(_haversine_m(mid[0], mid[1], c[0], c[1])
               >= config.V3_MIN_SEPARATION_M for c in centres):
            picked.append(w)
            centres.append(mid)
    return picked


def _make_block(city: str, info: dict, stratum: str,
                 way: dict, places: list[dict]) -> Block:
    name = way["tags"]["name"]
    tag = way["tags"]["highway"]
    pl = truncate_polyline_m(way["_polyline"], config.V3_MAX_WAY_LENGTH_M)
    length = geodesic_length_m(pl)
    mid = polyline_centroid(pl)
    hood, _d = nearest_neighborhood(mid[0], mid[1], places)
    bbox = polyline_bbox(pl)
    cells = h3_cells_along_polyline(pl, config.V3_WAY_BUFFER_M,
                                     config.V3_BLOCK_H3_RES)
    safe_name = name.replace(" ", "_").replace("/", "_")[:40]
    return Block(
        block_id=f"{city}__{stratum}__{way['id']}__{safe_name}",
        city=city,
        country=info["country"],
        driving_side=info["driving"],
        stratum=stratum,
        way_id=str(way["id"]),
        street_name=name,
        neighborhood=hood,
        highway_tag=tag,
        length_m=round(length, 1),
        polyline=[[round(la, 6), round(lo, 6)] for la, lo in pl],
        bbox=[round(b, 6) for b in bbox],
        centroid=[round(mid[0], 6), round(mid[1], 6)],
        h3_r9_cells=cells,
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None,
                    help="Subset of city keys; defaults to all 25")
    ap.add_argument("--out", default=str(config.V3_BLOCKS_JSON))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cities = args.cities or list(config.CITIES.keys())
    print(f"[carve] {len(cities)} cities x {config.V3_BLOCKS_PER_CITY} blocks "
          f"target = {len(cities) * config.V3_BLOCKS_PER_CITY} blocks")

    rng = random.Random(args.seed)
    all_blocks: list[Block] = []
    for city in tqdm(cities, desc="cities"):
        info = config.CITIES[city]
        try:
            blocks = select_blocks_for_city(city, info, rng)
        except Exception as exc:
            print(f"  {city}: FAILED ({exc})")
            continue
        print(f"  {city}: {len(blocks)} blocks "
              f"({', '.join(b.stratum[:3] for b in blocks)})")
        all_blocks.extend(blocks)

    out = {
        "name": "GeoFidelity-Bench",
        "version": "3.0.0",
        "num_cities": len({b.city for b in all_blocks}),
        "num_blocks": len(all_blocks),
        "h3_resolution_cells": config.V3_BLOCK_H3_RES,
        "blocks": [asdict(b) for b in all_blocks],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[carve] wrote {args.out}")
    print(f"[carve] per-city block counts:")
    by_city: dict[str, list[Block]] = {}
    for b in all_blocks:
        by_city.setdefault(b.city, []).append(b)
    for c in sorted(by_city):
        strata = {s: 0 for s, _, _ in config.V3_BLOCK_STRATA}
        for b in by_city[c]:
            strata[b.stratum] = strata.get(b.stratum, 0) + 1
        print(f"  {c:18s} {len(by_city[c]):2d} blocks  "
              f"(major={strata.get('major',0)} res={strata.get('residential',0)} "
              f"ped={strata.get('pedestrian',0)} ter={strata.get('tertiary',0)})")


if __name__ == "__main__":
    main()
