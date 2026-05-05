"""
Mapillary fetcher for GeoFidelity-Bench v2 (Tier 1 of the curation pipeline).

Query-time filters applied:
  * is_pano = False, camera_type = perspective
  * captured_at: sun elevation >= TIER1_MIN_SUN_ELEVATION_DEG at capture lat/lon

Over-fetches candidates (CANDIDATES_PER_TILE per H3 res-8 tile) so downstream
tiers (OSM road, SigLIP scene, Mask2Former semantic, low-level quality,
manual review) have material to work with. Final target per tile is
TARGET_IMAGES_PER_TILE after all six tiers.

Output:
  data/raw/mapillary_v2/{city}/{tile}/{id}.jpg
  data/processed/tier1_candidates.csv   (row per surviving image with metadata)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import math
import os
import time
from datetime import datetime, timezone

import h3
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

import config


MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not MAPILLARY_TOKEN:
    raise RuntimeError("Set MAPILLARY_TOKEN before downloading Mapillary images.")
BASE_URL = "https://graph.mapillary.com"
SESSION = requests.Session()


# ------------------------------ Solar geometry ------------------------------

def _julian_day(dt: datetime) -> float:
    y, m, d = dt.year, dt.month, dt.day
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    jd += (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    return jd


def sun_elevation_deg(lat: float, lon: float, captured_ms_utc: int) -> float:
    """NOAA solar-position formula. Returns sun elevation above horizon in deg."""
    if captured_ms_utc is None or captured_ms_utc <= 0:
        return float("nan")
    dt = datetime.fromtimestamp(captured_ms_utc / 1000.0, tz=timezone.utc)
    jd = _julian_day(dt)
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(
        math.sin(math.radians(eps)) * math.sin(math.radians(app_long))))
    y_ = math.tan(math.radians(eps / 2)) ** 2
    l0r = math.radians(l0)
    eqt = 4 * math.degrees(
        y_ * math.sin(2 * l0r) - 2 * e * math.sin(mr)
        + 4 * e * y_ * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y_ * y_ * math.sin(4 * l0r)
        - 1.25 * e * e * math.sin(2 * mr)
    )
    utc_min = dt.hour * 60 + dt.minute + dt.second / 60
    tst = (utc_min + eqt + 4 * lon) % 1440
    ha = tst / 4 - 180 if tst / 4 > 180 else tst / 4 + 180
    lat_r = math.radians(lat)
    decl_r = math.radians(decl)
    ha_r = math.radians(ha)
    cos_z = (math.sin(lat_r) * math.sin(decl_r)
             + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    cos_z = max(-1.0, min(1.0, cos_z))
    return 90.0 - math.degrees(math.acos(cos_z))


# ------------------------------ Mapillary API -------------------------------

def search_bbox(lat: float, lon: float, radius_m: int, limit: int = 500):
    """Box-search Mapillary for non-panoramic images around a center."""
    r_deg = radius_m / 111000.0
    bbox = f"{lon - r_deg},{lat - r_deg},{lon + r_deg},{lat + r_deg}"
    fields = "id,geometry,captured_at,is_pano,camera_type,compass_angle,quality_score"
    resp = SESSION.get(
        f"{BASE_URL}/images",
        params={
            "access_token": MAPILLARY_TOKEN,
            "fields": fields,
            "bbox": bbox,
            "limit": limit,
            "is_pano": "false",
        },
        timeout=45,
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("data", [])


def download_thumbnail(image_id: str) -> Image.Image | None:
    """Fetch 1024px thumbnail for an image id."""
    resp = SESSION.get(
        f"{BASE_URL}/{image_id}",
        params={"access_token": MAPILLARY_TOKEN, "fields": "thumb_1024_url"},
        timeout=30,
    )
    if resp.status_code != 200:
        return None
    url = resp.json().get("thumb_1024_url")
    if not url:
        return None
    try:
        r = SESSION.get(url, timeout=45)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    return None


# ------------------------------ Tile assembly -------------------------------

def discover_tiles(city: str, city_info: dict) -> list[dict]:
    """Return an ordered list of candidate image metadata rows for a city.

    Merges results over several concentric search radii so central tiles get
    filled even when the city center is image-dense.
    """
    lat, lon = city_info["lat"], city_info["lon"]
    seen: dict[str, dict] = {}
    for radius_m in config.TIER1_SEARCH_RADII_M:
        batch = search_bbox(lat, lon, radius_m, limit=500)
        for img in batch:
            seen.setdefault(str(img["id"]), img)
        time.sleep(0.3)
        if len(seen) > 4000:
            break
    return list(seen.values())


def tier1_filter(imgs: list[dict]) -> list[dict]:
    """Apply Tier 1: is_pano=False, camera_type=perspective, daylight."""
    out = []
    for img in imgs:
        if img.get("is_pano", False):
            continue
        ctype = img.get("camera_type", "perspective")
        if ctype and ctype not in ("perspective", "flat", None):
            continue
        cap_ms = img.get("captured_at")
        if not cap_ms:
            continue
        lon, lat = img["geometry"]["coordinates"]
        elev = sun_elevation_deg(lat, lon, cap_ms)
        if math.isnan(elev) or elev < config.TIER1_MIN_SUN_ELEVATION_DEG:
            continue
        img["_sun_elev_deg"] = elev
        img["_lat"] = lat
        img["_lon"] = lon
        out.append(img)
    return out


def bin_to_res8_tiles(imgs: list[dict]) -> dict[str, list[dict]]:
    """Group images by H3 resolution-8 cell."""
    tiles: dict[str, list[dict]] = {}
    for img in imgs:
        cell = h3.latlng_to_cell(img["_lat"], img["_lon"], config.H3_RESOLUTION)
        tiles.setdefault(cell, []).append(img)
    return tiles


def pick_tiles(tile_groups: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    """Choose up to TILES_PER_CITY tiles with most Tier 1 survivors, spread out.

    Greedy spatial-spread selection: require each chosen tile's center to be
    at least 350 m from any already-chosen tile center (res-8 tile edge is
    ~460 m, so this enforces roughly non-overlapping neighborhoods).
    """
    ranked = sorted(tile_groups.items(), key=lambda x: -len(x[1]))
    chosen: list[tuple[str, list[dict]]] = []
    min_sep_km = 0.35
    for cell, group in ranked:
        if len(chosen) >= config.TILES_PER_CITY:
            break
        lat, lon = h3.cell_to_latlng(cell)
        too_close = False
        for cc, _ in chosen:
            clat, clon = h3.cell_to_latlng(cc)
            d_km = _haversine_km(lat, lon, clat, clon)
            if d_km < min_sep_km:
                too_close = True
                break
        if not too_close:
            chosen.append((cell, group))
    return chosen


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ------------------------------ Download loop -------------------------------

def download_tile(city: str, tile: str, imgs: list[dict],
                  out_root: Path) -> list[dict]:
    """Download up to CANDIDATES_PER_TILE images for a tile."""
    tile_dir = out_root / city / tile
    tile_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = {p.stem.replace("mapillary_", "") for p in tile_dir.glob("*.jpg")}
    rows: list[dict] = []
    for img in sorted(imgs, key=lambda x: -x.get("_sun_elev_deg", 0))[:config.CANDIDATES_PER_TILE]:
        img_id = str(img["id"])
        path = tile_dir / f"mapillary_{img_id}.jpg"
        if img_id not in existing_ids:
            pil = download_thumbnail(img_id)
            if pil is None:
                continue
            pil = pil.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
            pil.save(path, quality=95)
            time.sleep(0.1)
        rows.append({
            "image_id": img_id,
            "image_path": str(path.relative_to(config.ROOT).as_posix()),
            "city": city,
            "country": config.CITIES[city]["country"],
            "driving_side": config.CITIES[city]["driving"],
            "h3_tile_res8": tile,
            "latitude": img["_lat"],
            "longitude": img["_lon"],
            "captured_at_utc_ms": img["captured_at"],
            "sun_elev_deg": round(img["_sun_elev_deg"], 2),
            "camera_type": img.get("camera_type", "perspective"),
            "compass_angle": img.get("compass_angle"),
            "mapillary_quality": img.get("quality_score"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None,
                    help="Optional subset of city keys to download")
    ap.add_argument("--out", default=str(config.MAPILLARY_V2_DIR),
                    help="Output image root")
    ap.add_argument("--meta", default=str(config.PROCESSED_DIR / "tier1_candidates.csv"))
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    cities = args.cities if args.cities else list(config.CITIES.keys())
    print(f"[tier1] Downloading for {len(cities)} cities -> {out_root}")
    print(f"[tier1] Target: {config.TILES_PER_CITY} tiles/city "
          f"x {config.CANDIDATES_PER_TILE} candidates/tile")

    # Resume: load existing CSV if present
    existing_df = None
    if meta_path.exists():
        existing_df = pd.read_csv(meta_path)
        done_cities = set(existing_df["city"].unique())
        print(f"[tier1] Resume: {len(existing_df)} rows already "
              f"for {len(done_cities)} cities")
    else:
        done_cities = set()

    all_rows: list[dict] = [] if existing_df is None else existing_df.to_dict("records")

    for city in tqdm(cities, desc="cities"):
        if city in done_cities:
            continue
        info = config.CITIES[city]
        raw = discover_tiles(city, info)
        kept = tier1_filter(raw)
        tile_groups = bin_to_res8_tiles(kept)
        chosen = pick_tiles(tile_groups)
        print(f"  {city}: {len(raw)} raw -> {len(kept)} tier1 -> "
              f"{len(tile_groups)} tiles -> {len(chosen)} chosen")
        city_rows: list[dict] = []
        for tile_id, imgs in chosen:
            try:
                city_rows.extend(download_tile(city, tile_id, imgs, out_root))
            except Exception as e:
                print(f"    tile {tile_id[:12]}: error {e}")
        all_rows.extend(city_rows)
        # Checkpoint after each city
        pd.DataFrame(all_rows).to_csv(meta_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(meta_path, index=False)
    print(f"\n[tier1] Done. {len(df)} rows across "
          f"{df['city'].nunique()} cities, {df['h3_tile_res8'].nunique()} tiles")
    print(f"[tier1] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
