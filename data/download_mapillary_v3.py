"""
Block-level Mapillary fetcher for GeoFidelity-Bench v3 (replaces
`download_mapillary.py` for the v3 benchmark; v2's file is untouched).

Pipeline for each block in `blocks_v3.json`:
  1. Walk the OSM polyline in V3_SEARCH_STEP_M increments.
  2. At each step, issue a Mapillary bbox search with side V3_SEARCH_BBOX_DEG
     (~500 m) — small bboxes avoid the "please reduce the amount of data"
     500s seen with city-radius searches.
  3. Every search already returns `thumb_1024_url`, so a single API call
     per image suffices (no separate meta lookup).
  4. Tier 1 filter: non-panoramic, perspective, sun elevation >=
     TIER1_MIN_SUN_ELEVATION_DEG.
  5. Deduplicate across polyline steps by image_id.
  6. Download thumbnails up to V3_CANDIDATES_PER_BLOCK per block.

Outputs:
  data/raw/mapillary_v3/{block_id}/mapillary_{id}.jpg
  data/processed/v3/tier1_candidates.csv

CSV columns include per-image GPS (latitude, longitude), capture time,
compass angle, sun elevation, block association — the metadata addressed
by user request #2 ("each photo's GPS position should be recorded").
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import h3
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import config
from data.download_mapillary import sun_elevation_deg


TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    raise RuntimeError("Set MAPILLARY_TOKEN before downloading Mapillary images.")
BASE_URL = "https://graph.mapillary.com"
SESSION = requests.Session()
SEARCH_FIELDS = ("id,geometry,captured_at,is_pano,camera_type,compass_angle,"
                 "thumb_1024_url,quality_score")


# --------------------------------------------------------------------------
# Polyline walk
# --------------------------------------------------------------------------

def polyline_steps(polyline_latlon: list[list[float]],
                    step_m: float) -> list[tuple[float, float]]:
    """Resample a polyline to points spaced by step_m along its length."""
    if not polyline_latlon:
        return []
    pts = [(p[0], p[1]) for p in polyline_latlon]
    out = [pts[0]]
    carry = 0.0
    for i in range(1, len(pts)):
        lat1, lon1 = pts[i - 1]
        lat2, lon2 = pts[i]
        seg_m = _haversine_m(lat1, lon1, lat2, lon2)
        if seg_m <= 0:
            continue
        dist_to_next = step_m - carry
        while dist_to_next <= seg_m:
            f = dist_to_next / seg_m
            out.append((lat1 + f * (lat2 - lat1), lon1 + f * (lon2 - lon1)))
            dist_to_next += step_m
        carry = seg_m - (dist_to_next - step_m)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# Mapillary search
# --------------------------------------------------------------------------

def search_bbox(lat: float, lon: float, half_deg: float, limit: int = 100,
                 retries: int = 2) -> list[dict]:
    bbox = (f"{lon - half_deg},{lat - half_deg},"
            f"{lon + half_deg},{lat + half_deg}")
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(
                f"{BASE_URL}/images",
                params={
                    "access_token": TOKEN,
                    "fields": SEARCH_FIELDS,
                    "bbox": bbox,
                    "limit": limit,
                    "is_pano": "false",
                },
                timeout=45,
            )
            if r.status_code == 200:
                return r.json().get("data", [])
            if r.status_code == 429:
                # quota hit — back off
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code in (500, 502, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            return []
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
    return []


def download_image(url: str, target_size: int = None) -> Image.Image | None:
    try:
        r = SESSION.get(url, timeout=45)
        if r.status_code != 200:
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if target_size:
            img = img.resize((target_size, target_size), Image.LANCZOS)
        return img
    except Exception:
        return None


# --------------------------------------------------------------------------
# Per-block fetch
# --------------------------------------------------------------------------

def tier1_pass(img: dict) -> float | None:
    """Return sun-elevation if image passes Tier 1 filters, else None."""
    if img.get("is_pano", False):
        return None
    ctype = img.get("camera_type", "perspective")
    if ctype and ctype not in ("perspective", "flat", None):
        return None
    cap_ms = img.get("captured_at")
    if not cap_ms:
        return None
    coords = img.get("geometry", {}).get("coordinates")
    if not coords or len(coords) != 2:
        return None
    lon, lat = coords
    elev = sun_elevation_deg(lat, lon, cap_ms)
    if math.isnan(elev) or elev < config.TIER1_MIN_SUN_ELEVATION_DEG:
        return None
    return elev


def _h3_r9_for_point(lat: float, lon: float) -> str:
    return h3.latlng_to_cell(lat, lon, config.V3_BLOCK_H3_RES)


def _point_in_block(lat: float, lon: float, block: dict) -> bool:
    """Keep images whose h3 r-9 cell is in the block's buffered cell set."""
    cell = _h3_r9_for_point(lat, lon)
    return cell in set(block["h3_r9_cells"])


def _dl_one(img: dict, block_dir: Path) -> tuple[str, bool]:
    iid = str(img["id"])
    path = block_dir / f"mapillary_{iid}.jpg"
    if path.exists():
        return iid, True
    thumb = img.get("thumb_1024_url")
    if not thumb:
        return iid, False
    pil = download_image(thumb, target_size=config.IMG_SIZE)
    if pil is None:
        return iid, False
    pil.save(path, quality=95)
    return iid, True


def fetch_block(block: dict, out_root: Path) -> list[dict]:
    block_dir = out_root / block["block_id"]
    block_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem.replace("mapillary_", "")
                for p in block_dir.glob("*.jpg")}

    # Parallel search over polyline steps
    steps = polyline_steps(block["polyline"], config.V3_SEARCH_STEP_M)
    cell_set = set(block["h3_r9_cells"])
    seen: dict[str, dict] = {}

    def _one_search(lat_lon):
        lat, lon = lat_lon
        return search_bbox(lat, lon, config.V3_SEARCH_BBOX_DEG, limit=100)

    with ThreadPoolExecutor(max_workers=config.V3_SEARCH_WORKERS) as ex:
        for batch in ex.map(_one_search, steps):
            for img in batch:
                iid = str(img["id"])
                if iid in seen:
                    continue
                coords = img.get("geometry", {}).get("coordinates")
                if not coords:
                    continue
                lon_i, lat_i = coords
                if _h3_r9_for_point(lat_i, lon_i) not in cell_set:
                    continue
                seen[iid] = img

    # Tier 1 + sort by sun elevation
    survivors = []
    for iid, img in seen.items():
        elev = tier1_pass(img)
        if elev is None:
            continue
        img["_sun_elev_deg"] = elev
        img["_lat"] = img["geometry"]["coordinates"][1]
        img["_lon"] = img["geometry"]["coordinates"][0]
        survivors.append(img)
    survivors.sort(key=lambda x: -x["_sun_elev_deg"])
    survivors = survivors[:config.V3_CANDIDATES_PER_BLOCK]

    # Parallel thumbnail downloads
    with ThreadPoolExecutor(max_workers=config.V3_DOWNLOAD_WORKERS) as ex:
        futures = [ex.submit(_dl_one, img, block_dir) for img in survivors]
        dl_ok: set[str] = set()
        for fut in as_completed(futures):
            iid, ok = fut.result()
            if ok:
                dl_ok.add(iid)

    rows = []
    for img in survivors:
        iid = str(img["id"])
        if iid not in dl_ok and iid not in existing:
            continue
        path = block_dir / f"mapillary_{iid}.jpg"
        try:
            rel_path = path.relative_to(config.ROOT).as_posix()
        except ValueError:
            rel_path = str(path.as_posix())
        rows.append({
            "image_id": iid,
            "image_path": rel_path,
            "block_id": block["block_id"],
            "city": block["city"],
            "country": block["country"],
            "driving_side": block["driving_side"],
            "stratum": block["stratum"],
            "street_name": block["street_name"],
            "neighborhood": block["neighborhood"],
            "highway_tag": block["highway_tag"],
            "latitude": img["_lat"],
            "longitude": img["_lon"],
            "h3_r9_cell": _h3_r9_for_point(img["_lat"], img["_lon"]),
            "captured_at_utc_ms": img["captured_at"],
            "sun_elev_deg": round(img["_sun_elev_deg"], 2),
            "camera_type": img.get("camera_type", "perspective"),
            "compass_angle": img.get("compass_angle"),
            "mapillary_quality": img.get("quality_score"),
        })
    return rows


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=str(config.V3_BLOCKS_JSON))
    ap.add_argument("--cities", nargs="*", default=None,
                    help="Subset of city keys; defaults to all")
    ap.add_argument("--out_root", default=str(config.V3_DATA_DIR))
    ap.add_argument("--meta_csv", default=str(config.V3_TIER1_CSV))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min_per_block", type=int,
                    default=config.V3_TARGET_IMAGES_PER_BLOCK,
                    help="Skip blocks that already have >= this many images "
                         "in the CSV when --resume is set")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_csv).resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.blocks, "r", encoding="utf-8") as f:
        blocks = json.load(f)["blocks"]
    if args.cities:
        want = set(args.cities)
        blocks = [b for b in blocks if b["city"] in want]
    print(f"[v3-dl] {len(blocks)} blocks queued")

    # Resume: load existing CSV
    existing_rows: list[dict] = []
    done_blocks: set[str] = set()
    if meta_path.exists():
        old = pd.read_csv(meta_path)
        existing_rows = old.to_dict("records")
        if args.resume:
            counts = old.groupby("block_id").size()
            done_blocks = set(counts[counts >= args.min_per_block].index)
            print(f"[v3-dl] resume: {len(done_blocks)} blocks already "
                  f">= {args.min_per_block} imgs; skipping")

    all_rows = existing_rows[:]
    for block in tqdm(blocks, desc="blocks"):
        if block["block_id"] in done_blocks:
            continue
        try:
            rows = fetch_block(block, out_root)
        except Exception as exc:
            print(f"  {block['block_id']}: FAILED ({exc})")
            continue
        all_rows.extend(rows)
        # Dedup by (block_id, image_id)
        df = pd.DataFrame(all_rows).drop_duplicates(
            subset=["block_id", "image_id"], keep="last")
        df.to_csv(meta_path, index=False)
        all_rows = df.to_dict("records")
        if len(rows):
            safe = block['block_id'][:48].encode('ascii',
                                                  errors='replace').decode('ascii')
            print(f"  {safe:48s} -> {len(rows):4d} imgs")

    df = pd.DataFrame(all_rows)
    df.to_csv(meta_path, index=False)
    if len(df) and "block_id" in df.columns:
        print(f"\n[v3-dl] {len(df)} rows across "
              f"{df['block_id'].nunique()} blocks, "
              f"{df['city'].nunique()} cities")
    else:
        print(f"\n[v3-dl] 0 rows written (all blocks failed)")
    print(f"[v3-dl] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
