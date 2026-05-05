"""
Re-pull Mapillary thumbnails for rows of a Tier-N CSV directly on the GPU
host, so we never need to ship the 1.5 GB image bundle over a slow home
uplink. Skips files that already exist and is safe to run repeatedly.

Usage:
    python data/fetch_from_csv.py \\
        --csv data/processed/tier2_osm.csv \\
        --filter-col tier2_pass \\
        --workers 16
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import config

MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not MAPILLARY_TOKEN:
    raise RuntimeError("Set MAPILLARY_TOKEN before downloading Mapillary images.")
BASE_URL = "https://graph.mapillary.com"


def _fetch_one(row: dict) -> tuple[str, bool]:
    rel = row["image_path"]
    dst = config.ROOT / rel
    if dst.exists() and dst.stat().st_size > 1000:
        return rel, True
    dst.parent.mkdir(parents=True, exist_ok=True)
    img_id = str(row["image_id"])
    try:
        r = requests.get(f"{BASE_URL}/{img_id}",
                         params={"access_token": MAPILLARY_TOKEN,
                                 "fields": "thumb_1024_url"}, timeout=30)
        if r.status_code != 200:
            return rel, False
        url = r.json().get("thumb_1024_url")
        if not url:
            return rel, False
        r2 = requests.get(url, timeout=45)
        if r2.status_code != 200:
            return rel, False
        img = Image.open(BytesIO(r2.content)).convert("RGB")
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
        img.save(dst, quality=95)
        return rel, True
    except Exception:
        return rel, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.PROCESSED_DIR / "tier2_osm.csv"))
    ap.add_argument("--filter-col", default="tier2_pass")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.filter_col and args.filter_col in df.columns:
        df = df[df[args.filter_col].fillna(False).astype(bool)].copy()
    print(f"[fetch] {len(df)} rows (filtered on {args.filter_col})")

    rows = df.to_dict("records")
    ok = 0; miss = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_fetch_one, r) for r in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="fetch"):
            _, success = fut.result()
            if success:
                ok += 1
            else:
                miss += 1
    print(f"[fetch] ok={ok}  miss={miss}")


if __name__ == "__main__":
    main()
