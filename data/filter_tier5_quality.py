"""
Tier 5 filter: low-level image quality gates.

Four cheap CPU-side checks:
  * Blur:         variance of Laplacian >= TIER5_MIN_LAPLACIAN_VAR
  * Exposure:     mean luminance in TIER5_LUMINANCE_RANGE
  * Colorfulness: Hasler & Susstrunk (2003) >= TIER5_MIN_COLORFULNESS
  * Aspect:       image aspect ratio in TIER5_ASPECT_RATIO_RANGE

This catches whatever slipped past the perceptual filters (Tiers 3 & 4):
motion-blurred drive-by shots, dusk / overexposed highlights, and foggy /
grayscale dashcam frames that can still semantically "look urban".

Input:  data/processed/tier4_segmentation.csv
Output: data/processed/tier5_quality.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler & Susstrunk 2003 (M3)."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def _metrics(path: Path) -> dict:
    # cv2.imread fails silently on Windows when the path contains non-ASCII
    # characters (GBK fallback), which otherwise drops whole cities like
    # Istanbul / Bangkok / Seoul / Tokyo whose OSM street names are Unicode.
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return {"lowlevel_error": True}
    if arr.size == 0:
        return {"lowlevel_error": True}
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"lowlevel_error": True}
    h, w = img.shape[:2]
    aspect = w / max(1, h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luma = float(gray.mean())
    cf = _colorfulness(img)
    return {
        "lap_var": round(lap_var, 2),
        "luma_mean": round(luma, 2),
        "colorfulness": round(cf, 2),
        "aspect_ratio": round(aspect, 3),
    }


def _passes(m: dict) -> bool:
    if m.get("lowlevel_error"):
        return False
    lo_l, hi_l = config.TIER5_LUMINANCE_RANGE
    lo_a, hi_a = config.TIER5_ASPECT_RATIO_RANGE
    if m["lap_var"] < config.TIER5_MIN_LAPLACIAN_VAR:
        return False
    if not (lo_l <= m["luma_mean"] <= hi_l):
        return False
    if m["colorfulness"] < config.TIER5_MIN_COLORFULNESS:
        return False
    if not (lo_a <= m["aspect_ratio"] <= hi_a):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(config.PROCESSED_DIR / "tier4_segmentation.csv"))
    ap.add_argument("--out_csv", default=str(config.PROCESSED_DIR / "tier5_quality.csv"))
    ap.add_argument("--ignore_prior", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.ignore_prior:
        active = df.copy()
    else:
        prior = [c for c in ("tier2_pass", "tier3_pass", "tier4_pass")
                 if c in df.columns]
        mask = pd.Series([True] * len(df))
        for c in prior:
            mask &= df[c].fillna(False).astype(bool)
        active = df[mask].copy()
    print(f"[tier5] {len(active)}/{len(df)} images to measure")

    results: list[dict] = []
    for p_rel in tqdm(active["image_path"].tolist(), desc="lowlevel"):
        p = config.ROOT / p_rel
        m = _metrics(p)
        m["image_path"] = p_rel
        m["tier5_pass"] = _passes(m)
        results.append(m)

    add = pd.DataFrame(results)
    merged = df.merge(add, on="image_path", how="left")
    merged["tier5_pass"] = merged["tier5_pass"].fillna(False).astype(bool)
    merged.to_csv(args.out_csv, index=False)

    n = int(merged["tier5_pass"].sum())
    print(f"[tier5] pass {n}/{len(merged)} ({100.0*n/max(1,len(merged)):.1f}%)")


if __name__ == "__main__":
    main()
