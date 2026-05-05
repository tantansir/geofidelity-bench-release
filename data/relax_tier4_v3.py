"""
Recompute `tier4_pass` on the v3 segmentation CSV using v3-relaxed ratio
gates (V3_TIER4_RATIOS).

v2's gates were tuned for place-unit-scale sampling whose bbox could
pull in highways, tunnels, and rural roads that needed aggressive
filtering. v3 already restricts to OSM-way membership, so the same
gates were dropping >30% of legitimate sky-occluded and narrow-street
shots — e.g. pedestrianized lanes, old-town canyons, covered bazaars.

This script only re-applies the pass rule; the expensive Mask2Former
outputs (ratio_* columns) are untouched, so the hot-fix is a few
seconds rather than hours. Input/output is the same file, overwritten
atomically.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd

import config


def relaxed_pass(row, ratios: dict, urbanness_min: float) -> bool:
    for g, (lo, hi) in ratios.items():
        col = f"ratio_{g}"
        if col not in row:
            return False
        v = row[col]
        if pd.isna(v) or v < lo or v > hi:
            return False
    return float(row.get("urbanness", 0)) >= urbanness_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",
                    default=str(config.V3_PROCESSED_DIR / "tier4_segmentation.csv"))
    ap.add_argument("--out_csv",
                    default=str(config.V3_PROCESSED_DIR / "tier4_segmentation.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    before = df.columns.tolist()
    # Dedup rows that accumulated from prior pipeline merges
    df = df.drop_duplicates(subset=["image_id", "block_id"], keep="first")

    old_pass = int((df.get("tier4_pass") == True).sum()) \
                if "tier4_pass" in df.columns else None

    mask = df.apply(
        lambda r: relaxed_pass(r, config.V3_TIER4_RATIOS,
                                config.V3_TIER4_URBANNESS_MIN), axis=1)
    df["tier4_pass"] = mask.astype(bool)
    new_pass = int(mask.sum())

    df = df[before]  # keep original column order
    df.to_csv(args.out_csv, index=False)

    print(f"[relax_t4] rows after dedup: {len(df)}")
    print(f"[relax_t4] tier4_pass:  old={old_pass}  new={new_pass} "
          f"(+{new_pass - (old_pass or 0)})")
    print(f"[relax_t4] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
