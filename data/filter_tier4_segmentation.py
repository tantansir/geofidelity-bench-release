"""
Tier 4 filter: Mask2Former Mapillary-Vistas semantic ratio gates.

Runs semantic segmentation on every Tier-3 survivor (or every image if
--ignore_prior) and applies min/max pixel-ratio gates on sky / building /
road / vehicle / vegetation plus an "urbanness" composite score. Images
failing any gate are marked tier4_pass=False.

Side effect: all per-image group ratios are persisted in the CSV so the
GAAS metric (which uses the same backbone) can reuse them without a
second inference pass.

Input:  data/processed/tier3_siglip.csv
Output: data/processed/tier4_segmentation.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from data.segmentation_backbone import (Mask2FormerVistas, VISTAS_GROUPS,
                                         ratios_from_seg, urbanness)


def passes_ratio_gates(ratios: dict[str, float]) -> bool:
    for group, (lo, hi) in config.TIER4_RATIOS.items():
        r = ratios.get(group, 0.0)
        if r < lo or r > hi:
            return False
    if urbanness(ratios) < config.TIER4_URBANNESS_MIN:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(config.PROCESSED_DIR / "tier3_siglip.csv"))
    ap.add_argument("--out_csv", default=str(config.PROCESSED_DIR / "tier4_segmentation.csv"))
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--batch_size", type=int, default=4)     # Mask2Former is heavy
    ap.add_argument("--ignore_prior", action="store_true",
                    help="Segment every image even if prior tier failed")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.ignore_prior:
        df_active = df.copy()
    else:
        prior_pass_cols = [c for c in ("tier2_pass", "tier3_pass") if c in df.columns]
        mask = pd.Series([True] * len(df))
        for c in prior_pass_cols:
            mask &= df[c].fillna(False).astype(bool)
        df_active = df[mask].copy()
    print(f"[tier4] segmenting {len(df_active)}/{len(df)} images")

    seg = Mask2FormerVistas(device=args.device)

    results: list[dict] = []
    paths = [config.ROOT / p for p in df_active["image_path"].tolist()]

    for i in tqdm(range(0, len(paths), args.batch_size), desc="segment"):
        batch = paths[i:i + args.batch_size]
        try:
            pils = [Image.open(p).convert("RGB") for p in batch]
        except Exception as e:
            print(f"  open error: {e}")
            for p in batch:
                results.append({"image_path": str(p.relative_to(config.ROOT).as_posix()),
                                "seg_error": True, "tier4_pass": False})
            continue
        try:
            segs = seg.segment_batch(pils)
        except Exception as e:
            print(f"  seg error: {e}")
            for p in batch:
                results.append({"image_path": str(p.relative_to(config.ROOT).as_posix()),
                                "seg_error": True, "tier4_pass": False})
            continue

        for p, s in zip(batch, segs):
            r = ratios_from_seg(s)
            row: dict = {"image_path": str(p.relative_to(config.ROOT).as_posix())}
            for g in VISTAS_GROUPS:
                row[f"ratio_{g}"] = round(r[g], 5)
            row["urbanness"] = round(urbanness(r), 5)
            row["tier4_pass"] = bool(passes_ratio_gates(r))
            results.append(row)

    add = pd.DataFrame(results)
    merged = df.merge(add, on="image_path", how="left")
    merged["tier4_pass"] = merged["tier4_pass"].fillna(False).astype(bool)
    merged.to_csv(args.out_csv, index=False)

    n_pass = int(merged["tier4_pass"].sum())
    print(f"[tier4] pass {n_pass}/{len(merged)} "
          f"({100.0 * n_pass / max(1, len(merged)):.1f}%)")


if __name__ == "__main__":
    main()
