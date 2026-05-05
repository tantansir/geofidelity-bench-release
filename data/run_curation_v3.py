"""
Run the v3 curation pipeline end-to-end.

v3 skips Tier 2 (OSM road-type) because images are already fetched
along named OSM ways in `download_mapillary_v3.py`; the highway tag is
known at download time and stored in tier1_candidates.csv.

Pipeline:
    tier1  (download_mapillary_v3.py)     -> data/processed/v3/tier1_candidates.csv
    tier3  (SigLIP scene classification)  -> data/processed/v3/tier3_siglip.csv
    tier4  (Mask2Former-Vistas seg)       -> data/processed/v3/tier4_segmentation.csv
    tier5  (blur / luminance / colour)    -> data/processed/v3/tier5_quality.csv
    curate (curate_blocks_v3.py)          -> data/processed/v3/benchmark_v3.json

Invoke with:
    python data/run_curation_v3.py all
    python data/run_curation_v3.py tier3            # individual stages
    python data/run_curation_v3.py curate
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess

import config


PY = sys.executable
ROOT = config.ROOT
V3P = config.V3_PROCESSED_DIR


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def download():
    _run([PY, str(ROOT / "data" / "download_mapillary_v3.py"), "--resume"])


def tier3():
    _run([PY, str(ROOT / "data" / "filter_tier3_siglip.py"),
          "--in_csv",  str(config.V3_TIER1_CSV),
          "--out_csv", str(V3P / "tier3_siglip.csv"),
          "--skip_failed_tiers"])


def tier4():
    _run([PY, str(ROOT / "data" / "filter_tier4_segmentation.py"),
          "--in_csv",  str(V3P / "tier3_siglip.csv"),
          "--out_csv", str(V3P / "tier4_segmentation.csv")])


def tier5():
    _run([PY, str(ROOT / "data" / "filter_tier5_quality.py"),
          "--in_csv",  str(V3P / "tier4_segmentation.csv"),
          "--out_csv", str(V3P / "tier5_quality.csv")])


def curate():
    _run([PY, str(ROOT / "data" / "curate_blocks_v3.py"),
          "--blocks_spec", str(config.V3_BLOCKS_JSON),
          "--in_csv",      str(V3P / "tier5_quality.csv"),
          "--pass_col",    "tier5_pass",
          "--out",         str(config.V3_BENCHMARK_JSON)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["all", "download", "tier3", "tier4",
                                      "tier5", "filter", "curate"])
    args = ap.parse_args()

    if args.step == "all":
        for fn in (download, tier3, tier4, tier5, curate):
            fn()
    elif args.step == "filter":
        for fn in (tier3, tier4, tier5):
            fn()
    else:
        globals()[args.step]()


if __name__ == "__main__":
    main()
