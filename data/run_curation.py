"""
Run the full 6-tier curation pipeline end-to-end.

Intended shell use:
    python data/run_curation.py all                      # all steps
    python data/run_curation.py download                 # Tier 1 only
    python data/run_curation.py filter                   # Tiers 2-5
    python data/run_curation.py review_build             # Tier 6 phase 1
    python data/run_curation.py review_apply             # Tier 6 phase 2
    python data/run_curation.py curate                   # benchmark spec

Heavy GPU steps (Tier 3 SigLIP and Tier 4 Mask2Former) can be run on
AutoDL by pointing PYTHONPATH at the repo and running the same commands.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess

import config


PY = sys.executable
ROOT = config.ROOT


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def download():
    run([PY, str(ROOT / "data" / "download_mapillary.py")])


def tier2():
    run([PY, str(ROOT / "data" / "filter_tier2_osm.py")])


def tier3():
    run([PY, str(ROOT / "data" / "filter_tier3_siglip.py"),
         "--skip_failed_tiers"])


def tier4():
    run([PY, str(ROOT / "data" / "filter_tier4_segmentation.py")])


def tier5():
    run([PY, str(ROOT / "data" / "filter_tier5_quality.py")])


def review_build():
    run([PY, str(ROOT / "data" / "filter_tier6_review.py"), "generate"])


def review_apply():
    run([PY, str(ROOT / "data" / "filter_tier6_review.py"), "apply"])


def curate():
    run([PY, str(ROOT / "data" / "curate_places.py")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=[
        "all", "download", "filter", "tier2", "tier3", "tier4", "tier5",
        "review_build", "review_apply", "curate",
    ])
    args = ap.parse_args()

    if args.step == "all":
        for fn in (download, tier2, tier3, tier4, tier5,
                    review_build, curate):  # review_apply is manual
            fn()
    elif args.step == "filter":
        for fn in (tier2, tier3, tier4, tier5):
            fn()
    else:
        globals()[args.step]()


if __name__ == "__main__":
    main()
