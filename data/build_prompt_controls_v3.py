"""
Build v3 control-prompt assignments for prompt-specificity ablations.

This script creates deterministic donor assignments for three controls:
  * C_WRONG_STREET: replace the target street name with a different one
  * C_SHUFFLED_NEIGHBORHOOD: replace the target neighborhood label
  * C_WRONG_STREET_NEIGHBORHOOD: replace both

The target city/country remain unchanged so the controls preserve prompt
length and named-entity structure while removing the correct local cue.
Assignments prefer donors from the same city, and then prefer the same
stratum to avoid turning the control into a coarse road-type change.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from collections import Counter, defaultdict

import config


CONTROL_LEVELS = (
    "C_WRONG_STREET",
    "C_SHUFFLED_NEIGHBORHOOD",
    "C_WRONG_STREET_NEIGHBORHOOD",
)


def _load_blocks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["blocks"]


def _pick_donor(block: dict, blocks: list[dict], seed_key: str,
                require_street_diff: bool = False,
                require_neighborhood_diff: bool = False) -> dict | None:
    candidates = []
    for cand in blocks:
        if cand["block_id"] == block["block_id"]:
            continue
        if require_street_diff and cand["street_name"] == block["street_name"]:
            continue
        if require_neighborhood_diff and cand["neighborhood"] == block["neighborhood"]:
            continue
        candidates.append(cand)
    if not candidates:
        return None

    def tier(cand: dict) -> tuple[int, int, str]:
        same_city = cand["city"] == block["city"]
        same_stratum = cand["stratum"] == block["stratum"]
        return (
            0 if same_city and same_stratum else
            1 if same_city else
            2 if same_stratum else
            3,
            0 if cand["city"] == block["city"] else 1,
            cand["block_id"],
        )

    ranked = sorted(candidates, key=tier)
    best_tier = tier(ranked[0])[0]
    pool = [cand for cand in ranked if tier(cand)[0] == best_tier]
    rng = random.Random(f"{block['block_id']}|{seed_key}")
    return rng.choice(pool)


def build_controls(blocks: list[dict]) -> dict:
    per_city = defaultdict(list)
    for block in blocks:
        per_city[block["city"]].append(block)

    out_blocks = {}
    summary = Counter()
    for block in blocks:
        controls = {}

        street_donor = _pick_donor(
            block,
            per_city[block["city"]],
            "street_same_city",
            require_street_diff=True,
        )
        if street_donor is None:
            street_donor = _pick_donor(
                block,
                blocks,
                "street_global_fallback",
                require_street_diff=True,
            )
        if street_donor is not None:
            controls["C_WRONG_STREET"] = {
                "street_name": street_donor["street_name"],
                "source_block_id": street_donor["block_id"],
                "source_city": street_donor["city"],
                "selection_scope": "same_city" if street_donor["city"] == block["city"] else "global_fallback",
                "selection_stratum_match": street_donor["stratum"] == block["stratum"],
            }
            summary["C_WRONG_STREET"] += 1

        neighborhood_donor = _pick_donor(
            block,
            per_city[block["city"]],
            "neighborhood_same_city",
            require_neighborhood_diff=True,
        )
        if neighborhood_donor is None:
            neighborhood_donor = _pick_donor(
                block,
                blocks,
                "neighborhood_global_fallback",
                require_neighborhood_diff=True,
            )
        if neighborhood_donor is not None:
            controls["C_SHUFFLED_NEIGHBORHOOD"] = {
                "neighborhood": neighborhood_donor["neighborhood"],
                "source_block_id": neighborhood_donor["block_id"],
                "source_city": neighborhood_donor["city"],
                "selection_scope": "same_city" if neighborhood_donor["city"] == block["city"] else "global_fallback",
                "selection_stratum_match": neighborhood_donor["stratum"] == block["stratum"],
            }
            summary["C_SHUFFLED_NEIGHBORHOOD"] += 1

        combo_street = street_donor or _pick_donor(
            block,
            blocks,
            "combo_street",
            require_street_diff=True,
        )
        combo_neighborhood = neighborhood_donor or _pick_donor(
            block,
            blocks,
            "combo_neighborhood",
            require_neighborhood_diff=True,
        )
        if combo_street is not None and combo_neighborhood is not None:
            controls["C_WRONG_STREET_NEIGHBORHOOD"] = {
                "street_name": combo_street["street_name"],
                "street_source_block_id": combo_street["block_id"],
                "street_source_city": combo_street["city"],
                "neighborhood": combo_neighborhood["neighborhood"],
                "neighborhood_source_block_id": combo_neighborhood["block_id"],
                "neighborhood_source_city": combo_neighborhood["city"],
                "street_scope": "same_city" if combo_street["city"] == block["city"] else "global_fallback",
                "neighborhood_scope": "same_city" if combo_neighborhood["city"] == block["city"] else "global_fallback",
            }
            summary["C_WRONG_STREET_NEIGHBORHOOD"] += 1

        out_blocks[block["block_id"]] = controls

    return {
        "name": "GeoFidelity-Bench prompt controls",
        "version": "1.0.0",
        "benchmark": str(config.V3_BENCHMARK_JSON.relative_to(config.ROOT).as_posix()),
        "levels": list(CONTROL_LEVELS),
        "summary": {
            "num_blocks": len(blocks),
            "available_per_level": dict(summary),
        },
        "blocks": out_blocks,
    }


def build_preview_rows(blocks: list[dict], controls: dict) -> list[dict]:
    by_id = {block["block_id"]: block for block in blocks}
    rows = []
    for block_id, spec in controls["blocks"].items():
        block = by_id[block_id]
        for level in CONTROL_LEVELS:
            ctrl = spec.get(level)
            if not ctrl:
                continue
            row = {
                "block_id": block_id,
                "city": block["city"],
                "stratum": block["stratum"],
                "level": level,
                "target_street_name": block["street_name"],
                "target_neighborhood": block["neighborhood"],
                "control_street_name": ctrl.get("street_name", block["street_name"]),
                "control_neighborhood": ctrl.get("neighborhood", block["neighborhood"]),
            }
            row.update({k: v for k, v in ctrl.items() if k not in row})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(config.V3_BENCHMARK_JSON))
    ap.add_argument(
        "--out_json",
        default=str(config.V3_PROCESSED_DIR / "prompt_controls_v3.json"),
    )
    ap.add_argument(
        "--out_csv",
        default=str(config.V3_PROCESSED_DIR / "prompt_controls_v3_preview.csv"),
    )
    args = ap.parse_args()

    blocks = _load_blocks(Path(args.benchmark))
    controls = build_controls(blocks)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(controls, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = build_preview_rows(blocks, controls)
    import pandas as pd
    pd.DataFrame(rows).to_csv(Path(args.out_csv), index=False)

    print(f"[build_prompt_controls_v3] wrote {out_json}")
    print(f"[build_prompt_controls_v3] wrote {args.out_csv}")
    print(json.dumps(controls["summary"], indent=2))


if __name__ == "__main__":
    main()
