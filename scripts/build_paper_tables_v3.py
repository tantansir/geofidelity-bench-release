"""
Populate v3 placeholders in paper/numbers.tex.

Extends `build_paper_tables.py` (v2) with v3-specific macros. The two
scripts coexist; run v2 first, then v3, which only overwrites the v3
macros and appends new ones without touching v2 lines.

New macros defined by v3:
    \\NUMBLOCKS           -- block units in v3 (was \\NUMPLACES in v2)
    \\NUMIMAGESV3         -- total curated reference images in v3
    \\NUMAVGBLOCK         -- mean images per block
    \\NUMCITIESV3         -- cities covered by v3 (may differ from 25 if
                              some were dropped at tier5)
    \\V3L0<STEM><M>, \\V3L1<STEM><M>, \\V3L2<STEM><M>
                            per-model × level × metric means (STEM = one
                            of SDXL/SDM/SDL/FLD/FLS/PX/HD; M = one of
                            C/DCSF/G/RET for CosSim/DCSF/GAAS/retrieval)
    \\V3DELTAL1L0C        -- mean CosSim gain L1 vs L0 across models
    \\V3DELTAL2L0C        -- mean CosSim gain L2 vs L0
    \\V3METRICHUMAN<M>    -- Spearman rho(metric, human preference)

Inputs (all optional, any missing keeps existing TODO):
    data/processed/v3/benchmark_v3.json
    outputs/eval_v3/raw_results.csv
    outputs/eval_v3/metric_human_correlation.csv

Output:
    paper/numbers.tex (append / update only v3 keys — v2 keys untouched)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config


# v2 macro stems reused for the six formal v3 generators.
MODEL_STEMS = {
    "sdxl_base":    "SDXL",
    "sd35_large":   "SDL",
    "flux_dev":     "FLD",
    "flux_schnell": "FLS",
    "pixart_sigma": "PX",
    "hunyuan_dit":  "HD",
}
METRIC_STEMS = {
    "cos_sim":       "C",
    "dcsf":          "DCSF",
    "gaas":          "G",
    "retrieval_acc": "RET",
    "mmd":           "MMD",
}


def _fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.{nd}f}"


def _cmd(name: str, value: str) -> str:
    # \providecommand defines the macro if absent (so paper compiles without
    # pre-declared placeholders) and is overwritten by a later \renewcommand
    # when the key is already bound in main.tex (e.g. NUMBLOCKS).
    return f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}"


def collect_bench(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    n_blocks = b["num_blocks"]
    n_cities = b["num_cities"]
    n_images = b["num_images"]
    out = {
        "NUMBLOCKS":   str(n_blocks),
        "NUMIMAGESV3": str(n_images),
        "NUMCITIESV3": str(n_cities),
        "NUMAVGBLOCK": f"{n_images / max(1, n_blocks):.1f}",
    }
    # per-city v3 table (street count + image count)
    by_city: dict[str, list[dict]] = {}
    for blk in b["blocks"]:
        by_city.setdefault(blk["city"], []).append(blk)
    rows = []
    for city in sorted(by_city):
        ps = by_city[city]
        n_img = sum(len(blk["images"]) for blk in ps)
        info = config.CITIES[city]
        drive = "L" if info["driving"] == "left" else "R"
        rows.append(f"{city.replace('_',' ').title()} & {info['country']} & "
                    f"{info['lat']:.2f} & {info['lon']:.2f} & {drive} & "
                    f"{len(ps)} & {n_img} \\\\")
    out["CITYTABLEV3"] = "\n".join(rows)
    return out


def collect_eval(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, str] = {}
    if df.empty:
        return out

    # Per (method, level) metric means across blocks
    grouped = df.groupby(["method", "level"]).mean(numeric_only=True)
    for (method, level), row in grouped.iterrows():
        if method not in MODEL_STEMS:
            continue
        ms = MODEL_STEMS[method]
        for metric, m_stem in METRIC_STEMS.items():
            if metric not in row.index:
                continue
            out[f"V3{level}{ms}{m_stem}"] = _fmt(row[metric], nd=3)

    # Prompt-level gains (mean across models of delta between levels)
    model_level = (df[df["method"].isin(MODEL_STEMS)]
                     .groupby(["method", "level"])["cos_sim"]
                     .mean().unstack("level"))
    if {"L0", "L1"} <= set(model_level.columns):
        out["V3DELTAL1L0C"] = _fmt((model_level["L1"]
                                    - model_level["L0"]).mean(), nd=3)
    if {"L0", "L2"} <= set(model_level.columns):
        out["V3DELTAL2L0C"] = _fmt((model_level["L2"]
                                    - model_level["L0"]).mean(), nd=3)
    if {"L1", "L2"} <= set(model_level.columns):
        out["V3DELTAL2L1C"] = _fmt((model_level["L2"]
                                    - model_level["L1"]).mean(), nd=3)

    # Retrieval saturation break — report mean retrieval_acc across methods
    if "retrieval_acc" in df.columns:
        # aggregate over main models only
        sub = df[df["method"].isin(MODEL_STEMS)]
        if len(sub):
            out["V3MEANRETACC"] = _fmt(sub["retrieval_acc"].mean(), nd=3)
        for method in ("oracle_nn", "random_global", "random_same_country"):
            m = df[df["method"] == method]
            if len(m):
                stem = {"oracle_nn": "OO", "random_global": "RG",
                        "random_same_country": "RC"}[method]
                out[f"V3{stem}RET"] = _fmt(m["retrieval_acc"].mean(), nd=3)
    return out


def collect_metric_human(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        metric = r["metric"]
        if metric not in METRIC_STEMS:
            continue
        stem = METRIC_STEMS[metric]
        out[f"V3METRICHUMAN{stem}"] = _fmt(r["spearman_rho"], nd=2)
        out[f"V3METRICHUMANP{stem}"] = _fmt(r["p_value"], nd=3)
    return out


def merge_into(numbers_path: Path, new_vals: dict[str, str]) -> None:
    """Update-or-append \\renewcommand lines in numbers.tex.

    Every key becomes `\\renewcommand{\\KEY}{VAL}`. Existing lines for
    the same key are replaced; new keys are appended before EOF.
    """
    if numbers_path.exists():
        text = numbers_path.read_text(encoding="utf-8")
    else:
        text = ("% Auto-generated from evaluation outputs by "
                "scripts/build_paper_tables_v3.py.\n")
    for key, val in new_vals.items():
        line = _cmd(key, val)
        # Match either the legacy "\renewcommand{\KEY}{...}" or the new
        # "\providecommand{\KEY}{}\renewcommand{\KEY}{...}" form, so we
        # overwrite cleanly regardless of which one numbers.tex already has.
        pat = re.compile(
            rf"(?:\\providecommand\{{\\{re.escape(key)}\}}\{{[^}}]*\}})?"
            rf"\\renewcommand\{{\\{re.escape(key)}\}}\{{[^}}]*\}}",
            re.DOTALL)
        # Use a function replacement to avoid `\p` / `\N` being interpreted
        # as regex backrefs in the replacement string.
        if pat.search(text):
            text = pat.sub(lambda _m: line, text)
        else:
            text = text.rstrip() + "\n" + line + "\n"
    numbers_path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(config.V3_BENCHMARK_JSON))
    ap.add_argument("--eval_csv",
                    default=str(config.OUTPUT_DIR / "eval_v3" / "raw_results.csv"))
    ap.add_argument("--human_csv",
                    default=str(config.OUTPUT_DIR / "eval_v3"
                                 / "metric_human_correlation.csv"))
    ap.add_argument("--numbers",
                    default=str(config.ROOT / "paper" / "numbers.tex"))
    args = ap.parse_args()

    vals: dict[str, str] = {}
    vals.update(collect_bench(Path(args.benchmark)))
    vals.update(collect_eval(Path(args.eval_csv)))
    vals.update(collect_metric_human(Path(args.human_csv)))

    if not vals:
        print("[build_v3] nothing to write (inputs all missing)")
        return

    merge_into(Path(args.numbers), vals)
    print(f"[build_v3] updated {len(vals)} macros in {args.numbers}")
    for k in sorted(vals):
        print(f"  {k}: {vals[k][:60]}{'...' if len(vals[k]) > 60 else ''}")


if __name__ == "__main__":
    main()
