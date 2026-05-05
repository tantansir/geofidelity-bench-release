"""
Populate paper/numbers.tex from evaluation CSVs.

Run after the metric-validity, within-synthetic, and main eval scripts
finish; it re-renders every \TODO placeholder in paper/main.tex with a
concrete numeric value. Missing inputs are tolerated: any placeholder
whose source is not yet available keeps its \TODO red text.

Inputs (all optional):
    outputs/validity_v2/metric_validity_summary.csv
    outputs/within_synthetic/within_synthetic_summary.csv
    outputs/eval_v2/raw_results.csv
    outputs/human_eval/analysis/*.csv
    data/processed/benchmark_v2.json

Output:
    paper/numbers.tex
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config


MODEL_MACRO_STEMS = {
    "sdxl_base":    "SDXL",
    "sd35_large":   "SDL",
    "flux_dev":     "FLD",
    "flux_schnell": "FLS",
    "pixart_sigma": "PX",
    "hunyuan_dit":  "HD",
}

CONDITIONS = ["same_place", "same_city_wrong_nbhd",
              "same_climate_wrong_city", "random_city"]
VALID_STEMS = {"same_place": "SP", "same_city_wrong_nbhd": "SC",
               "same_climate_wrong_city": "CL", "random_city": "RN"}
METRIC_STEMS = {"cos_sim": "COS", "dcsf": "DCSF", "mmd": "MMD", "gaas": "GAAS"}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.{nd}f}"


def cmd(name, value):
    return f"\\renewcommand{{\\{name}}}{{{value}}}"


def collect_benchmark_stats(bench_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not bench_path.exists():
        return out
    with open(bench_path, "r", encoding="utf-8") as f:
        b = json.load(f)
    n_places = b["num_places"]
    n_cities = b["num_cities"]
    n_images = b["num_images"]
    avg = n_images / max(1, n_places)
    out["NUMPLACES"] = str(n_places)
    out["NUMCITIES"] = str(n_cities)
    out["NUMIMAGES"] = str(n_images)
    out["NUMAVG"] = f"{avg:.1f}"
    # City table rows
    rows = []
    places_by_city = defaultdict(list)
    for p in b["places"]:
        places_by_city[p["city"]].append(p)
    for city in sorted(places_by_city):
        ps = places_by_city[city]
        lat, lon = config.CITIES[city]["lat"], config.CITIES[city]["lon"]
        country = config.CITIES[city]["country"]
        drive = "L" if config.CITIES[city]["driving"] == "left" else "R"
        n_tiles = len(ps)
        n_imgs = sum(len(p["image_paths"]) for p in ps)
        rows.append(f"{city.replace('_',' ').title()} & {country} & "
                    f"{lat:.2f} & {lon:.2f} & {drive} & {n_tiles} & {n_imgs} \\\\")
    out["CITYTABLE"] = "\n".join(rows)
    return out


def collect_validity(validity_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not validity_csv.exists():
        return out
    df = pd.read_csv(validity_csv)
    for _, r in df.iterrows():
        cond = r["condition"]; m = r["metric"]
        if cond not in VALID_STEMS or m not in METRIC_STEMS:
            continue
        macro = f"VALID{VALID_STEMS[cond]}{METRIC_STEMS[m]}"
        out[macro] = fmt(r["mean"], nd=3)
    # Spearman rho placeholders: fill after computing
    for m, ms in METRIC_STEMS.items():
        m_rows = df[df["metric"] == m]
        if len(m_rows) < 3:
            continue
        means = [float(m_rows[m_rows["condition"] == c]["mean"].values[0])
                 if c in m_rows["condition"].values else np.nan
                 for c in CONDITIONS]
        exp_rank = list(range(len(means)))
        if m == "cos_sim":
            obs_rank = (-np.array(means)).argsort().argsort().tolist()
        else:
            obs_rank = np.array(means).argsort().argsort().tolist()
        rho, _ = spearmanr(exp_rank, obs_rank)
        out[f"VALIDSP{METRIC_STEMS[m]}R"] = fmt(rho, nd=2)
    return out


def collect_within_synth(summary_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not summary_csv.exists():
        return out
    df = pd.read_csv(summary_csv)
    for model, stem in MODEL_MACRO_STEMS.items():
        sub = df[df["model"] == model]
        for m, ms in METRIC_STEMS.items():
            cell = sub[sub["metric"] == m]
            if len(cell):
                macro = f"WSGEO{stem}{ms[0]}"
                out[macro] = fmt(cell["mean_delta"].values[0], nd=3)
    return out


def collect_main_eval(csv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not csv_path.exists():
        return out
    df = pd.read_csv(csv_path)
    grouped = df.groupby("method").mean(numeric_only=True)
    baseline_map = {"oracle_nn": "OO",
                    "random_same_country": "RC",
                    "random_global": "RG"}
    for method, stem in {**baseline_map, **MODEL_MACRO_STEMS}.items():
        if method not in grouped.index:
            continue
        row = grouped.loc[method]
        prefix = "MAIN"
        out[f"{prefix}{stem}G"] = fmt(row.get("gaas"), nd=3)
        out[f"{prefix}{stem}C"] = fmt(row.get("cos_sim"), nd=3)
        out[f"{prefix}{stem}DCSF"] = fmt(row.get("dcsf"), nd=3)
        out[f"{prefix}{stem}MMD"] = fmt(row.get("mmd"), nd=3)
        out[f"{prefix}{stem}RET"] = fmt(row.get("retrieval_acc"), nd=3)

    # Rank Spearman across metrics (within open-source models)
    opens = [m for m in MODEL_MACRO_STEMS.keys() if m in grouped.index]
    matrix: list[tuple[str, list[float]]] = []
    for mname, col in [("gaas", "gaas"), ("cos_sim", "cos_sim"),
                       ("dcsf", "dcsf"), ("mmd", "mmd")]:
        vals = [grouped.loc[m, col] for m in opens if m in grouped.index]
        if len(vals) == len(opens):
            matrix.append((mname, vals))
    if len(matrix) >= 2:
        # Multiple-metric rank correlation (mean of all pairwise Spearmans)
        rhos = []
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                sign_i = -1 if matrix[i][0] in {"cos_sim", "ret"} else 1
                sign_j = -1 if matrix[j][0] in {"cos_sim", "ret"} else 1
                rho, _ = spearmanr(np.array(matrix[i][1]) * sign_i,
                                    np.array(matrix[j][1]) * sign_j)
                rhos.append(rho)
        out["RANKSPEARMAN"] = fmt(float(np.mean(rhos)), nd=2)

    # Top generator by mean normalized score
    if opens:
        score = {}
        for m in opens:
            if m not in grouped.index:
                continue
            # simple rank score: rank on each metric (lower-is-better inverted)
            r = 0
            for _, col, sign in [("gaas", "gaas", +1), ("cos_sim", "cos_sim", -1),
                                  ("dcsf", "dcsf", +1), ("mmd", "mmd", +1)]:
                series = grouped[col].loc[opens]
                rank = series.rank(ascending=(sign > 0)).loc[m]
                r += rank
            score[m] = r
        if score:
            top = min(score, key=score.get)
            out["TOPGENERATOR"] = top.replace("_", "-")
    return out


def collect_human(analysis_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    wg = analysis_dir / "within_geo_accuracy.csv"
    if wg.exists():
        df = pd.read_csv(wg)
        # Weighted mean accuracy across raters
        if {"mean", "count"}.issubset(df.columns):
            acc = float((df["mean"] * df["count"]).sum() / df["count"].sum())
            out["HUMANWITHINACC"] = fmt(100 * acc, nd=1)
    mr = analysis_dir / "model_ranking.csv"
    if mr.exists():
        df = pd.read_csv(mr)
        ordered = df.sort_values("mean_pref_score", ascending=False)["model"].tolist()
        out["HUMANRANKING"] = " $\\succ$ ".join(
            m.replace("_", "-") for m in ordered)
    rg = analysis_dir / "real_vs_gen_raw.csv"
    if rg.exists():
        df = pd.read_csv(rg)
        if "picked_real" in df.columns:
            out["HUMANREALVSGEN"] = fmt(100 * df["picked_real"].mean(), nd=1)
    return out


def write_numbers_file(defs: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Auto-generated from evaluation outputs by "
             "scripts/build_paper_tables.py.",
             "% DO NOT EDIT BY HAND -- rerun after new experiments.",
             ""]
    for name in sorted(defs):
        lines.append(cmd(name, defs[name]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark",
                    default=str(config.PROCESSED_DIR / "benchmark_v2.json"))
    ap.add_argument("--validity",
                    default=str(config.OUTPUT_DIR / "validity_v2"
                                 / "metric_validity_summary.csv"))
    ap.add_argument("--within_synth",
                    default=str(config.OUTPUT_DIR / "within_synthetic"
                                 / "within_synthetic_summary.csv"))
    ap.add_argument("--eval",
                    default=str(config.OUTPUT_DIR / "eval_v2" / "raw_results.csv"))
    ap.add_argument("--human",
                    default=str(config.OUTPUT_DIR / "human_eval" / "analysis"))
    ap.add_argument("--out",
                    default=str(config.ROOT / "paper" / "numbers.tex"))
    args = ap.parse_args()

    defs: dict[str, str] = {}
    defs.update(collect_benchmark_stats(Path(args.benchmark)))
    defs.update(collect_validity(Path(args.validity)))
    defs.update(collect_within_synth(Path(args.within_synth)))
    defs.update(collect_main_eval(Path(args.eval)))
    defs.update(collect_human(Path(args.human)))
    write_numbers_file(defs, Path(args.out))
    print(f"[build_paper_tables] wrote {len(defs)} macros -> {args.out}")


if __name__ == "__main__":
    main()
