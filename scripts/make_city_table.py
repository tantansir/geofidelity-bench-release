"""
Generate an expanded per-city LaTeX table with Population.
Populations taken from Jang et al. 2024 Table 1 style (metro-area
estimates, 2024) for the 25 cities in our benchmark.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict

import config


# Metropolitan population estimates (thousands), 2024 sources:
# UN World Urbanization Prospects + national census rounding
POPULATION_K = {
    "london":        9_748, "paris":         11_208, "berlin":        3_677,
    "rome":          2_873, "amsterdam":     1_174,
    "new_york":      19_034, "san_francisco": 3_318, "toronto":       6_432,
    "mexico_city":   22_750,
    "buenos_aires":  15_490, "sao_paulo":     22_807, "bogota":        11_344,
    "tokyo":         37_194, "singapore":     6_037, "mumbai":        21_296,
    "bangkok":       11_233, "seoul":         9_988, "shanghai":      29_868,
    "dubai":         3_720,
    "cape_town":     4_800, "nairobi":       5_325, "cairo":         22_183,
    "istanbul":      15_848,
    "sydney":        5_294, "melbourne":     5_150,
}


def main():
    bench = json.load(open(config.PROCESSED_DIR / "benchmark_v2.json"))
    places = bench["places"]
    by_city = defaultdict(list)
    for p in places:
        by_city[p["city"]].append(p)

    rows = []
    for city in sorted(by_city):
        ps = by_city[city]
        info = config.CITIES[city]
        tiles = len(ps)
        imgs = sum(len(p["image_paths"]) for p in ps)
        pop_k = POPULATION_K.get(city, 0)
        drive = "L" if info["driving"] == "left" else "R"
        if pop_k >= 1000:
            pop_str = f"{pop_k/1000:.1f}M"
        else:
            pop_str = f"{pop_k}K"
        rows.append(
            f"{city.replace('_',' ').title()} & {info['country']} & "
            f"{pop_str} & {info['lat']:.2f} & {info['lon']:.2f} & "
            f"{drive} & {tiles} & {imgs}"
        )
    # Join rows with \\ separators; final row has no trailing \\ to
    # avoid \midrule ' noalign ' errors after \input.
    out = " \\\\\n".join(rows)
    out_path = Path("paper/city_table_rows.tex")
    out_path.write_text(out, encoding="utf-8")
    print(out)
    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
