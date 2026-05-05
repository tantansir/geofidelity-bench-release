"""
Tier 6: manual review via local HTML grid.

Two-phase workflow:

Phase 1 (--generate):
  Take Tier-5 survivors, stratified-sample up to TIER6_SAMPLE_PER_CITY per
  city, and write one self-contained HTML page per city to data/review/.
  The page uses browser localStorage to track click-to-reject decisions
  and provides a "Download CSV" button.

Phase 2 (--apply):
  Read every data/review/rejects_*.csv downloaded by kaizhen, merge the
  rejection set with the Tier-5 CSV, and write tier6_review.csv with a
  final tier6_pass column (passed every prior tier AND not rejected).

Why a static HTML / localStorage workflow instead of a server?
  - No extra pip deps, no port to open, works over file:// on Windows.
  - Image thumbnails are inlined as base64 so a single .html is portable
    (kaizhen can review offline or on a tablet).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import random
from html import escape
from io import BytesIO

import pandas as pd
from PIL import Image

import config

THUMB_SIZE = 256


def _sample_per_city(df: pd.DataFrame, per_city: int, seed: int = 42
                     ) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for city, g in df.groupby("city"):
        n = min(per_city, len(g))
        rows.append(g.sample(n=n, random_state=seed))
    return pd.concat(rows, ignore_index=True) if rows else df.head(0)


def _thumb_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tier 6 review - {city}</title>
<style>
 body{{font-family:system-ui;margin:1rem;background:#111;color:#eee}}
 h1{{margin:0 0 .2rem}} .sub{{color:#888;margin-bottom:1rem}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.5rem}}
 .card{{position:relative;cursor:pointer;border:3px solid #444;border-radius:4px;overflow:hidden}}
 .card img{{width:100%;display:block}}
 .card .tag{{position:absolute;bottom:0;left:0;right:0;font-size:11px;padding:2px 4px;background:rgba(0,0,0,.6)}}
 .card.reject{{border-color:#e33;opacity:.55}}
 .card.reject::after{{content:"REJECTED";position:absolute;top:40%;left:0;right:0;text-align:center;color:#fff;font-weight:700;background:rgba(230,60,60,.6);padding:6px}}
 .bar{{position:sticky;top:0;background:#111;padding:.5rem 0;display:flex;gap:.5rem;align-items:center}}
 button{{background:#2a6;color:#fff;border:0;padding:.5rem 1rem;border-radius:4px;cursor:pointer;font-size:14px}}
 button.warn{{background:#c43}}
 .count{{color:#fc9;font-weight:700}}
</style></head><body>
<h1>{city}</h1>
<div class="sub">{n} images from Tier-5 survivors. Click any image to toggle REJECT. Decisions persist in localStorage. Download when done.</div>
<div class="bar">
  <button onclick="downloadCSV()">Download rejects_{city_slug}.csv</button>
  <button class="warn" onclick="if(confirm('Clear all decisions?')){{localStorage.clear();location.reload();}}">Clear</button>
  <span>Rejected: <span id="rej_count" class="count">0</span> / {n}</span>
</div>
<div class="grid">{cards}</div>
<script>
const CITY = "{city}";
function key(p){{return 'rej:' + CITY + ':' + p;}}
function update(){{
 const cards = document.querySelectorAll('.card');
 let k = 0;
 cards.forEach(c => {{
  const p = c.dataset.path;
  if (localStorage.getItem(key(p)) === '1') {{ c.classList.add('reject'); k++; }}
  else c.classList.remove('reject');
 }});
 document.getElementById('rej_count').textContent = k;
}}
document.querySelectorAll('.card').forEach(c => c.addEventListener('click', () => {{
 const p = c.dataset.path;
 const k = key(p);
 if (localStorage.getItem(k) === '1') localStorage.removeItem(k);
 else localStorage.setItem(k, '1');
 update();
}}));
function downloadCSV(){{
 const rows = [['image_path','reject']];
 document.querySelectorAll('.card').forEach(c => {{
  const p = c.dataset.path;
  const r = localStorage.getItem(key(p)) === '1' ? 1 : 0;
  rows.push([p, r]);
 }});
 const csv = rows.map(r => r.join(',')).join('\\n');
 const blob = new Blob([csv], {{type:'text/csv'}});
 const a = document.createElement('a');
 a.href = URL.createObjectURL(blob);
 a.download = 'rejects_{city_slug}.csv';
 a.click();
}}
update();
</script></body></html>
"""


def generate(in_csv: Path, out_dir: Path, per_city: int) -> None:
    df = pd.read_csv(in_csv)
    prior = [c for c in ("tier2_pass", "tier3_pass", "tier4_pass", "tier5_pass")
             if c in df.columns]
    mask = pd.Series([True] * len(df))
    for c in prior:
        mask &= df[c].fillna(False).astype(bool)
    df = df[mask].copy()
    sampled = _sample_per_city(df, per_city)
    print(f"[tier6-gen] sampled {len(sampled)} images across "
          f"{sampled['city'].nunique()} cities")

    out_dir.mkdir(parents=True, exist_ok=True)
    for city, g in sampled.groupby("city"):
        cards = []
        for _, r in g.iterrows():
            p = config.ROOT / r["image_path"]
            if not p.exists():
                continue
            try:
                b64 = _thumb_b64(p)
            except Exception:
                continue
            tag = (f"{escape(r.get('osm_highway','?'))} "
                   f"u{r.get('urbanness',0):.2f} "
                   f"lap{r.get('lap_var',0):.0f}")
            cards.append(
                f'<div class="card" data-path="{escape(r["image_path"])}">'
                f'<img src="data:image/jpeg;base64,{b64}" loading="lazy">'
                f'<div class="tag">{tag}</div></div>'
            )
        html = HTML_TEMPLATE.format(
            city=city, city_slug=city,
            n=len(cards), cards="\n".join(cards),
        )
        (out_dir / f"review_{city}.html").write_text(html, encoding="utf-8")
    print(f"[tier6-gen] wrote {len(list(out_dir.glob('review_*.html')))} pages "
          f"to {out_dir}")
    print(f"[tier6-gen] open each review_<city>.html, click to reject, "
          f"download CSVs back into {out_dir}")


def apply(in_csv: Path, review_dir: Path, out_csv: Path) -> None:
    df = pd.read_csv(in_csv)
    rejects: set[str] = set()
    rej_files = list(review_dir.glob("rejects_*.csv"))
    for rf in rej_files:
        r = pd.read_csv(rf)
        rejects |= set(r.loc[r["reject"].astype(int) == 1, "image_path"].tolist())
    print(f"[tier6-apply] {len(rejects)} rejects from {len(rej_files)} city files")

    prior = [c for c in ("tier2_pass", "tier3_pass", "tier4_pass", "tier5_pass")
             if c in df.columns]
    prior_ok = pd.Series([True] * len(df))
    for c in prior:
        prior_ok &= df[c].fillna(False).astype(bool)
    df["tier6_reject"] = df["image_path"].isin(rejects)
    df["tier6_pass"] = prior_ok & (~df["tier6_reject"])
    df.to_csv(out_csv, index=False)
    n = int(df["tier6_pass"].sum())
    print(f"[tier6-apply] final pass {n}/{len(df)} "
          f"({100.0*n/max(1,len(df)):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--in_csv", default=str(config.PROCESSED_DIR / "tier5_quality.csv"))
    g.add_argument("--out_dir", default=str(config.REVIEW_DIR))
    g.add_argument("--per_city", type=int, default=config.TIER6_SAMPLE_PER_CITY)

    a = sub.add_parser("apply")
    a.add_argument("--in_csv", default=str(config.PROCESSED_DIR / "tier5_quality.csv"))
    a.add_argument("--review_dir", default=str(config.REVIEW_DIR))
    a.add_argument("--out_csv", default=str(config.PROCESSED_DIR / "tier6_review.csv"))

    args = ap.parse_args()
    if args.cmd == "generate":
        generate(Path(args.in_csv), Path(args.out_dir), args.per_city)
    else:
        apply(Path(args.in_csv), Path(args.review_dir), Path(args.out_csv))


if __name__ == "__main__":
    main()
