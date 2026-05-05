"""
Human evaluation for GeoFidelity-Bench.

Supports both the older v2 benchmark structure (`places`) and the
current v3 structure (`blocks`). The generated HTML is self-contained
and optimized for one-trial-at-a-time rating with autosave, progress
navigation, and keyboard shortcuts.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import json
import random
from collections import defaultdict
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

import config


REF_PER_PANEL = 6
HUMAN_EVAL_DIR = config.OUTPUT_DIR / "human_eval"
CHOICE_LABELS = {
    -2: "A much better",
    -1: "A slightly better",
    0: "About the same",
    1: "B slightly better",
    2: "B much better",
}


def _load_ratings_dir(ratings_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv in sorted(ratings_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if not {"trial_id", "choice"}.issubset(df.columns):
            continue
        if "rater_id" not in df.columns:
            df["rater_id"] = csv.stem
        df["source_file"] = csv.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["trial_id", "choice"]).copy()
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    df["choice"] = pd.to_numeric(df["choice"], errors="coerce")
    df = df.dropna(subset=["trial_id", "choice"]).copy()
    df["trial_id"] = df["trial_id"].astype(int)
    df["choice"] = df["choice"].astype(int)
    df = df.sort_values(["trial_id", "rater_id", "source_file"])
    return df.drop_duplicates(subset=["trial_id", "rater_id"], keep="last")


def _load_benchmark(benchmark_path: Path) -> tuple[str, dict[str, list[dict]]]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    units_by_city: dict[str, list[dict]] = defaultdict(list)
    if "blocks" in bench:
        kind = "v3"
        for block in bench["blocks"]:
            units_by_city[block["city"]].append({
                "unit_id": block["block_id"],
                "city": block["city"],
                "ref_paths": [img["image_path"] for img in block["images"]],
            })
    elif "places" in bench:
        kind = "v2"
        for place in bench["places"]:
            units_by_city[place["city"]].append({
                "unit_id": place["place_id"],
                "city": place["city"],
                "ref_paths": list(place["image_paths"]),
            })
    else:
        raise ValueError("unrecognized benchmark format: expected `blocks` or `places`")
    return kind, units_by_city


def _resolve_gen_root(kind: str, gen_root: str | None) -> Path:
    if gen_root:
        return Path(gen_root)
    return config.V3_GEN_DIR if kind == "v3" else config.GEN_DIR


def _unit_gen_dir(gen_root: Path, kind: str, model: str,
                  unit_id: str, level: str | None) -> Path:
    if kind == "v3":
        if not level:
            raise ValueError("v3 human eval requires an explicit prompt level")
        return gen_root / model / level / unit_id
    return gen_root / model / unit_id


def _gen_paths_for_city(city: str, model: str, kind: str,
                        units_by_city: dict[str, list[dict]], gen_root: Path,
                        level: str | None) -> list[str]:
    out: list[str] = []
    for unit in units_by_city.get(city, []):
        pdir = _unit_gen_dir(gen_root, kind, model, unit["unit_id"], level)
        if pdir.exists():
            out.extend(
                str(jpg.relative_to(config.ROOT).as_posix())
                for jpg in sorted(pdir.glob("*.jpg"))
            )
    return out


def _ref_paths_for_city(city: str, units_by_city: dict[str, list[dict]],
                        n: int = REF_PER_PANEL) -> list[str]:
    pool: list[str] = []
    for unit in units_by_city.get(city, []):
        pool.extend(unit["ref_paths"])
    if len(pool) <= n:
        return pool
    return random.sample(pool, n)


def build_trials(benchmark_path: Path, counts: dict[str, int],
                 models: list[str], seed: int = 42,
                 level: str | None = None,
                 gen_root: Path | None = None) -> list[dict]:
    random.seed(seed)
    kind, units_by_city = _load_benchmark(benchmark_path)
    gen_root = gen_root or _resolve_gen_root(kind, None)
    if kind == "v3" and not level:
        level = "L1"

    cities = [
        city for city in units_by_city
        if all(_gen_paths_for_city(city, model, kind, units_by_city, gen_root, level)
               for model in models)
    ]
    if len(cities) < 2:
        print(f"[human_eval] only {len(cities)} cities have complete generations; can't build balanced trials")
        return []

    trials: list[dict] = []
    tid = 0

    for _ in range(counts["within_geo"]):
        model = random.choice(models)
        target = random.choice(cities)
        other = random.choice([city for city in cities if city != target])
        gen_tgt = _gen_paths_for_city(target, model, kind, units_by_city, gen_root, level)
        gen_oth = _gen_paths_for_city(other, model, kind, units_by_city, gen_root, level)
        if not gen_tgt or not gen_oth:
            continue
        cand_target = random.choice(gen_tgt)
        cand_other = random.choice(gen_oth)
        flip = random.random() < 0.5
        trials.append({
            "trial_id": tid,
            "type": "within_geo",
            "target_city": target,
            "candidate_A": cand_other if flip else cand_target,
            "candidate_B": cand_target if flip else cand_other,
            "candidate_A_is_target": not flip,
            "meta": {
                "model": model,
                "other_city": other,
                "benchmark_kind": kind,
                "level": level,
                "flipped": flip,
            },
            "ref_paths": _ref_paths_for_city(target, units_by_city),
        })
        tid += 1

    for _ in range(counts["model_pair"]):
        if len(models) < 2:
            break
        model_a, model_b = random.sample(models, 2)
        target = random.choice(cities)
        gen_a = _gen_paths_for_city(target, model_a, kind, units_by_city, gen_root, level)
        gen_b = _gen_paths_for_city(target, model_b, kind, units_by_city, gen_root, level)
        if not gen_a or not gen_b:
            continue
        flip = random.random() < 0.5
        trials.append({
            "trial_id": tid,
            "type": "model_pair",
            "target_city": target,
            "candidate_A": random.choice(gen_b if flip else gen_a),
            "candidate_B": random.choice(gen_a if flip else gen_b),
            "meta": {
                "model_A": model_b if flip else model_a,
                "model_B": model_a if flip else model_b,
                "benchmark_kind": kind,
                "level": level,
                "flipped": flip,
            },
            "ref_paths": _ref_paths_for_city(target, units_by_city),
        })
        tid += 1

    for _ in range(counts["real_vs_gen"]):
        model = random.choice(models)
        target = random.choice(cities)
        gen = _gen_paths_for_city(target, model, kind, units_by_city, gen_root, level)
        reals = [path for unit in units_by_city[target] for path in unit["ref_paths"]]
        if len(reals) < 8 or not gen:
            continue
        pool_ref = _ref_paths_for_city(target, units_by_city)
        holdout = [path for path in reals if path not in set(pool_ref)]
        if not holdout:
            continue
        real = random.choice(holdout)
        gen_pick = random.choice(gen)
        flip = random.random() < 0.5
        trials.append({
            "trial_id": tid,
            "type": "real_vs_gen",
            "target_city": target,
            "candidate_A": gen_pick if flip else real,
            "candidate_B": real if flip else gen_pick,
            "meta": {
                "model": model,
                "real_is_A": not flip,
                "benchmark_kind": kind,
                "level": level,
                "flipped": flip,
            },
            "ref_paths": pool_ref,
        })
        tid += 1

    random.shuffle(trials)
    for index, trial in enumerate(trials):
        trial["trial_id"] = index
    return trials


def _b64(path: Path, max_size: int = 360) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html(trials: list[dict], out_path: Path) -> None:
    def _data_url(rel_path: str) -> str:
        return f"data:image/jpeg;base64,{_b64(config.ROOT / rel_path)}"

    light_trials = []
    for trial in trials:
        light_trials.append({
            "trial_id": trial["trial_id"],
            "type": trial["type"],
            "target_city": trial["target_city"],
            "meta": trial["meta"],
            "ref_imgs": [_data_url(path) for path in trial["ref_paths"]],
            "a_img": _data_url(trial["candidate_A"]),
            "b_img": _data_url(trial["candidate_B"]),
        })

    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>GeoFidelity-Bench rating interface</title>
<style>
  :root{
    --bg:#f5f1e8;
    --panel:#fffdf8;
    --ink:#1f2a22;
    --muted:#6f776d;
    --line:#d6cec1;
    --accent:#9b2c2c;
    --accent-soft:#f4d9d5;
    --forest:#285943;
    --forest-soft:#d9ebe1;
    --shadow:0 10px 30px rgba(36,29,20,.08);
  }
  *{box-sizing:border-box}
  body{
    margin:0;
    font-family:Georgia,"Times New Roman",serif;
    color:var(--ink);
    background:
      radial-gradient(circle at top left, rgba(155,44,44,.08), transparent 32%),
      radial-gradient(circle at bottom right, rgba(40,89,67,.08), transparent 28%),
      var(--bg);
  }
  .shell{
    min-height:100vh;
    display:grid;
    grid-template-columns:320px minmax(0,1fr);
  }
  .side{
    border-right:1px solid var(--line);
    background:rgba(255,253,248,.82);
    backdrop-filter:blur(8px);
    padding:28px 24px 24px;
    display:flex;
    flex-direction:column;
    gap:18px;
    position:sticky;
    top:0;
    height:100vh;
  }
  .brand h1{
    margin:0 0 8px;
    font-size:1.55rem;
    line-height:1.1;
    color:var(--accent);
  }
  .brand p,.hint,.small{
    margin:0;
    color:var(--muted);
    line-height:1.45;
    font-size:.95rem;
  }
  .field label{
    display:block;
    font-size:.85rem;
    text-transform:uppercase;
    letter-spacing:.08em;
    color:var(--muted);
    margin-bottom:8px;
  }
  .field input{
    width:100%;
    border:1px solid var(--line);
    border-radius:12px;
    padding:12px 14px;
    background:#fff;
    font:inherit;
    color:var(--ink);
  }
  .panel{
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:18px;
    padding:16px;
    box-shadow:var(--shadow);
  }
  .progress-bar{
    width:100%;
    height:10px;
    border-radius:999px;
    background:#eee7dc;
    overflow:hidden;
    margin:10px 0 8px;
  }
  .progress-fill{
    height:100%;
    background:linear-gradient(90deg,var(--accent),#d97757);
  }
  .stats{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:8px;
    margin-top:12px;
  }
  .stat{
    background:#faf5ee;
    border:1px solid var(--line);
    border-radius:12px;
    padding:10px 12px;
  }
  .stat b{
    display:block;
    font-size:1.05rem;
    color:var(--forest);
  }
  .nav,.actions,.legend{
    display:grid;
    gap:10px;
  }
  button{
    border:0;
    border-radius:12px;
    padding:12px 14px;
    font:inherit;
    cursor:pointer;
    transition:transform .12s ease, box-shadow .12s ease, background .12s ease;
  }
  button:hover{transform:translateY(-1px)}
  .ghost{
    background:#fff;
    color:var(--ink);
    border:1px solid var(--line);
  }
  .primary{background:var(--forest); color:#fff}
  .danger{background:#7f1d1d; color:#fff}
  .workspace{
    padding:28px 34px 34px;
    min-width:0;
  }
  .trial{
    display:grid;
    gap:18px;
  }
  .trial-top{
    display:flex;
    flex-wrap:wrap;
    justify-content:space-between;
    gap:12px;
    align-items:flex-end;
  }
  .trial-top h2{
    margin:0;
    font-size:2rem;
    line-height:1.05;
  }
  .chips{display:flex; flex-wrap:wrap; gap:8px}
  .chip{
    display:inline-flex;
    align-items:center;
    gap:6px;
    border-radius:999px;
    padding:6px 10px;
    background:#fff;
    border:1px solid var(--line);
    color:var(--muted);
    font-size:.85rem;
    text-transform:uppercase;
    letter-spacing:.05em;
  }
  .references .grid{
    display:grid;
    grid-template-columns:repeat(6,minmax(0,1fr));
    gap:10px;
  }
  .references img,.candidate-card img{
    width:100%;
    display:block;
    border-radius:14px;
  }
  .candidates{
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:18px;
  }
  .candidate-card{
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:22px;
    padding:16px;
    box-shadow:var(--shadow);
    display:grid;
    gap:14px;
  }
  .candidate-card.selected-a{
    border-color:var(--accent);
    box-shadow:0 12px 30px rgba(155,44,44,.16);
  }
  .candidate-card.selected-b{
    border-color:var(--forest);
    box-shadow:0 12px 30px rgba(40,89,67,.16);
  }
  .candidate-head{
    display:flex;
    justify-content:space-between;
    align-items:baseline;
    gap:12px;
  }
  .candidate-head h3{margin:0; font-size:1.25rem}
  .choice-grid{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:10px;
  }
  .choice-btn{
    text-align:left;
    background:#fff;
    border:1px solid var(--line);
    color:var(--ink);
  }
  .choice-btn strong{
    display:block;
    margin-bottom:4px;
    font-size:.92rem;
    color:var(--accent);
  }
  .choice-btn.active{
    background:var(--accent-soft);
    border-color:var(--accent);
  }
  .choice-btn.active.b-side{
    background:var(--forest-soft);
    border-color:var(--forest);
  }
  .choice-btn.active.tie{
    background:#efe8da;
    border-color:#9b8c74;
  }
  .footer-note{
    margin-top:4px;
    color:var(--muted);
    font-size:.92rem;
  }
  @media (max-width: 1100px){
    .shell{grid-template-columns:1fr}
    .side{position:static; height:auto; border-right:0; border-bottom:1px solid var(--line)}
    .workspace{padding:20px}
  }
  @media (max-width: 760px){
    .references .grid{grid-template-columns:repeat(3,minmax(0,1fr))}
    .candidates{grid-template-columns:1fr}
    .choice-grid{grid-template-columns:1fr}
  }
</style>
</head>
<body>
<div class="shell">
  <aside class="side">
    <div class="brand">
      <h1>GeoFidelity-Bench Rater</h1>
      <p>Pick the candidate that looks more likely to belong to the target location shown in the reference panel.</p>
    </div>

    <div class="field">
      <label for="rater">Rater ID</label>
      <input id="rater" placeholder="e.g. kaizhen / A1 / reviewer-2" />
    </div>

    <div class="panel">
      <div class="small" id="progressText">Trial 1 / 1</div>
      <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
      <div class="stats">
        <div class="stat"><span class="small">Answered</span><b id="answeredCount">0</b></div>
        <div class="stat"><span class="small">Remaining</span><b id="remainingCount">0</b></div>
        <div class="stat"><span class="small">Current</span><b id="currentCount">1</b></div>
      </div>
    </div>

    <div class="panel legend">
      <div class="small"><b>Keyboard</b></div>
      <div class="small">1 / 2 / 3 / 4 / 5: choose rating</div>
      <div class="small">← / →: previous or next trial</div>
      <div class="small">U: jump to next unanswered trial</div>
      <div class="small">D: download CSV</div>
    </div>

    <div class="panel hint">
      Focus on architecture, road layout, vegetation, pole density, signage, and overall urban character. If both images look equally plausible or equally implausible, use the "about the same" option instead of forcing a side.
    </div>

    <div class="nav">
      <button class="ghost" id="prevBtn">Previous Trial</button>
      <button class="ghost" id="nextBtn">Next Trial</button>
      <button class="ghost" id="nextUnansweredBtn">Next Unanswered</button>
    </div>

    <div class="actions">
      <button class="primary" id="downloadBtn">Download Ratings CSV</button>
      <button class="danger" id="resetBtn">Reset Current Rater</button>
    </div>
  </aside>

  <main class="workspace">
    <div class="trial" id="trialRoot"></div>
  </main>
</div>

<script>
const TRIALS = __TRIALS__;
const CHOICES = [
  {value:-2, key:'1', label:'A much better', side:'A', tone:'a-side'},
  {value:-1, key:'2', label:'A slightly better', side:'A', tone:'a-side'},
  {value:0, key:'3', label:'About the same', side:'tie', tone:'tie'},
  {value:1, key:'4', label:'B slightly better', side:'B', tone:'b-side'},
  {value:2, key:'5', label:'B much better', side:'B', tone:'b-side'}
];

function cleanRater(){
  const raw = document.getElementById('rater').value.trim();
  return raw || 'anon';
}
function answerKey(trialId){
  return `geofidelity:${cleanRater()}:trial:${trialId}`;
}
function indexKey(){
  return `geofidelity:${cleanRater()}:currentIndex`;
}
function getChoice(trialId){
  const value = localStorage.getItem(answerKey(trialId));
  return value === null ? null : parseInt(value, 10);
}
function setChoice(trialId, value){
  localStorage.setItem(answerKey(trialId), String(value));
}
function currentIndex(){
  const value = localStorage.getItem(indexKey());
  const parsed = value === null ? 0 : parseInt(value, 10);
  if(Number.isNaN(parsed)) return 0;
  return Math.max(0, Math.min(TRIALS.length - 1, parsed));
}
function setCurrentIndex(idx){
  localStorage.setItem(indexKey(), String(Math.max(0, Math.min(TRIALS.length - 1, idx))));
}
function answeredCount(){
  return TRIALS.filter(t => getChoice(t.trial_id) !== null).length;
}
function nextUnanswered(startIdx){
  for(let step = 1; step <= TRIALS.length; step++){
    const idx = (startIdx + step) % TRIALS.length;
    if(getChoice(TRIALS[idx].trial_id) === null){
      return idx;
    }
  }
  return startIdx;
}
function chip(text){
  return `<span class="chip">${text}</span>`;
}
function trialSubtitle(trial){
  const bits = [chip(`trial ${trial.trial_id + 1}`), chip(trial.type.replace('_',' '))];
  if(trial.meta && trial.meta.level){
    bits.push(chip(trial.meta.level));
  }
  return bits.join('');
}
function renderTrial(){
  const idx = currentIndex();
  const trial = TRIALS[idx];
  const choice = getChoice(trial.trial_id);
  const selectedSide = choice === null ? null : (choice < 0 ? 'A' : (choice > 0 ? 'B' : 'tie'));
  const root = document.getElementById('trialRoot');
  const answered = answeredCount();
  const remaining = TRIALS.length - answered;
  const progress = TRIALS.length ? (answered / TRIALS.length) * 100 : 0;

  document.getElementById('progressText').textContent = `Trial ${idx + 1} / ${TRIALS.length}`;
  document.getElementById('progressFill').style.width = `${progress}%`;
  document.getElementById('answeredCount').textContent = answered;
  document.getElementById('remainingCount').textContent = remaining;
  document.getElementById('currentCount').textContent = idx + 1;

  root.innerHTML = `
    <div class="trial-top">
      <div>
        <div class="chips">${trialSubtitle(trial)}</div>
        <h2>Target city: ${trial.target_city.replace(/_/g, ' ')}</h2>
      </div>
      <div class="footer-note">Choose the candidate more likely to belong to this location.</div>
    </div>

    <section class="panel references">
      <div class="small" style="margin-bottom:10px;">Reference panel</div>
      <div class="grid">
        ${trial.ref_imgs.map(src => `<img src="${src}" alt="reference image">`).join('')}
      </div>
    </section>

    <section class="candidates">
      <article class="candidate-card ${selectedSide === 'A' ? 'selected-a' : ''}">
        <div class="candidate-head">
          <h3>Candidate A</h3>
          <span class="small">${selectedSide === 'A' ? 'selected' : 'compare visually'}</span>
        </div>
        <img src="${trial.a_img}" alt="candidate A">
      </article>
      <article class="candidate-card ${selectedSide === 'B' ? 'selected-b' : ''}">
        <div class="candidate-head">
          <h3>Candidate B</h3>
          <span class="small">${selectedSide === 'B' ? 'selected' : 'compare visually'}</span>
        </div>
        <img src="${trial.b_img}" alt="candidate B">
      </article>
    </section>

    <section class="panel">
      <div class="small" style="margin-bottom:12px;">Rating</div>
      <div class="choice-grid">
        ${CHOICES.map(opt => `
          <button class="choice-btn ${choice === opt.value ? `active ${opt.tone}` : ''}" data-choice="${opt.value}">
            <strong>[${opt.key}] ${opt.label}</strong>
            <span>${opt.side === 'A' ? 'Candidate A is better' : (opt.side === 'B' ? 'Candidate B is better' : 'Both look about equally plausible')}</span>
          </button>
        `).join('')}
      </div>
      <div class="footer-note" style="margin-top:12px;">
        ${choice === null ? 'No answer saved yet for this trial.' : `Saved answer: ${CHOICES.find(opt => opt.value === choice).label}.`}
      </div>
    </section>
  `;

  for(const button of root.querySelectorAll('[data-choice]')){
    button.addEventListener('click', () => {
      setChoice(trial.trial_id, parseInt(button.dataset.choice, 10));
      renderTrial();
    });
  }
}
function download(){
  const rater = cleanRater();
  const rows = [[
    'rater_id','trial_id','type','target_city','level','choice','choice_label','saved_at'
  ]];
  for(const trial of TRIALS){
    const choice = getChoice(trial.trial_id);
    rows.push([
      rater,
      trial.trial_id,
      trial.type,
      trial.target_city,
      (trial.meta && trial.meta.level) ? trial.meta.level : '',
      choice === null ? '' : choice,
      choice === null ? '' : (CHOICES.find(opt => opt.value === choice)?.label || ''),
      choice === null ? '' : new Date().toISOString()
    ]);
  }
  const csv = rows.map(row => row.map(cell => String(cell).replace(/,/g, ';')).join(',')).join('\\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `ratings_${rater}.csv`;
  a.click();
}
function go(delta){
  setCurrentIndex(currentIndex() + delta);
  renderTrial();
}
function jumpToNextUnanswered(){
  setCurrentIndex(nextUnanswered(currentIndex()));
  renderTrial();
}
document.getElementById('prevBtn').addEventListener('click', () => go(-1));
document.getElementById('nextBtn').addEventListener('click', () => go(1));
document.getElementById('nextUnansweredBtn').addEventListener('click', jumpToNextUnanswered);
document.getElementById('downloadBtn').addEventListener('click', download);
document.getElementById('resetBtn').addEventListener('click', () => {
  const rater = cleanRater();
  if(!confirm(`Clear all saved answers for ${rater}?`)) return;
  for(const trial of TRIALS){
    localStorage.removeItem(answerKey(trial.trial_id));
  }
  localStorage.removeItem(indexKey());
  renderTrial();
});
document.getElementById('rater').addEventListener('input', renderTrial);
document.addEventListener('keydown', (event) => {
  if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) return;
  if(event.key === 'ArrowLeft'){ event.preventDefault(); go(-1); }
  if(event.key === 'ArrowRight'){ event.preventDefault(); go(1); }
  if(event.key.toLowerCase() === 'u'){ event.preventDefault(); jumpToNextUnanswered(); }
  if(event.key.toLowerCase() === 'd'){ event.preventDefault(); download(); }
  const match = CHOICES.find(opt => opt.key === event.key);
  if(match){
    event.preventDefault();
    const trial = TRIALS[currentIndex()];
    setChoice(trial.trial_id, match.value);
    renderTrial();
  }
});
renderTrial();
</script>
</body>
</html>
"""
    html = html.replace("__TRIALS__", json.dumps(light_trials))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[human_eval] wrote {out_path} ({len(trials)} trials)")


def analyze(ratings_dir: Path, trials_path: Path, out_dir: Path) -> None:
    with open(trials_path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    tmap = {t["trial_id"]: t for t in trials}

    df = _load_ratings_dir(ratings_dir)
    if df.empty:
        print("[human_eval] no rating CSVs found")
        return
    df["B_wins"] = (df["choice"] > 0).astype(int)
    df["is_tie"] = (df["choice"] == 0).astype(int)
    trial_votes = df.groupby(["trial_id", "type", "target_city"], as_index=False).agg(
        choice_mean=("choice", "mean"),
        n_ratings=("choice", "size"),
        tie_rate=("is_tie", "mean"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_summary = pd.DataFrame([
        {"key": "n_unique_raters", "value": int(df["rater_id"].nunique())},
        {"key": "n_rating_rows", "value": int(len(df))},
        {"key": "n_answered_trials", "value": int(df["trial_id"].nunique())},
        {"key": "n_within_geo_trials", "value": int((trial_votes["type"] == "within_geo").sum())},
        {"key": "n_model_pair_trials", "value": int((trial_votes["type"] == "model_pair").sum())},
        {"key": "n_real_vs_gen_trials", "value": int((trial_votes["type"] == "real_vs_gen").sum())},
        {"key": "overall_tie_rate", "value": float(df["is_tie"].mean())},
    ])
    archive_summary.to_csv(out_dir / "archive_summary.csv", index=False)
    print("[human_eval] archive summary:")
    print(archive_summary.to_string(index=False))

    wg = df[df["type"] == "within_geo"].copy()
    wg["target_is_A"] = wg["trial_id"].map(lambda tid: tmap[tid].get("candidate_A_is_target", True))
    wg["is_tie"] = wg["choice"] == 0
    wg["correct"] = np.where(
        wg["is_tie"],
        np.nan,
        ((wg["target_is_A"]) & (wg["choice"] < 0)) | ((~wg["target_is_A"]) & (wg["choice"] > 0)),
    )
    by_rater = wg.groupby("rater_id").agg(
        decisive_accuracy=("correct", "mean"),
        decisive_count=("correct", "count"),
        tie_rate=("is_tie", "mean"),
        total_count=("choice", "size"),
    )
    by_rater.to_csv(out_dir / "within_geo_accuracy.csv")
    print("\nWithin-geo accuracy per rater:")
    print(by_rater.round(3).to_string())

    mp = trial_votes[trial_votes["type"] == "model_pair"].copy()
    mp["model_A"] = mp["trial_id"].map(lambda tid: tmap[tid]["meta"]["model_A"])
    mp["model_B"] = mp["trial_id"].map(lambda tid: tmap[tid]["meta"]["model_B"])
    records: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for _, row in mp.iterrows():
        model_a, model_b = row["model_A"], row["model_B"]
        strength = row["choice_mean"]
        records[model_a][model_b].append(-strength)
        records[model_b][model_a].append(strength)
    models = sorted(set(mp["model_A"]).union(mp["model_B"]))
    score = {
        model: np.mean([value for opp in records[model].values() for value in opp]) if records[model] else float("nan")
        for model in models
    }
    rank = pd.DataFrame(
        [{"model": model, "mean_pref_score": score[model]} for model in models]
    ).sort_values("mean_pref_score", ascending=False)
    rank.to_csv(out_dir / "model_ranking.csv", index=False)
    print("\nModel ranking (higher = humans prefer):")
    print(rank.round(3).to_string(index=False))

    rg = trial_votes[trial_votes["type"] == "real_vs_gen"].copy()
    if len(rg):
        rg["real_is_A"] = rg["trial_id"].map(lambda tid: tmap[tid]["meta"]["real_is_A"])
        rg["picked_real"] = np.where(
            rg["choice_mean"] == 0,
            np.nan,
            ((rg["real_is_A"]) & (rg["choice_mean"] < 0)) | ((~rg["real_is_A"]) & (rg["choice_mean"] > 0)),
        )
        pct = float(rg["picked_real"].mean())
        tie_rate = float((rg["choice_mean"] == 0).mean())
        print(
            f"\nReal vs generated: humans picked real {100 * pct:.1f}% of the time "
            f"on decisive trials, with tie rate {100 * tie_rate:.1f}% "
            f"(expected >>50% if generators are still distinguishable)"
        )
        rg.to_csv(out_dir / "real_vs_gen_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument(
        "--benchmark",
        default=str(config.V3_BENCHMARK_JSON if config.V3_BENCHMARK_JSON.exists()
                    else config.PROCESSED_DIR / "benchmark_v2.json"),
    )
    g.add_argument("--gen_root", default=None)
    g.add_argument("--level", default="L1", help="Used for v3 generations; ignored for v2")
    g.add_argument("--models", nargs="*", default=None)
    g.add_argument("--n_within_geo", type=int, default=100)
    g.add_argument("--n_model_pair", type=int, default=70)
    g.add_argument("--n_real_vs_gen", type=int, default=30)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out_dir", default=str(HUMAN_EVAL_DIR))

    a = sub.add_parser("analyze")
    a.add_argument("--ratings_dir", default=str(HUMAN_EVAL_DIR / "ratings"))
    a.add_argument("--trials", default=str(HUMAN_EVAL_DIR / "trials.json"))
    a.add_argument("--out_dir", default=str(HUMAN_EVAL_DIR / "analysis"))

    args = ap.parse_args()
    if args.cmd == "generate":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        benchmark_path = Path(args.benchmark)
        kind, _ = _load_benchmark(benchmark_path)
        gen_root = _resolve_gen_root(kind, args.gen_root)

        models = args.models
        if not models:
            if gen_root.exists():
                models = [d.name for d in gen_root.iterdir() if d.is_dir()]
            if not models:
                print("no models found in generation root -- generate images first")
                return

        counts = {
            "within_geo": args.n_within_geo,
            "model_pair": args.n_model_pair,
            "real_vs_gen": args.n_real_vs_gen,
        }
        trials = build_trials(
            benchmark_path=benchmark_path,
            counts=counts,
            models=models,
            seed=args.seed,
            level=args.level if kind == "v3" else None,
            gen_root=gen_root,
        )
        with open(out_dir / "trials.json", "w", encoding="utf-8") as f:
            lite = [{k: v for k, v in trial.items() if not k.startswith("_")} for trial in trials]
            json.dump(lite, f, indent=2, ensure_ascii=False)
        render_html(trials, out_dir / "human_eval.html")
        (out_dir / "ratings").mkdir(parents=True, exist_ok=True)
        print(
            f"\nNext: open {out_dir/'human_eval.html'} in browsers, collect "
            f"ratings_<rater>.csv files in {out_dir/'ratings'}/, then run "
            f"`python eval/human_eval.py analyze`."
        )
    else:
        analyze(Path(args.ratings_dir), Path(args.trials), Path(args.out_dir))


if __name__ == "__main__":
    main()
