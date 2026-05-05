# GeoFidelity-Bench

Anonymous code release for the NeurIPS 2026 submission
"GeoFidelity-Bench: Evaluating Geographic Fidelity in Block-Conditioned
Street-View Generation".

## Contents

- `data/`: block carving, Mapillary download, curation, prompt-control generation.
- `generation/`: deterministic text-to-image generation wrappers.
- `metrics/`: set fidelity, semantic agreement, and hard-negative retrieval metrics.
- `eval/`: main benchmark evaluation, controls, sensitivity analyses, and audits.
- `scripts/`: table and figure builders used for the paper.
- `croissant.json`: Croissant metadata for the dataset release.

Large artifacts are hosted in the dataset release, not in this code repository:
reference images, generated images, curation CSVs, evaluation outputs, and the
reviewer sample archive.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Mapillary download scripts require:

```bash
export MAPILLARY_TOKEN=<your-token>
```

Generation scripts also require access to the corresponding Hugging Face model
weights and enough GPU memory for the selected generator.

## Typical Commands

Run the v3 curation pipeline after configuring `config.py` and
`MAPILLARY_TOKEN`:

```bash
python data/run_curation_v3.py
```

Generate images for one model:

```bash
python generation/run_generation_v3.py --model sdxl_base --levels L0 L1 L2
```

Evaluate generated panels against the benchmark:

```bash
python eval/run_eval_v3.py \
  --benchmark data/processed/v3/benchmark_v3.json \
  --generations generations_v3 \
  --out outputs/eval_v3
```

Run prompt-specificity controls:

```bash
python eval/control_ablation_v3.py
```

## Data

Use the dataset URL from the OpenReview submission for the full release and
the sample URL for a compact reviewer-inspection archive. The full release is
required to reproduce aggregate paper scores.

