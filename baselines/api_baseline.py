"""
API-based generation baselines for GeoFidelity-Bench.
Supports: DALL-E 3 (OpenAI), Gemini Imagen (Google).
No GPU required — runs via API calls.

Usage:
    export OPENAI_API_KEY=sk-...
    python baselines/api_baseline.py --model dalle3 --benchmark data/processed/benchmark_mapillary.json
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import os
import json
import time
import base64
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

import config

# Prompt template — same for all models for fair comparison
PROMPT_TEMPLATE = (
    "A street-level photograph taken in {city}, {country}. "
    "The image shows a typical street scene with buildings, roads, "
    "and urban environment characteristic of this location. "
    "Photorealistic, daytime, clear weather."
)


class DallE3Baseline:
    """Generate street views using OpenAI DALL-E 3 API."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            print("WARNING: Set OPENAI_API_KEY environment variable")

    def generate(self, city, country, n=2, size="1024x1024"):
        prompt = PROMPT_TEMPLATE.format(
            city=city.replace("_", " ").title(), country=country)

        images = []
        for i in range(n):
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": "Bearer " + self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "dall-e-3",
                        "prompt": prompt,
                        "n": 1,
                        "size": size,
                        "response_format": "b64_json",
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    b64 = resp.json()["data"][0]["b64_json"]
                    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    img = img.resize((512, 512), Image.LANCZOS)
                    images.append(img)
                else:
                    print("  DALL-E error: %s" % resp.text[:200])
                time.sleep(1)  # rate limit
            except Exception as e:
                print("  DALL-E error: %s" % e)
        return images


class GeminiImagenBaseline:
    """Generate street views using Google Gemini Imagen API."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    def generate(self, city, country, n=2):
        prompt = PROMPT_TEMPLATE.format(
            city=city.replace("_", " ").title(), country=country)

        images = []
        for i in range(n):
            try:
                resp = requests.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    "imagen-3.0-generate-002:predict",
                    params={"key": self.api_key},
                    json={
                        "instances": [{"prompt": prompt}],
                        "parameters": {
                            "sampleCount": 1,
                            "aspectRatio": "1:1",
                        },
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    b64 = resp.json()["predictions"][0]["bytesBase64Encoded"]
                    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    img = img.resize((512, 512), Image.LANCZOS)
                    images.append(img)
                else:
                    print("  Gemini error: %s" % resp.text[:200])
                time.sleep(0.5)
            except Exception as e:
                print("  Gemini error: %s" % e)
        return images


def generate_for_benchmark(model_name, generator, benchmark_path, output_dir,
                            n_per_place=2):
    """Generate images for all places in the benchmark."""
    with open(str(benchmark_path)) as f:
        benchmark = json.load(f)

    gen_dir = output_dir / model_name
    total = 0

    for place in tqdm(benchmark["places"], desc=model_name):
        city = place["city"]
        tile = place["h3_tile"]
        country = place["country"]

        place_dir = gen_dir / city / tile
        place_dir.mkdir(parents=True, exist_ok=True)

        existing = list(place_dir.glob("*.jpg"))
        if len(existing) >= n_per_place:
            continue

        images = generator.generate(city, country, n=n_per_place)
        for i, img in enumerate(images):
            fname = "{0}_{1:03d}.jpg".format(model_name, i)
            img.save(str(place_dir / fname), quality=95)
            total += 1

    print("Generated %d images for %s" % (total, model_name))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dalle3", "gemini", "all"], default="all")
    parser.add_argument("--benchmark", default=str(
        config.PROCESSED_DIR / "benchmark_mapillary.json"))
    parser.add_argument("--output", default=str(
        config.OUTPUT_DIR / "generated_mapillary"))
    args = parser.parse_args()

    bm_path = Path(args.benchmark)
    out_dir = Path(args.output)

    if args.model in ["dalle3", "all"]:
        gen = DallE3Baseline()
        if gen.api_key:
            generate_for_benchmark("dalle3", gen, bm_path, out_dir)
        else:
            print("Skip DALL-E 3: no API key")

    if args.model in ["gemini", "all"]:
        gen = GeminiImagenBaseline()
        if gen.api_key:
            generate_for_benchmark("gemini", gen, bm_path, out_dir)
        else:
            print("Skip Gemini: no API key")


if __name__ == "__main__":
    main()
