"""Generate street views using SiliconFlow free API (Kolors, Qwen-Image)."""
import sys, os, json, time, requests
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import config

SILICONFLOW_TOKEN = os.environ.get(
    "SILICONFLOW_TOKEN",
    "sk-ttwdxtkhveydjiscoqkpftkuxwtcjxekaelnbleicstijeat"
)
API_URL = "https://api.siliconflow.cn/v1/images/generations"

PROMPT_TEMPLATE = (
    "A street-level photograph taken in {city}, {country}. "
    "The image shows a typical street scene with buildings, roads, "
    "and urban environment characteristic of this location. "
    "Photorealistic, daytime, clear weather."
)

def generate_image(model, prompt, seed=42):
    resp = requests.post(API_URL,
        headers={"Authorization": "Bearer " + SILICONFLOW_TOKEN,
                 "Content-Type": "application/json"},
        json={"model": model, "prompt": prompt,
              "image_size": "512x512", "seed": seed},
        timeout=90)
    if resp.status_code != 200:
        return None
    data = resp.json()
    url = (data.get("images", [{}])[0].get("url") or
           data.get("data", [{}])[0].get("url", ""))
    if not url:
        return None
    img_resp = requests.get(url, timeout=30)
    if img_resp.status_code == 200:
        return Image.open(BytesIO(img_resp.content)).convert("RGB").resize((512,512), Image.LANCZOS)
    return None

def generate_for_benchmark(model_id, short_name, benchmark_path, output_dir, n=2):
    with open(str(benchmark_path)) as f:
        bm = json.load(f)
    gen_dir = output_dir / short_name
    total = 0
    for place in tqdm(bm["places"], desc=short_name):
        city, tile = place["city"], place["h3_tile"]
        d = gen_dir / city / tile
        d.mkdir(parents=True, exist_ok=True)
        if list(d.glob("*.jpg")):
            continue
        prompt = PROMPT_TEMPLATE.format(
            city=city.replace("_", " ").title(), country=place["country"])
        for i in range(n):
            img = generate_image(model_id, prompt, seed=42+i)
            if img:
                img.save(str(d / "{0}_{1:03d}.jpg".format(short_name, i)), quality=95)
                total += 1
            time.sleep(0.5)
    print("Generated %d images for %s" % (total, short_name))

if __name__ == "__main__":
    bm = config.PROCESSED_DIR / "benchmark_mapillary.json"
    out = config.OUTPUT_DIR / "generated_mapillary"
    generate_for_benchmark("Kwai-Kolors/Kolors", "kolors", bm, out)
    generate_for_benchmark("Qwen/Qwen-Image", "qwen_image", bm, out)
