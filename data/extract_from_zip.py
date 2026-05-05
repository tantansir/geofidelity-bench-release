"""
Extract target city images from OSV-5M zip files.
Matches images by ID against the filtered metadata.
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import zipfile
import h3
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

import config


def extract_target_images(zip_path: str, metadata_csv: str, output_dir: Path):
    """Extract images from OSV-5M zip that match target cities."""
    print(f"Loading metadata from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)

    # Filter for target cities
    filtered_rows = []
    for city_name, city_info in config.CITIES.items():
        center = (city_info["lat"], city_info["lon"])
        mask = (
            (df["latitude"] > center[0] - 0.3) & (df["latitude"] < center[0] + 0.3) &
            (df["longitude"] > center[1] - 0.4) & (df["longitude"] < center[1] + 0.4)
        )
        city_df = df[mask].copy()
        city_df["target_city"] = city_name
        filtered_rows.append(city_df)

    filtered = pd.concat(filtered_rows, ignore_index=True)
    target_ids = set(str(int(x)) for x in filtered["id"].values)
    print(f"Looking for {len(target_ids)} target images in zip...")

    # Open zip and extract matching images
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"Zip contains {len(namelist)} files")

        for name in tqdm(namelist, desc="Extracting"):
            # Image files are named like {id}.jpg
            img_id = Path(name).stem
            if img_id not in target_ids:
                continue

            # Find the matching row
            row = filtered[filtered["id"].astype(str).str.startswith(img_id[:10])]
            if row.empty:
                # Try exact match
                row = filtered[filtered["id"] == int(img_id)]
            if row.empty:
                continue

            row = row.iloc[0]
            city = row["target_city"]
            tile = h3.latlng_to_cell(row["latitude"], row["longitude"], config.H3_RESOLUTION)

            tile_dir = output_dir / city / tile
            tile_dir.mkdir(parents=True, exist_ok=True)

            img_path = tile_dir / f"osv5m_{img_id}.jpg"
            if img_path.exists():
                continue

            try:
                img_data = zf.read(name)
                img = Image.open(BytesIO(img_data)).convert("RGB")
                img = img.resize((512, 512), Image.LANCZOS)
                img.save(str(img_path), quality=95)
                extracted += 1
            except Exception as e:
                print(f"  Error: {name}: {e}")

    print(f"\nExtracted {extracted} images to {output_dir}")

    # Save metadata for extracted images
    meta_rows = []
    for city_dir in output_dir.iterdir():
        if not city_dir.is_dir():
            continue
        for tile_dir in city_dir.iterdir():
            if not tile_dir.is_dir():
                continue
            for img_path in tile_dir.glob("*.jpg"):
                img_id = img_path.stem.replace("osv5m_", "")
                match = filtered[filtered["id"] == int(img_id)]
                if not match.empty:
                    r = match.iloc[0]
                    meta_rows.append({
                        "id": img_id,
                        "latitude": r["latitude"],
                        "longitude": r["longitude"],
                        "country": r.get("country", ""),
                        "target_city": r["target_city"],
                        "h3_tile": tile_dir.name,
                    })

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        meta_path = config.PROCESSED_DIR / "osv5m_real_metadata.csv"
        meta_df.to_csv(str(meta_path), index=False)
        print(f"Metadata saved: {meta_path} ({len(meta_df)} rows)")

    return extracted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, required=True, help="Path to OSV-5M zip file")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the OSV-5M metadata CSV.")
    args = parser.parse_args()

    extract_target_images(
        zip_path=args.zip,
        metadata_csv=args.csv,
        output_dir=config.DATA_DIR / "osv5m_real",
    )


if __name__ == "__main__":
    main()
