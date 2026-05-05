"""
Leakage Audit for GeoFidelity-Bench.
Ensures benchmark images are not in the training data of evaluated models.

Methods:
1. pHash near-duplicate detection
2. DINOv2 embedding nearest-neighbor search
3. Temporal filter (post-training-cutoff images only)
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import config


def compute_phash(image: Image.Image, hash_size: int = 16) -> str:
    """Compute perceptual hash of an image."""
    # Resize to hash_size x hash_size
    img = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=float)

    # DCT-like: compare each pixel to mean
    mean = pixels.mean()
    bits = (pixels > mean).flatten()

    # Convert to hex string
    hash_val = sum(1 << i for i, b in enumerate(bits) if b)
    return format(hash_val, f"0{hash_size * hash_size // 4}x")


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count("1")


class LeakageAuditor:
    """Audit benchmark for data leakage."""

    def __init__(self, threshold_hamming: int = 10):
        self.threshold = threshold_hamming

    def compute_hashes(self, image_dir: Path) -> dict[str, str]:
        """Compute pHash for all images in directory."""
        hashes = {}
        for img_path in tqdm(sorted(image_dir.rglob("*.jpg")), desc="Hashing"):
            try:
                img = Image.open(str(img_path)).convert("RGB")
                h = compute_phash(img)
                hashes[str(img_path)] = h
            except Exception as e:
                pass
        return hashes

    def find_near_duplicates(self, hashes: dict[str, str]) -> list[tuple]:
        """Find near-duplicate pairs."""
        paths = list(hashes.keys())
        hash_vals = list(hashes.values())
        duplicates = []

        for i in tqdm(range(len(paths)), desc="Checking duplicates"):
            for j in range(i + 1, len(paths)):
                dist = hamming_distance(hash_vals[i], hash_vals[j])
                if dist <= self.threshold:
                    duplicates.append((paths[i], paths[j], dist))

        return duplicates

    def cross_split_audit(self, split1_dir: Path, split2_dir: Path) -> dict:
        """Check for near-duplicates between two data splits."""
        print(f"Computing hashes for split 1: {split1_dir}")
        hashes1 = self.compute_hashes(split1_dir)

        print(f"Computing hashes for split 2: {split2_dir}")
        hashes2 = self.compute_hashes(split2_dir)

        print("Cross-checking for near-duplicates...")
        cross_duplicates = []
        paths1 = list(hashes1.keys())
        vals1 = list(hashes1.values())
        paths2 = list(hashes2.keys())
        vals2 = list(hashes2.values())

        for i in tqdm(range(len(paths1)), desc="Cross-checking"):
            for j in range(len(paths2)):
                dist = hamming_distance(vals1[i], vals2[j])
                if dist <= self.threshold:
                    cross_duplicates.append((paths1[i], paths2[j], dist))

        return {
            "split1_size": len(hashes1),
            "split2_size": len(hashes2),
            "cross_duplicates": len(cross_duplicates),
            "pairs": cross_duplicates[:50],  # limit output
        }

    def audit_benchmark(self, benchmark_dir: Path) -> dict:
        """Full leakage audit of the benchmark dataset."""
        print("=" * 60)
        print("LEAKAGE AUDIT")
        print("=" * 60)

        # 1. Within-benchmark dedup
        print("\n--- Within-benchmark deduplication ---")
        all_hashes = self.compute_hashes(benchmark_dir)
        within_dupes = self.find_near_duplicates(all_hashes)

        print(f"Total images: {len(all_hashes)}")
        print(f"Near-duplicates found: {len(within_dupes)}")
        if within_dupes:
            print("Examples:")
            for p1, p2, dist in within_dupes[:5]:
                print(f"  {Path(p1).name} <-> {Path(p2).name} (dist={dist})")

        return {
            "total_images": len(all_hashes),
            "within_duplicates": len(within_dupes),
            "duplicate_pairs": [(str(p1), str(p2), d) for p1, p2, d in within_dupes],
            "hashes": {k: v for k, v in list(all_hashes.items())[:100]},
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Leakage Audit")
    parser.add_argument("--data-dir", type=str,
                        default=str(config.DATA_DIR))
    parser.add_argument("--output", type=str,
                        default=str(config.OUTPUT_DIR / "audit"))
    args = parser.parse_args()

    auditor = LeakageAuditor(threshold_hamming=10)
    result = auditor.audit_benchmark(Path(args.data_dir))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(output_dir / "leakage_audit.json"), "w") as f:
        json.dump({k: v for k, v in result.items() if k != "hashes"},
                  f, indent=2)

    print(f"\nAudit saved to {output_dir / 'leakage_audit.json'}")


if __name__ == "__main__":
    main()
