"""
Trivial baselines for GeoFidelity-Bench.
These establish lower bounds and calibration points.
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional

import config


class RandomGlobalBaseline:
    """Return random images from the entire benchmark (any city)."""

    def __init__(self, all_image_paths: list[Path]):
        self.all_paths = all_image_paths

    def generate(self, n: int = 4, seed: int = 42) -> list[Image.Image]:
        rng = random.Random(seed)
        paths = rng.sample(self.all_paths, min(n, len(self.all_paths)))
        return [Image.open(str(p)).convert("RGB") for p in paths]


class RandomCountryBaseline:
    """Return random images from the same country."""

    def __init__(self, image_index: dict[str, list[Path]]):
        """image_index: {country_code: [image_paths]}"""
        self.index = image_index

    def generate(self, country: str, n: int = 4, seed: int = 42) -> list[Image.Image]:
        rng = random.Random(seed)
        pool = self.index.get(country, [])
        if not pool:
            return []
        paths = rng.sample(pool, min(n, len(pool)))
        return [Image.open(str(p)).convert("RGB") for p in paths]


class CityMedoidBaseline:
    """Return the city medoid — the most "average" image in the city."""

    def __init__(self, retriever):
        """retriever: PanelRetriever instance for computing embeddings."""
        self.retriever = retriever
        self.medoid_cache = {}

    def compute_medoid(self, images: list[Image.Image]) -> int:
        """Find the medoid (image closest to the mean embedding)."""
        embeddings = self.retriever.encode_batch(images)
        mean_emb = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - mean_emb, axis=1)
        return int(distances.argmin())

    def generate(self, city_images: list[Image.Image], n: int = 4) -> list[Image.Image]:
        """Return n copies of the city medoid (tests if "generic city" fools metrics)."""
        medoid_idx = self.compute_medoid(city_images)
        return [city_images[medoid_idx]] * n


class NearestNeighborBaseline:
    """Oracle: retrieve the closest real image from the target neighborhood.
    This is the upper bound — if a method can't beat this, it's just memorizing.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def generate(self, target_images: list[Image.Image],
                 n: int = 4, seed: int = 42) -> list[Image.Image]:
        """Return n random images from the target neighborhood itself."""
        rng = random.Random(seed)
        indices = rng.sample(range(len(target_images)), min(n, len(target_images)))
        return [target_images[i] for i in indices]


class WrongNeighborhoodBaseline:
    """Return images from a different neighborhood in the same city."""

    def generate(self, wrong_tile_images: list[Image.Image],
                 n: int = 4, seed: int = 42) -> list[Image.Image]:
        rng = random.Random(seed)
        indices = rng.sample(range(len(wrong_tile_images)),
                             min(n, len(wrong_tile_images)))
        return [wrong_tile_images[i] for i in indices]


def build_image_index(benchmark_path: Path, data_dir: Path) -> dict:
    """Build index from benchmark JSON for baseline use."""
    with open(str(benchmark_path)) as f:
        benchmark = json.load(f)

    # Index by country and city
    index = {
        "all_paths": [],
        "by_country": {},
        "by_city": {},
        "by_tile": {},
    }

    for place in benchmark["places"]:
        city = place["city"]
        country = place["country"]
        tile = place["h3_tile"]

        tile_dir = data_dir / "osv5m" / city / tile
        if not tile_dir.exists():
            continue

        paths = list(tile_dir.glob("*.jpg"))
        index["all_paths"].extend(paths)
        index["by_country"].setdefault(country, []).extend(paths)
        index["by_city"].setdefault(city, []).extend(paths)
        index["by_tile"][tile] = paths

    return index
