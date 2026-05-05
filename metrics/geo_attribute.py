"""
Geo-Attribute Agreement Score (GAAS) v2.

Swaps the v1 SegFormer-Cityscapes backbone for Mask2Former pretrained on
Mapillary Vistas (global street-view dataset, ~66 classes), eliminating
the v1 reviewer hazard where Cityscapes domain-shift could explain away
high GAAS on non-European cities.

GAAS is the mean Jensen-Shannon divergence across attribute groups
(road / sidewalk / building / vegetation / sky / pole / sign / vehicle)
between per-image pixel-ratio histograms of the generated set and the
reference set.

For the reference set we reuse ratios already persisted by the Tier-4
curation filter (data/processed/tier4_segmentation.csv) so no double
inference. Generated images are segmented fresh.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import jensenshannon

import config
from data.segmentation_backbone import (Mask2FormerVistas, VISTAS_GROUPS,
                                         ratios_from_seg)


# Attribute groups actually used in GAAS (ignore high-noise buckets)
GAAS_ATTRS = ["road", "sidewalk", "building", "vehicle",
              "vegetation", "sky", "pole", "sign"]


class GeoAttributeExtractor:
    """Extract Mapillary-Vistas attribute ratios with caching."""

    def __init__(self, device: str = config.DEVICE):
        self.device = device
        self._seg: Optional[Mask2FormerVistas] = None
        self._cache: dict[str, dict[str, float]] = {}

    def load(self):
        if self._seg is None:
            self._seg = Mask2FormerVistas(device=self.device)
            self._seg.load()

    def warm_from_tier4(self, tier4_csv: Path) -> int:
        """Populate cache with ratios already computed by Tier-4 curation."""
        df = pd.read_csv(tier4_csv)
        n = 0
        for _, r in df.iterrows():
            if pd.isna(r.get("urbanness")):
                continue
            p = str(r["image_path"])
            self._cache[p] = {g: float(r.get(f"ratio_{g}", 0.0))
                               for g in VISTAS_GROUPS}
            n += 1
        return n

    def extract(self, image_path: Path) -> dict[str, float]:
        rel = str(image_path.relative_to(config.ROOT).as_posix()) \
            if image_path.is_absolute() else str(image_path)
        if rel in self._cache:
            return self._cache[rel]
        self.load()
        img = Image.open(image_path).convert("RGB")
        seg = self._seg.segment(img)
        r = ratios_from_seg(seg)
        self._cache[rel] = r
        return r

    def extract_batch(self, paths: Iterable[Path],
                      batch_size: int = 4) -> list[dict[str, float]]:
        paths = list(paths)
        # Resolve relative keys and split cached vs uncached
        uncached: list[Path] = []
        out_map: dict[int, dict[str, float]] = {}
        for i, p in enumerate(paths):
            rel = str(p.relative_to(config.ROOT).as_posix()) \
                if p.is_absolute() else str(p)
            if rel in self._cache:
                out_map[i] = self._cache[rel]
            else:
                uncached.append((i, p))
        if uncached:
            self.load()
            for i in range(0, len(uncached), batch_size):
                chunk = uncached[i:i + batch_size]
                pils = [Image.open(p).convert("RGB") for _, p in chunk]
                segs = self._seg.segment_batch(pils)
                for (idx, p), s in zip(chunk, segs):
                    r = ratios_from_seg(s)
                    rel = str(p.relative_to(config.ROOT).as_posix()) \
                        if p.is_absolute() else str(p)
                    self._cache[rel] = r
                    out_map[idx] = r
        return [out_map[i] for i in range(len(paths))]


def _hist(values: list[float], n_bins: int = 10) -> np.ndarray:
    h, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    h = h.astype(np.float64) + 1e-8
    return h / h.sum()


def gaas(gen_ratios: list[dict[str, float]],
         ref_ratios: list[dict[str, float]],
         attrs: list[str] = GAAS_ATTRS,
         n_bins: int = 10) -> dict:
    """GAAS: mean per-attribute JS-divergence between generated and reference.

    Lower is better. Returns per-attribute divergences and the mean.
    """
    per_attr: dict[str, float] = {}
    for a in attrs:
        g = _hist([r.get(a, 0.0) for r in gen_ratios], n_bins)
        f = _hist([r.get(a, 0.0) for r in ref_ratios], n_bins)
        per_attr[a] = float(jensenshannon(g, f, base=2))
    return {"overall": float(np.mean(list(per_attr.values()))),
            "per_attribute": per_attr}


class GeoAttributeAgreementScore:
    """Thin orchestrator: extractor + per-place GAAS."""

    def __init__(self, device: str = config.DEVICE,
                 tier4_csv: Path | None = None):
        self.extractor = GeoAttributeExtractor(device=device)
        if tier4_csv is not None and Path(tier4_csv).exists():
            n = self.extractor.warm_from_tier4(Path(tier4_csv))
            print(f"[gaas] warm cache: {n} ref ratios")

    def evaluate_place(self, gen_paths: list[Path],
                       ref_paths: list[Path]) -> dict:
        gen_r = self.extractor.extract_batch(gen_paths)
        ref_r = self.extractor.extract_batch(ref_paths)
        return gaas(gen_r, ref_r)
