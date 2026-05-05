"""
Shared semantic segmentation backbone for GeoFidelity-Bench.

Uses Mask2Former Swin-Large pretrained on Mapillary Vistas (65 semantic
classes), chosen because:
  1. Mapillary Vistas is a global street-view dataset - no Cityscapes European
     bias that hurt v1's GAAS on African / Asian cities.
  2. The v2 reference data is Mapillary itself, so in-domain segmentation is
     more reliable than cross-domain Cityscapes -> global inference.
  3. 65-class label set is fine-grained enough for meaningful
     road / sidewalk / pole / traffic-sign distinctions.

This backbone is used both by the Tier-4 curation filter and by the GAAS
metric so evaluation stays consistent with curation.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

import config


# Mapillary Vistas v1.2 class IDs grouped into GAAS attribute buckets.
# Reference: https://research.mapillary.com/img/publications/MVD_ICCV17_supplementary.pdf
VISTAS_GROUPS = {
    "road":       [13, 14, 23, 24],               # road, service lane, lane markings
    "sidewalk":   [15, 11, 2, 8, 9],              # sidewalk, pedestrian area, curb, crosswalk
    "building":   [17, 6, 16, 18, 3, 5],          # building, wall, bridge, tunnel, fence, barrier
    "vehicle":    [52, 53, 54, 55, 56, 57, 58,
                   59, 60, 61, 62, 63, 64],       # all vehicle classes + ego + mount
    "person":     [19, 20, 21, 22],               # person + riders
    "vegetation": [30, 29, 25],                   # vegetation, terrain, mountain
    "sky":        [27],                           # sky
    "pole":       [45, 47, 44],                   # pole, utility pole, street light
    "sign":       [48, 49, 50, 46, 35, 32],       # traffic light, signs, banner, billboard
    "water":      [31],
    "snow":       [28],
    "other":      [0, 1, 4, 26, 33, 34, 36, 37,
                   38, 39, 40, 41, 42, 43, 51,
                   65],
}


class Mask2FormerVistas:
    """Thin wrapper around Mask2Former Swin-Large @ Mapillary Vistas."""

    def __init__(self, device: str = config.DEVICE,
                 model_name: str = config.TIER4_SEG_MODEL):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        from transformers import (Mask2FormerForUniversalSegmentation,
                                   Mask2FormerImageProcessor)
        print(f"[seg] loading {self.model_name} on {self.device}")
        self._processor = Mask2FormerImageProcessor.from_pretrained(self.model_name)
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_name).to(self.device).eval()

    @torch.no_grad()
    def segment(self, image: Image.Image) -> np.ndarray:
        """Return HxW int32 semantic segmentation map."""
        self.load()
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        out = self._model(**inputs)
        seg = self._processor.post_process_semantic_segmentation(
            out, target_sizes=[image.size[::-1]])[0]
        return seg.cpu().numpy().astype(np.int32)

    @torch.no_grad()
    def segment_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Batched inference. Images may be different sizes (resized internally)."""
        self.load()
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        out = self._model(**inputs)
        sizes = [img.size[::-1] for img in images]
        segs = self._processor.post_process_semantic_segmentation(
            out, target_sizes=sizes)
        return [s.cpu().numpy().astype(np.int32) for s in segs]


def ratios_from_seg(seg: np.ndarray) -> dict[str, float]:
    """Aggregate per-class pixel counts into attribute-group ratios."""
    total = seg.size
    if total == 0:
        return {k: 0.0 for k in VISTAS_GROUPS}
    out: dict[str, float] = {}
    for group, ids in VISTAS_GROUPS.items():
        mask = np.isin(seg, ids)
        out[group] = float(mask.sum()) / total
    return out


def urbanness(ratios: dict[str, float]) -> float:
    """Composite score: more building/sidewalk good; truck/highway road bad."""
    return (ratios.get("building", 0.0)
            + 0.3 * ratios.get("sidewalk", 0.0)
            - max(0.0, ratios.get("vehicle", 0.0) - 0.10)
            - max(0.0, ratios.get("road", 0.0) - 0.30))
