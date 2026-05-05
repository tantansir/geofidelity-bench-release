"""
Abstract base for T2I generators.

Each concrete generator wraps a single Diffusers pipeline. The runner
(generation/run_generation.py) instantiates one generator at a time,
calls load() / generate(...) / unload() so peak VRAM is bounded by the
biggest model (FLUX.1-dev ~24 GB fp16).

All generators must honor:
  * a fixed per-model seed per (place, k) so runs are reproducible
  * sizing to 512x512 JPEGs in the output folder
  * returning PIL.Image, not saving, so the runner can overlay metadata
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class GenConfig:
    """Per-model generation hyperparameters."""
    name: str
    repo: str
    num_inference_steps: int
    guidance_scale: float
    height: int = 1024
    width: int = 1024
    out_size: int = 512           # downsize before saving / eval
    dtype: str = "fp16"           # fp16 / bf16 / fp32
    pipeline_cls: str = ""         # diffusers pipeline class name
    enable_cpu_offload: bool = False
    enable_vae_tiling: bool = True
    extra_kwargs: dict | None = None


class Generator(ABC):
    def __init__(self, cfg: GenConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.pipe = None

    @abstractmethod
    def load(self) -> None: ...

    def unload(self) -> None:
        self.pipe = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def generate(self, prompt: str, seed: int,
                 negative_prompt: str | None = None) -> Image.Image: ...

    def _torch_dtype(self) -> torch.dtype:
        return {"fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32}[self.cfg.dtype]

    def _resize(self, img: Image.Image) -> Image.Image:
        s = self.cfg.out_size
        if img.size != (s, s):
            img = img.resize((s, s), Image.LANCZOS)
        return img


NEGATIVE_PROMPT_DEFAULT = (
    "panorama, fisheye, black and white, illustration, painting, cartoon, "
    "3d render, watermark, text overlay, blurry, low resolution, night, "
    "people close up, indoor"
)
