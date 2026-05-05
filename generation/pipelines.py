"""
Concrete Diffusers-based generators for the 7 open-source models in
GeoFidelity-Bench v2's generation roster.

Each class wraps one pipeline; all share the Generator interface defined
in base.py and are registered in registry.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

from .base import Generator, GenConfig, NEGATIVE_PROMPT_DEFAULT


def _cached_snapshot(repo: str) -> str:
    """Return a local HF snapshot path when available.

    AutoDL runs often preload model snapshots into a shared cache and then
    disable outbound Hub lookups. Passing the concrete snapshot directory to
    Diffusers avoids a repo-id resolution request while preserving the same
    model contents.
    """
    repo_dir = "models--" + repo.replace("/", "--")
    cache_roots = []
    if os.environ.get("HF_HUB_CACHE"):
        cache_roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    cache_roots.extend([
        Path("/autodl-fs/data/hf_cache/hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ])

    seen = set()
    for root in cache_roots:
        if root in seen:
            continue
        seen.add(root)
        snap_root = root / repo_dir / "snapshots"
        if not snap_root.exists():
            continue
        snapshots = [p for p in snap_root.iterdir() if p.is_dir()]
        if snapshots:
            return str(max(snapshots, key=lambda p: p.stat().st_mtime))
    return repo


def _pretrained_kwargs(gen: Generator, **kwargs) -> dict:
    merged = {
        "torch_dtype": gen._torch_dtype(),
        "local_files_only": True,
    }
    if gen.cfg.extra_kwargs:
        merged.update(gen.cfg.extra_kwargs)
    merged.update(kwargs)
    return merged


class _SDXLGen(Generator):
    def load(self) -> None:
        from diffusers import StableDiffusionXLPipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            _cached_snapshot(self.cfg.repo),
            **_pretrained_kwargs(
                self,
                use_safetensors=True,
                variant="fp16" if self.cfg.dtype == "fp16" else None,
            ),
        )
        if self.cfg.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        if self.cfg.enable_vae_tiling:
            self.pipe.vae.enable_tiling()

    def generate(self, prompt: str, seed: int,
                 negative_prompt: str | None = None) -> Image.Image:
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or NEGATIVE_PROMPT_DEFAULT,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            height=self.cfg.height, width=self.cfg.width,
            generator=g,
        ).images[0]
        return self._resize(out)


class _SD3Gen(Generator):
    def load(self) -> None:
        from diffusers import StableDiffusion3Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            _cached_snapshot(self.cfg.repo),
            **_pretrained_kwargs(self),
        )
        # Diffusers' SD3 loader sometimes leaves text_encoder_3 (T5) in fp32
        # or mixes fp16/bfloat16 across the three encoders, producing
        # "mat1 mat2 dtype mismatch" at encode_prompt. Force every submodule
        # onto the target dtype so the joint prompt embedding stays uniform.
        dt = self._torch_dtype()
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3",
                     "transformer", "vae"):
            mod = getattr(self.pipe, attr, None)
            if mod is not None and hasattr(mod, "to"):
                mod.to(dtype=dt)
        if self.cfg.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, seed: int,
                 negative_prompt: str | None = None) -> Image.Image:
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or NEGATIVE_PROMPT_DEFAULT,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            height=self.cfg.height, width=self.cfg.width,
            generator=g,
        ).images[0]
        return self._resize(out)


class _FluxGen(Generator):
    """FLUX dev or schnell. Note: schnell ignores guidance_scale."""

    def load(self) -> None:
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(
            _cached_snapshot(self.cfg.repo),
            **_pretrained_kwargs(self),
        )
        if self.cfg.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, seed: int,
                 negative_prompt: str | None = None) -> Image.Image:
        g = torch.Generator(device="cpu").manual_seed(seed)   # FLUX uses CPU rng
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            height=self.cfg.height, width=self.cfg.width,
            generator=g, max_sequence_length=256,
        ).images[0]
        return self._resize(out)


class _PixArtGen(Generator):
    def load(self) -> None:
        from diffusers import PixArtSigmaPipeline
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            _cached_snapshot(self.cfg.repo),
            **_pretrained_kwargs(self),
        )
        if self.cfg.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, seed: int,
                 negative_prompt: str | None = None) -> Image.Image:
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or NEGATIVE_PROMPT_DEFAULT,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            height=self.cfg.height, width=self.cfg.width,
            generator=g,
        ).images[0]
        return self._resize(out)


class _HunyuanDiTGen(Generator):
    def load(self) -> None:
        from diffusers import HunyuanDiTPipeline
        self.pipe = HunyuanDiTPipeline.from_pretrained(
            _cached_snapshot(self.cfg.repo),
            **_pretrained_kwargs(self),
        )
        if self.cfg.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, seed: int,
                 negative_prompt: str | None = None) -> Image.Image:
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or NEGATIVE_PROMPT_DEFAULT,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            height=self.cfg.height, width=self.cfg.width,
            generator=g,
        ).images[0]
        return self._resize(out)
