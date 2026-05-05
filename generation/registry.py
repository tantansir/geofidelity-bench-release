"""
Model registry for the 6-model open-source generation roster.

VRAM budgets are for fp16 with VAE tiling + attention slicing.
Enable `cpu_offload` in the GenConfig if the host GPU is smaller.
"""
from __future__ import annotations

from .base import GenConfig, Generator
from .pipelines import (_SDXLGen, _SD3Gen, _FluxGen, _PixArtGen,
                         _HunyuanDiTGen)


MODEL_REGISTRY: dict[str, tuple[type[Generator], GenConfig]] = {
    "sdxl_base": (_SDXLGen, GenConfig(
        name="sdxl_base",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        num_inference_steps=30, guidance_scale=5.0,
        height=1024, width=1024, dtype="fp16",
        pipeline_cls="StableDiffusionXLPipeline",
    )),
    "sd35_large": (_SD3Gen, GenConfig(
        name="sd35_large",
        repo="stabilityai/stable-diffusion-3.5-large",
        num_inference_steps=28, guidance_scale=3.5,
        height=1024, width=1024, dtype="bf16",
        pipeline_cls="StableDiffusion3Pipeline",
        enable_cpu_offload=True,   # needs CPU offload on 24 GB cards
    )),
    "flux_dev": (_FluxGen, GenConfig(
        name="flux_dev",
        repo="black-forest-labs/FLUX.1-dev",
        num_inference_steps=28, guidance_scale=3.5,
        height=1024, width=1024, dtype="bf16",
        pipeline_cls="FluxPipeline",
        enable_cpu_offload=True,
    )),
    "flux_schnell": (_FluxGen, GenConfig(
        name="flux_schnell",
        repo="black-forest-labs/FLUX.1-schnell",
        num_inference_steps=4, guidance_scale=0.0,
        height=1024, width=1024, dtype="bf16",
        pipeline_cls="FluxPipeline",
        enable_cpu_offload=True,
    )),
    "pixart_sigma": (_PixArtGen, GenConfig(
        name="pixart_sigma",
        repo="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        num_inference_steps=20, guidance_scale=4.5,
        height=1024, width=1024, dtype="fp16",
        pipeline_cls="PixArtSigmaPipeline",
    )),
    "hunyuan_dit": (_HunyuanDiTGen, GenConfig(
        name="hunyuan_dit",
        repo="Tencent-Hunyuan/HunyuanDiT-Diffusers",
        num_inference_steps=50, guidance_scale=5.0,
        height=1024, width=1024, dtype="fp16",
        pipeline_cls="HunyuanDiTPipeline",
    )),
}


def load_generator(name: str, device: str = "cuda") -> Generator:
    cls, cfg = MODEL_REGISTRY[name]
    gen = cls(cfg, device=device)
    gen.load()
    return gen
