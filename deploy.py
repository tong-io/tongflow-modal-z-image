"""Modal deploy entry for z-image.

Deploy:
  modal deploy deploy.py

Design constraints:
  - Keep this file mostly self-contained because Modal remote imports may mount
    only the entry file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
from tongflow import deploy
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset
from tongflow.slots import node_slot


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "Tongyi-MAI/Z-Image-Turbo")
MODEL_DIR = f"/models/{REPO_ID}"

# Diffusion sampling defaults — plugin-internal, not part of the ABI contract.
DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 0.0

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)


# ── app ──────────────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install(
        "tongflow==0.1.0",
        "diffusers==0.37.1",
        "transformers==5.4.0",
        "safetensors==0.7.0",
        "loguru==0.7.3",
        "pillow==12.1.1",
        "accelerate==1.13.0",
        "huggingface_hub==1.6.0",
        "tqdm==4.67.3",
        "sentencepiece==0.2.1",
    )
)

with image.imports():
    import torch
    from diffusers import ZImagePipeline


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
)
class Inference:
    @modal.enter()
    def load(self):
        self.pipe = ZImagePipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def _png_bytes(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        guidance_scale: float = 0.0,
        seed: int = 42,
    ) -> bytes:
        import io

        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps + 1,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        guidance_scale: float = 0.0,
        seed: int = 42,
    ) -> bytes:
        return self._png_bytes(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN)
    def image_gen(self, input: ImageGenInput) -> ImageGenOutput:
        text = (input.text or "").strip()
        if not text:
            return ImageGenOutput(success=False, error="Missing text prompt")

        raw = self._png_bytes(
            text,
            height=input.height if input.height is not None else 1024,
            width=input.width if input.width is not None else 1024,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=int(input.seed) if input.seed is not None else 42,
        )
        return ImageGenOutput(success=True, image=asset(raw, mime="image/png"))
