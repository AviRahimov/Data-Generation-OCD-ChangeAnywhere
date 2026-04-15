"""Stable Diffusion inpainting wrapper for generating synthetic 'after' images."""

import sys
import numpy as np
from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

_SD2_REPOS = [
    "stable-diffusion-2-inpainting",
    "sd2-community/stable-diffusion-2-inpainting",
]


class InpaintingModel:
    """Wraps a Stable Diffusion inpainting pipeline."""

    def __init__(self, model_id=None, device="cuda",
                 num_inference_steps=30, guidance_scale=7.5,
                 strength=0.85):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Install diffusers: pip install diffusers accelerate")

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.strength = strength

        repos = [model_id] if model_id else _SD2_REPOS
        pipe = None
        for repo in repos:
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    repo, torch_dtype=torch.float16,
                )
                break
            except Exception as e:
                print(f"  Could not load {repo}: {e}", file=sys.stderr)
                continue

        if pipe is None:
            raise RuntimeError(
                f"Failed to load any inpainting model from: {repos}"
            )

        self.pipe = pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def inpaint(self, image, mask, prompt,
                negative_prompt="blurry, low quality, cartoon, painting, watermark",
                seed=None):
        """Inpaint the masked region of `image` guided by `prompt`.

        Args:
            image: PIL RGB image (should be 512x512 for best results)
            mask: PIL L image or bool ndarray -- white/True = region to regenerate
            prompt: text description of what to generate in the masked area
            negative_prompt: what to avoid
            seed: optional int for reproducibility

        Returns:
            PIL RGB image with the masked region inpainted
        """
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray((mask.astype(np.uint8) * 255))
        else:
            mask_pil = mask.convert("L")

        image_rgb = image.convert("RGB")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_rgb,
            mask_image=mask_pil,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=generator,
        ).images[0]

        return result

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
