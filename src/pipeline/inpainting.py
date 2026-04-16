"""Stable Diffusion inpainting wrapper with Poisson seamless blending."""

import sys
import numpy as np
from PIL import Image, ImageFilter

try:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

import cv2
from scipy.ndimage import binary_dilation

_SD2_REPOS = [
    "stable-diffusion-2-inpainting",
    "sd2-community/stable-diffusion-2-inpainting",
]

_DEFAULT_NEG = (
    "blurry, low quality, cartoon, painting, watermark, "
    "blue tint, color fringe, edge artifact, seam, border artifact"
)


class InpaintingModel:
    """Wraps a Stable Diffusion inpainting pipeline with feathered blending."""

    def __init__(self, model_id=None, device="cuda",
                 num_inference_steps=30, guidance_scale=7.5,
                 strength=0.85,
                 mask_blur_radius=12, mask_dilate_px=8,
                 blend_mode="poisson"):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Install diffusers: pip install diffusers accelerate")

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.mask_blur_radius = mask_blur_radius
        self.mask_dilate_px = mask_dilate_px
        self.blend_mode = blend_mode

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

    def _feather_mask(self, mask_bool, dilate_px=None, blur_radius=None):
        """Dilate then Gaussian-blur a binary mask to produce soft edges.

        Args:
            mask_bool: 2-D bool ndarray
            dilate_px: override dilation iterations (None = use self.mask_dilate_px)
            blur_radius: override blur radius (None = use self.mask_blur_radius)

        Returns a PIL "L" image (0-255) with soft transitions at the boundary.
        """
        dilate = dilate_px if dilate_px is not None else self.mask_dilate_px
        blur = blur_radius if blur_radius is not None else self.mask_blur_radius

        if dilate > 0:
            mask_bool = binary_dilation(mask_bool, iterations=dilate)

        mask_pil = Image.fromarray((mask_bool.astype(np.uint8) * 255))

        if blur > 0:
            mask_pil = mask_pil.filter(
                ImageFilter.GaussianBlur(radius=blur)
            )

        return mask_pil

    @staticmethod
    def _alpha_blend(original_pil, inpainted_pil, feathered_mask_pil):
        """Blend inpainted result into original using the feathered mask as alpha."""
        alpha = np.array(feathered_mask_pil).astype(np.float32) / 255.0
        alpha = alpha[..., np.newaxis]  # (H, W, 1)
        orig = np.array(original_pil).astype(np.float32)
        inpainted = np.array(inpainted_pil).astype(np.float32)
        blended = orig * (1.0 - alpha) + inpainted * alpha
        return Image.fromarray(blended.astype(np.uint8))

    @staticmethod
    def _poisson_blend(original_pil, inpainted_pil, mask_bool, dilate_px=8):
        """Blend via Poisson equation (cv2.seamlessClone).

        Unlike alpha blending, Poisson blending adjusts colors of the source
        to match the destination at the boundary -- no ghost/halo pixels.
        Falls back to alpha blending if seamlessClone fails.
        """
        dilated = binary_dilation(mask_bool, iterations=max(dilate_px, 1))
        clone_mask = (dilated.astype(np.uint8) * 255)

        src = np.array(inpainted_pil)[:, :, ::-1]  # RGB -> BGR
        dst = np.array(original_pil)[:, :, ::-1]

        ys, xs = np.where(dilated)
        if len(ys) == 0:
            return original_pil
        center = (int(xs.mean()), int(ys.mean()))

        try:
            result_bgr = cv2.seamlessClone(
                src, dst, clone_mask, center, cv2.NORMAL_CLONE)
            return Image.fromarray(result_bgr[:, :, ::-1])  # BGR -> RGB
        except cv2.error:
            feathered = Image.fromarray(clone_mask).filter(
                ImageFilter.GaussianBlur(radius=12))
            return InpaintingModel._alpha_blend(
                original_pil, inpainted_pil, feathered)

    @torch.inference_mode()
    def _run_sd2(self, image_rgb, mask_bool, prompt, negative_prompt, seed):
        """Core SD2 pipeline call. Returns (raw_result, feathered_mask) at
        the original image size -- no alpha blending."""
        feathered = self._feather_mask(mask_bool)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        orig_size = image_rgb.size

        raw_result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_rgb,
            mask_image=feathered,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=generator,
        ).images[0]

        if raw_result.size != orig_size:
            raw_result = raw_result.resize(orig_size, Image.LANCZOS)

        return raw_result

    def inpaint(self, image, mask, prompt,
                negative_prompt=None,
                seed=None):
        """Inpaint the masked region with feathered blending.

        Convenience wrapper that runs SD2 and alpha-blends the result.
        Used by process_one.py and batch_generate. For object-centric
        inpainting use inpaint_raw() + manual blend instead.

        Returns:
            PIL RGB image with seamless inpainted region
        """
        if negative_prompt is None:
            negative_prompt = _DEFAULT_NEG

        if isinstance(mask, np.ndarray):
            mask_bool = mask.astype(bool)
        else:
            mask_bool = np.array(mask.convert("L")) > 127

        image_rgb = image.convert("RGB")
        raw_result = self._run_sd2(image_rgb, mask_bool, prompt,
                                   negative_prompt, seed)

        feathered = self._feather_mask(mask_bool)
        if feathered.size != image_rgb.size:
            feathered = feathered.resize(image_rgb.size, Image.LANCZOS)

        return self._alpha_blend(image_rgb, raw_result, feathered)

    def inpaint_raw(self, image, mask, prompt,
                    negative_prompt=None, seed=None):
        """Run SD2 inpainting and return the raw output without alpha-blending.

        The caller is responsible for blending. This avoids the double-blend
        problem when inpaint_object() needs to do its own crop-level blend.

        Returns:
            PIL RGB image -- raw SD2 output (fully regenerated in masked area,
            original pixels outside the mask region may differ due to SD2's
            global denoising)
        """
        if negative_prompt is None:
            negative_prompt = _DEFAULT_NEG

        if isinstance(mask, np.ndarray):
            mask_bool = mask.astype(bool)
        else:
            mask_bool = np.array(mask.convert("L")) > 127

        image_rgb = image.convert("RGB")
        return self._run_sd2(image_rgb, mask_bool, prompt,
                             negative_prompt, seed)

    def inpaint_object(self, full_image, obj_mask_fullimg, prompt,
                       bbox_fullimg, pad_ratio=0.3, feather_margin=48,
                       seed=None):
        """Inpaint a detected object directly on the full image.

        Uses Poisson blending (cv2.seamlessClone) by default to eliminate
        visible edges. The Poisson solver adjusts colors at the boundary
        so no ghost/halo pixels appear. Falls back to alpha blending
        if blend_mode="alpha" or if seamlessClone fails.

        Args:
            full_image: PIL RGB -- the full-resolution before image
            obj_mask_fullimg: bool ndarray (H_full, W_full) -- object mask
            prompt: str -- inpainting prompt
            bbox_fullimg: (x1, y1, x2, y2) tight bbox around the object
            pad_ratio: fraction of bbox size to pad on each side for context
            feather_margin: pixels over which crop edges fade (alpha fallback)
            seed: optional int for reproducibility

        Returns:
            dict with keys:
                inpainted_crop: PIL RGB -- blended result at crop resolution
                paste_box: (x1, y1, x2, y2) in full-image coords
                mask_crop: bool ndarray (crop_h, crop_w) of the object mask
        """
        fw, fh = full_image.size
        bx1, by1, bx2, by2 = bbox_fullimg
        bw, bh = bx2 - bx1, by2 - by1

        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        cx1 = max(bx1 - pad_x, 0)
        cy1 = max(by1 - pad_y, 0)
        cx2 = min(bx2 + pad_x, fw)
        cy2 = min(by2 + pad_y, fh)
        crop_w, crop_h = cx2 - cx1, cy2 - cy1

        crop_img = full_image.crop((cx1, cy1, cx2, cy2))
        mask_crop = obj_mask_fullimg[cy1:cy2, cx1:cx2]

        needs_scale = max(crop_w, crop_h) > 512
        if needs_scale:
            scale = 512 / max(crop_w, crop_h)
            sd2_w = int(crop_w * scale)
            sd2_h = int(crop_h * scale)
            sd2_img = crop_img.resize((sd2_w, sd2_h), Image.LANCZOS)
            sd2_mask_pil = Image.fromarray(
                mask_crop.astype(np.uint8) * 255
            ).resize((sd2_w, sd2_h), Image.NEAREST)
            sd2_mask = np.array(sd2_mask_pil) > 127
        else:
            sd2_img = crop_img
            sd2_mask = mask_crop

        raw_sd2 = self.inpaint_raw(
            image=sd2_img, mask=sd2_mask,
            prompt=prompt, seed=seed,
        )

        if needs_scale:
            raw_upscaled = raw_sd2.resize((crop_w, crop_h), Image.LANCZOS)
        else:
            raw_upscaled = raw_sd2

        crop_scale = max(1.0, max(crop_w, crop_h) / 512.0)
        scaled_dilate = int(self.mask_dilate_px * crop_scale)

        if self.blend_mode == "poisson":
            blended_crop = self._poisson_blend(
                crop_img, raw_upscaled, mask_crop,
                dilate_px=scaled_dilate)
        else:
            scaled_blur = int(self.mask_blur_radius * crop_scale)
            obj_feathered = self._feather_mask(
                mask_crop, dilate_px=scaled_dilate, blur_radius=scaled_blur)
            blended_crop = self._alpha_blend(
                crop_img, raw_upscaled, obj_feathered)

        return {
            "inpainted_crop": blended_crop,
            "paste_box": (cx1, cy1, cx2, cy2),
            "mask_crop": mask_crop,
        }

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def crop_edge_feather(crop_w, crop_h, margin=48):
    """Create a soft alpha ramp that fades to 0 at crop edges.

    Returns an (H, W) float32 array in [0, 1].
    Center pixels = 1.0, edges ramp linearly to 0 over *margin* pixels.
    """
    if margin <= 0:
        return np.ones((crop_h, crop_w), dtype=np.float32)

    ramp_y = np.ones(crop_h, dtype=np.float32)
    ramp_x = np.ones(crop_w, dtype=np.float32)

    m = min(margin, crop_h // 2, crop_w // 2)
    for i in range(m):
        v = i / m
        ramp_y[i] = min(ramp_y[i], v)
        ramp_y[-(i + 1)] = min(ramp_y[-(i + 1)], v)
        ramp_x[i] = min(ramp_x[i], v)
        ramp_x[-(i + 1)] = min(ramp_x[-(i + 1)], v)

    return ramp_y[:, np.newaxis] * ramp_x[np.newaxis, :]
