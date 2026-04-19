"""Inpainting wrapper with pluggable diffusers backends.

Supports three switchable backends behind one public API:
    - "sd2"             : Stable Diffusion 2 inpainting  (512 native)
    - "sd15_realistic"  : SD1.5 photorealistic fine-tune (512 native)
    - "sdxl"            : SDXL inpainting                (1024 native)

All backends share the same public methods (.inpaint, .inpaint_object,
.inpaint_raw, .cleanup) so callers don't need to branch. SDXL uses CPU
offload + VAE slicing to fit in ~8 GB VRAM.
"""

import sys
import numpy as np
from PIL import Image, ImageFilter

try:
    import torch
    import diffusers  # noqa: F401 -- presence check
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

import cv2
from scipy.ndimage import binary_dilation


_BACKENDS = {
    "sd2": {
        "pipeline_cls": "StableDiffusionInpaintPipeline",
        "default_repo": "sd2-community/stable-diffusion-2-inpainting",
        "fallback_repos": ["stabilityai/stable-diffusion-2-inpainting"],
        "max_resolution": 512,
        "default_steps": 50,
        "default_guidance": 12.0,
        "default_strength": 1.0,
        "needs_cpu_offload": False,
    },
    "sd15_realistic": {
        "pipeline_cls": "StableDiffusionInpaintPipeline",
        "default_repo": "Lykon/dreamshaper-8-inpainting",
        "fallback_repos": [],
        "max_resolution": 512,
        "default_steps": 40,
        "default_guidance": 7.5,
        "default_strength": 1.0,
        "needs_cpu_offload": False,
    },
    "sdxl": {
        "pipeline_cls": "StableDiffusionXLInpaintPipeline",
        "default_repo": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "fallback_repos": [],
        "max_resolution": 1024,
        "default_steps": 30,
        "default_guidance": 6.0,
        "default_strength": 1.0,
        "needs_cpu_offload": True,
    },
}


_DEFAULT_NEG = (
    "blurry, low quality, cartoon, painting, watermark, "
    "blue tint, color fringe, edge artifact, seam, border artifact"
)


def _resolve_pipeline_cls(name):
    """Lazily import a diffusers pipeline class by name."""
    import diffusers
    if not hasattr(diffusers, name):
        raise RuntimeError(
            f"diffusers.{name} not found. Upgrade diffusers: "
            f"pip install -U diffusers"
        )
    return getattr(diffusers, name)


class InpaintingModel:
    """Wraps a diffusers inpainting pipeline with feathered / Poisson blending.

    Args:
        backend: one of "sd2", "sd15_realistic", "sdxl". Selects which
            diffusers pipeline and default resolution to use.
        model_id: optional repo id override. If None, the backend's
            default_repo is used (with fallbacks for sd2).
        max_resolution: optional override for the backend's max_resolution.
            This governs the crop downsample in ``inpaint_object``.
        device: "cuda" or "cpu". SDXL will ignore this and use CPU offload
            if needs_cpu_offload is True in the registry.
        num_inference_steps / guidance_scale / strength: pipeline params.
            If None, the registry default for the backend is used.
        mask_blur_radius / mask_dilate_px / blend_mode: blending knobs.
    """

    def __init__(self, backend="sd2", model_id=None,
                 max_resolution=None, device="cuda",
                 num_inference_steps=None, guidance_scale=None,
                 strength=None,
                 mask_blur_radius=12, mask_dilate_px=8,
                 object_dilate_ratio=0.08, object_dilate_max_px=48,
                 blend_mode="poisson"):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Install diffusers: pip install diffusers accelerate")

        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown inpainting backend '{backend}'. "
                f"Valid options: {list(_BACKENDS.keys())}"
            )
        spec = _BACKENDS[backend]

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.backend = backend
        self.device = device
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None
            else spec["default_steps"]
        )
        self.guidance_scale = (
            guidance_scale if guidance_scale is not None
            else spec["default_guidance"]
        )
        self.strength = (
            strength if strength is not None else spec["default_strength"]
        )
        self.max_resolution = (
            max_resolution if max_resolution is not None
            else spec["max_resolution"]
        )
        self.mask_blur_radius = mask_blur_radius
        self.mask_dilate_px = mask_dilate_px
        self.object_dilate_ratio = object_dilate_ratio
        self.object_dilate_max_px = object_dilate_max_px
        self.blend_mode = blend_mode

        pipe_cls = _resolve_pipeline_cls(spec["pipeline_cls"])
        repos = [model_id] if model_id else (
            [spec["default_repo"]] + list(spec.get("fallback_repos", []))
        )
        pipe = None
        last_err = None
        for repo in repos:
            try:
                pipe = pipe_cls.from_pretrained(
                    repo, torch_dtype=torch.float16, use_safetensors=True,
                )
                print(f"  Loaded {backend} pipeline from '{repo}'")
                break
            except Exception as e:
                last_err = e
                print(f"  Could not load {repo}: {e}", file=sys.stderr)
                continue

        if pipe is None:
            raise RuntimeError(
                f"Failed to load any inpainting model for backend '{backend}' "
                f"from {repos}. Last error: {last_err}"
            )

        if spec["needs_cpu_offload"] and device == "cuda":
            pipe.enable_model_cpu_offload()
            if hasattr(pipe, "vae"):
                if hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
        else:
            pipe = pipe.to(device)

        # SD1/2 ship a CLIP-based NSFW safety checker that frequently
        # misfires on aerial/natural imagery and replaces outputs with black
        # frames. For synthetic-data generation we want every inpaint to
        # return real pixels, so disable it. SDXL inpaint has no such
        # attribute and is unaffected.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False

        self.pipe = pipe
        self.pipe.set_progress_bar_config(disable=True)

    def _object_dilate_iterations(self, mask_bool):
        """Compute bbox-aware binary_dilation iterations for an object mask.

        Tight SAM 3 masks cover only the object silhouette but objects in
        natural scenes have *shadows and reflections* that extend beyond
        that silhouette. Using a fixed ``mask_dilate_px`` leaves those halo
        pixels untouched during inpainting, producing a visible colored
        outline of the removed object. Scaling the dilation with the
        object's bounding box eliminates that artifact.

        Returns ``max(mask_dilate_px, ratio * min(bbox_w, bbox_h))`` capped
        at ``object_dilate_max_px``. Falls back to the fixed
        ``mask_dilate_px`` if ratio is 0 or the mask is empty.
        """
        if self.object_dilate_ratio <= 0:
            return self.mask_dilate_px
        ys, xs = np.where(mask_bool)
        if len(ys) == 0:
            return self.mask_dilate_px
        bw = int(xs.max() - xs.min()) + 1
        bh = int(ys.max() - ys.min()) + 1
        size = min(bw, bh)
        ratio_px = int(self.object_dilate_ratio * size)
        return max(self.mask_dilate_px,
                   min(self.object_dilate_max_px, ratio_px))

    def _feather_mask(self, mask_bool, dilate_px=None, blur_radius=None):
        """Dilate (bbox-aware by default) then Gaussian-blur a binary mask
        to produce soft edges."""
        if dilate_px is None:
            dilate = self._object_dilate_iterations(mask_bool)
        else:
            dilate = dilate_px
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
        alpha = alpha[..., np.newaxis]
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

        src = np.array(inpainted_pil)[:, :, ::-1]
        dst = np.array(original_pil)[:, :, ::-1]

        ys, xs = np.where(dilated)
        if len(ys) == 0:
            return original_pil
        center = (int(xs.mean()), int(ys.mean()))

        try:
            result_bgr = cv2.seamlessClone(
                src, dst, clone_mask, center, cv2.NORMAL_CLONE)
            return Image.fromarray(result_bgr[:, :, ::-1])
        except cv2.error:
            feathered = Image.fromarray(clone_mask).filter(
                ImageFilter.GaussianBlur(radius=12))
            return InpaintingModel._alpha_blend(
                original_pil, inpainted_pil, feathered)

    @torch.inference_mode()
    def _run_pipe(self, image_rgb, mask_bool, prompt, negative_prompt, seed):
        """Core diffusers pipeline call. Works for SD1.5, SD2, and SDXL
        inpainting since they share the same kwargs. Returns the raw
        pipeline output resized back to the original image size."""
        feathered = self._feather_mask(mask_bool)

        generator = None
        if seed is not None:
            gen_device = "cpu" if _BACKENDS[self.backend]["needs_cpu_offload"] else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

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

        Convenience wrapper that runs the selected backend and alpha-blends
        the result. Used by process_one.py and batch_generate. For
        object-centric inpainting use inpaint_raw() + manual blend instead.

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
        raw_result = self._run_pipe(image_rgb, mask_bool, prompt,
                                    negative_prompt, seed)

        feathered = self._feather_mask(mask_bool)
        if feathered.size != image_rgb.size:
            feathered = feathered.resize(image_rgb.size, Image.LANCZOS)

        return self._alpha_blend(image_rgb, raw_result, feathered)

    def inpaint_raw(self, image, mask, prompt,
                    negative_prompt=None, seed=None):
        """Run inpainting and return the raw output without alpha-blending.

        The caller is responsible for blending. This avoids the double-blend
        problem when inpaint_object() needs to do its own crop-level blend.

        Returns:
            PIL RGB image -- raw pipeline output (fully regenerated in masked
            area; original pixels outside the mask region may differ due to
            global denoising)
        """
        if negative_prompt is None:
            negative_prompt = _DEFAULT_NEG

        if isinstance(mask, np.ndarray):
            mask_bool = mask.astype(bool)
        else:
            mask_bool = np.array(mask.convert("L")) > 127

        image_rgb = image.convert("RGB")
        return self._run_pipe(image_rgb, mask_bool, prompt,
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

        max_res = self.max_resolution
        needs_scale = max(crop_w, crop_h) > max_res
        if needs_scale:
            scale = max_res / max(crop_w, crop_h)
            pipe_w = int(crop_w * scale)
            pipe_h = int(crop_h * scale)
            pipe_img = crop_img.resize((pipe_w, pipe_h), Image.LANCZOS)
            pipe_mask_pil = Image.fromarray(
                mask_crop.astype(np.uint8) * 255
            ).resize((pipe_w, pipe_h), Image.NEAREST)
            pipe_mask = np.array(pipe_mask_pil) > 127
        else:
            pipe_img = crop_img
            pipe_mask = mask_crop

        raw = self.inpaint_raw(
            image=pipe_img, mask=pipe_mask,
            prompt=prompt, seed=seed,
        )

        if needs_scale:
            raw_upscaled = raw.resize((crop_w, crop_h), Image.LANCZOS)
        else:
            raw_upscaled = raw

        # Use bbox-aware dilation at crop resolution so the Poisson clone
        # mask matches SD's effective feathered fill area (which also uses
        # _object_dilate_iterations). Matching dilations eliminates the
        # "SD filled more than Poisson blends back" gap that causes halos.
        obj_dilate = self._object_dilate_iterations(mask_crop)

        if self.blend_mode == "poisson":
            blended_crop = self._poisson_blend(
                crop_img, raw_upscaled, mask_crop,
                dilate_px=obj_dilate)
        else:
            crop_scale = max(1.0, max(crop_w, crop_h) / float(max_res))
            scaled_blur = int(self.mask_blur_radius * crop_scale)
            obj_feathered = self._feather_mask(
                mask_crop, dilate_px=obj_dilate, blur_radius=scaled_blur)
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


def build_inpainter_from_cfg(inpaint_cfg, backend_override=None):
    """Instantiate ``InpaintingModel`` from a config dict.

    Supports both the new per-backend schema:

        inpainting:
          backend: "sd2"
          sd2: {model_id, num_inference_steps, guidance_scale, strength}
          sd15_realistic: {...}
          sdxl: {...}
          device, mask_blur_radius, mask_dilate_px, blend_mode

    and the legacy flat schema (model_id/num_inference_steps/... at the top
    level), so existing configs keep working.

    Args:
        inpaint_cfg: the ``cfg.inpainting`` dict.
        backend_override: if given, selects this backend regardless of the
            ``backend`` field in the config (used by the A/B comparison
            script).
    """
    cfg = inpaint_cfg or {}
    shared = {
        "device": cfg.get("device", "cuda"),
        "mask_blur_radius": cfg.get("mask_blur_radius", 12),
        "mask_dilate_px": cfg.get("mask_dilate_px", 8),
        "object_dilate_ratio": cfg.get("object_dilate_ratio", 0.08),
        "object_dilate_max_px": cfg.get("object_dilate_max_px", 48),
        "blend_mode": cfg.get("blend_mode", "poisson"),
    }

    backend = backend_override or cfg.get("backend")
    is_nested = backend is not None and isinstance(cfg.get(backend), dict)

    if is_nested:
        sub = cfg[backend]
        return InpaintingModel(
            backend=backend,
            model_id=sub.get("model_id"),
            max_resolution=sub.get("max_resolution"),
            num_inference_steps=sub.get("num_inference_steps"),
            guidance_scale=sub.get("guidance_scale"),
            strength=sub.get("strength"),
            **shared,
        )

    # Legacy flat schema: all params at the top level; default to sd2.
    effective_backend = backend or "sd2"
    return InpaintingModel(
        backend=effective_backend,
        model_id=cfg.get("model_id"),
        max_resolution=cfg.get("max_resolution"),
        num_inference_steps=cfg.get("num_inference_steps"),
        guidance_scale=cfg.get("guidance_scale"),
        strength=cfg.get("strength"),
        **shared,
    )


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
