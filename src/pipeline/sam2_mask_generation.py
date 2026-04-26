"""SAM 2.1 automatic (prompt-free) masks via Hugging Face ``mask-generation``.

This path is for **comparison** with SAM 3 text/auto (e.g. ``eval_detection_modes.py``),
not for the main ChangeAnywhere inpainting flow. It uses Meta's segment-everything style
inference, which is a different design than SAM 3 box-prompted auto mode.

Config: ``segmentation.sam2`` in ``config.yaml`` (``enabled``, ``checkpoint``, etc.).

Requires a recent ``transformers`` with SAM2 and enough VRAM; otherwise loading fails
gracefully and the eval script can skip the third column.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_OCD_SAM2_NMS_PATCHED = False


def apply_sam2_huggingface_nms_postprocess_patch() -> None:
    """Work around a Hugging Face + torchvision issue in AMG NMS for SAM2.

    ``MaskGenerationPipeline`` ``torch.cat``'s per-minibatch ``iou_scores`` which can
    keep shape (N, 1). ``torchvision.ops.batched_nms`` requires **1D** scores; 2D
    tensors can cause opaque errors (e.g. about ``dets``/dtype). We also build
    ``idxs`` on the same **device** as ``mask_boxes`` for robustness on CUDA.
    """
    global _OCD_SAM2_NMS_PATCHED
    if _OCD_SAM2_NMS_PATCHED:
        return
    try:
        import torch
        from torchvision.ops import batched_nms
        from transformers.models.sam2 import image_processing_sam2 as s2
    except Exception:
        return

    if getattr(s2, "_ocd_patched_nms", False):
        _OCD_SAM2_NMS_PATCHED = True
        return

    _rle_to_mask = s2._rle_to_mask

    def _patched(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
        if iou_scores is not None and iou_scores.dim() > 1:
            iou_scores = iou_scores.reshape(-1)
        dev = mask_boxes.device
        idxs = torch.zeros(
            mask_boxes.shape[0], device=dev, dtype=torch.int64
        )
        keep_by_nms = batched_nms(
            boxes=mask_boxes.float(),
            scores=iou_scores.float().to(dev),
            idxs=idxs,
            iou_threshold=amg_crops_nms_thresh,
        )
        iou_scores = iou_scores[keep_by_nms]
        km = [int(x) for x in keep_by_nms.tolist()]
        rle_masks = [rle_masks[i] for i in km]
        mask_boxes = mask_boxes[keep_by_nms]
        masks = [_rle_to_mask(rle) for rle in rle_masks]
        return masks, iou_scores, rle_masks, mask_boxes

    s2._post_process_for_mask_generation = _patched
    s2._ocd_patched_nms = True
    _OCD_SAM2_NMS_PATCHED = True


def _mask_bbox(m: np.ndarray) -> Tuple[int, int, int, int]:
    """Tight (x1,y1,x2,y2) for a bool (H,W) mask; z-sized box if empty."""
    ys, xs = np.where(m)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _as_bool_mask(
        m, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Return (H,W) bool; resize if needed."""
    h, w = target_hw[1], target_hw[0]
    if m is None:
        return None
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[:, :, 0]
        else:
            m = m[0] if m.shape[0] <= 4 else m.mean(axis=0) > 0.5
    m = m.astype(np.float32)
    m_bool = m > 0.5
    if m_bool.shape[0] != h or m_bool.shape[1] != w:
        from PIL import Image as PILImage
        im = PILImage.fromarray((m_bool.astype(np.uint8) * 255))
        im = im.resize((w, h), PILImage.NEAREST)
        m_bool = np.array(im) > 127
    return m_bool


def build_sam2_mask_pipeline(
        sam2_cfg: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[str]]:
    """Load the ``mask-generation`` pipeline, or (None, error_message)."""
    try:
        import torch
        from transformers import pipeline
    except Exception as e:  # pragma: no cover
        return None, f"import: {e}"

    model_id = sam2_cfg.get("checkpoint", "facebook/sam2.1-hiera-base-plus")
    dev_str = (sam2_cfg.get("device") or "cuda").lower()
    if dev_str == "cuda" and torch.cuda.is_available():
        device = 0
    else:
        device = -1
    torch_dtype = None
    if device >= 0 and bool(sam2_cfg.get("use_bfloat16", False)):
        try:
            torch_dtype = torch.bfloat16
        except Exception:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    extra_kw_variants = (
        {"dtype": torch_dtype},
        {"torch_dtype": torch_dtype},
        {},
    )
    apply_sam2_huggingface_nms_postprocess_patch()
    last_err = None
    for xkw in extra_kw_variants:
        try:
            gen = pipeline(
                "mask-generation",
                model=model_id,
                device=device,
                **xkw,
            )
            return gen, None
        except Exception as e:
            last_err = e
    return None, str(last_err)


def sam2_detections_on_image(
        pil: Image.Image,
        pipe,
        sam2_cfg: Dict[str, Any],
) -> List[dict]:
    """Run automatic mask generation; return list like ``select_best_objects`` (full-image coords).

    Each dict: ``mask_fullimg``, ``bbox_fullimg``, ``centroid_fullimg``,
    ``label`` (``"sam2"``), ``score``, ``area_ratio`` — suitable for
    :func:`eval_comparison_viz.overlay_detections`.
    """
    apply_sam2_huggingface_nms_postprocess_patch()
    w, h = pil.size
    total_px = w * h
    ppb = int(sam2_cfg.get("points_per_batch", 32) or 32)
    pkw: Dict[str, Any] = dict(sam2_cfg.get("pipeline_kwargs") or {})
    pkw = {k: v for k, v in pkw.items() if v is not None}
    # Pipeline signature varies; keep common kwargs only if supported by caller
    call_kw = {"points_per_batch": ppb, **pkw}
    try:
        out = pipe(pil, **call_kw)
    except TypeError:
        try:
            out = pipe(pil, points_per_batch=ppb)
        except Exception as e:  # pragma: no cover
            print(f"  SAM2.1 mask-generation error: {e}", file=sys.stderr)
            return []
    except Exception as e:  # pragma: no cover
        print(f"  SAM2.1 mask-generation error: {e}", file=sys.stderr)
        return []

    masks = None
    scores = None
    if isinstance(out, dict):
        masks = out.get("masks")
        scores = out.get("scores")
    if masks is None:
        return []
    if hasattr(masks, "cpu"):
        t = masks.cpu()
        if t.dim() == 3:
            masks = [t[i] for i in range(t.shape[0])]
        else:
            masks = [t]
    n_m = len(masks)
    if scores is None:
        score_list = [1.0] * n_m
    elif hasattr(scores, "cpu"):
        score_list = scores.cpu().numpy().flatten().tolist()
    else:
        score_list = [float(x) for x in list(scores)]
    while len(score_list) < n_m:
        score_list.append(1.0)

    dets: List[dict] = []
    for i, m in enumerate(masks):
        sc = float(score_list[i]) if i < len(score_list) else 1.0
        mb = _as_bool_mask(m, (w, h))
        if mb is None or not mb.any():
            continue
        ar = float(mb.sum() / max(1, total_px))
        min_ar = float(sam2_cfg.get("min_area_ratio", 0.0) or 0.0)
        max_ar = float(sam2_cfg.get("max_area_ratio", 0.90) or 0.90)
        if not (min_ar <= ar <= max_ar):
            continue
        x1, y1, x2, y2 = _mask_bbox(mb)
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))
        dets.append({
            "mask_fullimg": mb,
            "bbox_fullimg": (x1, y1, x2, y2),
            "centroid_fullimg": (cx, cy),
            "label": "sam2",
            "score": sc,
            "area_ratio": ar,
        })

    dets.sort(key=lambda d: d["score"], reverse=True)
    cap = int(sam2_cfg.get("max_masks", 48) or 0)
    if cap > 0 and len(dets) > cap:
        dets = dets[:cap]
    return dets


def sam2_is_configured(sam2_cfg: Optional[Dict]) -> bool:
    return bool(sam2_cfg and sam2_cfg.get("enabled", False))


def promote_dets_to_full_space(
        dets: List[dict],
        small_w: int,
        small_h: int,
        full_w: int,
        full_h: int,
) -> List[dict]:
    """When SAM2.1 was run on a downscaled image, project masks and boxes to full resolution."""
    if not dets or (small_w == full_w and small_h == full_h):
        return dets
    sx = full_w / float(small_w)
    sy = full_h / float(small_h)
    out: List[dict] = []
    for d in dets:
        m = d.get("mask_fullimg")
        if m is not None:
            from PIL import Image as PILImage
            im = PILImage.fromarray((m.astype(np.uint8) * 255))
            im = im.resize((full_w, full_h), PILImage.NEAREST)
            m = np.array(im) > 127
        x1, y1, x2, y2 = d["bbox_fullimg"]
        out.append({
            **d,
            "mask_fullimg": m,
            "bbox_fullimg": (
                int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            ),
            "centroid_fullimg": (
                int((d["centroid_fullimg"][0]) * sx),
                int((d["centroid_fullimg"][1]) * sy),
            ),
        })
    return out
