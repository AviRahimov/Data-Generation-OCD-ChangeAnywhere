"""Full-image synthetic pipeline: SAM scanning, object selection, compositing.

Used by ``generate_pair.py`` and ``generate_dataset.py`` (object-centric path).
"""

import random
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from .change_simulator import _find_background_region
from .prompt_templates import (
    TERRAIN_BACKGROUND_CLASSES,
    get_appearance_prompt,
    get_disappearance_prompt,
    get_background_label,
    sample_object_type,
)


def _detection_visibility(det):
    import math
    return det["score"] * math.sqrt(max(det["area_ratio"], 1e-6) / 0.01)


def _bbox_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _mask_bbox(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _mask_compactness(mask_bool):
    area = int(mask_bool.sum())
    if area == 0:
        return 0.0
    eroded = ndimage.binary_erosion(mask_bool)
    perim = int(mask_bool.sum() - eroded.sum())
    if perim <= 0:
        return 0.0
    import math
    return float(min(1.0, 4.0 * math.pi * area / (perim * perim)))


def _mask_contrast(mask_bool, gray_arr, ring_px=4):
    if mask_bool.sum() == 0:
        return 0.0
    dilated = ndimage.binary_dilation(mask_bool, iterations=ring_px)
    ring = np.logical_and(dilated, np.logical_not(mask_bool))
    if ring.sum() == 0 or mask_bool.sum() == 0:
        return 0.0
    inside_mean = float(gray_arr[mask_bool].mean())
    ring_mean = float(gray_arr[ring].mean())
    return float(abs(inside_mean - ring_mean))


def _dominant_class(mask_bool, seg_map):
    if mask_bool.shape != seg_map.shape:
        return -1
    pixels = seg_map[mask_bool]
    if pixels.size == 0:
        return -1
    counts = np.bincount(pixels.astype(np.int64))
    return int(counts.argmax())


def _post_filter_auto(dets, crop_pil, seg_model=None, *,
                      ignore_terrain=True, min_compactness=0.25,
                      min_contrast=8.0):
    if not dets:
        return dets

    gray = np.asarray(crop_pil.convert("L"), dtype=np.float32)
    seg_map = None

    kept = []
    for d in dets:
        m = d["mask"]
        if m.shape != gray.shape:
            continue

        compact = _mask_compactness(m)
        if compact < min_compactness:
            continue

        contrast = _mask_contrast(m, gray)
        if contrast < min_contrast:
            continue

        if ignore_terrain and seg_model is not None:
            if seg_map is None:
                try:
                    seg_map = seg_model.segment(crop_pil)
                except Exception as e:
                    import sys as _sys
                    print(f"  SegFormer terrain-check failed: {e}",
                          file=_sys.stderr)
                    seg_map = np.zeros_like(gray, dtype=np.int32)
            cls = _dominant_class(m, seg_map)
            if cls in TERRAIN_BACKGROUND_CLASSES:
                continue

        out = dict(d)
        out["compactness"] = compact
        out["contrast"] = contrast
        kept.append(out)

    return kept


def select_best_objects(full_image, sam_model, max_objects=3,
                        scan_tile_size=1024, scan_overlap=128,
                        detection_prompts=None, detection_score=0.30,
                        variance_threshold=500,
                        min_object_distance=2000,
                        max_detections_per_crop=5,
                        max_per_label=0,
                        detection_mode="text",
                        seg_model=None,
                        auto_cfg=None):
    import math
    from .tiler import tile_image

    if auto_cfg is None:
        auto_cfg = {}
    auto_mode = (detection_mode == "auto")

    if auto_mode and max_per_label > 0:
        max_per_label = 0

    fw, fh = full_image.size

    all_dets = []
    crop_idx = 0
    for cx, cy, crop_pil in tile_image(full_image, tile_size=scan_tile_size,
                                        overlap=scan_overlap):
        crop_idx += 1
        arr = np.array(crop_pil.convert("L"), dtype=np.float32)
        if arr.var() < variance_threshold:
            continue

        try:
            if auto_mode:
                dets = sam_model.detect_objects_auto(
                    crop_pil,
                    min_score=detection_score,
                    min_area_ratio=float(auto_cfg.get("min_area_ratio", 0.0005)),
                    max_area_ratio=float(auto_cfg.get("max_area_ratio", 0.15)),
                    points_per_side=int(auto_cfg.get("points_per_side", 16)),
                )
                dets = _post_filter_auto(
                    dets, crop_pil, seg_model=seg_model,
                    ignore_terrain=bool(auto_cfg.get("ignore_terrain", True)),
                    min_compactness=float(auto_cfg.get("min_compactness", 0.25)),
                    min_contrast=float(auto_cfg.get("min_contrast", 8.0)),
                )
            else:
                dets = sam_model.detect_objects(
                    crop_pil, prompts=detection_prompts,
                    min_score=detection_score,
                )
        except Exception as e:
            print(f"  SAM3 scan error on crop ({cx},{cy}): {e}")
            continue

        if not dets:
            continue

        cw, ch = crop_pil.size
        for d in dets:
            local_mask = d["mask"]
            if local_mask.shape != (ch, cw):
                continue
            lbbox = _mask_bbox(local_mask)
            fbbox = (lbbox[0] + cx, lbbox[1] + cy,
                     lbbox[2] + cx, lbbox[3] + cy)
            lys, lxs = np.where(local_mask)
            fcx = int(lxs.mean()) + cx
            fcy = int(lys.mean()) + cy

            all_dets.append({
                "mask_local": local_mask,
                "crop_offset": (cx, cy),
                "crop_size": (cw, ch),
                "bbox_fullimg": fbbox,
                "centroid_fullimg": (fcx, fcy),
                "label": d["label"],
                "score": d["score"],
                "area_ratio": d["area_ratio"],
                "visibility": _detection_visibility(d),
            })

    if not all_dets:
        return []

    print(f"  SAM3 found {len(all_dets)} raw detections across {crop_idx} crops")

    all_dets.sort(key=lambda d: d["visibility"], reverse=True)

    kept = []
    for det in all_dets:
        duplicate = False
        for k in kept:
            if _bbox_iou(det["bbox_fullimg"], k["bbox_fullimg"]) > 0.3:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
            if len(kept) >= max_detections_per_crop * 10:
                break

    print(f"  After dedup: {len(kept)} unique detections")

    selected = []
    label_counts = {}
    for det in kept:
        lbl = det["label"]
        if max_per_label > 0 and label_counts.get(lbl, 0) >= max_per_label:
            continue

        dcx, dcy = det["centroid_fullimg"]
        too_close = any(
            math.hypot(dcx - sx, dcy - sy) < min_object_distance
            for sx, sy in [(s["centroid_fullimg"]) for s in selected]
        )
        if too_close:
            continue

        ox, oy = det["crop_offset"]
        cw, ch = det["crop_size"]
        full_mask = np.zeros((fh, fw), dtype=bool)
        local = det["mask_local"]
        y1 = oy
        y2 = min(oy + ch, fh)
        x1 = ox
        x2 = min(ox + cw, fw)
        lh = y2 - y1
        lw = x2 - x1
        full_mask[y1:y2, x1:x2] = local[:lh, :lw]

        selected.append({
            "mask_fullimg": full_mask,
            "bbox_fullimg": det["bbox_fullimg"],
            "centroid_fullimg": det["centroid_fullimg"],
            "label": det["label"],
            "score": det["score"],
            "visibility": det["visibility"],
        })
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        if len(selected) >= max_objects:
            break

    return selected


def select_appearance_locations(full_image, seg_model, max_count,
                                scan_tile_size=1024,
                                min_object_distance=2000,
                                min_radius=60, max_radius=180,
                                min_bg_fraction=0.25,
                                existing_centroids=None,
                                max_attempts_per_slot=25,
                                rng=None):
    import math
    from .change_simulator import _find_background_region, _random_blob_mask

    rng = rng or random.Random()
    fw, fh = full_image.size
    results = []
    existing = list(existing_centroids or [])

    scan_w = min(scan_tile_size, fw)
    scan_h = min(scan_tile_size, fh)

    for _slot in range(max_count):
        placed = False
        for _attempt in range(max_attempts_per_slot):
            cx0 = rng.randint(0, max(0, fw - scan_w))
            cy0 = rng.randint(0, max(0, fh - scan_h))
            crop = full_image.crop((cx0, cy0, cx0 + scan_w, cy0 + scan_h))

            try:
                seg_map = seg_model.segment(crop)
            except Exception as e:
                print(f"  SegFormer error during appearance scan: {e}")
                continue

            bg_regions = _find_background_region(seg_map,
                                                  min_area_ratio=min_bg_fraction)
            if not bg_regions:
                continue

            bg_class, _ = bg_regions[0]
            bg_mask = seg_map == bg_class
            ys, xs = np.where(bg_mask)
            if len(ys) < 200:
                continue

            idx = rng.randint(0, len(ys) - 1)
            ly, lx = int(ys[idx]), int(xs[idx])
            radius = rng.randint(min_radius, max_radius)
            if (ly - radius < 0 or ly + radius >= scan_h
                    or lx - radius < 0 or lx + radius >= scan_w):
                continue

            blob_local = _random_blob_mask(scan_h, scan_w, lx, ly, radius, rng)
            blob_in_bg = np.logical_and(blob_local, bg_mask)
            if blob_in_bg.sum() < blob_local.sum() * 0.7:
                continue

            fcx = lx + cx0
            fcy = ly + cy0

            too_close = any(
                math.hypot(fcx - ex, fcy - ey) < min_object_distance
                for ex, ey in existing
            )
            if too_close:
                continue

            full_mask = np.zeros((fh, fw), dtype=bool)
            full_mask[cy0:cy0 + scan_h, cx0:cx0 + scan_w] = blob_in_bg

            mys, mxs = np.where(full_mask)
            fbbox = (int(mxs.min()), int(mys.min()),
                     int(mxs.max()) + 1, int(mys.max()) + 1)

            obj_type = sample_object_type(rng, bg_class_id=bg_class)

            results.append({
                "kind": "appearance",
                "label": obj_type,
                "score": 1.0,
                "visibility": 1.0,
                "mask_fullimg": full_mask,
                "bbox_fullimg": fbbox,
                "centroid_fullimg": (fcx, fcy),
                "surround_class": int(bg_class),
            })
            existing.append((fcx, fcy))
            placed = True
            break

        if not placed:
            continue

    return results


def _paste_inpainted_crop(canvas, inpaint_result, edge_margin=8):
    crop_pil = inpaint_result["inpainted_crop"]
    x1, y1, x2, y2 = inpaint_result["paste_box"]
    crop_w, crop_h = x2 - x1, y2 - y1

    crop = np.array(crop_pil).astype(np.float32)
    region = np.array(canvas.crop((x1, y1, x2, y2))).astype(np.float32)

    m = min(edge_margin, crop_h // 2, crop_w // 2)
    alpha = np.ones((crop_h, crop_w), dtype=np.float32)
    for i in range(m):
        v = (i + 1) / (m + 1)
        alpha[i, :] = np.minimum(alpha[i, :], v)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], v)
        alpha[:, i] = np.minimum(alpha[:, i], v)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], v)

    alpha_3 = alpha[..., np.newaxis]
    blended = region * (1.0 - alpha_3) + crop * alpha_3
    canvas.paste(Image.fromarray(blended.astype(np.uint8)), (x1, y1))


def compute_local_ssim_change_mask(before_pil, after_pil, change_mask_arr, pad=64):
    """Local SSIM on a crop around the union of changed pixels (full-resolution)."""
    from .tile_synthetic import compute_local_ssim_tile
    m = np.asarray(change_mask_arr)
    if m.dtype != bool:
        m = m > 127
    return compute_local_ssim_tile(before_pil, after_pil, m, pad=pad)


def generate_full_image_pair(before_full, changes, inpaint_model,
                             seg_model=None, seed=42,
                             verbose=False, collect_crops=False,
                             gt_diff_thresh=15, gt_dilate_iterations=6):
    """Apply a uniform list of change dicts to a full-resolution image.

    Ground-truth ``change_mask`` merges SAM prompt masks with pixels that
    actually differ from ``before`` inside each paste box (bounded by a
    dilated prompt mask) so Poisson/SD edits outside the tight SAM silhouette
    are still labeled.
    """
    fw, fh = before_full.size
    canvas = before_full.copy()
    change_mask_full = np.zeros((fh, fw), dtype=bool)
    object_crops = []
    meta_entries = []

    before_arr = np.asarray(before_full, dtype=np.int16)

    for i, obj in enumerate(changes):
        kind = obj.get("kind", "disappearance")
        label = obj["label"]
        cx, cy = obj["centroid_fullimg"]
        bbox = obj["bbox_fullimg"]
        bx1, by1, bx2, by2 = bbox
        bw, bh = bx2 - bx1, by2 - by1
        mask_full = obj["mask_fullimg"]

        rng = random.Random(seed + i)

        surround_class = obj.get("surround_class")
        if surround_class is None:
            if seg_model is not None:
                sx1 = max(bx1 - 64, 0)
                sy1 = max(by1 - 64, 0)
                sx2 = min(bx2 + 64, fw)
                sy2 = min(by2 + 64, fh)
                seg_tile = before_full.crop((sx1, sy1, sx2, sy2))
                try:
                    seg_map = seg_model.segment(seg_tile)
                    bg_regions = _find_background_region(seg_map)
                    surround_class = bg_regions[0][0] if bg_regions else 13
                except Exception as e:
                    print(f"  SegFormer error looking up background: {e}")
                    surround_class = 13
            else:
                surround_class = 13
        surround_class = int(surround_class)

        if kind == "appearance":
            prompt = get_appearance_prompt(label, surround_class, rng)
        else:
            prompt = get_disappearance_prompt(surround_class, rng)

        if verbose:
            print(f"  [{i+1}/{len(changes)}] {kind} '{label}' at ({cx},{cy}), "
                  f"size {bw}x{bh}, bg={get_background_label(surround_class)}")

        before_crop_box = None
        before_crop = None
        if collect_crops:
            before_crop_box = (
                max(bx1 - int(bw * 0.3), 0),
                max(by1 - int(bh * 0.3), 0),
                min(bx2 + int(bw * 0.3), fw),
                min(by2 + int(bh * 0.3), fh),
            )
            before_crop = before_full.crop(before_crop_box)

        _t0 = time.time()
        try:
            result = inpaint_model.inpaint_object(
                full_image=before_full,
                obj_mask_fullimg=mask_full,
                prompt=prompt,
                bbox_fullimg=bbox,
                pad_ratio=0.3,
                feather_margin=48,
                seed=seed + i,
            )
        except Exception as e:
            print(f"  Inpainting failed for {kind} '{label}': {e}")
            continue
        if verbose:
            print(f"    inpaint done in {time.time() - _t0:.1f}s")

        _paste_inpainted_crop(canvas, result)

        px1, py1, px2, py2 = result["paste_box"]
        local_mask = result["mask_crop"]
        lh, lw = local_mask.shape

        slice_before = before_arr[py1:py2, px1:px2, :]
        slice_after = np.asarray(canvas, dtype=np.int16)[py1:py2, px1:px2, :]
        diff = np.abs(slice_before - slice_after).max(axis=2)

        prompt_dilated = ndimage.binary_dilation(
            local_mask[:lh, :lw], iterations=gt_dilate_iterations)
        local_gt = (diff > gt_diff_thresh) & prompt_dilated
        if local_gt.any():
            change_mask_full[py1:py2, px1:px2] |= local_gt
        else:
            change_mask_full[py1:py2, px1:px2] |= local_mask[:lh, :lw]

        if collect_crops and before_crop_box is not None:
            after_crop = canvas.crop(before_crop_box)
            object_crops.append({
                "before_crop": before_crop,
                "after_crop": after_crop,
                "label": label,
                "score": float(obj.get("score", 1.0)),
            })

        meta_entries.append({
            "kind": kind,
            "label": label,
            "score": round(float(obj.get("score", 1.0)), 3),
            "visibility": round(float(obj.get("visibility", 1.0)), 3),
            "centroid": [int(cx), int(cy)],
            "bbox": [int(v) for v in bbox],
            "paste_box": [int(v) for v in result["paste_box"]],
            "prompt": prompt,
            "bg_class": surround_class,
        })

    change_mask_pil = Image.fromarray(change_mask_full.astype(np.uint8) * 255)
    return {
        "after": canvas,
        "change_mask": change_mask_pil,
        "meta_entries": meta_entries,
        "object_crops": object_crops if collect_crops else None,
    }
