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


def _auto_objectness(det, weight=None):
    """Rank auto-mode candidates: SAM score, color separation, dissimilarity to local ring, size, edges.

    If ``weight`` (dict) is None, use defaults for backward compatibility.
    """
    import math
    w = weight or {}
    s_w = float(w.get("score_weight", 1.5))
    lab_w = float(w.get("lab_weight", 0.035))
    hist_w = float(w.get("hist_weight", 4.0))
    comp_w = float(w.get("comp_weight", 0.35))
    ctr_w = float(w.get("contrast_weight", 0.018))
    edge_b = float(w.get("edge_bonus_scale", 0.75))
    area_w = float(w.get("area_log_weight", 0.7))
    ar_target = float(w.get("min_area_ratio_target", 0.0))
    ar_pen = float(w.get("tiny_area_penalty", 0.0))
    nt_w = float(w.get("non_terrain_weight", 2.0))
    s = float(det.get("score", 0.0)) * s_w
    ar = float(det.get("area_ratio", 0.0))
    lab = float(det.get("lab_ring_delta_e", 0.0)) * lab_w
    hbc = float(det.get("ring_hist_bc", 1.0))
    hterm = (1.0 - hbc) * hist_w
    comp = float(det.get("compactness", 0.0)) * comp_w
    c = float(det.get("contrast", 0.0)) * ctr_w
    egr = float(det.get("edge_interior_grad_ratio", 0.0))
    edg = min(1.5, max(0.0, egr - 1.0)) * edge_b
    area_part = area_w * math.log(max(ar, 1e-6) / 0.01)
    if ar_target > 0.0 and ar < ar_target:
        area_part = area_part - ar_pen
    ntf = float(det.get("non_terrain_fraction", 0.0))
    nt_bonus = ntf * nt_w
    return s + hterm + lab + comp + c + edg + area_part + nt_bonus


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


def _pairwise_mmr_similarity(a, b, min_object_distance):
    """[0,1] high if boxes overlap a lot or centroids are very close (vs min_object_distance)."""
    import math
    iou = _bbox_iou(a["bbox_fullimg"], b["bbox_fullimg"])
    dcx, dcy = a["centroid_fullimg"]
    bcx, bcy = b["centroid_fullimg"]
    dist = math.hypot(dcx - bcx, dcy - bcy)
    sig = max(float(min_object_distance), 1.0) * 0.45
    s_d = float(math.exp(-dist / sig))
    return float(max(iou, s_d))


def _nms_by_visibility(dets, iou_threshold=0.3):
    """Classical NMS: each step keeps the highest-visibility det; remove overlapping lower scores."""
    if not dets:
        return []
    th = float(iou_threshold)
    work = sorted(dets, key=lambda d: d["visibility"], reverse=True)
    out = []
    while work:
        cur = work.pop(0)
        out.append(cur)
        work = [d for d in work
                if _bbox_iou(d["bbox_fullimg"], cur["bbox_fullimg"]) <= th]
    return out


def _mmr_select_topk(dets, k, lam, min_object_distance):
    """Maximal marginal relevance: balance visibility vs dissimilarity to already chosen."""
    if not dets or k <= 0:
        return []
    if len(dets) <= k:
        return list(dets)
    lam = max(0.0, min(1.0, float(lam)))
    remaining = list(dets)
    # seed with single best visibility
    remaining.sort(key=lambda d: d["visibility"], reverse=True)
    first = remaining.pop(0)
    selected = [first]
    if len(selected) >= k:
        return selected
    while remaining and len(selected) < k:
        best = None
        best_mmr = -1e18
        for c in remaining:
            vis = float(c["visibility"])
            max_sim = 0.0
            for s in selected:
                max_sim = max(
                    max_sim,
                    _pairwise_mmr_similarity(c, s, min_object_distance)
                )
            mmr = lam * vis - (1.0 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best = c
        if best is None:
            break
        selected.append(best)
        # Do not use list.remove(best): det dicts hold numpy arrays; == is ambiguous.
        remaining = [c for c in remaining if c is not best]
    return selected


def _segformer_nonterrain_seed_boxes(seg_map, *, min_cc_pixels=400, margin_px=8,
                                    max_seeds=12):
    """Build xyxy query boxes in crop pixels for SegFormer *non-terrain* CCs (no text to SAM)."""
    h, w = int(seg_map.shape[0]), int(seg_map.shape[1])
    terr = np.isin(
        seg_map.astype(np.int64),
        np.array(list(TERRAIN_BACKGROUND_CLASSES), dtype=np.int64),
    )
    pos = np.logical_not(terr)
    lab, nlab = ndimage.label(pos)
    out = []
    for k in range(1, int(nlab) + 1):
        m = (lab == k)
        area = int(m.sum())
        if area < int(min_cc_pixels):
            continue
        ys, xs = np.where(m)
        if len(xs) < 1:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        mg = int(max(0, margin_px))
        x1 = max(0, x1 - mg)
        y1 = max(0, y1 - mg)
        x2 = min(w, x2 + mg)
        y2 = min(h, y2 + mg)
        if (x2 - x1) * (y2 - y1) < 16:
            continue
        out.append([float(x1), float(y1), float(x2), float(y2)])
    out.sort(
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    ms = int(max(0, max_seeds))
    return out[:ms] if ms else out


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


def _mask_ring_in_out(mask_bool, ring_px=4):
    """Binary mask of dilated\mask minus mask (halo), or (None, None) if invalid."""
    if int(mask_bool.sum()) == 0:
        return None, None
    dilated = ndimage.binary_dilation(mask_bool, iterations=ring_px)
    ring = np.logical_and(dilated, np.logical_not(mask_bool))
    if int(ring.sum()) < 3:
        return None, None
    return mask_bool, ring


def _mask_contrast(mask_bool, gray_arr, ring_px=4):
    pair = _mask_ring_in_out(mask_bool, ring_px=ring_px)
    if pair[0] is None:
        return 0.0
    m, ring = pair
    inside_mean = float(gray_arr[m].mean())
    ring_mean = float(gray_arr[ring].mean())
    return float(abs(inside_mean - ring_mean))


def _mask_lab_ring_metrics(mask_bool, lab_img, ring_px=4):
    """Return (delta_E_CIE76, abs_chroma_diff) in skimage L*a*b* space, or (0,0) if invalid."""
    pair = _mask_ring_in_out(mask_bool, ring_px=ring_px)
    if pair[0] is None or lab_img.ndim != 3 or lab_img.shape[2] != 3:
        return 0.0, 0.0
    m, ring = pair
    ins = lab_img[m]
    out = lab_img[ring]
    l_in, a_in, b_in = ins[:, 0].mean(), ins[:, 1].mean(), ins[:, 2].mean()
    l_ou, a_ou, b_ou = out[:, 0].mean(), out[:, 1].mean(), out[:, 2].mean()
    d = float(
        (l_in - l_ou) ** 2
        + (a_in - a_ou) ** 2
        + (b_in - b_ou) ** 2
    ) ** 0.5
    c_in = float((a_in * a_in + b_in * b_in) ** 0.5)
    c_ou = float((a_ou * a_ou + b_ou * b_ou) ** 0.5)
    return d, float(abs(c_in - c_ou))


def _mask_rgb_ring_histogram_bc(mask_bool, rgb_u8, ring_px=4, bin_per_ch=8):
    """Bhattacharyya coeff. [0,1] (1 = identical) between inside vs ring color dists."""
    pair = _mask_ring_in_out(mask_bool, ring_px=ring_px)
    if pair[0] is None:
        return 1.0
    m, ring = pair
    bins = int(bin_per_ch)
    flat = rgb_u8.reshape(-1, 3)
    mi = m.ravel()
    ri = ring.ravel()
    e = 256.0 / bins
    r_i = (flat[mi, 0] // e).clip(0, bins - 1).astype(np.int32)
    g_i = (flat[mi, 1] // e).clip(0, bins - 1).astype(np.int32)
    b_i = (flat[mi, 2] // e).clip(0, bins - 1).astype(np.int32)
    idx_in = (r_i * (bins * bins) + g_i * bins + b_i)
    r_o = (flat[ri, 0] // e).clip(0, bins - 1).astype(np.int32)
    g_o = (flat[ri, 1] // e).clip(0, bins - 1).astype(np.int32)
    b_o = (flat[ri, 2] // e).clip(0, bins - 1).astype(np.int32)
    idx_ou = (r_o * (bins * bins) + g_o * bins + b_o)
    nb = bins ** 3
    c1 = np.bincount(idx_in, minlength=nb).astype(np.float64)
    c2 = np.bincount(idx_ou, minlength=nb).astype(np.float64)
    s1, s2 = float(c1.sum()), float(c2.sum())
    if s1 < 1.0 or s2 < 1.0:
        return 1.0
    p = c1 / s1
    q = c2 / s2
    return float(np.sqrt(p * q).sum())


def _mask_edge_interior_grad_ratio(gray_arr, mask_bool, erode_it=2):
    """Ratio mean(|grad|) on inner boundary band of mask vs eroded (core) interior.

    Tilled specks are often mottled with high texture inside (ratio ~1--1.2);
    more compact, entity-like regions often show a higher boundary/core contrast.
    """
    g = np.hypot(
        ndimage.sobel(gray_arr, axis=0, mode="nearest"),
        ndimage.sobel(gray_arr, axis=1, mode="nearest"),
    )
    ero = ndimage.binary_erosion(mask_bool, iterations=erode_it)
    edge_band = np.logical_and(mask_bool, np.logical_not(ero))
    if int(edge_band.sum()) < 3:
        return 0.0
    if int(ero.sum()) < 2:
        core = mask_bool
    else:
        core = ero
    ge = float(g[edge_band].mean())
    gi = float(g[core].mean()) + 1e-3
    return float(ge / gi)


def _dominant_class(mask_bool, seg_map):
    if mask_bool.shape != seg_map.shape:
        return -1
    pixels = seg_map[mask_bool]
    if pixels.size == 0:
        return -1
    counts = np.bincount(pixels.astype(np.int64))
    return int(counts.argmax())


def _mask_bbox_short_side(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(ys) < 1:
        return 0
    return int(min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))


def _mask_interior_variance(mask_bool, gray_arr):
    sel = gray_arr[mask_bool]
    if sel.size < 6:
        return 0.0
    return float(np.var(sel))


def _terrain_pixel_fraction(mask_bool, seg_map):
    px = seg_map[mask_bool].astype(np.int64)
    if px.size == 0:
        return 1.0
    return float(
        np.isin(px, np.array(list(TERRAIN_BACKGROUND_CLASSES), dtype=np.int64)).sum()
    ) / float(px.size)


def _post_filter_auto(dets, crop_pil, seg_model=None, *,
                      ignore_terrain=True, min_compactness=0.25,
                      min_contrast=8.0, min_area_ratio_post=None,
                      object_class_allowlist=None,
                      min_mask_interior_variance=0.0,
                      min_short_side_px=0,
                      max_terrain_pixel_fraction=1.0,
                      min_lab_ring_distance=0.0,
                      min_chroma_ring_delta=0.0,
                      color_contrast_mode="and",
                      ring_contrast_px=4,
                      min_non_terrain_class_fraction=0.0,
                      max_ring_histogram_bc=1.0,
                      ring_hist_bin_per_ch=8,
                      min_sam_mask_score=0.0,
                      min_mask_area_pixels=0,
                      min_edge_interior_grad_ratio=0.0,
                      edge_grad_erode=2,
                      precomputed_seg_map=None):
    if not dets:
        return dets

    gray = np.asarray(crop_pil.convert("L"), dtype=np.float32)
    h, w = int(gray.shape[0]), int(gray.shape[1])
    total_px = h * w
    seg_map = precomputed_seg_map
    allow = None
    if object_class_allowlist is not None and len(object_class_allowlist) > 0:
        allow = set(int(x) for x in object_class_allowlist)

    m_lab = float(min_lab_ring_distance)
    m_chr = float(min_chroma_ring_delta)
    cc_mode = (color_contrast_mode or "and").lower()
    rpx = int(ring_contrast_px)
    m_non = float(min_non_terrain_class_fraction)
    max_bh = float(max_ring_histogram_bc)
    hb = int(max(4, min(16, ring_hist_bin_per_ch)))
    m_sam = float(min_sam_mask_score)
    m_pix = int(min_mask_area_pixels)
    m_eg = float(min_edge_interior_grad_ratio)
    e_ero = int(max(0, int(edge_grad_erode)))

    use_lab = m_lab > 0.0 or m_chr > 0.0
    use_hist = max_bh < 0.999
    lab_arr = None
    rgb_u8 = None
    if use_lab or use_hist:
        rgb_u8 = np.asarray(crop_pil.convert("RGB"), dtype=np.uint8)
        if use_lab:
            from skimage.color import rgb2lab
            lab_arr = rgb2lab(rgb_u8 / 255.0)

    need_seg = (
        seg_model is not None
        and (
            ignore_terrain
            or (allow is not None)
            or (max_terrain_pixel_fraction < 0.999)
            or (m_non > 0.0)
        )
    )

    kept = []
    for d in dets:
        m = d["mask"]
        if m.shape != gray.shape:
            continue

        if min_short_side_px > 0 and _mask_bbox_short_side(m) < int(min_short_side_px):
            continue

        if min_mask_interior_variance > 0.0:
            if _mask_interior_variance(m, gray) < float(min_mask_interior_variance):
                continue

        if m_sam > 0.0 and float(d.get("score", 0.0)) < m_sam:
            continue
        if m_pix > 0 and int(m.sum()) < m_pix:
            continue

        compact = _mask_compactness(m)
        if compact < min_compactness:
            continue

        contrast = _mask_contrast(m, gray, ring_px=rpx)
        lab_de = 0.0
        chroma_d = 0.0
        if use_lab and lab_arr is not None and lab_arr.shape[:2] == m.shape:
            lab_de, chroma_d = _mask_lab_ring_metrics(m, lab_arr, ring_px=rpx)
        h_bc = 1.0
        if use_hist and rgb_u8 is not None and rgb_u8.shape[:2] == m.shape:
            h_bc = _mask_rgb_ring_histogram_bc(m, rgb_u8, ring_px=rpx, bin_per_ch=hb)
            if h_bc > max_bh:
                continue

        if use_lab and m_lab > 0.0:
            gray_ok = contrast >= min_contrast
            lab_ok = lab_de >= m_lab
            if cc_mode == "or":
                if not (gray_ok or lab_ok):
                    continue
            else:
                if not (gray_ok and lab_ok):
                    continue
        else:
            if contrast < min_contrast:
                continue

        if m_chr > 0.0 and chroma_d < m_chr:
            continue

        egr = _mask_edge_interior_grad_ratio(gray, m, erode_it=e_ero)
        if m_eg > 0.0 and egr < m_eg:
            continue

        if min_area_ratio_post is not None and total_px > 0:
            if float(m.sum()) / float(total_px) < float(min_area_ratio_post):
                continue

        if need_seg and seg_model is not None:
            if seg_map is None:
                try:
                    seg_map = seg_model.segment(crop_pil)
                except Exception as e:
                    import sys as _sys
                    print(f"  SegFormer terrain-check failed: {e}",
                          file=_sys.stderr)
                    seg_map = np.zeros_like(gray, dtype=np.int32)
            tpf = _terrain_pixel_fraction(m, seg_map)
            if tpf > float(max_terrain_pixel_fraction):
                continue
            if m_non > 0.0 and (1.0 - tpf) < m_non:
                continue
            cls = _dominant_class(m, seg_map)
            if ignore_terrain and cls in TERRAIN_BACKGROUND_CLASSES:
                continue
            if allow is not None and cls not in allow:
                continue

        ntf = 0.0
        if seg_map is not None:
            ntf = float(1.0 - _terrain_pixel_fraction(m, seg_map))

        out = dict(d)
        out["compactness"] = compact
        out["contrast"] = contrast
        out["lab_ring_delta_e"] = lab_de
        out["chroma_ring_delta"] = chroma_d
        out["ring_hist_bc"] = h_bc
        out["edge_interior_grad_ratio"] = egr
        out["non_terrain_fraction"] = ntf
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
                        auto_cfg=None,
                        debug_detection_stages=None):
    import math
    from .tiler import tile_image

    if auto_cfg is None:
        auto_cfg = {}
    auto_mode = (detection_mode == "auto")

    if debug_detection_stages is not None:
        log_stages = bool(debug_detection_stages)
    else:
        log_stages = bool(
            auto_mode and auto_cfg.get("log_detection_stages", False))

    if auto_mode and max_per_label > 0:
        max_per_label = 0

    fw, fh = full_image.size

    # Per-crop cap only for auto (text would inherit assembler max_detections_per_tile ~3 and starve the pool).
    if auto_mode:
        per_crop_limit = int(auto_cfg.get("max_detections_per_crop", 10))
    else:
        per_crop_limit = 0
    nms_iou = float(
        auto_cfg.get("nms_iou", 0.3) if auto_mode else 0.3
    )
    if auto_mode:
        max_global_candidates = int(auto_cfg.get("max_global_candidates", 150))
    else:
        max_global_candidates = int(max(1, max_detections_per_crop * 10))
    use_mmr = bool(auto_mode and auto_cfg.get("use_mmr", True))
    mmr_lambda = float(auto_cfg.get("mmr_lambda", 0.55) if auto_mode else 0.55)
    mmr_min_bbox_area_px = int(auto_cfg.get("mmr_min_bbox_area_px", 0) or 0)
    ow_cfg = (auto_cfg.get("objectness_weights") or {}) if auto_mode else {}

    if auto_mode:
        _ams = auto_cfg.get("auto_detection_score_threshold", None)
        if _ams is not None:
            try:
                auto_min_score = float(_ams)
            except (TypeError, ValueError):
                auto_min_score = float(detection_score)
        else:
            auto_min_score = float(detection_score)
    else:
        auto_min_score = float(detection_score)

    n_auto_sam = 0
    n_auto_post = 0
    n_pooled_capped = 0
    log_crop_interval = 0
    if auto_mode:
        log_crop_interval = int(auto_cfg.get("log_crop_interval", 0) or 0)

    all_dets = []
    crop_idx = 0
    for cx, cy, crop_pil in tile_image(full_image, tile_size=scan_tile_size,
                                        overlap=scan_overlap):
        crop_idx += 1
        if log_crop_interval > 0 and crop_idx % log_crop_interval == 0:
            print(
                f"  auto: progress — crop index {crop_idx} (low-variance tiles are skipped)",
                flush=True,
            )
        arr = np.array(crop_pil.convert("L"), dtype=np.float32)
        if arr.var() < variance_threshold:
            continue

        try:
            if auto_mode:
                pre_seg = None
                m_post = auto_cfg.get("min_area_ratio_post")
                if m_post is not None:
                    m_post = float(m_post)
                o_allow = auto_cfg.get("object_class_allowlist")
                ms = bool(auto_cfg.get("multi_scale", True))
                ms_runs = auto_cfg.get("multi_scale_runs")
                extra_boxes = None
                if (bool(auto_cfg.get("segformer_seed_boxes", True))
                        and seg_model is not None):
                    try:
                        pre_seg = seg_model.segment(crop_pil)
                        extra_boxes = _segformer_nonterrain_seed_boxes(
                            pre_seg,
                            min_cc_pixels=int(
                                auto_cfg.get("seed_min_cc_pixels", 400) or 400),
                            margin_px=int(
                                auto_cfg.get("seed_margin_px", 8) or 8),
                            max_seeds=int(
                                auto_cfg.get("max_seed_boxes", 12) or 12),
                        ) or None
                    except Exception as _e:
                        extra_boxes = None
                        pre_seg = None
                bfb = int(auto_cfg.get("box_forward_batch_size", 0) or 0)
                dets = sam_model.detect_objects_auto(
                    crop_pil,
                    min_score=auto_min_score,
                    min_area_ratio=float(auto_cfg.get("min_area_ratio", 0.002)),
                    max_area_ratio=float(auto_cfg.get("max_area_ratio", 0.15)),
                    points_per_side=int(auto_cfg.get("points_per_side", 16)),
                    box_scale=float(auto_cfg.get("box_scale", 1.35)),
                    dedup_iou=float(auto_cfg.get("dedup_iou", 0.65)),
                    multi_scale=ms,
                    multi_scale_runs=ms_runs,
                    merge_dedup_iou=float(auto_cfg.get("merge_dedup_iou", 0.52)),
                    extra_boxes=extra_boxes,
                    separate_seed_forward=bool(
                        auto_cfg.get("separate_seed_forward", True)),
                    box_forward_batch_size=bfb,
                )
                n_auto_sam += len(dets)
                miv = auto_cfg.get("min_mask_interior_variance")
                if miv is not None:
                    miv = float(miv)
                mss = auto_cfg.get("min_short_side_px")
                if mss is not None:
                    mss = int(mss)
                mtf = auto_cfg.get("max_terrain_pixel_fraction")
                if mtf is not None:
                    mtf = float(mtf)
                mlr = auto_cfg.get("min_lab_ring_distance", 0.0)
                if mlr is not None:
                    mlr = float(mlr)
                else:
                    mlr = 0.0
                mch = auto_cfg.get("min_chroma_ring_delta", 0.0)
                if mch is not None:
                    mch = float(mch)
                else:
                    mch = 0.0
                ccm = auto_cfg.get("color_contrast_mode", "and")
                rpx = int(auto_cfg.get("ring_contrast_px", 4) or 4)
                mnt = auto_cfg.get("min_non_terrain_class_fraction", 0.0)
                if mnt is not None:
                    mnt = float(mnt)
                else:
                    mnt = 0.0
                mxh = auto_cfg.get("max_ring_histogram_bc", 1.0)
                if mxh is not None:
                    mxh = float(mxh)
                else:
                    mxh = 1.0
                hbc = int(auto_cfg.get("ring_hist_bin_per_ch", 8) or 8)
                mss2 = auto_cfg.get("min_sam_mask_score", 0.0)
                if mss2 is not None:
                    mss2 = float(mss2)
                else:
                    mss2 = 0.0
                mpix = auto_cfg.get("min_mask_area_pixels", 0)
                if mpix is not None:
                    mpix = int(mpix)
                else:
                    mpix = 0
                meg = auto_cfg.get("min_edge_interior_grad_ratio", 0.0)
                if meg is not None:
                    meg = float(meg)
                else:
                    meg = 0.0
                ege = int(auto_cfg.get("edge_grad_erode", 2) or 2)
                dets = _post_filter_auto(
                    dets, crop_pil, seg_model=seg_model,
                    ignore_terrain=bool(auto_cfg.get("ignore_terrain", True)),
                    min_compactness=float(auto_cfg.get("min_compactness", 0.25)),
                    min_contrast=float(auto_cfg.get("min_contrast", 8.0)),
                    min_area_ratio_post=m_post,
                    object_class_allowlist=o_allow,
                    min_mask_interior_variance=miv if miv is not None else 0.0,
                    min_short_side_px=mss if mss is not None else 0,
                    max_terrain_pixel_fraction=mtf if mtf is not None else 1.0,
                    min_lab_ring_distance=mlr,
                    min_chroma_ring_delta=mch,
                    color_contrast_mode=str(ccm or "and"),
                    ring_contrast_px=rpx,
                    min_non_terrain_class_fraction=mnt,
                    max_ring_histogram_bc=mxh,
                    ring_hist_bin_per_ch=hbc,
                    min_sam_mask_score=mss2,
                    min_mask_area_pixels=mpix,
                    min_edge_interior_grad_ratio=meg,
                    edge_grad_erode=ege,
                    precomputed_seg_map=pre_seg,
                )
                n_auto_post += len(dets)
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
        crop_dets = []
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

            ovis = {
                "score": d["score"],
                "area_ratio": d["area_ratio"],
                "lab_ring_delta_e": d.get("lab_ring_delta_e", 0.0),
                "ring_hist_bc": d.get("ring_hist_bc", 1.0),
                "compactness": d.get("compactness", 0.0),
                "contrast": d.get("contrast", 0.0),
                "edge_interior_grad_ratio": d.get("edge_interior_grad_ratio", 0.0),
                "non_terrain_fraction": d.get("non_terrain_fraction", 0.0),
            }
            crop_dets.append({
                "mask_local": local_mask,
                "crop_offset": (cx, cy),
                "crop_size": (cw, ch),
                "bbox_fullimg": fbbox,
                "centroid_fullimg": (fcx, fcy),
                "label": d["label"],
                "score": d["score"],
                "area_ratio": d["area_ratio"],
                "visibility": (
                    _auto_objectness(ovis, weight=ow_cfg) if auto_mode
                    else _detection_visibility(d)
                ),
            })
        crop_dets.sort(key=lambda d: d["visibility"], reverse=True)
        if per_crop_limit > 0 and len(crop_dets) > per_crop_limit:
            crop_dets = crop_dets[:per_crop_limit]
        if auto_mode:
            n_pooled_capped += len(crop_dets)
        all_dets.extend(crop_dets)

    if not all_dets:
        return []

    if log_stages and auto_mode:
        print(f"  auto: after SAM (pre post_filter), summed: {n_auto_sam}")
        print(f"  auto: after post_filter, summed: {n_auto_post}")
        print(f"  auto: after per-crop cap, pooled: {n_pooled_capped}")
    if auto_mode:
        print(
            f"  Pooled {len(all_dets)} dets (post_filter + per-crop cap) "
            f"across {crop_idx} crops"
        )
    else:
        print(
            f"  SAM3 pooled {len(all_dets)} dets across {crop_idx} crops"
        )
    if log_stages and not auto_mode:
        print(f"  text: before global NMS: {len(all_dets)}")

    all_dets.sort(key=lambda d: d["visibility"], reverse=True)
    if log_stages and auto_mode:
        print(f"  auto: before global NMS: {len(all_dets)}")

    kept = _nms_by_visibility(all_dets, iou_threshold=nms_iou)
    if len(kept) > max_global_candidates:
        kept = sorted(
            kept, key=lambda d: d["visibility"], reverse=True)[:max_global_candidates]

    if log_stages and auto_mode:
        print(f"  auto: after NMS: {len(kept)}")
    else:
        print(f"  After NMS: {len(kept)} unique detections")

    def _build_selected_entry(det):
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
        return {
            "mask_fullimg": full_mask,
            "bbox_fullimg": det["bbox_fullimg"],
            "centroid_fullimg": det["centroid_fullimg"],
            "label": det["label"],
            "score": det["score"],
            "visibility": det["visibility"],
        }

    if auto_mode and use_mmr and kept:
        pool = list(kept)
        if log_stages:
            print(
                f"  auto: MMR pool (after NMS, before min_bbox_area): {len(pool)}"
            )
        if mmr_min_bbox_area_px > 0:
            def _bbox_area(fb):
                return max(0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
            pool = [d for d in pool
                    if _bbox_area(d["bbox_fullimg"]) >= mmr_min_bbox_area_px]
            if log_stages and mmr_min_bbox_area_px > 0:
                print(
                    f"  auto: MMR pool after min_bbox_area: {len(pool)}"
                )
        if not pool:
            pool = list(kept)
        pool = sorted(pool, key=lambda d: d["visibility"], reverse=True)
        picked = _mmr_select_topk(pool, max_objects, mmr_lambda, min_object_distance)
        if log_stages:
            print(f"  auto: after MMR: {len(picked)}")
        selected = [_build_selected_entry(p) for p in picked]
        return selected

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

        selected.append(_build_selected_entry(det))
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        if len(selected) >= max_objects:
            break

    if log_stages and auto_mode:
        print(f"  auto: after greedy selection: {len(selected)}")
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
