"""ChangeAnywhere-style synthetic pair generator.

Orchestrates: SAM3 detection -> semantic segmentation -> change simulation
-> SD inpainting with feathered blending to produce (before, synthetic_after,
change_mask) training triplets.
"""

import random
import csv
import numpy as np
from PIL import Image
from pathlib import Path

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

from .io import write_json
from .change_simulator import simulate_change
from .prompt_templates import TERRAIN_BACKGROUND_CLASSES


def generate_synthetic_pair(before_pil, seg_map, inpaint_model,
                            rng=None, seed=None, appearance_prob=0.20,
                            detected_objects=None):
    """Generate one synthetic (after, change_mask) pair from a before tile.

    Args:
        before_pil: PIL RGB image (the "before" tile)
        seg_map: integer ndarray (H, W) with ADE20K semantic class IDs
        inpaint_model: InpaintingModel instance
        rng: random.Random instance
        seed: int seed for the diffusion model
        appearance_prob: probability of appearance vs disappearance event
        detected_objects: optional list from SAMModel.detect_objects()

    Returns:
        (after_pil, change_mask_pil, meta_dict) or None if no change possible
    """
    rng = rng or random.Random()

    result = simulate_change(seg_map, rng=rng, appearance_prob=appearance_prob,
                             detected_objects=detected_objects)
    if result is None:
        return None

    change_mask, prompt, meta = result

    after_pil = inpaint_model.inpaint(
        image=before_pil,
        mask=change_mask,
        prompt=prompt,
        seed=seed,
    )

    change_mask_pil = Image.fromarray((change_mask.astype(np.uint8) * 255))

    if ssim is not None:
        before_gray = np.array(before_pil.convert("L"))
        after_gray = np.array(after_pil.convert("L"))
        score = ssim(before_gray, after_gray, data_range=255)
        meta["ssim"] = float(score)

    meta["prompt"] = prompt

    return after_pil, change_mask_pil, meta


def is_tile_interesting(seg_map, min_classes=2, min_bg_ratio=0.1):
    """Check if a tile has enough semantic variety for change simulation."""
    classes = np.unique(seg_map)
    if len(classes) < min_classes:
        return False

    h, w = seg_map.shape
    total = h * w
    bg_pixels = sum(
        (seg_map == c).sum() for c in classes if int(c) in TERRAIN_BACKGROUND_CLASSES
    )
    return bg_pixels / total >= min_bg_ratio


def batch_generate(tile_paths, seg_model, inpaint_model, output_dir,
                   sam_model=None, detection_prompts=None,
                   detection_score=0.30,
                   max_per_tile=2, seed=42, appearance_prob=0.20,
                   ssim_min=0.4, ssim_max=0.99):
    """Generate synthetic pairs for a batch of tiles.

    Args:
        tile_paths: list of Path to before tiles
        seg_model: SegmentationModel with .segment(pil) -> ndarray
        inpaint_model: InpaintingModel (with feathered blending)
        sam_model: optional SAMModel for targeted object detection
        detection_prompts: text prompts for SAM3 detection
        detection_score: minimum confidence for SAM3 detections
        output_dir: Path for output
        max_per_tile: how many change variants per tile
        seed: base random seed
        appearance_prob: probability of appearance events
        ssim_min/ssim_max: quality filter range

    Returns:
        list of dicts with provenance info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "provenance.csv"
    csv_exists = csv_path.exists()
    results = []

    with open(csv_path, "a" if csv_exists else "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "source_tile", "variant", "event", "object_type",
                "prompt", "ssim", "status", "output_after", "output_mask",
            ])

        for tile_path in tile_paths:
            tile_path = Path(tile_path)
            tile_name = tile_path.stem

            try:
                before = Image.open(tile_path).convert("RGB")
            except Exception as e:
                print(f"  Skip {tile_name}: {e}")
                continue

            seg_map = seg_model.segment(before)

            if not is_tile_interesting(seg_map):
                continue

            detected_objects = None
            if sam_model is not None:
                try:
                    detected_objects = sam_model.detect_objects(
                        before, prompts=detection_prompts,
                        min_score=detection_score,
                    )
                except Exception as e:
                    print(f"  SAM3 detection failed on {tile_name}: {e}")

            rng = random.Random(seed + hash(tile_name))

            for variant in range(max_per_tile):
                v_seed = seed + hash(tile_name) + variant * 1000
                try:
                    result = generate_synthetic_pair(
                        before, seg_map, inpaint_model,
                        rng=rng, seed=v_seed,
                        appearance_prob=appearance_prob,
                        detected_objects=detected_objects,
                    )
                except Exception as e:
                    print(f"  Error on {tile_name} v{variant}: {e}")
                    writer.writerow([
                        str(tile_path), variant, "", "", "", "", f"error: {e}", "", "",
                    ])
                    continue

                if result is None:
                    continue

                after_pil, mask_pil, meta = result

                score = meta.get("ssim", -1)
                if score != -1 and (score > ssim_max or score < ssim_min):
                    writer.writerow([
                        str(tile_path), variant, meta.get("event", ""),
                        meta.get("object_type", ""), meta.get("prompt", ""),
                        f"{score:.4f}", "filtered", "", "",
                    ])
                    continue

                var_dir = output_dir / tile_name / f"v{variant}"
                var_dir.mkdir(parents=True, exist_ok=True)

                before.save(var_dir / "before.png")
                after_pil.save(var_dir / "after_synth.png")
                mask_pil.save(var_dir / "change_mask.png")
                write_json(meta, var_dir / "meta.json")

                seg_vis = _colorize_seg(seg_map)
                seg_vis.save(var_dir / "seg_map.png")

                writer.writerow([
                    str(tile_path), variant, meta.get("event", ""),
                    meta.get("object_type", ""), meta.get("prompt", ""),
                    f"{score:.4f}" if score != -1 else "",
                    "kept", str(var_dir / "after_synth.png"),
                    str(var_dir / "change_mask.png"),
                ])

                results.append({
                    "tile": str(tile_path),
                    "variant": variant,
                    "output_dir": str(var_dir),
                    **meta,
                })

    return results


def _detection_visibility(det):
    """Score a single detection by how visible its removal would be.

    Larger objects with higher confidence produce more noticeable changes.
    The sqrt(area_ratio / 0.01) normalizes so that a 1% area object = 1.0.
    """
    import math
    return det["score"] * math.sqrt(max(det["area_ratio"], 1e-6) / 0.01)


def _bbox_iou(a, b):
    """Intersection-over-union of two (x1,y1,x2,y2) bounding boxes."""
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
    """Return (x1, y1, x2, y2) tight bounding box of a 2-D bool mask."""
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def select_best_objects(full_image, sam_model, max_objects=3,
                        scan_tile_size=1024, scan_overlap=128,
                        detection_prompts=None, detection_score=0.30,
                        variance_threshold=500,
                        min_object_distance=2000,
                        max_detections_per_crop=5,
                        max_per_label=0):
    """Scan the full image with 1024x1024 crops, detect objects, and select the best.

    Two-scale strategy: SAM3 sees 1024x1024 context to capture full objects,
    then the best individual objects are chosen based on visibility,
    spatial spread, and label diversity for downstream SD2 inpainting.

    Args:
        full_image: PIL RGB (the full-resolution before image)
        sam_model: SAMModel with detect_objects()
        max_objects: max objects to return
        scan_tile_size: crop size for SAM3 scanning
        scan_overlap: overlap between adjacent scanning crops
        detection_prompts: text prompts for SAM3
        detection_score: minimum confidence to keep
        variance_threshold: skip crops with pixel variance below this
        min_object_distance: min pixel distance between selected object centroids
        max_detections_per_crop: cap detections per crop before scoring
        max_per_label: max times the same object type can be selected
                       (0 = no limit, 1 = forces maximum variety)

    Returns:
        list of dicts (at most max_objects) sorted by visibility, each with:
            mask_fullimg  -- bool ndarray (H_full, W_full)
            bbox_fullimg  -- (x1, y1, x2, y2)
            centroid_fullimg -- (cx, cy)
            label         -- str
            score         -- float (SAM3 confidence)
            visibility    -- float (visibility metric)
    """
    import math
    from .tiler import tile_image

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
            dets = sam_model.detect_objects(
                crop_pil, prompts=detection_prompts, min_score=detection_score,
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


def _colorize_seg(seg):
    rng = np.random.RandomState(42)
    n = max(seg.max() + 1, 256)
    cmap = rng.randint(40, 255, size=(n, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]
    return Image.fromarray(cmap[seg % len(cmap)])
