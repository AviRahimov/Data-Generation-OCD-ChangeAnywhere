"""Tile-level synthetic generation: single tiles and batch_generate.

Used by legacy ``dataset.Pipeline`` and any workflow that runs on fixed-size
tiles rather than full-image SAM scanning + ``generate_full_image_pair``.
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

from .io import write_json
from .change_simulator import simulate_change
from .prompt_templates import TERRAIN_BACKGROUND_CLASSES


def _mask_bbox_bool(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def compute_local_ssim_tile(before_pil, after_pil, change_mask_bool, pad=32,
                            data_range=255.0):
    """SSIM on a tight crop around the change mask (handles small edits on large tiles)."""
    if ssim is None:
        return -1.0
    bbox = _mask_bbox_bool(change_mask_bool)
    if bbox is None:
        return -1.0
    y1, y2, x1, x2 = bbox
    h, w = change_mask_bool.shape
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(w, x2 + pad)
    if y2 <= y1 or x2 <= x1:
        return -1.0
    bg = np.array(before_pil.convert("L"), dtype=np.float64)[y1:y2, x1:x2]
    ag = np.array(after_pil.convert("L"), dtype=np.float64)[y1:y2, x1:x2]
    return float(ssim(bg, ag, data_range=data_range))


def generate_synthetic_pair(before_pil, seg_map, inpaint_model,
                            rng=None, seed=None, appearance_prob=0.20,
                            detected_objects=None):
    """Generate one synthetic (after, change_mask) pair from a before tile."""
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
        meta["ssim"] = compute_local_ssim_tile(
            before_pil, after_pil, change_mask.astype(bool), pad=32)

    meta["prompt"] = prompt

    return after_pil, change_mask_pil, meta


def generate_synthetic_after(before_pil, seg_map, inpaint_model, *,
                             max_modified_segments=3, color_jitter=0.2, seed=42,
                             appearance_prob=0.20, detected_objects=None):
    """Legacy API for ``dataset.Pipeline.generate_synthetic``.

    ``max_modified_segments`` and ``color_jitter`` are accepted for backward
    compatibility; they are not yet wired into ``simulate_change``.
    """
    del max_modified_segments, color_jitter  # reserved for future use
    rng = random.Random(seed)
    out = generate_synthetic_pair(
        before_pil, seg_map, inpaint_model,
        rng=rng, seed=seed, appearance_prob=appearance_prob,
        detected_objects=detected_objects,
    )
    if out is None:
        raise RuntimeError(
            "generate_synthetic_after: no change event (simulate_change returned None)")
    after_pil, change_pil, _meta = out
    return after_pil, change_pil


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


def _colorize_seg(seg):
    rng = np.random.RandomState(42)
    n = max(seg.max() + 1, 256)
    cmap = rng.randint(40, 255, size=(n, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]
    return Image.fromarray(cmap[seg % len(cmap)])


def batch_generate(tile_paths, seg_model, inpaint_model, output_dir,
                   sam_model=None, detection_prompts=None,
                   detection_score=0.30,
                   max_per_tile=2, seed=42, appearance_prob=0.20,
                   ssim_min=0.4, ssim_max=0.99):
    """Generate synthetic pairs for a batch of tiles."""
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
