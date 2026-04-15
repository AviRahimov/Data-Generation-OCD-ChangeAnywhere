"""ChangeAnywhere-style synthetic pair generator.

Orchestrates: semantic segmentation -> change simulation -> SD inpainting
to produce (before, synthetic_after, change_mask) training triplets.
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
                            rng=None, seed=None, appearance_prob=0.6):
    """Generate one synthetic (after, change_mask) pair from a before tile.

    Args:
        before_pil: PIL RGB image (the "before" tile)
        seg_map: integer ndarray (H, W) with ADE20K semantic class IDs
        inpaint_model: InpaintingModel instance
        rng: random.Random instance
        seed: int seed for the diffusion model
        appearance_prob: probability of appearance vs disappearance event

    Returns:
        (after_pil, change_mask_pil, meta_dict) or None if no change possible
    """
    rng = rng or random.Random()

    result = simulate_change(seg_map, rng=rng, appearance_prob=appearance_prob)
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
                   max_per_tile=2, seed=42, appearance_prob=0.6,
                   ssim_min=0.4, ssim_max=0.99):
    """Generate synthetic pairs for a batch of tiles.

    Args:
        tile_paths: list of Path to before tiles
        seg_model: SegmentationModel with .segment(pil) -> ndarray
        inpaint_model: InpaintingModel
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

            rng = random.Random(seed + hash(tile_name))

            for variant in range(max_per_tile):
                v_seed = seed + hash(tile_name) + variant * 1000
                try:
                    result = generate_synthetic_pair(
                        before, seg_map, inpaint_model,
                        rng=rng, seed=v_seed,
                        appearance_prob=appearance_prob,
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

                # Save segmentation visualization
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


def _colorize_seg(seg):
    rng = np.random.RandomState(42)
    n = max(seg.max() + 1, 256)
    cmap = rng.randint(40, 255, size=(n, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]
    return Image.fromarray(cmap[seg % len(cmap)])
