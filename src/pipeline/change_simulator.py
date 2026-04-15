"""Simulate realistic change events on semantic segmentation maps.

Follows the ChangeAnywhere paper's two event types:
  - Object APPEARANCE: a new object materializes on background terrain
  - Object DISAPPEARANCE: an existing non-background object is removed

Each event returns a binary change mask and a text prompt for the inpainting model.
"""

import random
import numpy as np
from PIL import Image, ImageDraw

from .prompt_templates import (
    is_terrain_background,
    get_background_label,
    get_appearance_prompt,
    get_disappearance_prompt,
    sample_object_type,
    TERRAIN_BACKGROUND_CLASSES,
)


def _random_blob_mask(h, w, cx, cy, avg_radius, rng):
    """Generate an irregular blob mask centered at (cx, cy)."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    n_vertices = rng.randint(6, 12)
    angles = sorted(rng.uniform(0, 2 * np.pi) for _ in range(n_vertices))
    points = []
    for a in angles:
        r = avg_radius * rng.uniform(0.5, 1.5)
        px = cx + r * np.cos(a)
        py = cy + r * np.sin(a)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        points.append((px, py))
    draw.polygon(points, fill=255)
    return np.array(mask) > 127


def _find_background_region(seg_map, min_area_ratio=0.05):
    """Find the dominant background class in the tile."""
    h, w = seg_map.shape
    total = h * w
    classes, counts = np.unique(seg_map, return_counts=True)
    bg_candidates = []
    for cls, cnt in zip(classes, counts):
        if is_terrain_background(int(cls)) and cnt / total >= min_area_ratio:
            bg_candidates.append((int(cls), int(cnt)))
    bg_candidates.sort(key=lambda x: x[1], reverse=True)
    return bg_candidates


def _find_foreground_objects(seg_map, min_pixels=100):
    """Find non-background segments suitable for removal."""
    classes, counts = np.unique(seg_map, return_counts=True)
    fg = []
    for cls, cnt in zip(classes, counts):
        cls_int = int(cls)
        if not is_terrain_background(cls_int) and cls_int != 0 and cnt >= min_pixels:
            fg.append((cls_int, int(cnt)))
    fg.sort(key=lambda x: x[1])
    return fg


def simulate_appearance(seg_map, rng=None, min_radius=40, max_radius=100):
    """Simulate an object appearing on background terrain.

    Returns:
        change_mask: bool array (H, W) -- True where the change is
        prompt: str -- text prompt for the inpainting model
        meta: dict -- metadata about the event
    Returns None if no suitable background region found.
    """
    rng = rng or random.Random()
    h, w = seg_map.shape

    bg_regions = _find_background_region(seg_map)
    if not bg_regions:
        return None

    bg_class, _ = bg_regions[0]
    bg_mask = seg_map == bg_class

    ys, xs = np.where(bg_mask)
    if len(ys) < 100:
        return None

    for _ in range(20):
        idx = rng.randint(0, len(ys) - 1)
        cy, cx = int(ys[idx]), int(xs[idx])
        radius = rng.randint(min_radius, max_radius)
        if cy - radius < 0 or cy + radius >= h or cx - radius < 0 or cx + radius >= w:
            continue

        blob = _random_blob_mask(h, w, cx, cy, radius, rng)
        blob_in_bg = np.logical_and(blob, bg_mask)
        if blob_in_bg.sum() < blob.sum() * 0.7:
            continue

        change_mask = blob_in_bg
        obj_type = sample_object_type(rng, bg_class_id=bg_class)
        prompt = get_appearance_prompt(obj_type, bg_class, rng)

        return change_mask, prompt, {
            "event": "appearance",
            "object_type": obj_type,
            "bg_class": bg_class,
            "bg_label": get_background_label(bg_class),
            "center": (cx, cy),
            "radius": radius,
            "change_pixels": int(change_mask.sum()),
        }

    return None


def simulate_disappearance(seg_map, rng=None, max_object_ratio=0.15, min_object_pixels=1500):
    """Simulate an object disappearing (removed, replaced by background).

    Returns same format as simulate_appearance, or None.
    """
    from scipy import ndimage
    rng = rng or random.Random()
    h, w = seg_map.shape
    total = h * w

    fg_objects = _find_foreground_objects(seg_map, min_pixels=min_object_pixels)
    if not fg_objects:
        return None

    bg_regions = _find_background_region(seg_map)
    if not bg_regions:
        surround_class = 13
    else:
        surround_class = bg_regions[0][0]

    candidates = [(c, n) for c, n in fg_objects if n / total <= max_object_ratio]
    if not candidates:
        return None

    fg_class, fg_count = rng.choice(candidates)
    change_mask = seg_map == fg_class

    change_mask = ndimage.binary_dilation(change_mask, iterations=8)

    prompt = get_disappearance_prompt(surround_class, rng)

    return change_mask, prompt, {
        "event": "disappearance",
        "removed_class": fg_class,
        "bg_class": surround_class,
        "bg_label": get_background_label(surround_class),
        "change_pixels": int(change_mask.sum()),
    }


def simulate_change(seg_map, rng=None, appearance_prob=0.6):
    """Simulate a random change event (appearance or disappearance).

    Returns (change_mask, prompt, meta) or None if the tile has no
    suitable regions.
    """
    rng = rng or random.Random()

    if rng.random() < appearance_prob:
        result = simulate_appearance(seg_map, rng)
        if result is not None:
            return result

    result = simulate_disappearance(seg_map, rng)
    if result is not None:
        return result

    return simulate_appearance(seg_map, rng)
