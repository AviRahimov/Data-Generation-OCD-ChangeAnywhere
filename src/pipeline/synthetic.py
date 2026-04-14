# ...existing code...
"""Synthetic generator stub for ChangeAnywhere-style workflow.
This module provides a simple, deterministic way to create a synthetic "after" image
from a single "before" image and its segmentation map. It is not a latent diffusion
model implementation — instead it's a lightweight, fast-to-run simulator you can use
for prototyping and testing the rest of the pipeline. Replace this with an LDM-based
generator later.

Behavior:
- Accepts a PIL RGB before image and an integer segmentation mask (H,W)
- Randomly selects up to `max_modified_segments` segment ids to modify
- For each selected segment, either copy pixels from a nearby segment (swap) or apply
  a color jitter to simulate change
- Returns the synthetic after image (PIL) and a binary change mask (PIL L)
"""
from PIL import Image, ImageEnhance
import numpy as np
import random


def generate_synthetic_after(before_pil, segmask, max_modified_segments=3, color_jitter=0.2, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    arr = np.array(before_pil)
    h, w = segmask.shape
    assert arr.shape[0] == h and arr.shape[1] == w, 'size mismatch between image and segmask'

    unique_segments = np.unique(segmask)
    unique_segments = unique_segments[unique_segments != 0]
    if len(unique_segments) == 0:
        # nothing to change
        return before_pil.copy(), Image.new('L', (w, h), 0)

    n_mod = min(max_modified_segments, max(1, len(unique_segments)//10))
    chosen = list(np.random.choice(unique_segments, size=n_mod, replace=False))

    after = arr.copy()
    change_mask = np.zeros((h, w), dtype=np.uint8)

    for segid in chosen:
        mask = (segmask == segid)
        # choose whether to swap with another segment or jitter color
        if len(unique_segments) > 1 and random.random() < 0.5:
            other = int(random.choice([s for s in unique_segments if s != segid]))
            other_mask = (segmask == other)
            # copy texture from other segment into this one (sample pixels)
            # sample pixels from other segment and assign them to segid pixels
            other_pixels = arr[other_mask]
            if other_pixels.size == 0:
                continue
            # if other region smaller, tile samples
            sampled = other_pixels[np.random.choice(other_pixels.shape[0], size=mask.sum(), replace=True)]
            after[mask] = sampled
        else:
            # color jitter: adjust brightness & color of segment region
            factor = 1.0 + (np.random.randn() * color_jitter)
            seg_pixels = after[mask].astype(np.float32)
            seg_pixels = np.clip(seg_pixels * factor, 0, 255).astype(np.uint8)
            after[mask] = seg_pixels
        change_mask[mask] = 255

    after_pil = Image.fromarray(after)
    change_pil = Image.fromarray(change_mask)
    return after_pil, change_pil

