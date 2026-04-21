"""Synthetic pair generation: re-exports for backward-compatible imports.

Implementation is split into:

- ``tile_synthetic`` — per-tile ``generate_synthetic_pair``, ``batch_generate``, legacy ``generate_synthetic_after``
- ``full_image`` — ``select_best_objects``, ``select_appearance_locations``, ``generate_full_image_pair``
"""

from .tile_synthetic import (
    batch_generate,
    compute_local_ssim_tile,
    generate_synthetic_after,
    generate_synthetic_pair,
    is_tile_interesting,
)
from .full_image import (
    compute_local_ssim_change_mask,
    generate_full_image_pair,
    select_appearance_locations,
    select_best_objects,
)

__all__ = [
    "batch_generate",
    "compute_local_ssim_change_mask",
    "compute_local_ssim_tile",
    "generate_full_image_pair",
    "generate_synthetic_after",
    "generate_synthetic_pair",
    "is_tile_interesting",
    "select_appearance_locations",
    "select_best_objects",
]
