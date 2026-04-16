from pathlib import Path
import re
from PIL import Image
import numpy as np

from .io import save_image, pil_to_numpy, numpy_to_pil

_COORD_RE = re.compile(r"_x(\d{4,})_y(\d{4,})")


def tile_image(img, tile_size=512, overlap=64):
    """Yield (x,y,tile) where x,y are top-left coordinates for each tile."""
    w, h = img.size
    step = tile_size - overlap
    for y in range(0, max(1, h - overlap), step):
        for x in range(0, max(1, w - overlap), step):
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tile = img.crop(box)
            yield x, y, tile


def is_tile_nonempty(tile, min_nonempty_ratio=0.02):
    arr = np.array(tile.convert('L'))
    # compute fraction of pixels that deviate from median by >10
    med = np.median(arr)
    mask = (np.abs(arr - med) > 10).astype(np.uint8)
    ratio = mask.mean()
    return ratio >= min_nonempty_ratio


def save_tiles_for_image(img_path, out_dir, tile_size=512, overlap=64, min_nonempty_ratio=0.02):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert('RGB')
    basename = Path(img_path).stem
    saved = []
    for x, y, tile in tile_image(img, tile_size=tile_size, overlap=overlap):
        if not is_tile_nonempty(tile, min_nonempty_ratio=min_nonempty_ratio):
            continue
        tile_name = f"{basename}_x{x:04d}_y{y:04d}.png"
        out_path = out_dir / tile_name
        save_image(tile, out_path)
        saved.append(str(out_path))
    return saved


def _parse_tile_coords(filename):
    """Extract (x, y) pixel coordinates from a tile filename."""
    m = _COORD_RE.search(str(filename))
    if m is None:
        raise ValueError(f"Cannot parse tile coordinates from: {filename}")
    return int(m.group(1)), int(m.group(2))


def reassemble_tiles(tile_dir, original_size, tile_size=512,
                     replacements=None):
    """Reassemble tiles back into a full image.

    Tiles are pasted in sorted (y, x) order so overlap regions are handled
    consistently with how they were cut.

    Args:
        tile_dir: Path to directory containing tiles named *_xNNNN_yNNNN.png
        original_size: (W, H) tuple -- size of the original full image
        tile_size: int (for reference; actual tile size is read from file)
        replacements: dict mapping tile stem (e.g. "before_x3584_y3136")
                      to a PIL RGB image to use instead of the saved tile

    Returns:
        PIL RGB image at original_size
    """
    tile_dir = Path(tile_dir)
    replacements = replacements or {}
    canvas = Image.new("RGB", original_size, (0, 0, 0))

    tile_files = sorted(tile_dir.glob("*.png"))

    entries = []
    for tf in tile_files:
        try:
            x, y = _parse_tile_coords(tf.stem)
            entries.append((y, x, tf))
        except ValueError:
            continue

    entries.sort()

    for y, x, tf in entries:
        stem = tf.stem
        if stem in replacements:
            tile = replacements[stem]
        else:
            tile = Image.open(tf).convert("RGB")
        canvas.paste(tile, (x, y))

    return canvas


def build_change_mask(original_size, changed_tile_stems, tile_dir,
                      tile_masks=None):
    """Build a full-resolution binary change mask.

    Args:
        original_size: (W, H) of the full image
        changed_tile_stems: list of tile stems that were changed
        tile_dir: Path to tile directory (for reading tile dimensions)
        tile_masks: optional dict mapping tile stem -> bool ndarray (H, W)
                    with per-pixel change mask. If not provided, the entire
                    tile region is marked as changed.

    Returns:
        PIL "L" image (0=unchanged, 255=changed) at original_size
    """
    tile_dir = Path(tile_dir)
    mask = Image.new("L", original_size, 0)

    for stem in changed_tile_stems:
        x, y = _parse_tile_coords(stem)

        if tile_masks and stem in tile_masks:
            tile_mask_bool = tile_masks[stem]
            tile_mask_pil = Image.fromarray(
                (tile_mask_bool.astype(np.uint8) * 255)
            )
        else:
            tf = tile_dir / f"{stem}.png"
            if tf.exists():
                tw, th = Image.open(tf).size
            else:
                tw, th = 512, 512
            tile_mask_pil = Image.new("L", (tw, th), 255)

        mask.paste(tile_mask_pil, (x, y))

    return mask

