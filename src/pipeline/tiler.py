# ...existing code...
from pathlib import Path
from PIL import Image
import numpy as np

from .io import save_image, pil_to_numpy, numpy_to_pil


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

