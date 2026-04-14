# ...existing code...
from pathlib import Path
from PIL import Image, ImageDraw
import json
import numpy as np


def load_image(path):
    path = Path(path)
    return Image.open(path).convert('RGB')


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def polygons_to_mask(polygons, size):
    """Rasterize a list of polygons (each polygon is list of [x,y]) into a binary PIL Image mask of given (W,H).
    Coordinates are assumed in image pixel coordinates (x,y).
    """
    w, h = size
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        # flatten coordinates
        coords = [(float(x), float(y)) for x, y in poly]
        try:
            draw.polygon(coords, outline=255, fill=255)
        except Exception:
            # ignore invalid polys
            continue
    return mask


def pil_to_numpy(img):
    return np.array(img)


def numpy_to_pil(arr):
    return Image.fromarray(arr)

