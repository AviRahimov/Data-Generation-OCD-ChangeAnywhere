"""A/B compare SD2 / SD1.5-realistic / SDXL inpainting backends on the same tiles.

For each sampled interesting tile, we run SegFormer + SAM 3 + change simulation
ONCE to produce a shared (before_tile, change_mask, prompt) triple. Then we
iterate the inpainting backends serially (loading + cleaning up between
backends, because 8 GB VRAM can't hold all three at once) and render the
inpainted result for the same triple.

Output: one grid per tile in src/data/workspace/synthetic/<pair>/<tile>_compare.png
with panels:

    [before | SD2 | SD1.5-realistic | SDXL | change_mask]

Usage:
    python src/scripts/compare_inpaint_backends.py
    python src/scripts/compare_inpaint_backends.py pair_0003
    python src/scripts/compare_inpaint_backends.py --tile before_x3584_y3136
    python src/scripts/compare_inpaint_backends.py pair_0003 --tile before_x1024_y0512
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pipeline.config import Config
from pipeline.tiler import save_tiles_for_image
from pipeline.segmentation import get_segmentation_model
from pipeline.inpainting import build_inpainter_from_cfg
from pipeline.change_simulator import simulate_change


TITLE_H = 48
PAD = 4
BG_COLOR = (30, 30, 30)
WHITE = (255, 255, 255)


def _font(size=18):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_grid(panels, titles, path):
    pw, ph = panels[0].size
    cols = min(len(panels), 4)
    rows = (len(panels) + cols - 1) // cols
    gw = cols * pw + (cols + 1) * PAD
    gh = rows * (ph + TITLE_H) + (rows + 1) * PAD
    grid = Image.new("RGB", (gw, gh), BG_COLOR)
    draw = ImageDraw.Draw(grid)
    fnt = _font(18)
    for i, (img, title) in enumerate(zip(panels, titles)):
        c, r = i % cols, i // cols
        x = PAD + c * (pw + PAD)
        y = PAD + r * (ph + TITLE_H + PAD)
        bbox = draw.textbbox((0, 0), title, font=fnt)
        tw = bbox[2] - bbox[0]
        draw.text((x + (pw - tw) // 2, y + 10), title, fill=WHITE, font=fnt)
        grid.paste(img, (x, y + TITLE_H))
    grid.save(path, quality=95)


def _contour_overlay(base, mask_bool, color=(255, 40, 40),
                     thickness=2, fill_alpha=0.25):
    from scipy import ndimage
    arr = np.array(base.copy())
    if fill_alpha > 0:
        overlay_color = np.array(color, dtype=np.float32)
        arr[mask_bool] = (
            arr[mask_bool] * (1 - fill_alpha) + overlay_color * fill_alpha
        ).astype(np.uint8)
    eroded = ndimage.binary_erosion(mask_bool, iterations=thickness)
    contour = mask_bool & ~eroded
    arr[contour] = color
    return Image.fromarray(arr)


def _pre_filter_tiles(tile_paths, seg_model, min_classes=2):
    """Return only tiles where SegFormer finds >= min_classes semantic classes."""
    interesting = []
    for tp in tile_paths:
        tile = Image.open(tp).convert("RGB")
        seg = seg_model.segment(tile)
        n = len(np.unique(seg))
        if n >= min_classes:
            interesting.append((tp, seg))
    return interesting


BACKENDS = ["sd2", "sd15_realistic", "sdxl"]
BACKEND_TITLES = {
    "sd2": "SD2 (512)",
    "sd15_realistic": "SD1.5 Realistic (512)",
    "sdxl": "SDXL (1024)",
}


def _prepare_triples(pair_name, cfg, n_samples=3, tile_filter=None):
    """Produce a list of (tile_path, before_pil, change_mask, prompt, meta)
    triples for the A/B test. Loads SegFormer + SAM 3, simulates one change
    event per sampled tile, then unloads both models from VRAM so the
    inpainting backends have maximum memory headroom.

    Args:
        pair_name: e.g. "pair_0000".
        cfg: ``Config`` instance.
        n_samples: how many tiles to randomly sample if ``tile_filter`` is None.
        tile_filter: optional tile stem (e.g. "before_x3584_y3136"). When
            provided, random sampling is skipped and only that tile is used.
    """
    raw = Path(cfg.data["raw_root"])
    pair_dir = raw / pair_name
    before_path = pair_dir / "before.jpg"
    if not before_path.exists():
        raise FileNotFoundError(f"{before_path} not found")

    tiles_out = Path(cfg.data["tiles_dir"]) / pair_name / "before"
    saved = save_tiles_for_image(
        before_path, tiles_out,
        tile_size=cfg.tiling.get("tile_size", 512),
        overlap=cfg.tiling.get("overlap", 64),
        min_nonempty_ratio=cfg.tiling.get("min_nonempty_ratio", 0.02),
    )
    print(f"  Tiles: {len(saved)}")

    print("  Loading SegFormer...")
    seg_model = get_segmentation_model(
        cfg.segmentation.get("active_model", "segformer"), cfg.segmentation
    )

    print("  Loading SAM3 (object detection)...")
    sam_cfg = cfg.segmentation.get("sam", {})
    sam_model = get_segmentation_model("sam", cfg.segmentation)
    detection_prompts = sam_cfg.get(
        "detection_prompts", ["rock", "person", "car", "box", "bag", "bush"]
    )
    detection_score = sam_cfg.get("detection_score_threshold", 0.30)

    if tile_filter:
        matches = [tp for tp in saved if Path(tp).stem == tile_filter]
        if not matches:
            raise FileNotFoundError(
                f"Tile '{tile_filter}' not found under {tiles_out}. "
                f"Available tiles start with: "
                f"{[Path(t).stem for t in saved[:3]]}..."
            )
        print(f"  Pinned to tile: {tile_filter}")
        tile = Image.open(matches[0]).convert("RGB")
        seg = seg_model.segment(tile)
        sampled = [(matches[0], seg)]
    else:
        print("  Pre-filtering tiles...")
        interesting = _pre_filter_tiles(saved, seg_model, min_classes=2)
        print(f"  Interesting tiles (>= 2 classes): {len(interesting)} / {len(saved)}")

        if not interesting:
            return []

        rng = random.Random(42)
        n_samples = min(n_samples, len(interesting))
        sampled = rng.sample(interesting, n_samples)

    app_prob = cfg.synthetic.get("appearance_prob", 0.20)
    triples = []
    for i, (tile_path, seg) in enumerate(sampled):
        name = Path(tile_path).stem
        print(f"\n  [{i+1}/{n_samples}] {name}")

        before = Image.open(tile_path).convert("RGB")

        print("    SAM3 detection...")
        detections = sam_model.detect_objects(
            before, prompts=detection_prompts, min_score=detection_score,
        )
        print(f"    SAM3 found {len(detections)} objects")

        tile_rng = random.Random(42 + i)
        result = simulate_change(
            seg, rng=tile_rng, appearance_prob=app_prob,
            detected_objects=detections,
        )
        if result is None:
            print("    No change possible, skipping")
            continue

        change_mask, prompt, meta = result
        print(f"    Event: {meta['event']}")
        print(f"    Prompt: {prompt[:80]}...")
        triples.append((tile_path, before, change_mask, prompt, meta))

    del seg_model, sam_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return triples


def _run_backend(backend, cfg, triples, seed_base=42):
    """Load one inpainting backend, run it on every triple, unload. Returns
    a dict mapping tile_path -> synthetic_after_pil."""
    print(f"\n  === Backend: {backend} ===")
    inpaint = build_inpainter_from_cfg(cfg.inpainting, backend_override=backend)

    results = {}
    for i, (tile_path, before, change_mask, prompt, _meta) in enumerate(triples):
        name = Path(tile_path).stem
        print(f"    [{i+1}/{len(triples)}] {name} -> inpainting...")
        try:
            synth_after = inpaint.inpaint(
                before, change_mask, prompt, seed=seed_base + i,
            )
            results[str(tile_path)] = synth_after
        except Exception as e:
            print(f"      ERROR: {e}")
            results[str(tile_path)] = None

    inpaint.cleanup()
    del inpaint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _mask_to_rgb(change_mask, size):
    mask_pil = Image.fromarray((change_mask.astype(np.uint8) * 255)).convert("RGB")
    if mask_pil.size != size:
        mask_pil = mask_pil.resize(size, Image.NEAREST)
    return mask_pil


def main():
    cfg = Config("src/config.yaml")
    import yaml
    yml = yaml.safe_load(cfg.path.open().read())
    default_pair = yml.get("testing", {}).get("test_pair", "pair_0000")

    parser = argparse.ArgumentParser(
        description="A/B compare inpainting backends on one or more tiles.",
    )
    parser.add_argument(
        "pair_name", nargs="?", default=default_pair,
        help=f"Pair directory name under raw_root (default: {default_pair}).",
    )
    parser.add_argument(
        "--tile", default=None,
        help="Optional tile stem to pin, e.g. 'before_x3584_y3136'. "
             "When set, random sampling is skipped and only this tile is used.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of tiles to randomly sample when --tile is not set.",
    )
    args = parser.parse_args()

    pair_name = args.pair_name
    tag = f"{pair_name}" + (f" tile={args.tile}" if args.tile else "")
    print(f"=== A/B inpaint compare on {tag} ===")

    out_dir = Path(cfg.data["synthetic_dir"]) / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = _prepare_triples(
        pair_name, cfg, n_samples=args.n_samples, tile_filter=args.tile,
    )
    if not triples:
        print("  No triples produced, exiting.")
        return

    per_backend = {}
    for backend in BACKENDS:
        per_backend[backend] = _run_backend(backend, cfg, triples)

    print("\n  Building comparison grids...")
    for (tile_path, before, change_mask, prompt, meta) in triples:
        name = Path(tile_path).stem
        change_overlay = _contour_overlay(before, change_mask)

        panels = [before, change_overlay]
        titles = ["1. Before", "2. Change Region"]

        for backend in BACKENDS:
            result = per_backend[backend].get(str(tile_path))
            if result is None:
                placeholder = Image.new("RGB", before.size, (60, 30, 30))
                panels.append(placeholder)
                titles.append(f"{BACKEND_TITLES[backend]} (FAILED)")
            else:
                if result.size != before.size:
                    result = result.resize(before.size, Image.LANCZOS)
                panels.append(result)
                titles.append(BACKEND_TITLES[backend])

        panels.append(_mask_to_rgb(change_mask, before.size))
        titles.append("Change Mask")

        grid_path = out_dir / f"{name}_compare.png"
        build_grid(panels, titles, grid_path)
        print(f"    {grid_path}  (event={meta['event']})")

    print(f"\n=== Done. Comparison grids in {out_dir} ===")


if __name__ == "__main__":
    main()
