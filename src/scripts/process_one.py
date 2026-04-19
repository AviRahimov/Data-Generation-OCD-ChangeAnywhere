"""End-to-end test: tile one pair, detect objects with SAM3, segment with
SegFormer, simulate changes (80% disappearance), inpaint with SD2 using
feathered masks, and produce comparison grids for visual inspection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from pipeline.config import Config
from pipeline.tiler import save_tiles_for_image
from pipeline.segmentation import get_segmentation_model
from pipeline.inpainting import build_inpainter_from_cfg
from pipeline.change_simulator import simulate_change
from pipeline.prompt_templates import get_background_label, TERRAIN_BACKGROUND_CLASSES

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


def _colorize_seg(seg):
    rng = np.random.RandomState(42)
    n = max(seg.max() + 1, 256)
    cmap = rng.randint(40, 255, size=(n, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]
    return Image.fromarray(cmap[seg % len(cmap)])


def _contour_overlay(base, mask_bool, color=(255, 40, 40), thickness=2, fill_alpha=0.25):
    """Draw a thin contour outline around masked regions with light semi-transparent fill."""
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


def _visualize_detections(base, detections):
    """Draw SAM3 detections as colored overlays with labels."""
    arr = np.array(base.copy())
    colors = [
        (0, 255, 0), (255, 255, 0), (0, 200, 255),
        (255, 0, 255), (255, 128, 0), (0, 255, 128),
    ]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        mask = det["mask"]
        overlay_color = np.array(color, dtype=np.float32)
        arr[mask] = (arr[mask] * 0.6 + overlay_color * 0.4).astype(np.uint8)

    vis = Image.fromarray(arr)
    draw = ImageDraw.Draw(vis)
    fnt = _font(14)
    for i, det in enumerate(detections):
        ys, xs = np.where(det["mask"])
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        label = f"{det['label']} ({det['score']:.2f})"
        draw.text((cx, cy), label, fill=(255, 255, 255), font=fnt)

    return vis


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


def build_gt_overview(pair_dir, out_path, thumb_width=1024):
    """Build an overview using the REAL GT data from the original pair."""
    before_path = pair_dir / "before.jpg"
    after_path = pair_dir / "after.jpg"
    mask_path = pair_dir / "after_binary_mask.png"
    polygons_path = pair_dir / "after_with_polygons.jpg"

    if not all(p.exists() for p in [before_path, after_path, mask_path]):
        print("  Skipping GT overview: missing files")
        return

    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    aspect = before.height / before.width
    thumb_h = int(thumb_width * aspect)

    before_t = before.resize((thumb_width, thumb_h), Image.LANCZOS)
    after_t = after.resize((thumb_width, thumb_h), Image.LANCZOS)
    mask_t = mask.resize((thumb_width, thumb_h), Image.NEAREST)

    mask_bool = np.array(mask_t) > 127
    gt_overlay = _contour_overlay(after_t, mask_bool, color=(0, 255, 0), thickness=3, fill_alpha=0.15)

    panels = [before_t, after_t, gt_overlay]
    titles = ["Before", "After (Ground Truth)", "GT Changes (Green Contour)"]

    if polygons_path.exists():
        poly_t = Image.open(polygons_path).convert("RGB").resize((thumb_width, thumb_h), Image.LANCZOS)
        panels.append(poly_t)
        titles.append("Original Annotated Polygons")

    build_grid(panels, titles, out_path)
    print(f"  GT overview: {out_path}")


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


def main():
    cfg = Config("src/config.yaml")
    yml = yaml.safe_load(cfg.path.open().read())
    pair_name = yml.get("testing", {}).get("test_pair", "pair_0000")

    raw = Path(cfg.data["raw_root"])
    pair_dir = raw / pair_name
    before_path = pair_dir / "before.jpg"
    if not before_path.exists():
        print(f"Error: {before_path} not found")
        return

    print(f"=== {pair_name} ===")

    # --- Tile ---
    tiles_out = Path(cfg.data["tiles_dir"]) / pair_name / "before"
    saved = save_tiles_for_image(
        before_path, tiles_out,
        tile_size=cfg.tiling.get("tile_size", 512),
        overlap=cfg.tiling.get("overlap", 64),
        min_nonempty_ratio=cfg.tiling.get("min_nonempty_ratio", 0.02),
    )
    print(f"  Tiles: {len(saved)}")

    out_dir = Path(cfg.data["synthetic_dir"]) / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- GT overview from real annotations ---
    build_gt_overview(pair_dir, out_dir / f"{pair_name}_gt_overview.png")

    # --- Load models ---
    print("  Loading SegFormer...")
    seg_model = get_segmentation_model(
        cfg.segmentation.get("active_model", "segformer"), cfg.segmentation
    )

    print("  Loading SAM3 (object detection)...")
    sam_cfg = cfg.segmentation.get("sam", {})
    sam_model = get_segmentation_model("sam", cfg.segmentation)
    detection_prompts = sam_cfg.get("detection_prompts",
                                    ["rock", "person", "car", "box", "bag", "bush"])
    detection_score = sam_cfg.get("detection_score_threshold", 0.30)

    backend = cfg.inpainting.get("backend", "sd2")
    print(f"  Loading inpainting backend: {backend}...")
    inpaint = build_inpainter_from_cfg(cfg.inpainting)

    # --- Pre-filter tiles to those with >= 2 semantic classes ---
    print("  Pre-filtering tiles (segmenting all to find interesting ones)...")
    interesting = _pre_filter_tiles(saved, seg_model, min_classes=2)
    print(f"  Interesting tiles (>= 2 classes): {len(interesting)} / {len(saved)}")

    if not interesting:
        print("  No interesting tiles found. Exiting.")
        inpaint.cleanup()
        return

    # --- Pick sample tiles from the interesting set ---
    rng = random.Random(42)
    n_samples = min(5, len(interesting))
    sampled = rng.sample(interesting, n_samples)

    for i, (tile_path, seg) in enumerate(sampled):
        name = Path(tile_path).stem
        n_cls = len(np.unique(seg))
        print(f"\n  [{i+1}/{n_samples}] {name} ({n_cls} classes)")

        before = Image.open(tile_path).convert("RGB")
        seg_vis = _colorize_seg(seg)

        # --- SAM3 object detection ---
        print("    Running SAM3 detection...")
        detections = sam_model.detect_objects(
            before, prompts=detection_prompts, min_score=detection_score,
        )
        det_summary = [f"{d['label']}({d['score']:.2f})" for d in detections]
        print(f"    SAM3 found {len(detections)} objects: {det_summary}")

        det_vis = _visualize_detections(before, detections)

        # --- Simulate change (80% disappearance / 20% appearance) ---
        tile_rng = random.Random(42 + i)
        app_prob = cfg.synthetic.get("appearance_prob", 0.20)
        result = simulate_change(
            seg, rng=tile_rng, appearance_prob=app_prob,
            detected_objects=detections,
        )
        if result is None:
            print("    No change possible on this tile, skipping")
            continue

        change_mask, prompt, meta = result
        print(f"    Event: {meta['event']}, prompt: {prompt[:80]}...")

        change_mask_pil = Image.fromarray((change_mask.astype(np.uint8) * 255))

        # --- Inpaint with feathered blending ---
        print("    Inpainting (feathered)...")
        synth_after = inpaint.inpaint(
            before, change_mask, prompt, seed=42 + i,
        )

        overlay = _contour_overlay(before, change_mask)
        after_overlay = _contour_overlay(synth_after, change_mask)

        panels = [
            before,
            seg_vis,
            det_vis,
            overlay,
            synth_after,
            change_mask_pil.convert("RGB"),
            after_overlay,
        ]
        titles = [
            "1. Original Tile",
            "2. Semantic Map (SegFormer)",
            f"3. SAM3 Detections ({len(detections)})",
            "4. Change Region",
            "5. Synthetic After (SD2)",
            "6. Change Mask",
            "7. After + Change Overlay",
        ]
        grid_path = out_dir / f"{name}_grid.png"
        build_grid(panels, titles, grid_path)
        print(f"    Grid: {grid_path}")

    inpaint.cleanup()
    print(f"\n=== Done! Outputs in {out_dir} ===")


if __name__ == "__main__":
    main()
