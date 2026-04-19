"""Generate a full-resolution synthetic (before, after, change_mask) triplet.

Object-centric pipeline:
  1. Scan the full 8K image with 1024x1024 crops using SAM3
  2. Select the best 1-3 objects (visibility + spatial spread)
  3. For each object, crop a padded region from the full image,
     run SD2 inpainting with the SAM3 mask, and feathered-paste back
  4. Produce overview comparison

Usage:
    python src/scripts/generate_pair.py                # uses pair from config
    python src/scripts/generate_pair.py pair_0005      # specific pair
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from pipeline.config import Config
from pipeline.segmentation import get_segmentation_model
from pipeline.inpainting import build_inpainter_from_cfg
from pipeline.synthetic import (
    select_best_objects,
    generate_full_image_pair,
)
from pipeline.io import write_json

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


def _thumb(img, width):
    aspect = img.height / img.width
    h = int(width * aspect)
    return img.resize((width, h), Image.LANCZOS)


def _diff_overlay(before_pil, after_pil, alpha=0.5):
    before_arr = np.array(before_pil).astype(np.float32)
    after_arr = np.array(after_pil).astype(np.float32)
    diff = np.abs(before_arr - after_arr).mean(axis=2)
    if diff.max() > 0:
        diff_norm = np.clip(diff / diff.max() * 255, 0, 255).astype(np.uint8)
    else:
        diff_norm = np.zeros_like(diff, dtype=np.uint8)

    overlay = np.array(after_pil).copy()
    mask = diff_norm > 15
    red = np.array([255, 60, 60], dtype=np.float32)
    overlay[mask] = (overlay[mask] * (1 - alpha) + red * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def _contour_overlay(base, mask_bool, color=(0, 255, 0), thickness=3, fill_alpha=0.15):
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


def build_overview(before_full, after_full, change_mask_full,
                   object_crops, overview_width, out_path):
    """Build a comparison overview with full-image thumbnails + object crops."""
    tw = overview_width
    before_t = _thumb(before_full, tw)
    after_t = _thumb(after_full, tw)
    mask_t = _thumb(change_mask_full.convert("RGB"), tw)

    mask_bool_t = np.array(_thumb(change_mask_full, tw)) > 127
    overlay_t = _contour_overlay(after_t, mask_bool_t)
    diff_t = _diff_overlay(before_t, after_t, alpha=0.5)

    fnt = _font(22)
    th = before_t.height

    full_panels = [before_t, after_t, overlay_t, diff_t]
    full_titles = ["Before (Full)", "Synthetic After (Full)",
                   "Change Overlay", "Pixel Difference"]

    cols = len(full_panels)
    row_h = th + TITLE_H
    total_h = row_h + PAD

    crop_size = 400
    if object_crops:
        crop_row_h = crop_size + TITLE_H
        total_h += crop_row_h + PAD

    canvas_w = cols * tw + (cols + 1) * PAD
    canvas = Image.new("RGB", (canvas_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for i, (img, title) in enumerate(zip(full_panels, full_titles)):
        x = PAD + i * (tw + PAD)
        y = PAD
        bbox = draw.textbbox((0, 0), title, font=fnt)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (tw - text_w) // 2, y + 8), title, fill=WHITE, font=fnt)
        canvas.paste(img, (x, y + TITLE_H))

    if object_crops:
        crop_y = row_h + PAD
        fnt_small = _font(14)
        for i, info in enumerate(object_crops):
            if i >= cols:
                break
            before_c = info["before_crop"].resize(
                (crop_size, crop_size), Image.LANCZOS)
            after_c = info["after_crop"].resize(
                (crop_size, crop_size), Image.LANCZOS)

            pair_w = crop_size * 2 + PAD
            x = PAD + i * (pair_w + PAD * 2)
            if x + pair_w > canvas_w:
                break

            title = f"{info['label']} (score={info['score']:.2f})"
            bbox = draw.textbbox((0, 0), title, font=fnt)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (pair_w - text_w) // 2, crop_y + 8),
                      title, fill=WHITE, font=fnt)

            canvas.paste(before_c, (x, crop_y + TITLE_H))
            canvas.paste(after_c, (x + crop_size + PAD, crop_y + TITLE_H))

            draw.text((x + 4, crop_y + TITLE_H + 4), "Before",
                      fill=(200, 200, 200), font=fnt_small)
            draw.text((x + crop_size + PAD + 4, crop_y + TITLE_H + 4), "After",
                      fill=(200, 200, 200), font=fnt_small)

    canvas.save(out_path, quality=95)


def main():
    cfg = Config("src/config.yaml")
    yml = yaml.safe_load(cfg.path.open().read())

    pair_name = (sys.argv[1] if len(sys.argv) > 1
                 else yml.get("testing", {}).get("test_pair", "pair_0000"))

    raw = Path(cfg.data["raw_root"])
    pair_dir = raw / pair_name
    before_path = pair_dir / "before.jpg"
    if not before_path.exists():
        print(f"Error: {before_path} not found")
        return

    before_full = Image.open(before_path).convert("RGB")
    full_size = before_full.size
    print(f"=== {pair_name} ({full_size[0]}x{full_size[1]}) ===")

    # --- 1. Load SAM3 ---
    print("  Loading SAM3...")
    sam_cfg = cfg.segmentation.get("sam", {})
    sam_model = get_segmentation_model("sam", cfg.segmentation)
    detection_prompts = sam_cfg.get("detection_prompts",
                                    ["rock", "person", "car", "box", "bag", "bush"])
    detection_score = sam_cfg.get("detection_score_threshold", 0.30)
    scan_tile_size = sam_cfg.get("scan_tile_size", 1024)
    scan_overlap = sam_cfg.get("scan_overlap", 128)

    # --- 2. Scan full image with 1024 crops, select best objects ---
    syn_cfg = cfg.synthetic
    asm_cfg = cfg.assembler
    max_objects = syn_cfg.get("max_changes", asm_cfg.get("max_changed_tiles", 3))
    min_obj_dist = syn_cfg.get("min_object_distance", asm_cfg.get("min_tile_distance", 2000))
    max_per_label = syn_cfg.get("max_per_label", 0)
    variance_thresh = asm_cfg.get("variance_prefilter", 500)
    max_dets_crop = asm_cfg.get("max_detections_per_tile", 3)

    print(f"  Scanning full image with {scan_tile_size}x{scan_tile_size} "
          f"crops (overlap={scan_overlap})...")
    print(f"  Config: max_changes={max_objects}, max_per_label={max_per_label}, "
          f"min_distance={min_obj_dist}px")
    best_objects = select_best_objects(
        before_full, sam_model,
        max_objects=max_objects,
        scan_tile_size=scan_tile_size,
        scan_overlap=scan_overlap,
        detection_prompts=detection_prompts,
        detection_score=detection_score,
        variance_threshold=variance_thresh,
        min_object_distance=min_obj_dist,
        max_detections_per_crop=max_dets_crop,
        max_per_label=max_per_label,
    )

    print(f"  Selected {len(best_objects)} objects for removal:")
    for obj in best_objects:
        cx, cy = obj["centroid_fullimg"]
        bx1, by1, bx2, by2 = obj["bbox_fullimg"]
        print(f"    {obj['label']} (score={obj['score']:.2f}, "
              f"vis={obj['visibility']:.2f}) "
              f"center=({cx},{cy}) bbox={bx2-bx1}x{by2-by1}")

    if not best_objects:
        print("  No objects detected. Exiting.")
        return

    # --- 3. Load inpainting backend ---
    backend = cfg.inpainting.get("backend", "sd2")
    blend_mode = cfg.inpainting.get("blend_mode", "poisson")
    print(f"  Loading inpainting backend: {backend} (blend_mode={blend_mode})...")
    inpaint = build_inpainter_from_cfg(cfg.inpainting)

    # --- 4. Load SegFormer for background context ---
    print("  Loading SegFormer (background context)...")
    seg_model = get_segmentation_model(
        cfg.segmentation.get("active_model", "segformer"), cfg.segmentation
    )

    # --- 5. Object-centric inpainting (shared with generate_dataset.py) ---
    seed = cfg.synthetic.get("seed", 42)
    result = generate_full_image_pair(
        before_full=before_full,
        changes=best_objects,
        inpaint_model=inpaint,
        seg_model=seg_model,
        seed=seed,
        verbose=True,
        collect_crops=True,
    )
    canvas = result["after"]
    mask_pil = result["change_mask"]
    meta_entries = result["meta_entries"]
    object_crops = result["object_crops"] or []

    inpaint.cleanup()

    if not meta_entries:
        print("  No changes produced. Exiting.")
        return

    # --- 6. Save outputs ---
    out_dir = Path(cfg.data["synthetic_dir"]) / pair_name / "full"
    out_dir.mkdir(parents=True, exist_ok=True)

    before_out = out_dir / "before.jpg"
    shutil.copy2(before_path, before_out)

    after_out = out_dir / "synthetic_after.jpg"
    canvas.save(after_out, quality=95)
    print(f"\n  Saved: {after_out}")

    mask_out = out_dir / "change_mask.png"
    mask_pil.save(mask_out)
    print(f"  Saved: {mask_out}")

    meta_out = out_dir / "meta.json"
    write_json({
        "pair": pair_name,
        "full_size": list(full_size),
        "objects_changed": len(meta_entries),
        "pipeline": "object_centric_v2",
        "sam3_scan_tile": scan_tile_size,
        "changes": meta_entries,
    }, meta_out)

    # --- 7. Overview ---
    overview_width = asm_cfg.get("overview_width", 2048)
    overview_path = out_dir / "overview.png"
    build_overview(
        before_full, canvas, mask_pil,
        object_crops, overview_width, overview_path,
    )
    print(f"  Saved: {overview_path}")

    print(f"\n=== Done! Full outputs in {out_dir} ===")


if __name__ == "__main__":
    main()
