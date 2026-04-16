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

import random
import shutil
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from pipeline.config import Config
from pipeline.segmentation import get_segmentation_model
from pipeline.inpainting import InpaintingModel
from pipeline.synthetic import select_best_objects
from pipeline.prompt_templates import get_disappearance_prompt, get_background_label
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


def _paste_inpainted(canvas, inpaint_result):
    """Paste an inpainted crop back onto the full-image canvas.

    With Poisson blending the crop edges already contain unchanged original
    pixels (object mask never reaches the padded crop boundary), so a direct
    paste is seamless. A minimal 8px safety feather is applied to avoid any
    sub-pixel rounding artifacts at the very edge of the crop.
    """
    crop_pil = inpaint_result["inpainted_crop"]
    x1, y1, x2, y2 = inpaint_result["paste_box"]
    crop_w, crop_h = x2 - x1, y2 - y1

    crop = np.array(crop_pil).astype(np.float32)
    region = np.array(canvas.crop((x1, y1, x2, y2))).astype(np.float32)

    margin = 8
    m = min(margin, crop_h // 2, crop_w // 2)
    alpha = np.ones((crop_h, crop_w), dtype=np.float32)
    for i in range(m):
        v = (i + 1) / (m + 1)
        alpha[i, :] = np.minimum(alpha[i, :], v)
        alpha[-(i + 1), :] = np.minimum(alpha[-(i + 1), :], v)
        alpha[:, i] = np.minimum(alpha[:, i], v)
        alpha[:, -(i + 1)] = np.minimum(alpha[:, -(i + 1)], v)

    alpha_3 = alpha[..., np.newaxis]
    blended = region * (1.0 - alpha_3) + crop * alpha_3
    canvas.paste(Image.fromarray(blended.astype(np.uint8)), (x1, y1))


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

    # --- 3. Load SD2-Inpainting ---
    blend_mode = cfg.inpainting.get("blend_mode", "poisson")
    print(f"  Loading SD2-Inpainting (blend_mode={blend_mode})...")
    inpaint = InpaintingModel(
        model_id=cfg.inpainting.get("model_id"),
        device=cfg.inpainting.get("device", "cuda"),
        num_inference_steps=cfg.inpainting.get("num_inference_steps", 50),
        guidance_scale=cfg.inpainting.get("guidance_scale", 12.0),
        strength=cfg.inpainting.get("strength", 1.0),
        mask_blur_radius=cfg.inpainting.get("mask_blur_radius", 12),
        mask_dilate_px=cfg.inpainting.get("mask_dilate_px", 8),
        blend_mode=blend_mode,
    )

    # --- 4. Load SegFormer for background context ---
    print("  Loading SegFormer (background context)...")
    seg_model = get_segmentation_model(
        cfg.segmentation.get("active_model", "segformer"), cfg.segmentation
    )

    # --- 5. Object-centric inpainting ---
    canvas = before_full.copy()
    change_mask_full = np.zeros(
        (full_size[1], full_size[0]), dtype=bool
    )
    object_crops = []
    meta_entries = []
    seed = cfg.synthetic.get("seed", 42)

    for i, obj in enumerate(best_objects):
        label = obj["label"]
        cx, cy = obj["centroid_fullimg"]
        bbox = obj["bbox_fullimg"]
        bx1, by1, bx2, by2 = bbox
        bw, bh = bx2 - bx1, by2 - by1

        print(f"\n  [{i+1}/{len(best_objects)}] Inpainting {label} "
              f"at ({cx},{cy}), size {bw}x{bh}...")

        seg_crop_x1 = max(bx1 - 64, 0)
        seg_crop_y1 = max(by1 - 64, 0)
        seg_crop_x2 = min(bx2 + 64, full_size[0])
        seg_crop_y2 = min(by2 + 64, full_size[1])
        seg_tile = before_full.crop((seg_crop_x1, seg_crop_y1,
                                     seg_crop_x2, seg_crop_y2))
        seg_map = seg_model.segment(seg_tile)

        from pipeline.change_simulator import _find_background_region
        bg_regions = _find_background_region(seg_map)
        surround_class = bg_regions[0][0] if bg_regions else 13
        rng = random.Random(seed + i)
        prompt = get_disappearance_prompt(surround_class, rng)

        print(f"    Background: {get_background_label(surround_class)}")
        print(f"    Prompt: {prompt[:80]}...")

        before_crop_box = (
            max(bx1 - int(bw * 0.3), 0),
            max(by1 - int(bh * 0.3), 0),
            min(bx2 + int(bw * 0.3), full_size[0]),
            min(by2 + int(bh * 0.3), full_size[1]),
        )
        before_crop = before_full.crop(before_crop_box)

        result = inpaint.inpaint_object(
            full_image=before_full,
            obj_mask_fullimg=obj["mask_fullimg"],
            prompt=prompt,
            bbox_fullimg=bbox,
            pad_ratio=0.3,
            feather_margin=48,
            seed=seed + i,
        )

        _paste_inpainted(canvas, result)

        px1, py1, px2, py2 = result["paste_box"]
        local_mask = result["mask_crop"]
        change_mask_full[py1:py2, px1:px2] |= local_mask

        after_crop = canvas.crop(before_crop_box)
        object_crops.append({
            "before_crop": before_crop,
            "after_crop": after_crop,
            "label": label,
            "score": obj["score"],
        })
        meta_entries.append({
            "label": label,
            "score": round(obj["score"], 3),
            "visibility": round(obj["visibility"], 3),
            "centroid": [cx, cy],
            "bbox": list(bbox),
            "paste_box": list(result["paste_box"]),
            "prompt": prompt,
            "bg_class": surround_class,
        })

        print(f"    Done. Paste region: {px2-px1}x{py2-py1}")

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

    mask_pil = Image.fromarray(change_mask_full.astype(np.uint8) * 255)
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
