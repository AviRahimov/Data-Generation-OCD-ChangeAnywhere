"""Batch-generate synthetic (before, after, change_mask) pairs.

Reads a folder of aerial images, loops N times, and on each iteration:
  1. Randomly picks a source image (balanced sampler: each input gets either
     floor(N/K) or ceil(N/K) draws, so no image is over-picked)
  2. Randomly picks a number of objects k ~ Uniform[min_objects, max_objects]
  3. For each of the k slots, flips the appearance_prob coin to decide
     'add a new object' vs 'remove an existing object'
  4. Calls select_best_objects() for removals and select_appearance_locations()
     for appearances, then composites the final pair via generate_full_image_pair
  5. Writes gen_{i:05d}/{before.jpg, synthetic_after.jpg, change_mask.png,
     comparison.png, meta.json, overview.png?} and appends a row to
     manifest.csv. comparison.png is a 2x2 grid (before / after / mask /
     overlay) with large titles for quick at-a-glance review.

Models (SAM 3 + SegFormer + inpainter) are loaded ONCE at startup and reused
for every iteration -- loading them per-sample would dominate runtime.

Auto-detects two input layouts:
  - pair_*/before.jpg (our repo layout)
  - a flat folder of .jpg / .jpeg / .png files

Example:
    python src/scripts/generate_dataset.py `
        --input-dir src/data/original_OCD_dataset `
        --n-images 100 `
        --output-dir src/data/workspace/dataset_v1 `
        --min-objects 1 --max-objects 20 --seed 42
"""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # src/     -> for `pipeline.*`
sys.path.insert(0, str(_HERE.parents[0]))  # scripts/ -> for `generate_pair.build_overview`

import argparse
import csv
import random
import shutil
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from pipeline.config import Config
from pipeline.segmentation import get_segmentation_model
from pipeline.inpainting import build_inpainter_from_cfg
from pipeline.synthetic import (
    select_best_objects,
    select_appearance_locations,
    generate_full_image_pair,
)
from pipeline.io import write_json


IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-generate synthetic change-detection pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Folder of input images. Supports pair_*/before.jpg "
                        "layout OR a flat folder of .jpg/.png files.")
    p.add_argument("--n-images", type=int, required=True,
                   help="How many synthetic pairs to generate.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write gen_*/ subfolders and manifest.csv.")
    p.add_argument("--min-objects", type=int, default=1,
                   help="Minimum number of object changes per generated pair.")
    p.add_argument("--max-objects", type=int, default=20,
                   help="Maximum number of object changes per generated pair.")
    p.add_argument("--appearance-prob", type=float, default=None,
                   help="P(add) per object slot (rest is remove). "
                        "If omitted, uses synthetic.appearance_prob from config.")
    p.add_argument("--backend", choices=["sd2", "sd15_realistic", "sdxl"],
                   default=None,
                   help="Override the inpainting.backend from config.")
    p.add_argument("--seed", type=int, default=42,
                   help="Master RNG seed; makes sampling reproducible.")
    p.add_argument("--save-overview", action="store_true",
                   help="Also save overview.png per sample (object close-ups, "
                        "doubles disk use).")
    p.add_argument("--no-comparison", action="store_true",
                   help="Skip the default comparison.png (before / after / "
                        "mask / overlay 2x2 grid) for each sample.")
    p.add_argument("--comparison-width", type=int, default=1600,
                   help="Per-panel width of comparison.png in pixels. "
                        "Final image is ~2x this wide.")
    p.add_argument("--config", type=Path, default=Path("src/config.yaml"),
                   help="Path to the pipeline config YAML.")
    p.add_argument("--overview-width", type=int, default=None,
                   help="Width of saved overview thumbnails (defaults to "
                        "assembler.overview_width in config).")
    return p.parse_args()


def discover_inputs(root: Path):
    """Return a list of Paths pointing to 'before' images.

    Priority:
      1. If root contains pair_*/before.jpg folders, use those.
      2. Otherwise recursively glob image files under root.
    Raises FileNotFoundError if neither layout yields anything.
    """
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    pair_befores = sorted(root.glob("pair_*/before.jpg"))
    if pair_befores:
        return pair_befores

    flat = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXTS:
            flat.append(p)
    flat.sort()
    if flat:
        return flat

    raise FileNotFoundError(
        f"No images found under {root}. Expected either pair_*/before.jpg or "
        f"a folder of .jpg/.jpeg/.png files."
    )


def balanced_draws(inputs, n, rng):
    """Return a length-n list where each item in ``inputs`` appears either
    floor(n/k) or ceil(n/k) times. Order is shuffled."""
    k = len(inputs)
    base, rem = divmod(n, k)
    draws = []
    for path in inputs:
        draws.extend([path] * base)
    extras = rng.sample(list(inputs), rem) if rem > 0 else []
    draws.extend(extras)
    rng.shuffle(draws)
    return draws


def _bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


_COMPARISON_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _comparison_font(size):
    """Load a readable bold font across platforms, falling back to PIL default
    (which is tiny but never missing)."""
    for candidate in _COMPARISON_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _thumb(img, width):
    aspect = img.height / img.width
    return img.resize((width, int(width * aspect)), Image.LANCZOS)


def _contour_overlay_on_before(before_pil, mask_bool,
                                color=(255, 60, 60), thickness=5,
                                fill_alpha=0.25):
    """Paint the change region in red on the BEFORE image so reviewers can
    instantly see where the edits landed even if the synthetic output looks
    subtle."""
    from scipy import ndimage
    arr = np.array(before_pil.copy())
    if mask_bool.shape != arr.shape[:2]:
        mask_pil = Image.fromarray(mask_bool.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(
            (arr.shape[1], arr.shape[0]), Image.NEAREST)
        mask_bool = np.array(mask_pil) > 127

    if fill_alpha > 0:
        overlay_color = np.array(color, dtype=np.float32)
        arr[mask_bool] = (
            arr[mask_bool] * (1 - fill_alpha) + overlay_color * fill_alpha
        ).astype(np.uint8)

    if mask_bool.any():
        eroded = ndimage.binary_erosion(mask_bool, iterations=thickness)
        contour = mask_bool & ~eroded
        arr[contour] = color

    return Image.fromarray(arr)


def build_comparison(before_pil, after_pil, mask_pil, out_path,
                     panel_width=1600, title_font_size=42):
    """Render a 2x2 comparison grid for quick visual QA.

    Layout::

        +----------------------+----------------------+
        |       BEFORE         |   SYNTHETIC AFTER    |
        +----------------------+----------------------+
        |      CHANGE MASK     |    CHANGE OVERLAY    |
        +----------------------+----------------------+

    Titles sit on a dark banner above each panel and use a large bold font
    (``title_font_size`` default 42 px) so the image is legible even when
    viewed at thumbnail size in a file browser.
    """
    pw = panel_width
    before_t = _thumb(before_pil, pw)
    after_t = _thumb(after_pil, pw)
    mask_rgb = mask_pil.convert("RGB") if mask_pil.mode != "RGB" else mask_pil
    mask_t = _thumb(mask_rgb, pw)

    mask_bool_full = np.array(mask_pil.convert("L")) > 127
    overlay_full = _contour_overlay_on_before(before_pil, mask_bool_full)
    overlay_t = _thumb(overlay_full, pw)

    panel_h = before_t.height
    title_h = title_font_size + 20
    pad = 8
    bg = (28, 28, 32)
    fg = (245, 245, 245)

    cols, rows = 2, 2
    canvas_w = cols * pw + (cols + 1) * pad
    canvas_h = rows * (panel_h + title_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)
    fnt = _comparison_font(title_font_size)

    cells = [
        ("BEFORE", before_t),
        ("SYNTHETIC AFTER", after_t),
        ("CHANGE MASK (binary)", mask_t),
        ("CHANGE OVERLAY (red = edited region)", overlay_t),
    ]

    for idx, (title, img) in enumerate(cells):
        col = idx % cols
        row = idx // cols
        x = pad + col * (pw + pad)
        y = pad + row * (panel_h + title_h + pad)

        draw.rectangle(
            [x, y, x + pw, y + title_h],
            fill=(44, 44, 52),
        )
        try:
            bbox = draw.textbbox((0, 0), title, font=fnt)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = fnt.getsize(title)
        tx = x + (pw - text_w) // 2
        ty = y + (title_h - text_h) // 2 - 2
        draw.text((tx, ty), title, fill=fg, font=fnt)

        canvas.paste(img, (x, y + title_h))

    canvas.save(out_path, quality=92)


def main():
    args = parse_args()
    cfg = Config(str(args.config))
    master_rng = random.Random(args.seed)

    if args.min_objects < 1:
        raise ValueError("--min-objects must be >= 1")
    if args.max_objects < args.min_objects:
        raise ValueError("--max-objects must be >= --min-objects")

    inputs = discover_inputs(args.input_dir)
    draws = balanced_draws(inputs, args.n_images, master_rng)
    k_inputs = len(inputs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    manifest_exists = manifest_path.exists()

    print(f"=== Batch dataset generation ===")
    print(f"  input-dir     : {args.input_dir}  ({k_inputs} images found)")
    print(f"  output-dir    : {args.output_dir}")
    print(f"  n-images      : {args.n_images}")
    print(f"  objects/image : {args.min_objects}..{args.max_objects}")
    print(f"  manifest      : {manifest_path}")

    appearance_prob = args.appearance_prob
    if appearance_prob is None:
        appearance_prob = cfg.synthetic.get("appearance_prob", 0.20)
    print(f"  appearance_prob: {appearance_prob:.2f} "
          f"(rest = {1.0 - appearance_prob:.2f} disappearance)")

    # --- Load models once ---
    print("\n  Loading SAM 3...")
    sam_cfg = cfg.segmentation.get("sam", {})
    sam_model = get_segmentation_model("sam", cfg.segmentation)
    detection_prompts = sam_cfg.get(
        "detection_prompts", ["rock", "person", "car", "box", "bag", "bush"])
    detection_score = sam_cfg.get("detection_score_threshold", 0.30)
    scan_tile_size = sam_cfg.get("scan_tile_size", 1024)
    scan_overlap = sam_cfg.get("scan_overlap", 128)

    print("  Loading SegFormer...")
    seg_model = get_segmentation_model(
        cfg.segmentation.get("active_model", "segformer"), cfg.segmentation)

    backend = args.backend or cfg.inpainting.get("backend", "sd2")
    print(f"  Loading inpainting backend: {backend}...")
    inpaint = build_inpainter_from_cfg(cfg.inpainting, backend_override=args.backend)

    syn_cfg = cfg.synthetic
    asm_cfg = cfg.assembler
    min_obj_dist = syn_cfg.get(
        "min_object_distance", asm_cfg.get("min_tile_distance", 2000))
    max_per_label = syn_cfg.get("max_per_label", 0)
    variance_thresh = asm_cfg.get("variance_prefilter", 500)
    max_dets_crop = asm_cfg.get("max_detections_per_tile", 3)
    overview_width = args.overview_width or asm_cfg.get("overview_width", 2048)

    # --- Manifest writer ---
    manifest_f = open(manifest_path, "a" if manifest_exists else "w",
                      newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_f)
    if not manifest_exists:
        manifest_writer.writerow([
            "id", "source_path", "n_requested", "n_applied",
            "n_add_requested", "n_add_applied",
            "n_remove_requested", "n_remove_applied",
            "labels", "status", "elapsed_s",
        ])
        manifest_f.flush()

    # --- Main loop ---
    try:
        for i, src_path in enumerate(tqdm(draws, desc="Generating", unit="pair")):
            gen_id = f"gen_{i:05d}"
            start = time.time()
            sample_rng = random.Random(args.seed + 1 + i)
            status = "ok"
            meta_entries = []
            n_total = sample_rng.randint(args.min_objects, args.max_objects)
            n_add_req = sum(1 for _ in range(n_total)
                            if sample_rng.random() < appearance_prob)
            n_rem_req = n_total - n_add_req

            try:
                before_full = Image.open(src_path).convert("RGB")

                removals = []
                if n_rem_req > 0:
                    removals = select_best_objects(
                        before_full, sam_model,
                        max_objects=n_rem_req,
                        scan_tile_size=scan_tile_size,
                        scan_overlap=scan_overlap,
                        detection_prompts=detection_prompts,
                        detection_score=detection_score,
                        variance_threshold=variance_thresh,
                        min_object_distance=min_obj_dist,
                        max_detections_per_crop=max_dets_crop,
                        max_per_label=max_per_label,
                    )
                    for r in removals:
                        r["kind"] = "disappearance"

                appearances = []
                if n_add_req > 0:
                    existing = [r["centroid_fullimg"] for r in removals]
                    appearances = select_appearance_locations(
                        before_full, seg_model,
                        max_count=n_add_req,
                        scan_tile_size=scan_tile_size,
                        min_object_distance=min_obj_dist,
                        existing_centroids=existing,
                        rng=sample_rng,
                    )

                changes = removals + appearances
                n_rem_applied_pre = len(removals)
                n_add_applied_pre = len(appearances)

                if not changes:
                    status = "empty"
                    result = {
                        "after": before_full.copy(),
                        "change_mask": Image.new(
                            "L", before_full.size, 0),
                        "meta_entries": [],
                        "object_crops": None,
                    }
                else:
                    result = generate_full_image_pair(
                        before_full=before_full,
                        changes=changes,
                        inpaint_model=inpaint,
                        seg_model=seg_model,
                        seed=args.seed + i,
                        verbose=False,
                        collect_crops=args.save_overview,
                    )

                meta_entries = result["meta_entries"]

                out_dir = args.output_dir / gen_id
                out_dir.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy2(src_path, out_dir / "before.jpg")
                except Exception:
                    before_full.save(out_dir / "before.jpg", quality=95)

                result["after"].save(out_dir / "synthetic_after.jpg", quality=95)
                result["change_mask"].save(out_dir / "change_mask.png")

                write_json({
                    "id": gen_id,
                    "source_path": str(src_path),
                    "full_size": list(before_full.size),
                    "n_requested": n_total,
                    "n_applied": len(meta_entries),
                    "n_add_requested": n_add_req,
                    "n_add_applied": sum(
                        1 for m in meta_entries if m["kind"] == "appearance"),
                    "n_remove_requested": n_rem_req,
                    "n_remove_applied": sum(
                        1 for m in meta_entries if m["kind"] == "disappearance"),
                    "backend": backend,
                    "seed": args.seed + i,
                    "changes": meta_entries,
                }, out_dir / "meta.json")

                if not args.no_comparison:
                    try:
                        build_comparison(
                            before_full, result["after"], result["change_mask"],
                            out_dir / "comparison.png",
                            panel_width=args.comparison_width,
                        )
                    except Exception as e:
                        print(f"\n  {gen_id}: comparison render failed ({e})")

                if args.save_overview and result.get("object_crops"):
                    from generate_pair import build_overview
                    try:
                        build_overview(
                            before_full, result["after"], result["change_mask"],
                            result["object_crops"], overview_width,
                            out_dir / "overview.png",
                        )
                    except Exception as e:
                        print(f"\n  {gen_id}: overview render failed ({e})")

            except Exception as e:
                status = f"error: {type(e).__name__}: {e}"
                print(f"\n  {gen_id}: {status}")
                traceback.print_exc()

            elapsed = time.time() - start

            n_add_applied = sum(
                1 for m in meta_entries if m.get("kind") == "appearance")
            n_rem_applied = sum(
                1 for m in meta_entries if m.get("kind") == "disappearance")
            labels = ";".join(m["label"] for m in meta_entries) if meta_entries else ""

            manifest_writer.writerow([
                gen_id, str(src_path), n_total, len(meta_entries),
                n_add_req, n_add_applied,
                n_rem_req, n_rem_applied,
                labels, status, f"{elapsed:.1f}",
            ])
            manifest_f.flush()
    finally:
        manifest_f.close()
        inpaint.cleanup()

    print(f"\n=== Done. Manifest: {manifest_path} ===")


if __name__ == "__main__":
    main()
