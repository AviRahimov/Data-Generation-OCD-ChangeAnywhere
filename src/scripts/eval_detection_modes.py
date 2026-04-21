"""Evaluate text-mode vs auto-mode detection across a sample of pairs.

Runs `select_best_objects` in BOTH detection modes on the full `before.jpg`
of each sampled pair, records per-pair detection counts and spatial overlaps,
and writes a CSV + printed summary so you (or the supervisor) can judge
whether auto mode provides useful "novel" coverage beyond the fixed text
prompt list.

Examples:
    python -u src/scripts/eval_detection_modes.py --num-pairs 20 \\
        --output src/data/workspace/eval_text_vs_auto.csv

    # Regenerate side-by-side images for specific pairs (CSV + JPGs):
    python -u src/scripts/eval_detection_modes.py \\
        --pairs pair_0003,pair_0014 \\
        --output src/data/workspace/eval_text_vs_auto_smoke.csv

    # Skip JPGs to save time / disk:
    python -u src/scripts/eval_detection_modes.py --vis-dir ""
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Line-buffer stdout so that per-pair progress is visible in real time when
# this script is redirected to a file (e.g. a background job).
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import argparse
import csv
import gc
import random
import time
from statistics import mean, median

import numpy as np

try:
    import torch
except ImportError:
    torch = None
import yaml
from PIL import Image, ImageDraw, ImageFont

from pipeline.config import Config
from pipeline.segmentation import get_segmentation_model
from pipeline.synthetic import select_best_objects


TEXT_COLOR = (255, 70, 70)   # red for text-mode detections
AUTO_COLOR = (70, 140, 255)  # blue for auto-mode detections
PANEL_TITLE_H = 56


def _font(size=24):
    for p in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _downscale_for_vis(pil_image, target_width=1200):
    """Resize to a manageable size for per-pair visualizations."""
    w, h = pil_image.size
    if w <= target_width:
        return pil_image, 1.0
    scale = target_width / w
    new_size = (target_width, int(h * scale))
    return pil_image.resize(new_size, Image.LANCZOS), scale


def _cuda_gc():
    """Free fragmented CUDA allocations between heavy steps (8 GB GPUs)."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _overlay_detections(base_pil, dets, color, scale):
    """Draw mask fills + bbox outlines + labels for a detection list.

    ``scale`` is the image downscale factor (dets are in full-image coords).
    """
    out = base_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(20)

    color_fill = (*color, 90)
    color_line = (*color, 255)

    for i, d in enumerate(dets, 1):
        mask = d.get("mask_fullimg")
        if mask is not None:
            mask_small = np.array(Image.fromarray(
                (mask.astype(np.uint8) * 255)
            ).resize(out.size, Image.NEAREST))
            alpha = np.zeros(out.size[::-1] + (4,), dtype=np.uint8)
            fill = np.array(color_fill, dtype=np.uint8)
            alpha[mask_small > 127] = fill
            overlay = Image.alpha_composite(overlay, Image.fromarray(alpha))
            draw = ImageDraw.Draw(overlay)

        bx1, by1, bx2, by2 = d["bbox_fullimg"]
        sx1, sy1 = int(bx1 * scale), int(by1 * scale)
        sx2, sy2 = int(bx2 * scale), int(by2 * scale)
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color_line, width=3)
        label = f"{i}. {d.get('label', '?')} {d.get('score', 0):.2f}"
        tw = (draw.textlength(label, font=font)
              if hasattr(draw, "textlength")
              else (draw.textbbox((0, 0), label, font=font)[2]
                    - draw.textbbox((0, 0), label, font=font)[0]))
        draw.rectangle([sx1, max(sy1 - 24, 0), sx1 + tw + 8, max(sy1, 24)],
                       fill=color_line)
        draw.text((sx1 + 4, max(sy1 - 22, 2)), label, fill=(255, 255, 255), font=font)

    out = Image.alpha_composite(out, overlay).convert("RGB")
    return out


def _build_comparison_panel(before_pil, text_dets, auto_dets, pair_id, scale):
    """Return a side-by-side (text | auto) comparison panel with titles."""
    title_font = _font(30)
    small_font = _font(22)

    left = _overlay_detections(before_pil, text_dets, TEXT_COLOR, scale)
    right = _overlay_detections(before_pil, auto_dets, AUTO_COLOR, scale)

    pw, ph = left.size
    gap = 12
    out_w = pw * 2 + gap
    out_h = ph + PANEL_TITLE_H + 36
    canvas = Image.new("RGB", (out_w, out_h), (25, 25, 25))

    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"{pair_id}", fill=(255, 255, 255), font=title_font)
    left_title = f"text mode ({len(text_dets)} objs)"
    right_title = f"auto mode ({len(auto_dets)} objs)"
    draw.text((16, PANEL_TITLE_H - 8), left_title, fill=TEXT_COLOR, font=small_font)
    draw.text((pw + gap + 16, PANEL_TITLE_H - 8), right_title,
              fill=AUTO_COLOR, font=small_font)

    canvas.paste(left, (0, PANEL_TITLE_H + 24))
    canvas.paste(right, (pw + gap, PANEL_TITLE_H + 24))
    return canvas


def _bbox_iou(a, b):
    """Intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _summarize(dets):
    return {
        "count": len(dets),
        "mean_score": (mean(d["score"] for d in dets) if dets else 0.0),
        "bboxes": [d["bbox_fullimg"] for d in dets],
        "labels": [d["label"] for d in dets],
    }


def _match_counts(text_summary, auto_summary, iou_threshold):
    """For each auto box, check if any text box overlaps above threshold.

    Returns: (matched, auto_novel, text_only)
        matched    -- # auto boxes that spatially match a text box
        auto_novel -- # auto boxes NOT matched (unique to auto mode)
        text_only  -- # text boxes that no auto box matched (unique to text)
    """
    text_boxes = text_summary["bboxes"]
    auto_boxes = auto_summary["bboxes"]

    matched = 0
    text_matched = [False] * len(text_boxes)
    for ab in auto_boxes:
        best_idx = -1
        best_iou = 0.0
        for ti, tb in enumerate(text_boxes):
            iou = _bbox_iou(ab, tb)
            if iou > best_iou:
                best_iou = iou
                best_idx = ti
        if best_iou >= iou_threshold and best_idx >= 0:
            matched += 1
            text_matched[best_idx] = True
    auto_novel = len(auto_boxes) - matched
    text_only = sum(1 for m in text_matched if not m)
    return matched, auto_novel, text_only


def run_mode(mode, before_full, sam_model, seg_model, sam_cfg, syn_cfg, asm_cfg):
    """Run select_best_objects in one mode, returning detections + timing."""
    detection_prompts = sam_cfg.get("detection_prompts",
                                    ["rock", "person", "car", "box", "bag", "bush"])
    detection_score = sam_cfg.get("detection_score_threshold", 0.30)
    scan_tile_size = sam_cfg.get("scan_tile_size", 1024)
    scan_overlap = sam_cfg.get("scan_overlap", 128)
    auto_cfg = sam_cfg.get("auto", {}) or {}

    max_objects = syn_cfg.get("max_changes", asm_cfg.get("max_changed_tiles", 3))
    min_obj_dist = syn_cfg.get("min_object_distance",
                               asm_cfg.get("min_tile_distance", 2000))
    max_per_label = syn_cfg.get("max_per_label", 0)
    variance_thresh = asm_cfg.get("variance_prefilter", 500)
    max_dets_crop = asm_cfg.get("max_detections_per_tile", 3)

    t0 = time.time()
    dets = select_best_objects(
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
        detection_mode=mode,
        seg_model=seg_model,
        auto_cfg=auto_cfg,
    )
    elapsed = time.time() - t0
    return dets, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Compare text-mode vs auto-mode detection on N sample pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=None,
                        help="Directory containing pair_* folders "
                             "(defaults to data.raw_root in config).")
    parser.add_argument("--num-pairs", type=int, default=20,
                        help="How many pairs to sample for the evaluation.")
    parser.add_argument("--output", default="src/data/workspace/eval_text_vs_auto.csv",
                        help="CSV output path.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for pair sampling.")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="IoU threshold for deciding two detections match.")
    parser.add_argument("--vis-dir", default="src/data/workspace/eval_vis",
                        help="Directory to save per-pair side-by-side "
                             "comparison images (set to '' to disable).")
    parser.add_argument("--vis-width", type=int, default=1200,
                        help="Downscaled width for visualization panels "
                             "(height preserves aspect).")
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated pair folder names (e.g. pair_0003,pair_0014) "
             "to evaluate instead of random sampling. Use this to regenerate "
             "visuals for specific pairs.",
    )
    args = parser.parse_args()

    cfg = Config("src/config.yaml")
    yml = yaml.safe_load(cfg.path.open().read())

    input_dir = Path(args.input_dir) if args.input_dir else Path(cfg.data["raw_root"])
    if not input_dir.exists():
        print(f"Error: input dir {input_dir} not found")
        return

    pairs = sorted(p for p in input_dir.iterdir()
                   if p.is_dir() and p.name.startswith("pair_")
                   and (p / "before.jpg").exists())
    if not pairs:
        print(f"Error: no pair_* subfolders with before.jpg in {input_dir}")
        return

    if args.pairs:
        names = [s.strip() for s in args.pairs.split(",") if s.strip()]
        sampled = []
        for name in names:
            p = input_dir / name
            if p.is_dir() and (p / "before.jpg").exists():
                sampled.append(p)
            else:
                print(f"Warning: skip missing or invalid pair path {p}")
        n = len(sampled)
        if not sampled:
            print("Error: --pairs did not resolve to any valid pair folders.")
            return
        print(f"Using {n} explicit pair(s) from --pairs.")
    else:
        rng = random.Random(args.seed)
        n = min(args.num_pairs, len(pairs))
        sampled = sorted(rng.sample(pairs, n), key=lambda p: p.name)
        print(f"Sampled {n}/{len(pairs)} pairs (seed={args.seed}).")

    sam_cfg = cfg.segmentation.get("sam", {})
    syn_cfg = cfg.synthetic
    asm_cfg = cfg.assembler

    print("Loading SAM 3...")
    sam_model = get_segmentation_model("sam", cfg.segmentation)
    print("Loading SegFormer...")
    seg_model = get_segmentation_model(
        cfg.segmentation.get("active_model", "segformer"), cfg.segmentation
    )

    if torch is not None and torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "mem_get_info"):
                free_b, total_b = torch.cuda.mem_get_info()
                print(f"GPU mem after load: {(total_b - free_b) / 2**20:.0f} MiB used, "
                      f"{free_b / 2**20:.0f} MiB free (of {total_b / 2**20:.0f} MiB).")
            else:
                print("GPU: CUDA active (install a recent PyTorch for free/total VRAM stats).")
        except Exception as e:
            print(f"GPU memory query failed: {e}")
        print("Text mode runs ~6 SAM prompt passes per crop vs 1 for auto; expect "
              "~2-4 min of silence per 8K pair during the text scan (not a hang).")
    _cuda_gc()

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Per-pair comparison images -> {vis_dir}")

    rows = []
    totals = {
        "text_count": 0, "auto_count": 0,
        "matched": 0, "auto_novel": 0, "text_only": 0,
        "text_secs": 0.0, "auto_secs": 0.0,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id", "text_count", "auto_count", "matched", "auto_novel",
        "text_only", "mean_text_score", "mean_auto_score", "text_labels",
        "text_secs", "auto_secs",
    ]
    csv_file = out_path.open("w", newline="", encoding="utf-8")
    try:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_file.flush()

        for i, pair_dir in enumerate(sampled, 1):
            before_path = pair_dir / "before.jpg"
            before_full = Image.open(before_path).convert("RGB")
            print(f"\n[{i}/{n}] {pair_dir.name} "
                  f"({before_full.size[0]}x{before_full.size[1]})")

            print("  text mode...")
            text_dets, text_secs = run_mode(
                "text", before_full, sam_model, seg_model,
                sam_cfg, syn_cfg, asm_cfg,
            )
            print(f"    -> {len(text_dets)} objs in {text_secs:.1f}s")
            _cuda_gc()

            print("  auto mode...")
            auto_dets, auto_secs = run_mode(
                "auto", before_full, sam_model, seg_model,
                sam_cfg, syn_cfg, asm_cfg,
            )
            print(f"    -> {len(auto_dets)} objs in {auto_secs:.1f}s")
            _cuda_gc()

            text_summary = _summarize(text_dets)
            auto_summary = _summarize(auto_dets)
            matched, auto_novel, text_only = _match_counts(
                text_summary, auto_summary, args.iou_threshold
            )

            row = {
                "pair_id": pair_dir.name,
                "text_count": text_summary["count"],
                "auto_count": auto_summary["count"],
                "matched": matched,
                "auto_novel": auto_novel,
                "text_only": text_only,
                "mean_text_score": round(text_summary["mean_score"], 3),
                "mean_auto_score": round(auto_summary["mean_score"], 3),
                "text_labels": "|".join(text_summary["labels"]),
                "text_secs": round(text_secs, 1),
                "auto_secs": round(auto_secs, 1),
            }
            rows.append(row)
            csv_writer.writerow(row)
            csv_file.flush()

            totals["text_count"] += text_summary["count"]
            totals["auto_count"] += auto_summary["count"]
            totals["matched"] += matched
            totals["auto_novel"] += auto_novel
            totals["text_only"] += text_only
            totals["text_secs"] += text_secs
            totals["auto_secs"] += auto_secs

            if vis_dir:
                try:
                    before_small, scale = _downscale_for_vis(
                        before_full, args.vis_width)
                    panel = _build_comparison_panel(
                        before_small, text_dets, auto_dets, pair_dir.name, scale
                    )
                    panel_path = vis_dir / f"{pair_dir.name}_text_vs_auto.jpg"
                    panel.save(panel_path, quality=85)
                    print(f"    vis saved: {panel_path.name}")
                except Exception as e:
                    print(f"    vis failed: {e}")
                finally:
                    for _d in text_dets:
                        _d.pop("mask_fullimg", None)
                    for _d in auto_dets:
                        _d.pop("mask_fullimg", None)
                    _cuda_gc()
            else:
                for _d in text_dets:
                    _d.pop("mask_fullimg", None)
                for _d in auto_dets:
                    _d.pop("mask_fullimg", None)

            del before_full, text_dets, auto_dets
            _cuda_gc()

    finally:
        csv_file.close()

    if not rows:
        print("No pairs were evaluated; CSV contains header only.")
        return

    text_counts = [r["text_count"] for r in rows]
    auto_counts = [r["auto_count"] for r in rows]
    novel_rate = (totals["auto_novel"] / totals["auto_count"]
                  if totals["auto_count"] else 0.0)

    print("\n" + "=" * 64)
    print(f"Evaluation summary (N={n} pairs, IoU>={args.iou_threshold})")
    print("=" * 64)
    print(f"  text objs/pair:   mean={mean(text_counts):.2f}  "
          f"median={median(text_counts):.1f}  total={totals['text_count']}")
    print(f"  auto objs/pair:   mean={mean(auto_counts):.2f}  "
          f"median={median(auto_counts):.1f}  total={totals['auto_count']}")
    print(f"  spatial match:    matched={totals['matched']}  "
          f"auto_novel={totals['auto_novel']}  text_only={totals['text_only']}")
    print(f"  novel-rate:       {novel_rate:.1%} of auto detections were NOT "
          f"found by text mode")
    print(f"  avg time/pair:    text={totals['text_secs']/n:.1f}s  "
          f"auto={totals['auto_secs']/n:.1f}s")
    print(f"\n  CSV written:      {out_path}")


if __name__ == "__main__":
    main()
