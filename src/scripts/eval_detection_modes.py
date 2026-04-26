"""Compare SAM3 text vs SAM3 auto; optional column: SAM2.1 automatic masks.

See ``segmentation.sam.auto`` and ``segmentation.sam2`` in ``config.yaml``.
CLI: ``--with-sam2``, ``--no-sam2``. Multi-column figure saved as
``*_text_sam3auto_sam2.jpg`` when SAM2.1 is active, else ``*_text_vs_auto.jpg``.

Examples:
    python -u src/scripts/eval_detection_modes.py --num-pairs 20 \\
        --output src/data/workspace/eval_text_vs_auto.csv
    python -u src/scripts/eval_detection_modes.py --with-sam2 --pairs pair_0003
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

try:
    import torch
except ImportError:
    torch = None
import yaml
from PIL import Image

from pipeline.config import Config
from pipeline.eval_comparison_viz import (
    COLOR_AUTO,
    COLOR_SAM2,
    COLOR_TEXT,
    build_multi_column_panel,
    downscale_for_vis,
)
from pipeline.sam2_mask_generation import (
    build_sam2_mask_pipeline,
    promote_dets_to_full_space,
    sam2_detections_on_image,
    sam2_is_configured,
)
from pipeline.segmentation import get_segmentation_model
from pipeline.synthetic import select_best_objects


def _cuda_gc():
    """Free fragmented CUDA allocations between heavy steps (8 GB GPUs)."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def run_mode(mode, before_full, sam_model, seg_model, sam_cfg, syn_cfg, asm_cfg,
             debug_detection_stages=None):
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
        debug_detection_stages=debug_detection_stages,
    )
    elapsed = time.time() - t0
    return dets, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAM3 text vs SAM3 auto; optional SAM2.1 automask column.",
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
    parser.add_argument(
        "--debug-detection-stages",
        action="store_true",
        help="Print SAM vs post_filter vs NMS/MMR stage counts (auto) and text "
             "pre-NMS count. Same as sam.auto.log_detection_stages in config.",
    )
    s2 = parser.add_mutually_exclusive_group()
    s2.add_argument(
        "--with-sam2",
        action="store_true",
        help="Add SAM2.1 automatic mask column (overrides config if set).",
    )
    s2.add_argument(
        "--no-sam2",
        action="store_true",
        help="Force-disable SAM2.1 even if enabled in config.",
    )
    args = parser.parse_args()
    # None = honor sam.auto.log_detection_stages; True = force verbose.
    eval_debug_stages = True if args.debug_detection_stages else None

    cfg = Config("src/config.yaml")

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
    sam2_cfg = cfg.segmentation.get("sam2") or {}
    syn_cfg = cfg.synthetic
    asm_cfg = cfg.assembler

    if args.no_sam2:
        use_sam2 = False
    elif args.with_sam2:
        use_sam2 = True
    else:
        use_sam2 = sam2_is_configured(sam2_cfg)

    print("Loading SAM 3...")
    sam_model = get_segmentation_model("sam", cfg.segmentation)
    print("Loading SegFormer...")
    seg_model = get_segmentation_model(
        cfg.segmentation.get("active_model", "segformer"), cfg.segmentation
    )

    sam2_pipe = None
    if use_sam2:
        print("Loading SAM2.1 (mask-generation pipeline)...")
        sam2_pipe, err = build_sam2_mask_pipeline(sam2_cfg)
        if sam2_pipe is None:
            print(f"  SAM2.1 disabled (load failed): {err}")
            use_sam2 = False
        else:
            print("  SAM2.1 ready.")
        _cuda_gc()

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
        "sam2_count": 0, "sam2_secs": 0.0,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id", "text_count", "auto_count", "matched", "auto_novel",
        "text_only", "mean_text_score", "mean_auto_score", "text_labels",
        "text_secs", "auto_secs", "sam2_count", "mean_sam2_score",
        "sam2_secs",
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
                debug_detection_stages=eval_debug_stages,
            )
            print(f"    -> {len(text_dets)} objs in {text_secs:.1f}s")
            _cuda_gc()

            print("  auto mode...")
            auto_dets, auto_secs = run_mode(
                "auto", before_full, sam_model, seg_model,
                sam_cfg, syn_cfg, asm_cfg,
                debug_detection_stages=eval_debug_stages,
            )
            print(f"    -> {len(auto_dets)} objs in {auto_secs:.1f}s")
            _cuda_gc()

            sam2_dets = []
            sam2_secs = 0.0
            if use_sam2 and sam2_pipe is not None:
                print("  SAM2.1 automask...")
                t2 = time.time()
                before_small, _vscale = downscale_for_vis(
                    before_full, args.vis_width)
                sw, sh = before_small.size
                fw, fh = before_full.size
                if sam2_cfg.get("run_on_vis_resolution", True):
                    sam2_dets = sam2_detections_on_image(
                        before_small, sam2_pipe, sam2_cfg)
                    sam2_dets = promote_dets_to_full_space(
                        sam2_dets, sw, sh, fw, fh)
                else:
                    sam2_dets = sam2_detections_on_image(
                        before_full, sam2_pipe, sam2_cfg)
                sam2_secs = time.time() - t2
                print(f"    -> {len(sam2_dets)} masks in {sam2_secs:.1f}s")
                _cuda_gc()

            text_summary = _summarize(text_dets)
            auto_summary = _summarize(auto_dets)
            sam2_summary = _summarize(sam2_dets) if use_sam2 else {
                "count": 0, "mean_score": 0.0, "bboxes": [], "labels": [],
            }
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
                "sam2_count": sam2_summary["count"],
                "mean_sam2_score": round(sam2_summary["mean_score"], 3),
                "sam2_secs": round(sam2_secs, 1),
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
            totals["sam2_count"] += sam2_summary["count"]
            totals["sam2_secs"] += sam2_secs

            if vis_dir:
                try:
                    before_small, scale = downscale_for_vis(
                        before_full, args.vis_width)
                    if use_sam2 and sam2_pipe is not None:
                        cols = [
                            ("text", text_dets, COLOR_TEXT),
                            ("SAM3 auto", auto_dets, COLOR_AUTO),
                            ("SAM2.1 auto", sam2_dets, COLOR_SAM2),
                        ]
                        vis_name = f"{pair_dir.name}_text_sam3auto_sam2.jpg"
                    else:
                        cols = [
                            ("text", text_dets, COLOR_TEXT),
                            ("SAM3 auto", auto_dets, COLOR_AUTO),
                        ]
                        vis_name = f"{pair_dir.name}_text_vs_auto.jpg"
                    panel = build_multi_column_panel(
                        before_small, cols, pair_dir.name, scale,
                    )
                    panel_path = vis_dir / vis_name
                    panel.save(panel_path, quality=85)
                    print(f"    vis saved: {panel_path.name}")
                except Exception as e:
                    print(f"    vis failed: {e}")
                finally:
                    for _d in text_dets:
                        _d.pop("mask_fullimg", None)
                    for _d in auto_dets:
                        _d.pop("mask_fullimg", None)
                    for _d in sam2_dets:
                        _d.pop("mask_fullimg", None)
                    _cuda_gc()
            else:
                for _d in text_dets:
                    _d.pop("mask_fullimg", None)
                for _d in auto_dets:
                    _d.pop("mask_fullimg", None)
                for _d in sam2_dets:
                    _d.pop("mask_fullimg", None)

            del before_full, text_dets, auto_dets, sam2_dets
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
          f"auto={totals['auto_secs']/n:.1f}s", end="")
    if totals.get("sam2_count", 0) > 0 or totals.get("sam2_secs", 0) > 0:
        print(f"  sam2.1={totals['sam2_secs']/n:.1f}s")
    else:
        print()
    print(f"\n  CSV written:      {out_path}")


if __name__ == "__main__":
    main()
