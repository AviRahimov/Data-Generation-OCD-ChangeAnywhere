import sys
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import Sam3Model, Sam3Processor as HFSam3Processor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMModel:
    """SAM 3 segmentation using the HuggingFace Transformers local checkpoint.

    Supports two modes:
      - segment(): merges all detected masks into a single integer segmentation map
      - detect_objects(): returns individual instance masks with labels and scores
    """

    DEFAULT_PROMPTS = [
        "tree", "bush", "vegetation", "road", "car", "building",
        "house", "ground", "fence", "pole", "sidewalk", "dirt",
    ]

    DEFAULT_DETECTION_PROMPTS = [
        "rock", "person", "car", "box", "bag", "bush",
    ]

    def __init__(self, checkpoint="SAM3", device="cuda",
                 prompts=None, score_threshold=0.10, mask_threshold=0.5,
                 **kwargs):
        if not SAM_AVAILABLE:
            raise RuntimeError(
                "SAM 3 requires: pip install transformers torch\n"
                "And a local SAM3 model folder."
            )

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.device = device
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.prompts = prompts or self.DEFAULT_PROMPTS

        self.model = Sam3Model.from_pretrained(
            checkpoint, torch_dtype=torch.float16, use_safetensors=True
        ).to(device).eval()
        self.processor = HFSam3Processor.from_pretrained(checkpoint)

    @torch.inference_mode()
    def _run_prompt(self, pil_image, prompt):
        """Run a single text prompt and return (masks_bool, scores, label)."""
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        outputs = self.model(**inputs)

        target_sizes = [pil_image.size[::-1]]
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )
        return results[0]["masks"], results[0]["scores"], prompt

    def segment(self, pil_image):
        """Segment all objects using multiple text prompts.

        Returns an integer mask (H, W) where each pixel gets a unique
        segment ID (1-indexed); background is 0.
        """
        h, w = pil_image.size[1], pil_image.size[0]
        all_masks = []
        all_scores = []

        for prompt in self.prompts:
            masks, scores, _ = self._run_prompt(pil_image, prompt)
            for m, s in zip(masks, scores):
                m_np = m.cpu().numpy().astype(bool)
                area_ratio = m_np.sum() / (h * w)
                if area_ratio < 0.0005 or area_ratio > 0.95:
                    continue
                all_masks.append(m_np)
                all_scores.append(float(s.cpu()))

        if not all_masks:
            return np.zeros((h, w), dtype=np.int32)

        order = sorted(range(len(all_masks)),
                       key=lambda i: all_masks[i].sum(), reverse=True)

        seg = np.zeros((h, w), dtype=np.int32)
        for rank, idx in enumerate(order):
            seg[all_masks[idx]] = rank + 1

        return seg

    @torch.inference_mode()
    def detect_objects(self, pil_image, prompts=None, min_score=0.30,
                       min_area_ratio=0.005, max_area_ratio=0.30):
        """Detect individual objects using targeted text prompts.

        Unlike segment(), this returns each detection separately so callers
        can pick specific objects for removal or other operations.

        Args:
            pil_image: PIL RGB image
            prompts: list of text prompts (defaults to DEFAULT_DETECTION_PROMPTS)
            min_score: minimum confidence score to keep a detection
            min_area_ratio: discard detections smaller than this fraction of the tile
            max_area_ratio: discard detections larger than this fraction of the tile

        Returns:
            list of dicts, each with keys:
                mask: bool ndarray (H, W)
                label: str (the prompt that found it)
                score: float (confidence)
                area_ratio: float (fraction of image covered)
        """
        prompts = prompts or self.DEFAULT_DETECTION_PROMPTS
        h, w = pil_image.size[1], pil_image.size[0]
        total_pixels = h * w
        detections = []

        for prompt in prompts:
            try:
                masks, scores, _ = self._run_prompt(pil_image, prompt)
            except Exception as e:
                print(f"  SAM3 detect_objects error for '{prompt}': {e}",
                      file=sys.stderr)
                continue

            for m, s in zip(masks, scores):
                score_val = float(s.cpu()) if hasattr(s, "cpu") else float(s)
                if score_val < min_score:
                    continue

                m_np = m.cpu().numpy().astype(bool) if hasattr(m, "cpu") else np.asarray(m, dtype=bool)
                area_ratio = m_np.sum() / total_pixels

                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue

                detections.append({
                    "mask": m_np,
                    "label": prompt,
                    "score": score_val,
                    "area_ratio": area_ratio,
                })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    @torch.inference_mode()
    def detect_objects_auto(
            self, pil_image, min_score=0.30,
            min_area_ratio=0.0005, max_area_ratio=0.15,
            points_per_side=16, box_scale=1.35, dedup_iou=0.65,
            multi_scale=True, multi_scale_runs=None, merge_dedup_iou=0.52,
            extra_boxes=None, separate_seed_forward=False,
            box_forward_batch_size=0):
        """Prompt-free grid (``input_boxes``). Optional multi-scale grids.

        ``extra_boxes`` lists [x1,y1,x2,y2] in pixel coords (e.g. SegFormer CC
        bboxes). If ``separate_seed_forward`` is True, grid and seed boxes are
        processed in separate forwards, then deduplicated (reduces seed dilution).

        ``box_forward_batch_size`` > 0 runs chunk forwards + merge (VRAM cap).
        """
        extra = []
        if extra_boxes:
            for b in extra_boxes:
                if b is None or len(b) < 4:
                    continue
                extra.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        w, h = pil_image.size
        bsz = int(box_forward_batch_size) if box_forward_batch_size else 0

        def _merge_dedup_pair(da, db, iou):
            if not da:
                return list(db) if db else []
            if not db:
                return list(da)
            return self._dedup_by_iou(da + db, iou_thresh=float(iou))

        if not multi_scale:
            grid = self._auto_box_grid(w, h, int(points_per_side), float(box_scale))
            if separate_seed_forward and extra:
                dets_g = self._auto_from_boxes(
                    pil_image, grid, min_score, min_area_ratio, max_area_ratio,
                    float(dedup_iou), box_forward_batch_size=bsz,
                )
                dets_e = self._auto_from_boxes(
                    pil_image, extra, min_score, min_area_ratio, max_area_ratio,
                    float(dedup_iou), box_forward_batch_size=bsz,
                )
                dets = _merge_dedup_pair(dets_g, dets_e, merge_dedup_iou)
            else:
                boxes = list(grid) + extra
                dets = self._auto_from_boxes(
                    pil_image, boxes, min_score, min_area_ratio, max_area_ratio,
                    float(dedup_iou), box_forward_batch_size=bsz,
                )
            dets.sort(key=lambda d: d["score"], reverse=True)
            return dets

        runs = multi_scale_runs
        if not runs:
            runs = [(1.22, 12), (1.72, 10)]
        grid = []
        for run in runs:
            if not (isinstance(run, (list, tuple)) and len(run) >= 2):
                continue
            grid.extend(self._auto_box_grid(w, h, int(run[1]), float(run[0])))

        if separate_seed_forward and extra:
            dets_g = self._auto_from_boxes(
                pil_image, grid, min_score, min_area_ratio, max_area_ratio,
                float(merge_dedup_iou), box_forward_batch_size=bsz,
            )
            dets_e = self._auto_from_boxes(
                pil_image, extra, min_score, min_area_ratio, max_area_ratio,
                float(merge_dedup_iou), box_forward_batch_size=bsz,
            )
            dets = _merge_dedup_pair(dets_g, dets_e, merge_dedup_iou)
        else:
            boxes = list(grid) + extra
            dets = self._auto_from_boxes(
                pil_image, boxes, min_score, min_area_ratio, max_area_ratio,
                float(merge_dedup_iou), box_forward_batch_size=bsz,
            )
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def _grid_auto(self, pil_image, *, points_per_side,
                   min_score, min_area_ratio, max_area_ratio,
                   box_scale=1.35, dedup_iou=0.65):
        """Prompt-free detection via a dense box grid.

        SAM 3's HuggingFace processor supports ``input_boxes`` (it does
        *not* accept ``input_points``), so we build a
        ``points_per_side`` x ``points_per_side`` grid of overlapping
        boxes and feed them in a single forward pass. ``box_scale`` > 1
        widens each query box so the decoder sees more context, which
        usually yields *larger* masks and less tiny-fragment noise.

        Near-duplicates are collapsed with IoU NMS (``dedup_iou``).
        """
        w, h = pil_image.size
        boxes = self._auto_box_grid(w, h, int(points_per_side), float(box_scale))
        return self._auto_from_boxes(
            pil_image, boxes, min_score, min_area_ratio, max_area_ratio, dedup_iou)


    def _auto_box_grid(self, w, h, pps, box_scale):
        pps = int(pps)
        step_x = w / float(pps)
        step_y = h / float(pps)
        half = float(box_scale) * 0.75 * min(step_x, step_y)
        xs = np.linspace(step_x / 2.0, w - step_x / 2.0, pps)
        ys = np.linspace(step_y / 2.0, h - step_y / 2.0, pps)
        out = []
        for y in ys:
            for x in xs:
                x1, y1 = max(float(x - half), 0.0), max(float(y - half), 0.0)
                x2, y2 = min(float(x + half), float(w)), min(float(y + half), float(h))
                out.append([x1, y1, x2, y2])
        return out

    @torch.inference_mode()
    def _auto_forward_one_batch(
            self, pil_image, boxes, min_score, min_area_ratio, max_area_ratio):
        """Run one input_boxes batch; no inter-batch mask deduplication."""
        if not boxes:
            return []
        h = pil_image.size[1]
        w = pil_image.size[0]
        total_pixels = h * w
        try:
            inputs = self.processor(
                images=pil_image, input_boxes=[boxes], return_tensors="pt")
        except Exception as e:
            print(f"  SAM3 grid auto processor error: {e}", file=sys.stderr)
            return []
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        model_dtype = next(self.model.parameters()).dtype
        if "input_boxes" in inputs and inputs["input_boxes"].dtype != model_dtype:
            inputs["input_boxes"] = inputs["input_boxes"].to(dtype=model_dtype)
        try:
            outputs = self.model(**inputs)
        except Exception as e:
            print(f"  SAM3 grid auto forward error: {e}", file=sys.stderr)
            return []
        try:
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=min_score, mask_threshold=self.mask_threshold,
                target_sizes=[pil_image.size[::-1]],
            )
        except Exception as e:
            print(f"  SAM3 grid auto post-process error: {e}", file=sys.stderr)
            return []
        if not results:
            return []
        masks, scores = results[0].get("masks"), results[0].get("scores")
        if masks is None or scores is None:
            return []
        dets = []
        for m, s in zip(masks, scores):
            score_val = (float(s.cpu()) if hasattr(s, "cpu") else float(s))
            if score_val < min_score:
                continue
            m_np = (m.cpu().numpy().astype(bool) if hasattr(m, "cpu")
                    else np.asarray(m, dtype=bool))
            area_ratio = m_np.sum() / total_pixels
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            dets.append({
                "mask": m_np, "label": "object", "score": score_val,
                "area_ratio": float(area_ratio),
            })
        return dets

    @torch.inference_mode()
    def _auto_from_boxes(
            self, pil_image, boxes, min_score, min_area_ratio, max_area_ratio,
            dedup_iou, box_forward_batch_size=0):
        if not boxes:
            return []
        bsz = int(box_forward_batch_size) if box_forward_batch_size else 0
        if bsz > 0 and len(boxes) > bsz:
            merged = []
            for i in range(0, len(boxes), bsz):
                chunk = boxes[i:i + bsz]
                merged.extend(
                    self._auto_forward_one_batch(
                        pil_image, chunk, min_score, min_area_ratio, max_area_ratio
                    )
                )
            dets = merged
        else:
            dets = self._auto_forward_one_batch(
                pil_image, boxes, min_score, min_area_ratio, max_area_ratio
            )
        return self._dedup_by_iou(dets, iou_thresh=float(dedup_iou))

    @staticmethod
    def _dedup_by_iou(dets, iou_thresh=0.7):
        """Drop near-duplicate masks produced by adjacent grid seeds hitting
        the same object. Keeps the higher-scoring mask in each duplicate pair.
        """
        if not dets:
            return dets

        dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
        kept = []
        kept_masks = []
        for d in dets_sorted:
            m = d["mask"]
            area_m = m.sum()
            is_dup = False
            for km in kept_masks:
                inter = np.logical_and(m, km).sum()
                if inter == 0:
                    continue
                union = area_m + km.sum() - inter
                if union > 0 and inter / union >= iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(d)
                kept_masks.append(m)
        return kept
