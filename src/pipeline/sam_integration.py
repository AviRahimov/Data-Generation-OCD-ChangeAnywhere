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
    def detect_objects_auto(self, pil_image, min_score=0.30,
                            min_area_ratio=0.0005, max_area_ratio=0.15,
                            points_per_side=16):
        """Prompt-free "segment-everything" detection.

        SAM 3's HuggingFace API requires prompts (text or boxes) -- it has
        no native "automatic" mode like SAM 2. We approximate
        segment-everything by feeding a dense ``points_per_side`` x
        ``points_per_side`` grid of overlapping boxes as input prompts.
        SAM 3's DETR decoder returns all candidate masks above the score
        threshold; we then IoU-dedup near-duplicates.

        The returned dict schema is identical to ``detect_objects()`` so
        downstream code does not need to branch:
            mask        -- bool ndarray (H, W)
            label       -- always "object" (no text prompt)
            score       -- float confidence
            area_ratio  -- float fraction of the image covered

        Args:
            pil_image: PIL RGB image
            min_score: minimum confidence to keep a detection
            min_area_ratio: drop masks smaller than this fraction of the image
            max_area_ratio: drop masks larger than this fraction of the image
            points_per_side: edge length of the box grid (16 -> 256 boxes)
        """
        dets = self._grid_auto(
            pil_image, points_per_side=points_per_side,
            min_score=min_score,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def _grid_auto(self, pil_image, *, points_per_side,
                   min_score, min_area_ratio, max_area_ratio):
        """Prompt-free detection via a dense box grid.

        SAM 3's HuggingFace processor supports ``input_boxes`` (it does
        *not* accept ``input_points``), so we build a
        ``points_per_side`` x ``points_per_side`` grid of overlapping
        boxes and feed them in a single forward pass. SAM 3 uses a DETR
        decoder with a fixed number of object-query slots and returns
        *all* candidate detections above the score threshold regardless
        of exactly how many boxes we pass -- the boxes just act as
        attention hints directing the decoder toward every part of the
        image.

        Near-duplicates are collapsed by IoU NMS. Conceptually identical
        to SAM 2's ``AutomaticMaskGenerator``, adapted for SAM 3's
        prompt API.
        """
        h = pil_image.size[1]
        w = pil_image.size[0]
        total_pixels = h * w

        step_x = w / points_per_side
        step_y = h / points_per_side
        half = 0.75 * min(step_x, step_y)

        xs = np.linspace(step_x / 2.0, w - step_x / 2.0, points_per_side)
        ys = np.linspace(step_y / 2.0, h - step_y / 2.0, points_per_side)

        boxes = []
        for y in ys:
            for x in xs:
                x1 = max(float(x - half), 0.0)
                y1 = max(float(y - half), 0.0)
                x2 = min(float(x + half), float(w))
                y2 = min(float(y + half), float(h))
                boxes.append([x1, y1, x2, y2])

        try:
            inputs = self.processor(
                images=pil_image,
                input_boxes=[boxes],
                return_tensors="pt",
            )
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
                outputs,
                threshold=min_score,
                mask_threshold=self.mask_threshold,
                target_sizes=[pil_image.size[::-1]],
            )
        except Exception as e:
            print(f"  SAM3 grid auto post-process error: {e}", file=sys.stderr)
            return []

        if not results:
            return []

        masks = results[0].get("masks")
        scores = results[0].get("scores")
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
                "mask": m_np,
                "label": "object",
                "score": score_val,
                "area_ratio": float(area_ratio),
            })

        return self._dedup_by_iou(dets, iou_thresh=0.7)

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
