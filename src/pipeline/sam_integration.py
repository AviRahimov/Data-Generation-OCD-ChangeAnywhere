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
