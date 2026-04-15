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

    Runs text-prompted instance segmentation with multiple class prompts,
    then merges all detected masks into a single integer segmentation map.
    """

    DEFAULT_PROMPTS = [
        "tree", "bush", "vegetation", "road", "car", "building",
        "house", "ground", "fence", "pole", "sidewalk", "dirt",
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
            checkpoint, torch_dtype=torch.float16
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
