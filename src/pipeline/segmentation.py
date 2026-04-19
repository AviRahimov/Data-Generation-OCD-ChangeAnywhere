from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from PIL import Image
import sys

try:
    from skimage.segmentation import slic
    from skimage.color import rgb2lab
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    import sys
    print(f"Transformers/Torch import error: {e}", file=sys.stderr)
    TRANSFORMERS_AVAILABLE = False


class SegmentationModel(ABC):
    @abstractmethod
    def segment(self, pil_image):
        """Return an integer mask (H,W) where each pixel has a segment id or class id."""
        pass


class FallbackSLIC(SegmentationModel):
    def __init__(self, n_segments=200, compactness=10.0):
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError('scikit-image not available; install scikit-image to use FallbackSLIC')
        self.n_segments = n_segments
        self.compactness = compactness

    def segment(self, pil_image):
        arr = np.array(pil_image)
        # convert to LAB for better color clustering
        try:
            lab = rgb2lab(arr)
            seg = slic(lab, n_segments=self.n_segments, compactness=self.compactness, start_label=1)
        except Exception:
            # fallback to slic on RGB
            seg = slic(arr, n_segments=self.n_segments, compactness=self.compactness, start_label=1)
        return seg.astype(np.int32)


class SegformerModel(SegmentationModel):
    def __init__(self, checkpoint="nvidia/segformer-b2-finetuned-ade-512-512", device="cuda"):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers and torch are required for SegformerModel.")

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.device = device
        self.image_processor = SegformerImageProcessor.from_pretrained(checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint, use_safetensors=True
        ).to(self.device)
        self.model.eval()

    def segment(self, pil_image):
        arr = np.array(pil_image)
        # Segformer expects RGB images
        inputs = self.image_processor(images=arr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # shape: (batch_size, num_classes, H/4, W/4)

        # Resize logits to original image size
        logits = torch.nn.functional.interpolate(
            logits,
            size=pil_image.size[::-1], # PIL size is (W, H), interpolate needs (H, W)
            mode="bilinear",
            align_corners=False,
        )

        # Get the predicted class for each pixel
        seg = logits.argmax(dim=1).squeeze().cpu().numpy()
        return seg.astype(np.int32)


def get_segmentation_model(name, cfg=None):
    name = (name or '').lower()
    if name == 'fallback' or name == '' or name is None:
        n = 200
        if cfg and cfg.get('fallback'):
            n = cfg['fallback'].get('slic_segments', n)
        return FallbackSLIC(n_segments=n)

    if name == 'segformer':
        checkpoint = "nvidia/segformer-b5-finetuned-ade-640-640"
        device = "cuda"
        if cfg and cfg.get('segformer'):
            checkpoint = cfg['segformer'].get('checkpoint', checkpoint)
            device = cfg['segformer'].get('device', device)
        return SegformerModel(checkpoint=checkpoint, device=device)

    if name == 'sam':
        from .sam_integration import SAMModel
        checkpoint = "SAM3"
        device = "cuda"
        score_threshold = 0.10
        mask_threshold = 0.5
        prompts = None
        if cfg and cfg.get('sam'):
            sam_cfg = cfg['sam']
            checkpoint = sam_cfg.get('checkpoint', checkpoint)
            device = sam_cfg.get('device', device)
            score_threshold = sam_cfg.get('score_threshold', score_threshold)
            mask_threshold = sam_cfg.get('mask_threshold', mask_threshold)
            prompts = sam_cfg.get('prompts', prompts)
        return SAMModel(
            checkpoint=checkpoint,
            device=device,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            prompts=prompts,
        )

    raise ValueError(f'Unknown segmentation model: {name}')
