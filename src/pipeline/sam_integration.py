import numpy as np
from PIL import Image

try:
    import torch
    # Import SAM dependencies here based on your actual library
    # For SAM 2/3.1 you might use a specific repository
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMModel:
    def __init__(self, checkpoint="", model_type="sam3.1", device="cuda"):
        if not SAM_AVAILABLE:
            raise RuntimeError("Segment Anything is not installed. Please install it.")

        if device == "cuda" and not torch.cuda.is_available():
            import sys
            print("CUDA is not available. Falling back to CPU.", file=sys.stderr)
            device = "cpu"

        self.device = device
        # Note: change sam_model_registry according to the actual package of SAM 3.1
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def segment(self, pil_image):
        arr = np.array(pil_image)
        masks = self.mask_generator.generate(arr)

        # Sort masks by area to overlay smaller masks on top of larger ones
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        seg = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.int32)
        for i, mask in enumerate(sorted_masks):
            # assign a unique integer to each mask (1-indexed, background is 0)
            seg[mask['segmentation']] = i + 1

        return seg

