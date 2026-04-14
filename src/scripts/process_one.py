# Quick runner to smoke-test tiling, segmentation (SLIC), and synthetic generation on pair_0000
import sys
from pathlib import Path
import random
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import Config
from pipeline.tiler import save_tiles_for_image
from pipeline.segmentation import get_segmentation_model
from pipeline.synthetic import generate_synthetic_after
from PIL import Image
import numpy as np
import yaml


def main():
    cfg = Config('src/config.yaml')

    # Check if testing config added in python object manually or load from raw section
    testing_cfg = cfg.path.open('r').read()
    yml_data = yaml.safe_load(testing_cfg)
    test_pair_name = yml_data.get('testing', {}).get('test_pair', 'pair_0064')

    raw_root = Path(cfg.data['raw_root'])
    pair = raw_root / test_pair_name
    before = pair / 'before.jpg'

    if not before.exists():
        print(f"Error: {test_pair_name} before.jpg not found at {before}")
        return

    pair_name = pair.name
    out_dir = Path(cfg.data['tiles_dir']) / pair_name / 'before'
    saved = save_tiles_for_image(before, out_dir, tile_size=cfg.tiling.get('tile_size',512), overlap=cfg.tiling.get('overlap',64), min_nonempty_ratio=cfg.tiling.get('min_nonempty_ratio',0.02))
    print('Saved tiles for', pair_name, ':', len(saved))
    if len(saved) == 0:
        print('No tiles saved — check min_nonempty_ratio or image contents')
        return

    # Randomly pick a few tiles to inspect
    random.seed(42)
    sample_tiles = random.sample(saved, min(5, len(saved)))

    seg_model = get_segmentation_model(cfg.segmentation.get('active_model','fallback'), cfg.segmentation)

    out_synth_dir = Path(cfg.data['synthetic_dir']) / pair_name / 'before'
    out_synth_dir.mkdir(parents=True, exist_ok=True)

    for i, tile_path in enumerate(sample_tiles):
        print(f"\nProcessing tile {i+1}/5: {Path(tile_path).name}")
        img = Image.open(tile_path).convert('RGB')
        seg = seg_model.segment(img)
        unique_classes = np.unique(seg)
        print(f"  Segmentation classes present: {unique_classes}")

        # Save a colorful visualization of the semantic mask
        cmap = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        cmap[0] = [0, 0, 0] # Make class 0 black for visibility
        seg_color = cmap[seg % 256]
        seg_vis = Image.fromarray(seg_color)

        after_pil, change_pil = generate_synthetic_after(img, seg, max_modified_segments=cfg.synthetic.get('max_modified_segments',3), color_jitter=cfg.synthetic.get('color_jitter',0.2), seed=123 + i)

        base_name = Path(tile_path).stem

        # Create a side-by-side comparison grid: Original | Segmentation | Synthetic After | Change Mask
        w, h = img.size
        from PIL import ImageDraw, ImageFont

        combo = Image.new('RGB', (w * 4, h + 40), color='white')
        draw = ImageDraw.Draw(combo)

        # Add labels
        labels = ["Original Tile", "Segmentation Map", "Synthetic After", "Change Mask"]
        for idx, label in enumerate(labels):
            draw.text((w * idx + 10, 10), label, fill="black")

        combo.paste(img, (0, 40))
        combo.paste(seg_vis, (w, 40))
        combo.paste(after_pil, (w * 2, 40))
        combo.paste(change_pil.convert('RGB'), (w * 3, 40))
        combo.save(out_synth_dir / f'{base_name}_comparison_grid.png')

    print('\nComparison grids saved to', out_synth_dir)

if __name__ == '__main__':
    main()
