from pathlib import Path
import os
import csv
from PIL import Image
import numpy as np
import json

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

from .config import Config
from .tiler import save_tiles_for_image
from .segmentation import get_segmentation_model
from .inpainting import build_inpainter_from_cfg
from .io import save_image, read_json, write_json
from .synthetic import generate_synthetic_after


def discover_pairs(raw_root):
    raw_root = Path(raw_root)
    pairs = []
    for p in sorted(raw_root.iterdir()):
        if p.is_dir() and p.name.startswith('pair_'):
            pairs.append(p)
    return pairs


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.raw_root = Path(cfg.data['raw_root'])
        self.tiles_dir = Path(cfg.data['tiles_dir'])
        self.masks_dir = Path(cfg.data['masks_dir'])
        self.synthetic_dir = Path(cfg.data['synthetic_dir'])
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.seg_model = get_segmentation_model(cfg.segmentation.get('active_model', 'fallback'), cfg.segmentation)
        self.inpaint_model = build_inpainter_from_cfg(cfg.inpainting)

    def tile_all(self, use_after_as_base=True):
        pairs = discover_pairs(self.raw_root)
        saved = []
        for pair in pairs:
            before = pair / 'before.jpg'
            after = pair / 'after.jpg'
            if before.exists():
                out = self.tiles_dir / pair.name / 'before'
                s = save_tiles_for_image(before, out, tile_size=self.cfg.tiling.get('tile_size',512), overlap=self.cfg.tiling.get('overlap',64), min_nonempty_ratio=self.cfg.tiling.get('min_nonempty_ratio',0.02))
                saved.extend(s)
            if use_after_as_base and after.exists():
                out = self.tiles_dir / pair.name / 'after'
                s = save_tiles_for_image(after, out, tile_size=self.cfg.tiling.get('tile_size',512), overlap=self.cfg.tiling.get('overlap',64), min_nonempty_ratio=self.cfg.tiling.get('min_nonempty_ratio',0.02))
                saved.extend(s)
        return saved

    def segment_tiles(self):
        # find all tiles
        tiles = list(Path(self.tiles_dir).rglob('*.png'))
        processed = []
        for t in tiles:
            try:
                img = Image.open(t).convert('RGB')
                seg = self.seg_model.segment(img)
                # save segmask as npy
                out_npy = Path(self.masks_dir) / (t.stem + '_seg.npy')
                out_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_npy, seg)
                processed.append(str(out_npy))
            except Exception as e:
                # skip failures but report
                print(f"segmentation failed for {t}: {e}")
        return processed

    def generate_synthetic(self):
        tiles = list(Path(self.tiles_dir).rglob('*.png'))
        generated = []
        csv_path = self.synthetic_dir / 'provenance.csv'
        csv_exists = csv_path.exists()

        with open(csv_path, 'a' if csv_exists else 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['source_tile', 'after_synth', 'change_mask', 'ssim_score', 'status'])

            for t in tiles:
                # only generate from 'before' tiles (contain '/before/' in path)
                if '/before/' not in str(t).replace('\\','/'):
                    continue
                try:
                    seg_npy = Path(self.masks_dir) / (t.stem + '_seg.npy')
                    if not seg_npy.exists():
                        print(f"missing seg for {t}, skipping")
                        continue
                    seg = np.load(seg_npy)
                    before = Image.open(t).convert('RGB')
                    after_pil, change_pil = generate_synthetic_after(
                        before, seg, self.inpaint_model,
                        max_modified_segments=self.cfg.synthetic.get(
                            'max_modified_segments', 3),
                        color_jitter=self.cfg.synthetic.get('color_jitter', 0.2),
                        seed=42,
                        appearance_prob=self.cfg.synthetic.get(
                            'appearance_prob', 0.20),
                    )

                    # Calculate SSIM
                    ssim_score = -1.0
                    status = 'kept'
                    if ssim is not None:
                        before_gray = np.array(before.convert('L'))
                        after_gray = np.array(after_pil.convert('L'))
                        score = ssim(before_gray, after_gray, data_range=255)
                        ssim_score = float(score)
                        # Filter out pairs that are exactly the same (SSIM == 1.0) or too corrupted/different (e.g. SSIM < 0.6)
                        if score >= 0.99 or score < 0.6:
                            status = 'discarded'

                    if status == 'discarded':
                        writer.writerow([str(t), '', '', f"{ssim_score:.4f}", status])
                        continue

                    # save
                    rel = t.relative_to(self.tiles_dir)
                    out_dir = self.synthetic_dir / rel.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    after_path = out_dir / (t.stem + '_after_synth.png')
                    change_path = out_dir / (t.stem + '_change_mask.png')
                    after_pil.save(after_path)
                    change_pil.save(change_path)
                    # metadata
                    meta = {
                        'source_tile': str(t),
                        'seg_npy': str(seg_npy),
                        'after': str(after_path),
                        'change_mask': str(change_path),
                        'ssim_score': ssim_score
                    }
                    write_json(meta, out_dir / (t.stem + '_meta.json'))
                    generated.append(str(after_path))
                    writer.writerow([str(t), str(after_path), str(change_path), f"{ssim_score:.4f}", status])
                except Exception as e:
                    print(f"synthetic generation failed for {t}: {e}")
                    writer.writerow([str(t), '', '', '', f"error: {e}"])
        return generated


if __name__ == '__main__':
    # Run from repo root:  $env:PYTHONPATH="src"; python -m pipeline.dataset
    _src = Path(__file__).resolve().parents[1]
    cfg_path = _src / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Expected config at {cfg_path} (run from repo root with PYTHONPATH=src).")
    cfg = Config(str(cfg_path))
    p = Pipeline(cfg)
    print("Tiling...")
    p.tile_all()
    print("Segmenting...")
    p.segment_tiles()
    print("Generating synthetic...")
    p.generate_synthetic()
    print("Done")
