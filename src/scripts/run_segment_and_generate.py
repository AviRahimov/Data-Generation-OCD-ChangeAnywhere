# Run segmentation on all tiles and generate synthetic after images for 'before' tiles.
# Legacy tile-batch path (see docs/PIPELINES.md). For full-image datasets prefer
# generate_dataset.py / generate_pair.py.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import Config
from pipeline.dataset import Pipeline


def main():
    cfg = Config('src/config.yaml')
    p = Pipeline(cfg)
    print('Segmenting tiles...')
    segs = p.segment_tiles()
    print('Segmented:', len(segs))
    print('Generating synthetic images...')
    gens = p.generate_synthetic()
    print('Generated synthetic images:', len(gens))

if __name__ == '__main__':
    main()

