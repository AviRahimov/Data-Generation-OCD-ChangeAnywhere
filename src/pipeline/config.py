# ...existing code...
import yaml
from pathlib import Path

class Config:
    def __init__(self, path='src/config.yaml'):
        self.path = Path(path)
        with open(self.path, 'r', encoding='utf8') as f:
            raw = yaml.safe_load(f)
        # resolve simple templated paths
        work_root = raw['data'].get('work_root', 'src/data/workspace')
        self.data = raw['data']
        self.data['work_root'] = work_root
        for k in ['tiles_dir','masks_dir','synthetic_dir','dataset_dir']:
            if self.data.get(k):
                self.data[k] = self.data[k].format(work_root=work_root)
        self.tiling = raw.get('tiling', {})
        self.segmentation = raw.get('segmentation', {})
        self.inpainting = raw.get('inpainting', {})
        self.synthetic = raw.get('synthetic', {})
        self.logging = raw.get('logging', {})

    def __repr__(self):
        return f"Config(path={self.path})"

