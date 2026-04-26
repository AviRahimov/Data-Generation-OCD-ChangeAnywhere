# Pipelines: which script does what

This repository supports **three separate workflows**. They are not interchangeable; pick the row that matches your goal.

**Working directory:** run the commands below from the **repository root** (the folder that contains `src/` and `docs/`), unless noted otherwise. Paths like `src/config.yaml` assume that layout.

---

## How to run each track (CLI)

### 1. Full-image: batch dataset (production)

Uses SAM scan + `select_best_objects` + `generate_full_image_pair`. Writes `gen_00000/`, `gen_00001/`, … plus `manifest.csv`.

```powershell
cd "C:\path\to\Data-Generation-OCD-ChangeAnywhere"

python src/scripts/generate_dataset.py `
  --input-dir src/data/original_OCD_dataset `
  --n-images 100 `
  --output-dir src/data/workspace/dataset_v1 `
  --min-objects 1 `
  --max-objects 3 `
  --seed 42
```

Useful options:

- `--detection-mode text` or `--detection-mode auto` (overrides `config.yaml` `segmentation.sam.detection_mode`). For **auto**, tune `segmentation.sam.auto` in `src/config.yaml` (`auto_detection_score_threshold`, `separate_seed_forward`, `box_forward_batch_size`, `log_crop_interval`, etc.). **Multi-scale** can use two `box_scale` / `points_per_side` runs; with `separate_seed_forward: true` (default) SegFormer **seed** boxes and the **grid** are merged after separate forwards. **Post-filters** (no text to SAM) include `min_mask_interior_variance` (drop flat soil), `max_terrain_pixel_fraction` / `min_short_side_px` (drop specks), plus `min_area_ratio` and optional `object_class_allowlist` (ADE20K ids; set to `null` to disable that filter). Also tune `box_scale` / `dedup_iou` for grid behavior.
- `--backend sd2` / `sd15_realistic` / `sdxl` (overrides inpaint backend).
- `--source-frames before` / `after` / `both` — for `pair_*` folders, which real frames to draw from (`both` doubles the pool; manifest gets `source_frame`). Flat image folders always tag `before`.
- `--save-overview` — extra `overview.png` per sample.
- `--no-comparison` — skip `comparison.png`.
- `--config path\to\config.yaml` — non-default config.

After each sample the driver runs `gc.collect()` and `torch.cuda.empty_cache()` when CUDA is available to reduce fragmentation.

Optional quality gate: in `src/config.yaml` set `synthetic.full_image_quality.enabled: true` (local SSIM on the change region; rejected samples get `manifest` status `quality_reject` and no output folder).

### 2. Full-image: single pair (demo / one `pair_*`)

Uses `testing.test_pair` from config when you omit the positional argument, or pass `pair_0005` explicitly.

```powershell
python src/scripts/generate_pair.py
python src/scripts/generate_pair.py pair_0011
python src/scripts/generate_pair.py pair_0000 --detection-mode auto --max-objects 1
```

### 3. Tile debug / smoke (`process_one.py`)

Tiles **`testing.test_pair`** from `src/config.yaml` (default `pair_0000`), samples up to 5 interesting 512×512 tiles, runs per-tile SegFormer + SAM + `simulate_change` + inpaint, saves `*_grid.png` under `data.synthetic_dir` / `pair_*`. **No CLI arguments** — change `testing.test_pair` (and paths) in config to point at another pair.

```powershell
python src/scripts/process_one.py
```

### 4. Legacy tile batch (`run_segment_and_generate.py`)

Expects **existing** PNG tiles under `data.tiles_dir` (see `src/config.yaml`). Runs SegFormer on all tiles, then `Pipeline.generate_synthetic()` for `before` tiles only. Does **not** run `tile_all()`; use the full driver below if you need to rebuild tiles first.

```powershell
python src/scripts/run_segment_and_generate.py
```

**Full legacy loop** (tile all pairs from `raw_root`, segment, generate) in one shot:

```powershell
cd "C:\path\to\Data-Generation-OCD-ChangeAnywhere"
$env:PYTHONPATH = "src"
python -m pipeline.dataset
```

(`pipeline.dataset` `__main__` calls `tile_all()` → `segment_tiles()` → `generate_synthetic()`.)

Linux / macOS equivalent:

```bash
cd /path/to/Data-Generation-OCD-ChangeAnywhere
PYTHONPATH=src python -m pipeline.dataset
```

### 5. Evaluation: SAM3 text vs SAM3 auto (optional SAM2.1 column)

Random sample of N pairs from `data.raw_root` (or set `--input-dir`), full `before.jpg` only — no inpainting. Compares `select_best_objects` in **text** and **auto** modes. Optionally adds a **third** panel using Hugging Face **`mask-generation`** (Meta SAM2.1, e.g. `facebook/sam2.1-hiera-base-plus`) for **qualitative** comparison only; it is not used by `generate_pair` / `generate_dataset`.

**Config:** set `segmentation.sam2.enabled: true` in `src/config.yaml`, and tune `segmentation.sam2` (`checkpoint`, `points_per_batch`, `max_masks`, `use_bfloat16: false` recommended, `run_on_vis_resolution: true` to run on the same downscale as the JPEG to save VRAM). **CLI:** `--with-sam2` forces the third column; `--no-sam2` disables it even if enabled in config.

```powershell
python -u src/scripts/eval_detection_modes.py `
  --num-pairs 20 `
  --output src/data/workspace/eval_text_vs_auto.csv `
  --vis-dir src/data/workspace/eval_vis `
  --seed 42

# Three columns (SAM2.1): either enable in config or:
python -u src/scripts/eval_detection_modes.py --with-sam2 --pairs pair_0003 --vis-dir src/data/workspace/eval_vis
```

Explicit pairs (e.g. regenerate visuals for a smoke list):

```powershell
python -u src/scripts/eval_detection_modes.py `
  --pairs pair_0003,pair_0014 `
  --output src/data/workspace/eval_smoke.csv `
  --vis-dir src/data/workspace/eval_vis_smoke
```

- **Vis filenames:** `pair_*_text_sam3auto_sam2.jpg` when SAM2.1 is active, else `pair_*_text_vs_auto.jpg`.
- **CSV:** includes `sam2_count`, `mean_sam2_score`, `sam2_secs` (zeros when SAM2.1 is off).
- **Debug (SAM3 auto stages):** `--debug-detection-stages` or `sam.auto.log_detection_stages: true`.

Disable JPEG panels (CSV only): `--vis-dir ""`.

---

## Quick reference

| Goal | Entry point | Resolution | Main pipeline code |
|------|-------------|------------|-------------------|
| **Production dataset (recommended)** | [`src/scripts/generate_dataset.py`](../src/scripts/generate_dataset.py) | Full image per sample | SAM scan → `select_best_objects` → `generate_full_image_pair` in [`src/pipeline/full_image.py`](../src/pipeline/full_image.py) |
| **Single pair / supervisor demo** | [`src/scripts/generate_pair.py`](../src/scripts/generate_pair.py) | Full image | Same as above |
| **Tile debug / visual grids** | [`src/scripts/process_one.py`](../src/scripts/process_one.py) | 512×512 tiles | Tiling → per-tile SegFormer + SAM + `simulate_change` → inpaint |
| **Legacy tile batch (folder of tiles)** | [`src/scripts/run_segment_and_generate.py`](../src/scripts/run_segment_and_generate.py) | Tiles | [`src/pipeline/dataset.py`](../src/pipeline/dataset.py) `Pipeline`: tile → segment `.npy` → `generate_synthetic_after` |

## Full-image track (primary)

- **Outputs:** `before.jpg` (or copy), `synthetic_after.jpg`, `change_mask.png`, `meta.json`, optional `comparison.png`.
- **SSIM:** `src/config.yaml` keys `synthetic.ssim_min` / `synthetic.ssim_max` do **not** apply to `generate_dataset.py` / `generate_pair.py` unless you enable the optional `synthetic.full_image_quality` block (see config comments).
- **Detection:** Text prompts and/or auto mode come from `segmentation.sam` in config; see README.

## Tile debug track

- **Purpose:** Fast qualitative checks on small crops, SAM/SegFormer overlays, and grid layouts.
- **Not** the same code path as full-image dataset generation (no `select_best_objects` scan over the whole frame in the same way).

## Legacy tile batch track

- **Purpose:** Operate on an existing tree of PNG tiles under `data.tiles_dir`, with SegFormer masks under `data.masks_dir`.
- **`generate_synthetic_after`:** Implemented in [`src/pipeline/tile_synthetic.py`](../src/pipeline/tile_synthetic.py); requires an `InpaintingModel` (the `Pipeline` class loads one from `inpainting` config).
- If you only care about full-image datasets, you can ignore this track entirely.

## Tile-level SSIM (batch / `batch_generate` only)

Functions [`batch_generate`](../src/pipeline/tile_synthetic.py) and [`generate_synthetic_pair`](../src/pipeline/tile_synthetic.py) compute **local** SSIM on a padded bounding box of the change mask (not the whole tile). That avoids discarding small edits on large tiles. Thresholds `ssim_min` / `ssim_max` still come from the caller / config.

## Evaluation (not training)

- [`src/scripts/eval_detection_modes.py`](../src/scripts/eval_detection_modes.py): compares SAM3 **text** vs **auto**; optional **SAM2.1** `mask-generation` column (see section 5). Overlays are built by [`src/pipeline/eval_comparison_viz.py`](../src/pipeline/eval_comparison_viz.py). SAM2.1 integration and a small Transformers NMS patch live in [`src/pipeline/sam2_mask_generation.py`](../src/pipeline/sam2_mask_generation.py).

## Related docs

- [`docs/EXPERIMENT_REAL_VAL.md`](EXPERIMENT_REAL_VAL.md) — how to check whether synthetic data helps your change detector on real validation data.
