# Data-Generation-OCD-ChangeAnywhere

Synthetic **change-detection** dataset generator for aerial/drone imagery,
inspired by the *ChangeAnywhere* paper. The pipeline takes a real `before`
image, programmatically invents a plausible change (a new rock appears on
the ground, a car disappears from the dirt road, ...), renders the
corresponding `after` image with a diffusion inpainter, and emits the
pixel-perfect binary change mask that pairs the two. The result is a
free, unlimited supply of `(before, after, change_mask)` training triplets
that look real enough to train change-detection networks on.

---

## Table of contents

1. [Why this exists](#why-this-exists)
2. [High-level idea](#high-level-idea)
3. [Project layout](#project-layout)
4. [File-by-file tour](#file-by-file-tour)
5. [Models used](#models-used)
6. [Installation](#installation)
7. [Quick start](#quick-start)
8. [Configuration reference](#configuration-reference)
9. [Design decisions worth knowing](#design-decisions-worth-knowing)
10. [Outputs on disk](#outputs-on-disk)
11. [Troubleshooting](#troubleshooting)

---

## Why this exists

Training a change-detection model needs thousands of `(before, after, mask)`
triplets where the two images are *perfectly aligned* and the mask
*perfectly labels* every changed pixel. Collecting such data from the real
world is expensive: you need two flights over the same location taken on
different days with sub-pixel registration plus human annotators marking
every rock, car, or bush that moved.

This repo sidesteps the problem: starting from a *single* real aerial
image (`before`), it synthesizes a realistic `after` by *adding* or
*removing* a few objects using a diffusion model, and records the exact
change region as the ground-truth mask. The geometry is identical to the
original frame by construction, so alignment is free.

## High-level idea

```mermaid
flowchart LR
    raw["Before image (8K drone photo)"]
    tiler["Tile into 512x512 chunks"]
    seg["SegFormer semantic map<br/>(ADE20K classes)"]
    sam["SAM 3 object detections<br/>(rock, car, bush, ...)"]
    sim["Change simulator<br/>(80% remove, 20% add)"]
    inpaint["Diffusion inpainter<br/>(SD2 / SD1.5 / SDXL)"]
    after["After image + change mask"]

    raw --> tiler --> seg
    tiler --> sam
    seg --> sim
    sam --> sim
    sim -->|"prompt + mask"| inpaint
    inpaint --> after
```

Two operating modes coexist:

- **Tile-based** ([`src/scripts/process_one.py`](src/scripts/process_one.py)):
  cuts the full image into 512x512 tiles, processes each tile independently,
  and produces visual comparison grids. Good for quick qualitative checks.
- **Object-centric / full-image** ([`src/scripts/generate_pair.py`](src/scripts/generate_pair.py)):
  scans the whole image with 1024x1024 crops, picks the 1-3 most visible
  objects, inpaints each one directly at full resolution, and emits a single
  aligned `(before.jpg, synthetic_after.jpg, change_mask.png, meta.json)`
  training triplet. This is the path used for dataset generation.

## Project layout

```
Data-Generation-OCD-ChangeAnywhere/
  requirements.txt                  # Python dependencies
  README.md                         # this file
  src/
    config.yaml                     # single source of truth for all knobs
    data/
      original_OCD_dataset/         # real input pairs (pair_0000/, pair_0001/, ...)
        pair_0000/
          before.jpg                # aerial image (input)
          after.jpg                 # real paired "after" (reference only)
          after_binary_mask.png     # human-annotated GT mask (reference only)
          after_with_polygons.jpg   # annotated overlay (reference only)
          annotations.json
          metadata.json
      workspace/                    # all generated artefacts (gitignored)
        tiles/                      # 512x512 crops of each before image
        masks/                      # cached SegFormer semantic maps
        synthetic/                  # generated pairs + comparison grids
    pipeline/                       # reusable library code
      __init__.py
      config.py                     # YAML loader -> Config object
      io.py                         # image / JSON / polygon helpers
      tiler.py                      # tile + reassemble utilities
      segmentation.py               # SegFormer + SAM 3 wrappers + fallback
      sam_integration.py            # SAM 3 detect_objects() + segment()
      inpainting.py                 # diffusion wrapper (3 backends, bbox dilation)
      change_simulator.py           # picks WHAT to change and builds the mask
      prompt_templates.py           # ADE20K -> prompts for SD
      synthetic.py                  # pair generator + whole-image object scanner
      dataset.py                    # batch driver (tile -> segment -> generate)
    scripts/                        # end-user entry points
      process_one.py                # tile-based pipeline (one pair, grid outputs)
      generate_pair.py              # object-centric full-resolution pair
      compare_inpaint_backends.py   # A/B compare SD2 vs SD1.5 vs SDXL
      run_segment_and_generate.py   # batch driver over all pairs
```

## File-by-file tour

### Configuration and I/O primitives

- **[`src/config.yaml`](src/config.yaml)** - all tunable parameters live
  here: paths, tile size, which segmentation model to use, SAM 3 prompts
  and thresholds, diffusion backend selection and per-backend sampling
  settings, synthetic event probabilities, and the bbox-aware mask
  dilation knobs. Every script reads this file through `Config`.

- **[`src/pipeline/config.py`](src/pipeline/config.py)** - thin YAML
  loader. Resolves `{work_root}` templating in the `data:` paths and
  exposes named sections (`cfg.data`, `cfg.tiling`, `cfg.segmentation`,
  `cfg.inpainting`, `cfg.synthetic`, `cfg.assembler`).

- **[`src/pipeline/io.py`](src/pipeline/io.py)** - small helpers:
  `load_image`, `save_image`, `read_json`, `write_json`,
  `polygons_to_mask` (rasterize a list of polygon vertices to a binary
  mask), `pil_to_numpy`, `numpy_to_pil`.

### Tiling

- **[`src/pipeline/tiler.py`](src/pipeline/tiler.py)** - cuts a full image
  into overlapping 512x512 tiles named `basename_xNNNN_yNNNN.png`. Key
  functions:
  - `tile_image(img, tile_size, overlap)` - generator yielding
    `(x, y, tile_pil)`.
  - `is_tile_nonempty(tile, min_nonempty_ratio)` - drops blank or
    near-uniform tiles using a simple median-deviation heuristic.
  - `save_tiles_for_image(img_path, out_dir, ...)` - convenience wrapper
    that writes every non-empty tile to disk.
  - `reassemble_tiles(...)` - glues tiles back into a full image,
    optionally substituting a `replacements` dict (`stem -> PIL`). Used
    when only a few tiles were modified and we need to stitch the full
    synthetic `after`.
  - `build_change_mask(...)` - composes a full-resolution binary mask
    from per-tile masks for the tiles that were modified.

### Segmentation

- **[`src/pipeline/segmentation.py`](src/pipeline/segmentation.py)** -
  defines a small `SegmentationModel` ABC and three concrete
  implementations:
  - `FallbackSLIC` - pure scikit-image SLIC superpixels. No deps, no GPU,
    used when transformers is unavailable.
  - `SegformerModel` - wraps `nvidia/segformer-b5-finetuned-ade-640-640`
    and returns a per-pixel ADE20K class ID map. This is what the change
    simulator uses to decide which region is "background terrain".
  - Factory `get_segmentation_model(name, cfg)` - picks the right model
    based on `segmentation.active_model` in config (`fallback` /
    `segformer` / `sam`).

- **[`src/pipeline/sam_integration.py`](src/pipeline/sam_integration.py)** -
  wraps `facebook/sam3` with two different entry points:
  - `SAMModel.segment(pil)` - open-vocabulary segmentation using a list
    of text prompts (tree, road, car, ...), returns a merged integer
    class map.
  - `SAMModel.detect_objects(pil, prompts, min_score, ...)` - returns
    per-instance dicts with `{mask, label, score, area_ratio}`. This is
    the **primary way** the pipeline locates removable objects.

### The core idea: change simulation and prompts

- **[`src/pipeline/change_simulator.py`](src/pipeline/change_simulator.py)** -
  where the `(change_mask, prompt, meta)` triplet is born. Three event
  types:
  - `simulate_appearance(seg_map)` - picks a large terrain region from
    SegFormer (grass, dirt, ...), draws a random irregular blob in it,
    and asks SD to *paint* a rock / bush / box onto that ground.
  - `simulate_disappearance(seg_map)` - picks a non-background segment,
    dilates it slightly, and asks SD to *erase* it back to ground.
  - `simulate_disappearance_targeted(detected_objects, seg_map)` - given
    SAM 3 instance masks, picks 1-2 detected objects (weighted by
    confidence) and builds a combined removal mask. Used whenever SAM 3
    actually found something concrete in the tile.
  - `simulate_change(...)` - top-level dispatcher: flips a coin against
    `synthetic.appearance_prob` (default 20%) and picks the right
    function.

- **[`src/pipeline/prompt_templates.py`](src/pipeline/prompt_templates.py)** -
  maps ADE20K class IDs to natural-language backgrounds (`class 9 ->
  "green grass field"`, `class 13 -> "dry brown earth ground"`, ...) and
  stores prompt templates per object type. It also tone-matches: on a
  green field it prefers rocks and bags, on a brown field it prefers
  bushes. The negative prompt is fixed and discourages seams and color
  fringes.

### Inpainting (the model-heavy part)

- **[`src/pipeline/inpainting.py`](src/pipeline/inpainting.py)** - the
  single workhorse class `InpaintingModel` plus its config factory
  `build_inpainter_from_cfg()`. Features:
  - **Three switchable backends** via the `_BACKENDS` registry:
    - `sd2` - `sd2-community/stable-diffusion-2-inpainting`, 512 native.
    - `sd15_realistic` - `Lykon/dreamshaper-8-inpainting`, 512 native,
      photorealistic SD1.5 fine-tune.
    - `sdxl` - `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`,
      1024 native. Enables `model_cpu_offload` + VAE slicing/tiling to
      fit on 8 GB VRAM.
  - **Three public entry points:**
    - `inpaint(image, mask, prompt)` - tile-level convenience: feathers
      mask, runs SD, alpha-blends the result back.
    - `inpaint_raw(image, mask, prompt)` - returns the raw SD output
      without any blending. The caller is responsible for blending.
    - `inpaint_object(full_image, obj_mask_fullimg, prompt, bbox)` -
      crops a padded region around the object, downsamples to the
      backend's native resolution if needed, inpaints, upsamples back,
      and returns a Poisson-blended crop ready to paste onto the full
      canvas.
  - **Bbox-aware mask dilation** (`_object_dilate_iterations`): instead
    of dilating the mask by a fixed 8 px, the dilation scales with
    object size (`max(8, min(48, 0.08 * min(bbox_w, bbox_h)))`). This
    eliminates the colored-edge artefact where a removed car's shadow
    was left visible just outside a too-small mask.
  - **Seamless blending** via `cv2.seamlessClone` (Poisson), with an
    alpha-feather fallback. The clone mask uses the same bbox-aware
    dilation as the SD feather, so SD's fill area and Poisson's blend
    area are always perfectly aligned.
  - **Safety checker disabled** on SD1/SD2 pipelines because the CLIP
    NSFW filter false-fires on plain aerial imagery and returns black
    frames.

### Synthetic pair generation

- **[`src/pipeline/synthetic.py`](src/pipeline/synthetic.py)** - higher-level
  orchestration that combines segmentation, detection, simulation, and
  inpainting into reusable building blocks:
  - `generate_synthetic_pair(before_pil, seg_map, inpaint_model, ...)` -
    one tile, one change, returns `(after, change_mask, meta)` with
    SSIM score attached.
  - `batch_generate(tile_paths, seg_model, inpaint_model, ...)` - runs
    over a list of tiles, writes `before/after_synth/change_mask/meta`
    per variant plus a `provenance.csv` log, filters by SSIM band.
  - `select_best_objects(full_image, sam_model, ...)` - the key function
    for object-centric mode: tiles the full image into 1024x1024 scan
    crops, runs SAM 3 detection on each, scores detections by visibility
    (`score * sqrt(area_ratio / 0.01)`), de-duplicates by IoU, and picks
    up to `max_objects` while enforcing spatial spread and label
    diversity.

- **[`src/pipeline/dataset.py`](src/pipeline/dataset.py)** - `Pipeline`
  class wrapping the classic three-stage batch:
  `tile_all() -> segment_tiles() -> generate_synthetic()`. Writes cached
  SegFormer maps to `workspace/masks/*_seg.npy`, writes synthetic pairs
  under `workspace/synthetic/<pair>/`, and appends every result to
  `provenance.csv` (kept / filtered / error).

### Scripts (entry points)

- **[`src/scripts/process_one.py`](src/scripts/process_one.py)** - runs
  the *tile-based* pipeline on a single pair (default `pair_0000` from
  config). For each of 5 randomly-sampled interesting tiles it produces a
  7-panel grid (`before`, semantic map, SAM 3 detections, change region,
  synthetic after, change mask, after+overlay). Good for eyeballing
  quality while tuning.

- **[`src/scripts/generate_pair.py`](src/scripts/generate_pair.py)** -
  runs the *object-centric* pipeline on a single pair at full
  resolution. Scans the whole image with SAM 3 at 1024x1024, picks the
  best 1-3 objects, inpaints each one directly on the full canvas
  using `inpaint_object`, and writes:
  - `synthetic_after.jpg` - the full-resolution synthetic after.
  - `change_mask.png` - full-resolution binary mask.
  - `overview.png` - side-by-side comparison with per-object crops.
  - `meta.json` - one entry per removed object with label, score,
    bbox, prompt.

- **[`src/scripts/compare_inpaint_backends.py`](src/scripts/compare_inpaint_backends.py)** -
  A/B-tests the three inpainting backends on identical input. Loads
  SegFormer + SAM 3 once, simulates one change per sampled tile, then
  loads each backend serially (they cannot co-exist in 8 GB VRAM),
  renders the same change with each, and saves
  `<tile>_compare.png = [Before | Change Region | SD2 | SD1.5 Realistic | SDXL | Mask]`.
  Supports `--tile <stem>` to pin a specific tile and `--n-samples N` to
  change random sampling.

- **[`src/scripts/run_segment_and_generate.py`](src/scripts/run_segment_and_generate.py)** -
  simplest batch driver: runs `Pipeline.segment_tiles()` + `Pipeline.generate_synthetic()`
  on everything already tiled.

## Models used

| Role | Model | Repo | VRAM (fp16) | Why this one |
|---|---|---|---|---|
| Semantic segmentation | SegFormer-B5 | `nvidia/segformer-b5-finetuned-ade-640-640` | ~0.6 GB | ADE20K gives ~150 classes; we only need the terrain categories to decide background tone. Fast (~60 ms per tile). |
| Object detection + masks | SAM 3 | `facebook/sam3` (gated) | ~1.5 GB | Open-vocabulary - takes text prompts like "rock" or "car" and returns instance masks. Strictly stronger than ad-hoc CNN detectors for this use case. |
| Inpainting (default) | SD2-Inpaint | `sd2-community/stable-diffusion-2-inpainting` | ~4.5 GB | 512 native, fast, good baseline. |
| Inpainting (alt 1) | DreamShaper-8 Inpaint | `Lykon/dreamshaper-8-inpainting` | ~3.0 GB | SD1.5 photorealistic fine-tune; often cleaner on natural outdoor scenes. |
| Inpainting (alt 2) | SDXL-Inpaint | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | ~5.5 GB with CPU offload | 1024 native; preserves detail on large crops. |

SAM 3 is a **gated** Hugging Face repo, so you must accept its license
on the web and authenticate once with a token (see
[Installation](#installation)).

## Installation

### 1. Python and CUDA

- Python 3.11 (the project is developed against cpython 3.11).
- A CUDA GPU with >= 6 GB VRAM is strongly recommended. The project is
  regularly tested on an RTX 4060 (8 GB).
- The transformers >= 5 security gate requires torch >= 2.6, so install
  the CUDA 12.4 wheels (not 12.1):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Everything else

```powershell
pip install -r requirements.txt
```

### 3. Hugging Face authentication (once)

SAM 3 is gated; accept the license at
<https://huggingface.co/facebook/sam3> using the same account whose
token you pass below, then persist the token:

```powershell
python -c "from huggingface_hub import login; login(token='hf_YOUR_REAL_TOKEN')"
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

The SegFormer and diffusion models are ungated and will download
automatically on first use.

### 4. Verify CUDA

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.6.0+cu124 True NVIDIA ...`.

## Quick start

### Put one real pair in

Drop any real aerial image pair under
`src/data/original_OCD_dataset/pair_0000/` with at least `before.jpg` (the
`after.jpg`, `annotations.json`, `after_binary_mask.png` files are only
used as reference for the GT overview panel; they're not needed to
generate synthetic data).

### Tile-based test (grid outputs for quick inspection)

```powershell
python .\src\scripts\process_one.py
```

Outputs: `src/data/workspace/synthetic/pair_0000/<tile>_grid.png` plus a
`pair_0000_gt_overview.png`.

### Full-resolution synthetic pair (the main dataset-generation path)

```powershell
python .\src\scripts\generate_pair.py          # defaults to testing.test_pair
python .\src\scripts\generate_pair.py pair_0011
```

Outputs: `src/data/workspace/synthetic/<pair>/full/{synthetic_after.jpg,
change_mask.png, overview.png, meta.json}`.

### Compare the three inpainters on the same tile

```powershell
python .\src\scripts\compare_inpaint_backends.py                           # 3 random tiles
python .\src\scripts\compare_inpaint_backends.py --tile before_x3584_y3136 # one specific tile
python .\src\scripts\compare_inpaint_backends.py pair_0003 --n-samples 5
```

Outputs: `src/data/workspace/synthetic/<pair>/<tile>_compare.png`.

### Switch inpaint backend

Edit a single line in [`src/config.yaml`](src/config.yaml):

```yaml
inpainting:
  backend: "sdxl"    # or "sd2" (default), "sd15_realistic"
```

All three scripts (`process_one`, `generate_pair`,
`compare_inpaint_backends`) honor this immediately.

## Configuration reference

See [`src/config.yaml`](src/config.yaml). The most common knobs:

| Section | Key | Effect |
|---|---|---|
| `tiling` | `tile_size`, `overlap` | 512 / 64 are a good default for 8K drone frames. Raising `overlap` helps seam-hiding at the cost of more tiles. |
| `segmentation.sam` | `detection_prompts` | What SAM 3 looks for. Domain-specific: add "person", "pallet", etc. as needed. |
| `segmentation.sam` | `detection_score_threshold` | Minimum SAM 3 confidence. Higher = fewer, more certain objects. |
| `inpainting` | `backend` | `sd2` / `sd15_realistic` / `sdxl`. Switches the whole pipeline. |
| `inpainting` | `object_dilate_ratio` | Fraction of object bbox that gets added to the mask before inpainting. 0.08 = 8%. Raise if you still see halos; set to 0 to revert to fixed-size dilation. |
| `inpainting` | `object_dilate_max_px` | Hard cap on the bbox-aware dilation. Prevents oversmoothing on huge objects. |
| `inpainting` | `blend_mode` | `poisson` (seamlessClone) or `alpha` (feathered blend). |
| `inpainting.<backend>` | `num_inference_steps`, `guidance_scale`, `strength` | Standard diffusion knobs. Defaults tuned per backend. |
| `synthetic` | `appearance_prob` | P(appearance event). Rest is disappearance. |
| `synthetic` | `max_changes`, `min_object_distance`, `max_per_label` | Variety and spatial-spread controls for `generate_pair.py`. |

## Design decisions worth knowing

- **Why SAM 3 instead of YOLO or a CNN detector?** SAM 3 accepts
  free-form text prompts, so adding a new object category is just adding
  a string to `detection_prompts`. No retraining.

- **Why SegFormer if SAM 3 already segments?** SegFormer gives *semantic*
  classes (ADE20K's grass, dirt, sand, ...) that we need to decide what
  kind of background a removed object should be replaced with, and to
  pick prompt wording that contrasts with that background. SAM 3 gives
  *instances* but not abstract class categories.

- **Why three inpaint backends?** Different trade-offs. SD2 is fast and
  robust, DreamShaper (SD1.5) is often more photorealistic on natural
  scenes, SDXL preserves the most detail at 1024 but needs CPU offload
  on small cards. The factory keeps the public API identical so
  downstream code never branches.

- **Why Poisson blending?** `cv2.seamlessClone` matches the source's
  gradients to the destination's colors at the boundary. For object
  removal this hides the transition between SD's freshly-painted ground
  and the original surrounding ground even if they differ slightly in
  hue. Alpha blending is kept as a fallback.

- **Why bbox-aware mask dilation?** SAM 3 traces the object silhouette,
  but real objects have shadows and ground reflections a few percent of
  their own size outside that silhouette. A fixed 8 px dilation was too
  small for cars (~500 px wide) and left a colored edge halo. Scaling
  the dilation with the bounding box (8% of the shorter side, capped at
  48 px) erases the shadow ring cleanly.

- **Object-centric vs tile-based?** Tile-based is simpler and yields
  nice grids for inspection but every change is tile-local. Object-centric
  runs at full resolution, picks 1-3 globally-best objects, and emits a
  clean `(before, after, mask)` triplet suitable for dataset assembly.
  Use `generate_pair.py` for real dataset generation.

## Outputs on disk

Running `process_one.py` produces (for pair `pair_0000`):

```
src/data/workspace/
  tiles/pair_0000/before/before_xNNNN_yNNNN.png     # cut tiles
  synthetic/pair_0000/
    pair_0000_gt_overview.png                       # real GT reference
    before_xNNNN_yNNNN_grid.png                     # per-tile 7-panel grid
```

Running `generate_pair.py pair_0011`:

```
src/data/workspace/synthetic/pair_0011/full/
  synthetic_after.jpg     # full-resolution synthetic after
  change_mask.png         # full-resolution binary mask (0 / 255)
  overview.png            # side-by-side comparison with per-object crops
  meta.json               # [{label, score, bbox, prompt, ...}, ...]
```

Running `compare_inpaint_backends.py --tile before_x3584_y3136`:

```
src/data/workspace/synthetic/pair_0000/
  before_x3584_y3136_compare.png
    # panels: [ Before | Change Region | SD2 | SD1.5 Realistic | SDXL | Mask ]
```

## Troubleshooting

- **`401 Cannot access gated repo facebook/sam3`** - you haven't accepted
  the license or your token is wrong. Re-run the `huggingface_hub.login`
  one-liner from [Installation](#installation) with a fresh token.

- **`Torch not compiled with CUDA enabled`** - you installed the CPU
  wheel. Reinstall with the cu124 index (see [Installation](#installation)).

- **`ValueError: ... upgrade torch to at least v2.6 ...`
  (CVE-2025-32434)** - same root cause: cu121 wheels max out at
  torch 2.5. Use the cu124 index.

- **Black frames from SD1.5** - the CLIP safety checker fired. This is
  already disabled in `InpaintingModel`, but if you bypass the class and
  use diffusers directly, remember to `pipe.safety_checker = None`.

- **OOM when loading SDXL on 8 GB** - make sure `needs_cpu_offload` is
  True in the `_BACKENDS` registry (it is, by default). The pipeline
  calls `enable_model_cpu_offload` + `pipe.vae.enable_slicing` /
  `enable_tiling` which together fit SDXL under 7 GB at runtime.

- **Visible object halo after removal** - your `object_dilate_ratio` is
  too small for the object type. Raise it to 0.10 or 0.12 and bump
  `object_dilate_max_px` to 64.
