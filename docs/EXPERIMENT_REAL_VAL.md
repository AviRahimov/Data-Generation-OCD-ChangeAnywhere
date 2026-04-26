# Experiment: do synthetic pairs help on real validation?

This is the highest-value check before scaling generation to thousands of pairs. Visual quality of synthetics does not prove they improve your **downstream change detector** on **real** before/after imagery.

## Goal

On a **fixed real validation split** (registered pairs you trust), compare detector performance when training with:

- **A.** Real training data only (baseline)
- **B.** Real training data + N synthetic triplets from this repo (`before`, `synthetic_after`, `change_mask`)

Use the **same** architecture, training budget, and val split for A vs B.

## Suggested protocol

1. **Freeze the val set** — e.g. 10–20 held-out real pairs, never used in training.
2. **Build two train pools**
   - Train-A: your usual real training pairs.
   - Train-B: Train-A plus N synthetics from `generate_dataset.py` (match resolution / preprocessing to what the model expects).
3. **Train two checkpoints** — same hyperparameters, seeds where possible, same number of steps or early-stop on the same monitored metric.
4. **Report** precision/recall/F1 (or official metric for your benchmark) on the **same val split** for A and B.
5. **Decide** — if B does not beat A, inspect failure modes (domain gap, mask noise, class imbalance) before generating more data.

## Practical notes

- Start with **small N** (e.g. 200–500 synthetics) to see a trend before paying for 3k+ generations.
- If synthetic masks are tighter than blended regions, prefer the **diff-based** `change_mask` from `generate_full_image_pair` (current default) or tune inpainting/blend so labels match what the network sees.
- Optionally log **per-image** val errors to see if synthetics correlate with specific terrain or failure cases.

## Repo artifacts

- Full-image outputs: `gen_*/before.jpg`, `synthetic_after.jpg`, `change_mask.png`, `meta.json`
- Batch manifest: `manifest.csv` (includes `status` such as `ok`, `empty`, `quality_reject` when quality gate is enabled)

## Optional: comparing detectors / segmenters (this repo, not the val F1 test)

`src/scripts/eval_detection_modes.py` is only for **developing** the SAM3 detection stack: it scores **text** vs **auto** (and optionally **SAM2.1** automatic masks via `segmentation.sam2` + `--with-sam2`) on real `before.jpg` files and writes CSV + comparison JPEGs. That does **not** replace the protocol above; it is a separate tooling track documented in [`docs/PIPELINES.md`](PIPELINES.md) and the README.

This document is a **protocol only**; training code for BAN, CDMamba, ChangeMamba, etc. lives in your detector repository.
