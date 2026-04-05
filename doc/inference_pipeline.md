# Cell Segmentation Benchmarking Inference Pipeline

## Purpose

This repo now includes a standardized Python inference pipeline for four model families that were previously run from notebooks:

- Cellpose SAM
- CellSAM
- CellViT v3 using the SAM backbone only
- StarDist

The goal is to run them consistently on fixed-size tile datasets, keep filenames stable, and write outputs in a layout that can be consumed directly by the next benchmarking step.

## Input Expectations

- Default input root: `data/Monusac/tiles_256/`
- Expected data shape: RGB image tiles, typically `256x256`
- Tile content can vary from empty background to dense multi-cell patches
- If an input manifest such as `all_patches_dataset.csv` is present, the scripts reuse it and propagate its columns into the per-model prediction manifest

If you are benchmarking a different tiled dataset, change `INPUT_DIR` at the top of the script you plan to run or pass `--input-dir /your/path`.

## Scripts

- `scripts/02-inference/run_cellpose_sam.py`
  - Runs Cellpose SAM in batch mode over tiles
  - Uses the notebook thresholds as defaults: `flow_threshold=0.4`, `cellprob_threshold=0.0`
- `scripts/02-inference/run_cellsam.py`
  - Runs `cellsam_pipeline(...)` per tile
  - Uses `DEEPCELL_ACCESS_TOKEN` from the environment when available
- `scripts/02-inference/run_cellvit_sam.py`
  - Uses CellViT SAM only
  - Pads each tile to `1024x1024`, runs inference, then crops the predicted label map back to the original tile bounds
- `scripts/02-inference/run_stardist.py`
  - Uses the pretrained `2D_versatile_he` StarDist model
  - Normalizes tiles with percentile normalization before prediction
- `scripts/02-inference/run_all.py`
  - Launches all four scripts through their matching Pixi environments
  - Uses multiprocessing-based scheduling with a top-level default of `8` workers and `24GB` RAM
  - Uses `GPU_SLOTS = 1` by default, so jobs stay single-GPU safe unless you explicitly raise the slot count on a multi-GPU machine

## Output Contract

Each model writes to its own directory:

- `inference/benchmarking/monusac/cellpose_sam/`
- `inference/benchmarking/monusac/cellsam/`
- `inference/benchmarking/monusac/cellvit_sam/`
- `inference/benchmarking/monusac/stardist/`

Within each directory:

- One `uint16` instance mask per input tile
- The relative filename is preserved from the original tile
- `predictions.csv` stores:
  - original source path
  - relative tile path
  - predicted mask path
  - image dimensions and metadata
  - instance count
  - runtime per tile
  - any carried-through manifest columns
- `failed.csv` stores per-tile failures without stopping the rest of the run

This layout keeps evaluation simple because the prediction tree mirrors the source tile tree rather than inventing a new naming scheme.

## Expected Mask Semantics

All four scripts export instance label masks:

- background is `0`
- each detected cell instance receives a positive integer ID
- empty tiles should therefore produce all-zero masks

The masks are written as 16-bit PNGs so the label IDs survive round-tripping cleanly.

## Quick Start

### 1. Install environments on the target Linux machine

```bash
pixi install
```

### 2. Run one model

```bash
pixi run -e cellpose python scripts/02-inference/run_cellpose_sam.py
pixi run -e cellsam python scripts/02-inference/run_cellsam.py
pixi run -e cellvit python scripts/02-inference/run_cellvit_sam.py
pixi run -e stardist python scripts/02-inference/run_stardist.py
```

### 3. Run all four models

```bash
pixi run python scripts/02-inference/run_all.py
```

## Common Adjustments

Every script keeps its editable settings at the very top. The most common values to change are:

- `INPUT_DIR`
- `OUTPUT_DIR` or `OUTPUT_ROOT`
- `WORKERS`
- `RAM_LIMIT_GB`
- `BATCH_SIZE` for Cellpose SAM or CellViT SAM
- `DEEPCELL_ACCESS_TOKEN` for CellSAM if you prefer to hardwire it locally instead of exporting it in the shell

## Notes and Constraints

- The Pixi workspace in this repo is pinned to `linux-64`, so full model execution is expected on the target HPC/Linux environment rather than macOS.
- `run_all.py` is multiprocessing-based, but the actual number of simultaneous model launches is also gated by `MODEL_RAM_GB` and `GPU_SLOTS`.
- CellViT’s upstream tooling is WSI-oriented. This repo’s `run_cellvit_sam.py` therefore implements direct patch inference by reusing the model loader and post-processing code while handling `256px` tiles locally.
