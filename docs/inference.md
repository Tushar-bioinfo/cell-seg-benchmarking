# Cell Segmentation Benchmarking Inference

## Scope

This doc is the source of truth for the standardized Python inference pipeline in `scripts/02-inference/`.

Supported model families:

- Cellpose SAM
- CellSAM
- CellViT v3 using the SAM backbone only
- StarDist

## Inputs

- Default input root: `data/Monusac/tiles_256/`
- Expected input: tiled RGB image patches, typically `256x256`
- If `all_patches_dataset.csv` or another tile manifest is present, inference scripts reuse it and carry its columns into `predictions.csv`
- For another tiled dataset, change `INPUT_DIR` in the target script or pass `--input-dir`

## Paths

- Inference scripts: `scripts/02-inference/`
- Shared helpers: `scripts/02-inference/benchmark_inference_utils.py`
- Output root: `inference/benchmarking/monusac/`

Per-model output directories:

- `inference/benchmarking/monusac/cellpose_sam/`
- `inference/benchmarking/monusac/cellsam/`
- `inference/benchmarking/monusac/cellvit_sam/`
- `inference/benchmarking/monusac/stardist/`

## Scripts

- `run_cellpose_sam.py`
  - Batch Cellpose SAM inference
  - Default notebook-aligned thresholds: `flow_threshold=0.4`, `cellprob_threshold=0.0`
- `run_cellsam.py`
  - Runs `cellsam_pipeline(...)` per tile
  - Uses `DEEPCELL_ACCESS_TOKEN` from the environment when available
- `run_cellvit_sam.py`
  - Uses CellViT SAM only
  - Pads each tile to `1024x1024`, runs inference, then crops back to the original tile size
- `run_stardist.py`
  - Uses pretrained `2D_versatile_he`
  - Applies percentile normalization before prediction
- `run_all.py`
  - Launches all four scripts through their matching Pixi environments
  - Uses multiprocessing scheduling
  - Default scheduler limits: `WORKERS = 8`, `RAM_LIMIT_GB = 24`, `GPU_SLOTS = 1`

## Outputs

Each model writes:

- One `uint16` instance mask per input tile
- The same relative filename layout as the source tile tree
- `predictions.csv`
- `failed.csv`

`predictions.csv` includes:

- original source path
- relative tile path
- predicted mask path
- image dimensions and metadata
- instance count
- runtime per tile
- any propagated manifest columns

`failed.csv` records per-tile failures without aborting the full run.

## Mask Semantics

- Background label is `0`
- Each detected instance gets a positive integer ID
- Empty tiles should produce all-zero masks
- Masks are written as 16-bit PNGs

## Dependencies and Execution

Install environments from the repo root:

```bash
pixi install
```

Run one model:

```bash
pixi run -e cellpose python scripts/02-inference/run_cellpose_sam.py
pixi run -e cellsam python scripts/02-inference/run_cellsam.py
pixi run -e cellvit python scripts/02-inference/run_cellvit_sam.py
pixi run -e stardist python scripts/02-inference/run_stardist.py
```

Run all models:

```bash
pixi run python scripts/02-inference/run_all.py
```

## Config Knobs

Common top-of-file settings:

- `INPUT_DIR`
- `OUTPUT_DIR` or `OUTPUT_ROOT`
- `WORKERS`
- `RAM_LIMIT_GB`
- `BATCH_SIZE` for Cellpose SAM and CellViT SAM
- `DEEPCELL_ACCESS_TOKEN` for CellSAM

Keep these config blocks intact when editing the inference scripts.

## Constraints

- The Pixi workspace is pinned to `linux-64`
- Full execution is expected on Linux or HPC rather than macOS
- `run_all.py` concurrency is gated by both `MODEL_RAM_GB` and `GPU_SLOTS`
- `GPU_SLOTS = 1` keeps GPU-backed jobs serialized unless explicitly raised
- `run_cellvit_sam.py` uses local patch inference to adapt CellViT’s WSI-oriented tooling to `256px` tiles
