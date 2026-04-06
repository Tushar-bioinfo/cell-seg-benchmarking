# Flow 1 Context

`flow_1` is the reusable, dataset-generic benchmarking workflow under `scripts/benchmarking/flow_1/`.
It is meant to start from paired image and mask files, standardize them into manifests, run inference, and then evaluate predictions from the recorded manifests.

## Scope

- Use this doc for the generic `flow_1` path and data contract.
- This flow is not tied to a specific dataset name or folder like `Monusac`.
- The raw starting point is paired images and masks, usually:
  - images: `.png`
  - masks: `.png`, `.tif`, or `.tiff`
- The current flow does not implement dataset download. It starts from local files that already exist.

## Workflow Order

1. Raw paired dataset root
2. Optional rescale with `scripts/benchmarking/flow_1/rescale_dataset.py`
3. Optional tiling with `scripts/benchmarking/flow_1/tile_dataset.py`
4. Model prediction with `scripts/benchmarking/flow_1/run_*.py` or `scripts/benchmarking/flow_1/run_all.py`
5. Evaluation with `scripts/benchmarking/flow_1/evaluate_predictions.py`

## Entry Points

- Orchestrator: `scripts/benchmarking/flow_1/run_workflow.py`
- Rescale stage: `scripts/benchmarking/flow_1/rescale_dataset.py`
- Tile stage: `scripts/benchmarking/flow_1/tile_dataset.py`
- Batch inference: `scripts/benchmarking/flow_1/run_all.py`
- Per-model inference:
  - `scripts/benchmarking/flow_1/run_cellpose_sam.py`
  - `scripts/benchmarking/flow_1/run_cellsam.py`
  - `scripts/benchmarking/flow_1/run_cellvit_sam.py`
  - `scripts/benchmarking/flow_1/run_stardist.py`
- Evaluation: `scripts/benchmarking/flow_1/evaluate_predictions.py`

## Raw Input Contract

- Raw input can come from:
  - one tree containing paired files such as `sample_image.png` and `sample_mask.tiff`
  - separate image and mask subdirectories
  - a CSV manifest with image and mask path columns
- Generic pairing defaults:
  - `pair_mode = suffix`
  - `image_suffix_token = _image`
  - `mask_suffix_token = _mask`
- Manifest auto-discovery for raw dataset input checks:
  - `dataset_manifest.csv`
  - `extraction_summary.csv`

## Pair Resolution Rules

- In manifest mode, the prep stages look for image columns in this order:
  - `image_path`
  - `source_image_path`
  - `path`
- In manifest mode, the prep stages look for mask columns in this order:
  - `mask_path`
  - `source_mask_path`
- Sample ID is chosen from, in order:
  - explicit sample-id column if added later in code
  - `sample_id`
  - `unique_id`
  - `image_id`
  - `id`
  - otherwise derived from the relative image path
- Derived sample IDs are sanitized to `[A-Za-z0-9._-]` and path separators are flattened with `__`.

## Path Flow

Default path flow if you use `run_workflow.py` without overrides:

1. Raw dataset root:
   - `data/benchmark_input/`
2. Rescaled dataset root:
   - `data/benchmark_input/rescaled/`
3. Tiled dataset root:
   - `data/benchmark_input/rescaled/tiles_256/`
   - or `data/benchmark_input/tiles_256/` if tiling without rescale
4. Prediction root:
   - `inference/benchmarking/benchmark_input/`
5. Evaluation output root:
   - `<prediction-root>/_evaluation/`

The path flow is override-friendly. Treat those defaults as placeholders, not fixed dataset rules.

## Shapes And Dtypes

- Raw images are loaded as RGB `uint8` arrays with shape `(H, W, 3)`.
- Raw masks are loaded as 2D `uint16` instance-label arrays with shape `(H, W)`.
- Mask loading accepts:
  - already-2D masks
  - singleton dimensions that squeeze to 2D
  - grayscale-replicated RGB/RGBA masks
- Mask loading rejects ambiguous multi-channel masks.
- Background label is `0`.
- Any positive integer label is treated as an instance.

## Rescale Stage

- Script: `scripts/benchmarking/flow_1/rescale_dataset.py`
- Purpose:
  - rescale paired image/mask samples
  - preserve instance-label semantics
  - write a dataset-level manifest for later stages
- Default output root:
  - `<input-root>/rescaled/`

### Rescale Paths

- Output images go under:
  - `images/<original-relative-image-path>.png`
- Output masks go under:
  - `masks/<original-relative-mask-path>.png`
- Stage manifest:
  - `dataset_manifest.csv`

### Rescale Shape Rules

- Image output shape is `(round(H * scale_factor), round(W * scale_factor), 3)`.
- Mask output shape is `(round(H * scale_factor), round(W * scale_factor))`.
- Output image format is always PNG.
- Output mask format is always PNG with `uint16` labels.

### Rescale Semantics

- `scale_factor = target_magnification / source_magnification`
- RGB images use PIL `BOX` resampling.
- Instance masks are resized label-by-label, not by interpolating the merged label image directly.
- Downscaling uses `BOX` for binary masks.
- Upscaling uses `NEAREST` for binary masks.
- If labels compete for the same output pixel, the label with the higher resized occupancy score wins.

### Rescale Manifest Columns

Core columns written by `dataset_manifest.csv`:

- `sample_id`
- `sample_image_path`
- `sample_mask_path`
- `sample_image_relative_path`
- `sample_mask_relative_path`
- `image_path`
- `mask_path`
- `source_height`
- `source_width`
- `target_height`
- `target_width`
- `source_magnification`
- `target_magnification`
- `scale_factor`
- `min_instance_fraction`
- `original_instance_count`
- `resized_instance_count`
- `dropped_instance_count`
- `dropped_instance_labels`
- `status`

Any metadata propagated from the input manifest is preserved. If a propagated column conflicts with a reserved output name, it is renamed with an `input_` prefix.

## Tile Stage

- Script: `scripts/benchmarking/flow_1/tile_dataset.py`
- Purpose:
  - split paired images and masks into fixed-size, full patches
  - write one global patch manifest and one per-sample manifest
- Default output root:
  - `<input-root>/tiles_<patch_size>/`

### Tile Paths

- Per-sample patch directory:
  - `samples/<sample_id>/`
- Per-sample patch manifest:
  - `samples/<sample_id>/dataset.csv`
- Global patch manifest:
  - `all_patches_dataset.csv`
- Global sample summary:
  - `sample_patch_summary.csv`

### Tile Naming Rules

- Patch descriptor:
  - `{x0:05d}x_{y0:05d}y`
- Patch image file:
  - `<descriptor>_image.png`
- Patch mask file:
  - `<descriptor>_mask.png`
- Patch ID:
  - `<sample_id>.<descriptor>`

### Tile Shape Rules

- Only full patches are written.
- Default stride is the patch size.
- If `stride == patch_size`, tiling is non-overlapping.
- Right-edge and bottom-edge remainders are dropped.
- If `H < patch_size` or `W < patch_size`, patch count is `0`.
- Even with `0` patches, the stage still writes:
  - a per-sample `dataset.csv`
  - a row in `sample_patch_summary.csv`

### Tile Manifest Columns

Global and per-sample patch manifests include at least:

- `sample_id`
- `sample_image_path`
- `sample_mask_path`
- `sample_image_relative_path`
- `sample_mask_relative_path`
- `patch_id`
- `patch_index`
- `patch_row`
- `patch_col`
- `x0`
- `y0`
- `x1`
- `y1`
- `patch_size`
- `stride`
- `source_height`
- `source_width`
- `image_path`
- `mask_path`

Sample summary rows include at least:

- `sample_id`
- `sample_image_path`
- `sample_mask_path`
- `sample_image_relative_path`
- `sample_mask_relative_path`
- `patches_dir`
- `dataset_csv`
- `patch_size`
- `stride`
- `source_height`
- `source_width`
- `patch_rows`
- `patch_cols`
- `patch_count`

As with rescaling, passthrough input metadata is preserved, with `input_` prefixing on name collisions.

## Inference Stage

- Scripts:
  - `run_cellpose_sam.py`
  - `run_cellsam.py`
  - `run_cellvit_sam.py`
  - `run_stardist.py`
  - `run_all.py`
- Intended input:
  - the tiled dataset root
  - usually through `all_patches_dataset.csv`

### Inference Input Discovery

The shared inference helper looks for a manifest in this order:

- `<input-dir>/all_patches_dataset.csv`
- `<input-dir>/dataset.csv`
- `<input-dir>/patches.csv`
- `<input-dir>/../all_patches_dataset.csv`

If no manifest is found, it scans the input directory directly.

### Inference Image Filtering

- Supported image inputs:
  - `.png`
  - `.jpg`
  - `.jpeg`
  - `.tif`
  - `.tiff`
  - `.bmp`
- Files whose stem ends with these tokens are skipped during image discovery:
  - `_mask`
  - `_masks`
  - `_flows`

### Inference Path Rules

- One model output directory is created per model under the prediction root.
- Output mask paths preserve the input image relative layout.
- The output mask extension is `.png`.
- Important detail:
  - the inference helper changes the extension, but does not rewrite `_image` to `_mask`
  - if the input tile is `samples/foo/00000x_00000y_image.png`, the predicted mask path will also use the same relative stem under the model output directory

### Inference Outputs

Each model writes:

- one predicted instance mask per input image
- `predictions.csv`
- `failed.csv`

### Predictions Manifest Columns

`predictions.csv` includes the propagated tile-manifest metadata plus at least:

- `model_name`
- `status`
- `source_image_path`
- `relative_image_path`
- `source_image_name`
- `source_image_stem`
- `predicted_mask_path`
- `predicted_mask_name`
- `predicted_mask_relative_path`
- `image_width`
- `image_height`
- `image_mode`
- `image_format`
- `image_metadata_json`
- `instance_count`
- `runtime_seconds`

`failed.csv` includes the propagated metadata plus at least:

- `model_name`
- `status`
- `source_image_path`
- `relative_image_path`
- `source_image_name`
- `source_image_stem`
- `predicted_mask_path`
- `instance_count`
- `runtime_seconds`
- `error`

## Evaluation Stage

- Script: `scripts/benchmarking/flow_1/evaluate_predictions.py`
- Wrapper target:
  - `scripts/benchmarking/monusac_segmentation_evaluation.py`
- Even though the backing evaluator filename is legacy, the evaluation semantics are generic for instance masks.

### Evaluation Assumptions

- GT and predicted masks must have identical shapes.
- Masks are treated as 2D instance-label arrays.
- Background is `0`.
- Any positive integer is foreground.
- Numeric label IDs do not need to match between GT and prediction.
- Default instance match threshold is `0.5`.

### Evaluation Path Rules

- Evaluation reads model directories under the prediction root.
- It uses each model directory’s `predictions.csv` as the source of truth.
- Required `predictions.csv` columns:
  - `predicted_mask_path`
  - `mask_path`
- If `mask_path` is relative, the evaluator derives the tile root from:
  - `source_image_path`
  - plus `relative_image_path` or `image_path`

### Evaluation Outputs

- Per-model CSV:
  - `<prediction-root>/_evaluation/<model_name>_evaluation.csv`
- Optional combined CSV:
  - `<prediction-root>/_evaluation/all_models_evaluation.csv`
  - or a custom name from `--combined-csv-name`

### Evaluation Status Rules

- The evaluator processes successful prediction rows from `predictions.csv`.
- It does not merge in `failed.csv`.
- Row-level errors are recorded in the evaluation CSV instead of aborting the entire model.

## Important Arguments For Fast Reuse

Use these first before editing code.

### Generic Dataset Pairing

- `--in`
- `--manifest`
- `--images-subdir`
- `--masks-subdir`
- `--pair-mode`
- `--image-suffix-token`
- `--mask-suffix-token`
- `--image-exts`
- `--mask-exts`
- `--sample-id`
- `--non-recursive`

### Rescaling

- `--out`
- `--source-magnification`
- `--target-magnification`
- `--min-instance-fraction`
- `--overwrite`

### Tiling

- `--out`
- `--patch-size`
- `--stride`
- `--overwrite`

### Batch Inference

- `--in`
- `--out`
- `--manifest`
- `--models`
- `--workers`
- `--ram-limit-gb`
- `--gpu-slots`
- `--limit`
- `--overwrite`

### Per-Model Inference

Common:

- `--in`
- `--out`
- `--manifest`
- `--workers`
- `--ram-limit-gb`
- `--gpu-index`
- `--limit`
- `--overwrite`
- `--non-recursive`

Model-specific wrapper args:

- Cellpose SAM:
  - `--batch-size`
- CellViT SAM:
  - `--batch-size`

### Evaluation

- `--in`
- `--out`
- `--model-names`
- `--threshold`
- `--save-combined-csv`
- `--combined-csv-name`

## Orchestrator Notes

`run_workflow.py` is the fastest way to run the full path, but the stages are intentionally separable.

Useful flags:

- `--skip-rescale`
- `--skip-tile`
- `--skip-predict`
- `--skip-evaluate`

Important path behavior:

- If you skip tiling, prediction input falls back to `--predict-in` or the workflow default prediction input root.
- If you skip prediction, evaluation reads from `--predict-out`.
- If you skip tiling but still predict, passing `--manifest` can still control which images become inference records.

## Model-Specific Caveats

- `run_cellvit_sam.py` depends on the original CellViT SAM pipeline and requires CUDA.
- The original CellViT script pads tiles to `1024x1024` and crops predictions back to the original tile size.
- If an input tile is larger than the fixed CellViT padded size, that model path can fail.
- `run_all.py` schedules models with both RAM and GPU slot limits.
- Default `gpu_slots = 1` serializes GPU-backed jobs.

## Logging And Runtime Behavior

- All `flow_1` stage scripts log:
  - start
  - key input and output paths
  - selected sample count or model list
  - end summary
  - elapsed time
- `run_all.py` also writes one log per model under:
  - `<prediction-root>/_logs/`

## Practical Defaults

- Default dataset tag used in top-of-file config:
  - `benchmark_input`
- That tag only sets placeholder default paths.
- For reuse on a new dataset, prefer passing CLI paths first.
- Only edit top-of-file defaults if the same dataset layout will be reused repeatedly.

## Quick Examples

Raw paired files in one tree:

```bash
python scripts/benchmarking/flow_1/run_workflow.py \
  --in /abs/path/to/dataset \
  --pair-mode suffix
```

Separate image and mask folders:

```bash
python scripts/benchmarking/flow_1/run_workflow.py \
  --in /abs/path/to/dataset \
  --images-subdir images \
  --masks-subdir masks
```

Manifest-driven raw input:

```bash
python scripts/benchmarking/flow_1/run_workflow.py \
  --in /abs/path/to/dataset \
  --manifest /abs/path/to/dataset_manifest.csv
```
