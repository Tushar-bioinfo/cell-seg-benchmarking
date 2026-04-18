# Morphology Labels Context

## Scope

- This file is the compact source of truth for `scripts/benchmarking/morphology/extract_morphology_features.py`.
- Use it when working on morphology feature extraction, manifest interpretation, path handling, output joins, or HPC execution for mask-derived features.
- Prefer this file over reopening the full morphology script unless implementation details must be verified.

## Inputs

- Input is one manifest CSV with one row per image or image patch.
- Required columns:
  - `image_path`
  - `mask_path`
- Common passthrough metadata that should be preserved in outputs when present:
  - `patch_id`
  - `unique_id`
  - `image_id`
  - `patient_id`
  - `split`
  - `model_name`
- The workflow adds:
  - `input_row_index`

## Path Resolution

- Recorded manifest paths are the source of truth.
- `mask_path` is required for every processed row.
- `image_path` is required only when `--include-intensity` is enabled.
- Relative paths are resolved in this order:
  - manifest parent directory
  - project root
  - current working directory
- Absolute paths are used directly.
- Preserve recorded manifest path strings in outputs; do not rewrite them to absolute paths in the main CSV outputs.

## Mask And Image Formats

- Supported mask suffixes:
  - `.png`
  - `.tif`
  - `.tiff`
- Masks must decode to an unambiguous 2D label array.
- Accepted mask layouts:
  - plain 2D arrays
  - singleton-expandable arrays that squeeze to 2D
  - grayscale RGB or RGBA arrays with replicated channels
- Reject ambiguous multichannel masks.
- Accepted mask dtypes:
  - boolean
  - integer
  - integer-valued floating point
- Reject:
  - negative labels
  - non-finite values
  - non-integer floating labels
- Image loading for `--include-intensity` converts the recorded image to RGB.
- Image and mask shapes must match when intensity features are requested.

## Label Semantics

- Background label is `0`.
- Any positive value is foreground.
- Binary masks are converted to connected-component labels with 8-connectivity before morphology extraction.
- Multi-label instance masks keep their original positive label IDs.
- `--min-area` filtering is applied after labeling and before both instance-level and patch-level summaries.
- `--max-objects-per-image` is enforced after filtering on the retained object set.

## Features

- Instance-level output includes:
  - `label_id`
  - `area_pixels`
  - `perimeter_pixels`
  - `equivalent_diameter`
  - `major_axis_length`
  - `minor_axis_length`
  - `eccentricity`
  - `solidity`
  - `extent`
  - `orientation_deg`
  - `bbox_min_row`
  - `bbox_min_col`
  - `bbox_max_row`
  - `bbox_max_col`
  - `centroid_row`
  - `centroid_col`
  - `aspect_ratio`
  - `circularity`
  - `perimeter_area_ratio`
  - `bbox_area`
  - `fill_ratio`
- Patch-level summary output includes:
  - `num_objects`
  - `total_mask_area`
  - `foreground_fraction`
  - `mean_area`
  - `median_area`
  - `std_area`
  - `mean_eccentricity`
  - `mean_solidity`
  - `mean_circularity`
- Optional `--include-intensity` patch-level RGB summaries:
  - `image_mean_r`, `image_mean_g`, `image_mean_b`
  - `image_std_r`, `image_std_g`, `image_std_b`
  - `foreground_mean_r`, `foreground_mean_g`, `foreground_mean_b`
  - `foreground_std_r`, `foreground_std_g`, `foreground_std_b`

## Outputs

- Output root is user-supplied through `--outdir`.
- Managed outputs:
  - `<outdir>/instance_features.csv`
  - `<outdir>/patch_features.csv`
  - `<outdir>/failed_rows.csv`
  - `<outdir>/processing_summary.json`
  - `<outdir>/run.log`
- Output join contract:
  - preserve manifest metadata columns
  - preserve recorded path strings
  - keep `input_row_index`
  - prefer `patch_id` or `unique_id` when downstream joins need a stable upstream key
- Level behavior:
  - `--level instance` writes one row per retained object
  - `--level patch` writes one row per successfully processed manifest row
  - `--level both` writes both tables

## Failure Semantics

- Default behavior is skip-with-warning.
- `--fail-fast` stops on the first bad row.
- `failed_rows.csv` includes manifest passthrough columns plus:
  - `input_row_index`
  - `stage`
  - `error_message`
- Failure stages:
  - `path_resolution`
  - `mask_decode`
  - `mask_validation`
  - `image_decode`
  - `feature_extraction`

## Performance And Execution

- Processing is row-by-row; images and masks are not preloaded for the whole manifest.
- Multiprocessing uses `Pool.imap_unordered(...)`.
- `--batch-size` maps to the multiprocessing chunk size.
- CSV outputs are written incrementally to keep memory bounded.
- `--ram-gb` is a logging and Slurm guidance hint only.
- Primary entrypoint:
  - `scripts/benchmarking/morphology/extract_morphology_features.py`
- HPC wrapper:
  - `scripts/benchmarking/morphology/run_extract_morphology_features.slurm`

## Practical Defaults

- Keep manifest metadata untouched unless a new derived field is explicitly required.
- Prefer manifest-relative or repo-relative recorded paths over reconstructing dataset roots.
- Treat `patch_id` as the preferred patch-level key when available.
- Treat empty retained-object sets as valid rows:
  - `num_objects = 0`
  - `total_mask_area = 0`
  - `foreground_fraction = 0`
  - aggregate object statistics stay `NaN`
