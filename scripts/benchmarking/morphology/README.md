# Morphology Feature Extraction

This workflow extracts instance-level and patch-level morphology features from segmentation masks listed in a manifest CSV. It is designed for benchmarking-style datasets where `image_path` and `mask_path` are already recorded in the manifest and should remain the source of truth.

## Expected manifest columns

Required columns:

- `image_path`
- `mask_path`

All other manifest columns are preserved in the output CSVs when possible. Common passthrough metadata includes `image_id`, `patch_id`, `patient_id`, `split`, `model_name`, `unique_id`, and related dataset fields.

## Example commands

Extract both instance and patch summaries:

```bash
pixi run python scripts/benchmarking/morphology/extract_morphology_features.py \
  --manifest data/conic_lizard/dataset_manifest.csv \
  --outdir outputs/morphology/conic_lizard \
  --level both \
  --workers 8 \
  --batch-size 16
```

Patch summaries only, with RGB intensity features:

```bash
pixi run python scripts/benchmarking/morphology/extract_morphology_features.py \
  --manifest data/Monusac/tiles_256/all_patches_dataset.csv \
  --outdir outputs/morphology/monusac_tiles \
  --level patch \
  --workers 12 \
  --include-intensity \
  --min-area 20
```

Stop immediately on the first bad row:

```bash
pixi run python scripts/benchmarking/morphology/extract_morphology_features.py \
  --manifest path/to/dataset_manifest.csv \
  --outdir outputs/morphology/run_a \
  --level both \
  --fail-fast
```

## Output files

The script writes a small output tree under `--outdir`:

- `instance_features.csv`
  - written when `--level instance` or `--level both`
  - one row per retained labeled object
- `patch_features.csv`
  - written when `--level patch` or `--level both`
  - one row per successfully processed manifest row
- `failed_rows.csv`
  - one row per skipped manifest row with `stage` and `error_message`
- `processing_summary.json`
  - total rows, processed rows, skipped rows, total objects, runtime, average time per row
- `run.log`
  - timestamped log with major stage timing and warnings

## Main feature columns

Instance-level columns:

- shape and size: `area_pixels`, `perimeter_pixels`, `equivalent_diameter`
- ellipse-style summaries: `major_axis_length`, `minor_axis_length`, `eccentricity`, `orientation_deg`
- compactness and occupancy: `solidity`, `extent`, `circularity`, `perimeter_area_ratio`, `fill_ratio`
- location: bounding box fields and centroid fields

Patch-level summary columns:

- object count and total area: `num_objects`, `total_mask_area`, `foreground_fraction`
- size summaries: `mean_area`, `median_area`, `std_area`
- mean object-shape summaries: `mean_eccentricity`, `mean_solidity`, `mean_circularity`

Optional RGB intensity columns added by `--include-intensity`:

- whole-image RGB mean/std: `image_mean_*`, `image_std_*`
- retained-foreground RGB mean/std: `foreground_mean_*`, `foreground_std_*`

`orientation_deg` is the `skimage` orientation converted from radians to degrees.

## Binary vs labeled masks

- PNG and TIFF masks are supported.
- Masks must decode to an unambiguous 2D label array.
- Boolean masks, `0/1` masks, and single-positive-value masks such as `0/255` are treated as binary foreground/background masks.
- Binary masks are converted to labeled instances with 8-connectivity before feature extraction.
- Multi-label instance masks keep their original positive label IDs.
- `--min-area` filtering is applied after labeling and before both instance and patch summaries.

## Performance and workers

- The script processes one manifest row at a time and does not preload all images or masks into memory.
- `--workers` controls the number of Python worker processes.
- `--batch-size` maps to the multiprocessing chunk size used by `imap_unordered()`.
- Outputs are written incrementally, which keeps memory use stable even when `instance_features.csv` becomes large.
- `--ram-gb` is a logging hint only. It does not enforce memory limits.

## Failure modes

Bad rows are skipped with warnings unless `--fail-fast` is enabled.

Failure stages reported in `failed_rows.csv`:

- `path_resolution`
  - missing or unresolvable `mask_path`, or missing `image_path` when `--include-intensity` is enabled
- `mask_decode`
  - unreadable, unsupported, empty, or ambiguous mask file
- `mask_validation`
  - negative labels, non-integer floating labels, or other invalid mask contents
- `image_decode`
  - image file exists but cannot be loaded as RGB for intensity summaries
- `feature_extraction`
  - shape mismatches, guardrail failures, or unexpected feature computation issues
