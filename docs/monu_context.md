# Monu Data Context

This file is the dataset-specific source of truth for Monu-family data in this repo.

Current scope:

- The repo currently documents and processes MoNuSAC exports plus derived assets.
- There is not a separate MoNuSeg pipeline in this repo right now.
- If MoNuSeg or another Monu-family dataset is added later, extend this file instead of moving that detail back into `AGENTS.md`.

## Canonical Roots

- Original export root: `data/Monusac/`
- Rescaled derivative root: `data/Monusac/rescaled/`
- Tiled derivative root pattern: `data/Monusac/tiles_<patch_size>/`
- Shared helper module: `scripts/benchmarking/monusac_visualization_utils.py`
- Original export notebook: `scripts/benchmarking/monusac_download_extract.ipynb`
- Rescale script: `scripts/benchmarking/monusac_rescale_export.py`
- Tiling script: `scripts/benchmarking/monusac_tile_export.py`

## Original MoNuSAC Export

Source dataset and fields:

- Hugging Face dataset: `RationAI/MoNuSAC`
- Source fields used by the export notebook: `patient`, `image`, `instances`, `categories`, `tissue`

Folder layout:

- `data/Monusac/all_merged/`
- `data/Monusac/kidney_only/`
- `data/Monusac/extraction_summary.csv`

Canonical source-image key:

- Every exported sample uses one shared `unique_id` for both the RGB image and the merged instance mask.
- `unique_id` format:
  - `{merged_index:04d}_{source_split}_{source_index:04d}_{patient_id}_{tissue_lower}`
- Filename pair format:
  - `{unique_id}_image.png`
  - `{unique_id}_mask.png`
- Example:
  - `0007_train_0007_TCGA-AB-1234_kidney_image.png`
  - `0007_train_0007_TCGA-AB-1234_kidney_mask.png`

Naming and sanitization rules:

- `patient_id` and tissue strings are sanitized to safe filename characters.
- Any character outside `[A-Za-z0-9._-]` is replaced with `_`.
- The original export keeps the image and mask filename stems identical except for the `_image` and `_mask` suffixes.

Image and mask semantics:

- Images are RGB PNG files.
- Masks are single merged instance-label PNG files stored as `uint16`.
- Background label is `0`.
- Foreground instance labels are `1..N` in dataset order.
- This is one merged mask per sample, not one mask file per nucleus.
- If instance masks overlap, later instances in dataset order overwrite earlier labels at overlapping pixels.

Folder semantics:

- `all_merged/` is the canonical full export.
- `kidney_only/` contains copies of already-exported files from `all_merged/` for rows where `tissue == Kidney`.
- `folder` should be preserved whenever a manifest needs to distinguish `all_merged` from `kidney_only`.

Original summary CSV:

- Path: `data/Monusac/extraction_summary.csv`
- Canonical columns:
  - `unique_id`
  - `patient`
  - `tissue`
  - `source_split`
  - `source_index`
  - `num_instances`
  - `overlap_pixels`
  - `image_path`
  - `mask_path`
- In the original export, `image_path` and `mask_path` point to the exported PNG pair for that sample.

## Rescaled Export

Purpose:

- The rescaled export converts the exported MoNuSAC patches from 40x to 20x while preserving instance-label semantics as well as possible.

Default workflow:

```bash
pixi run python scripts/benchmarking/monusac_rescale_export.py
```

Default output layout:

- Root: `data/Monusac/rescaled/`
- Subdirectories:
  - `data/Monusac/rescaled/all_merged/`
  - `data/Monusac/rescaled/kidney_only/`
- Summary CSV:
  - `data/Monusac/rescaled/extraction_summary.csv`

Naming rules:

- The rescaled export keeps the same `unique_id`.
- The rescaled image and mask filenames keep the same MoNuSAC-style pair naming:
  - `{unique_id}_image.png`
  - `{unique_id}_mask.png`
- This means `unique_id` remains the join key across the original export and the rescaled export.

Default magnification behavior:

- Source magnification default: `40.0`
- Target magnification default: `20.0`
- Default scale factor for the standard workflow: `0.5`
- Default `min_instance_fraction`: `0.25`

Resizing semantics:

- RGB images are resized with PIL `BOX` resampling.
- Instance masks are resized label-by-label, not by directly interpolating the merged label image.
- Each label is converted to a binary mask, resized independently, and written back only where its resized occupancy score passes the configured threshold.
- When downscaling, binary masks use PIL `BOX` resampling.
- When upscaling, binary masks use PIL `NEAREST` resampling.
- If multiple labels compete for the same target pixel, the label with the highest resized score wins.
- Background remains `0`.

Rescaled summary CSV:

- The summary row starts from the original export metadata and adds derived fields.
- Important added fields include:
  - `source_image_path`
  - `source_mask_path`
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
- In the rescaled summary, `source_*_path` refers back to the original exported sample and `image_path` or `mask_path` refers to the rescaled files.

## Tiled Export

Purpose:

- The tiled export splits exported MoNuSAC RGB and mask pairs into fixed-size patches while keeping patch-to-source mapping explicit and easy to join back to source metadata.

Default workflow:

```bash
pixi run python scripts/benchmarking/monusac_tile_export.py --patch-size 256
```

Default output layout:

- Root pattern: `data/Monusac/tiles_<patch_size>/`
- Example default root for `256`:
  - `data/Monusac/tiles_256/`
- Default folder selection:
  - `all_merged`
- Global manifests at the tiling root:
  - `all_patches_dataset.csv`
  - `image_patch_summary.csv`

Patch extraction behavior:

- Patch size is fixed.
- Stride defaults to the patch size.
- The default workflow is non-overlapping full patches only.
- Right-edge and bottom-edge remainders are discarded if they do not fit a full patch.
- An image smaller than the patch size yields zero patches.
- Even when zero patches are produced, the tiling workflow still writes:
  - a per-image `dataset.csv` with headers
  - an `image_patch_summary.csv` row with `patch_count = 0`

Per-image folder layout:

- Each source image gets its own folder:
  - `tiles_<patch_size>/<folder>/<unique_id>/`
- Example:
  - `data/Monusac/tiles_256/all_merged/0007_train_0007_TCGA-AB-1234_kidney/`

Patch naming rules:

- Patch descriptor format:
  - `{x0:05d}x_{y0:05d}y`
- Patch filename pair format:
  - `{descriptor}_image.png`
  - `{descriptor}_mask.png`
- Example patch pair:
  - `00000x_00000y_image.png`
  - `00000x_00000y_mask.png`
- Patch ID format:
  - `{unique_id}.{descriptor}`

Patch manifest schema:

- `all_patches_dataset.csv` has one row per patch across the tiling run.
- `<folder>/<unique_id>/dataset.csv` has one row per patch for that one source image.
- Important patch columns:
  - original source metadata copied from the input export row, such as `unique_id`, `patient`, `tissue`, `source_split`, `source_index`, `num_instances`, and `folder`
  - `source_image_path`
  - `source_mask_path`
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

Image-level tiling summary:

- `image_patch_summary.csv` has one row per source image.
- Important columns:
  - original source metadata copied from the input export row
  - `source_image_path`
  - `source_mask_path`
  - `patches_dir`
  - `dataset_csv`
  - `patch_size`
  - `stride`
  - `source_height`
  - `source_width`
  - `patch_rows`
  - `patch_cols`
  - `patch_count`

Tiling path semantics:

- In the tiled manifests, `source_image_path` and `source_mask_path` refer to the source sample used for tiling.
- In the tiled manifests, `image_path`, `mask_path`, `patches_dir`, and `dataset_csv` are relative to the tiling output root.
- Downstream code should join source metadata by `unique_id` and distinguish the source folder by `folder`.
- Downstream code should use `patch_id` when it needs a unique patch-level key.

## Organization Rules For Future Work

When creating new files or workflows around these assets:

- Keep `unique_id` as the canonical source-image key across original, rescaled, and tiled data.
- Keep image and mask stems paired. Do not invent mismatched image and mask names.
- Preserve the `folder` column whenever `all_merged` and `kidney_only` can both appear.
- For patch-level derivatives, preserve `patch_id`, coordinates, and patch geometry fields.
- Prefer one top-level summary CSV per derived dataset root.
- If the workflow splits one source image into many outputs, also write a per-source manifest such as `dataset.csv`.
- Do not rename or reorganize existing MoNuSAC-derived folders lightly; downstream scripts are likely to rely on the current schema.
- If a future workflow changes naming or metadata structure, document that change here at the same time as the code change.
