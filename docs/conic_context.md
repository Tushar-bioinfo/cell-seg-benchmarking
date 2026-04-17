# CoNIC Context

## Scope

- Use this file for CoNIC/Lizard dataset conversion and `flow_1` usage.
- Treat it as the source of truth for `data/conic_lizard/`.

## Raw Release

- Raw CoNIC data arrives as ordered arrays plus metadata tables:
  - `images.npy`: `(N, 256, 256, 3)` RGB `uint8`
  - `labels.npy`: `(N, 256, 256, 2)` `uint16`
  - `labels[..., 0]`: instance map
  - `labels[..., 1]`: class map
  - `patch_info.csv`: stable sample IDs
  - `counts.csv`: per-patch class counts

## Export Contract

- Convert raw arrays into paired PNGs under:
  - `data/conic_lizard/images/*_image.png`
  - `data/conic_lizard/masks/*_mask.png`
  - `data/conic_lizard/class_labels/*_class_labels.png`
- Build `data/conic_lizard/dataset_manifest.csv`.
- Use `patch_info` as `sample_id`.
- Keep manifest paths relative to the project root, for example `data/conic_lizard/images/foo_image.png`.
- Required manifest columns for this dataset:
  - `sample_id`
  - `image_path`
  - `mask_path`
  - `class_label_path`
- Keep all `counts.csv` columns and `count_total` in the manifest as passthrough metadata.

## Flow 1 Notes

- `flow_1` uses `image_path` and `mask_path`; `class_label_path` is metadata only.
- Raw CoNIC patches are already `256x256`. Default `flow_1` rescaling (`40x -> 20x`) shrinks them to `128x128`, which yields zero `256`-pixel tiles.
- For raw CoNIC manifests, prefer `--skip-rescale`.
- If rescaling is intentional, pass `--images-subdir images --masks-subdir masks` so rescaled outputs stay under `images/...` and `masks/...` instead of `images/images/...`.
- Raw CoNIC manifests keep project-relative paths such as `data/conic_lizard/...`; the evaluator now accepts those recorded paths directly, but the standard full workflow still prefers evaluating from tiled `predictions.csv`.

## Stable Paths

- Raw/export root: `data/conic_lizard/`
- Manifest: `data/conic_lizard/dataset_manifest.csv`
- Prep script: `scripts/benchmarking/flow_1/prepare_conic_lizard.py`
