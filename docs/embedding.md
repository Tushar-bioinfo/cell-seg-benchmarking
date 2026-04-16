# Tile Embedding Extraction

## Scope

- Use this file for tile-level embedding extraction in `scripts/benchmarking/embeddings/`.
- Treat it as the source of truth for input CSV expectations, Prov-GigaPath tile-encoder loading, output layout, and resume behavior.
- This workflow is tile-level only. It does not compute slide-level embeddings.

## Inputs

- Input is a CSV with one row per image patch.
- Required:
  - one image-path column that resolves to an RGB patch image
- Default image-column inference order:
  - `image_path`
  - `source_image_path`
  - `patch_image_path`
  - `path`
- Default ID-column preference for stable tile keys:
  - `patch_id`
  - `sample_id`
  - `unique_id`
  - `image_id`
  - `id`
- If no usable ID column is present, the extractor derives `embedding_id` from the image stem plus `input_row_index`.

Most common local inputs:

- `flow_1` tiled manifests such as `all_patches_dataset.csv`
  - expected local join key: `patch_id`
  - expected image column: `image_path`
- raw CoNIC/Lizard manifest `data/conic_lizard/dataset_manifest.csv`
  - expected local join key: `sample_id`
  - expected image column: `image_path`

## Path Resolution

- Image paths may be absolute or relative.
- Relative image paths are resolved in this order:
  - `--path-base-dir` if provided
  - the input CSV parent directory
  - the project root
  - the current working directory
- Rows with missing or unresolvable files go to the failure report and do not abort the run.

## Prov-GigaPath Tile Encoder Contract

- Use the official tile-encoder loading path:
  - `timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)`
- Use the official preprocessing path:
  - `Resize(256, interpolation=BICUBIC)`
  - `CenterCrop(224)`
  - `ToTensor()`
  - `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`
- The project treats the returned tensor as one compact embedding per tile.
- The released tile embeddings are `1536`-dimensional to match the official slide-encoder interface.
- Treat this output as a global tile representation.
- Do not treat it as:
  - a patch-token grid
  - an attention matrix
  - a slide-level embedding

## Paths

- Main script: `scripts/benchmarking/embeddings/gigapath_extract_embeddings.py`
- Helper module: `scripts/benchmarking/embeddings/gigapath_embedding_utils.py`
- Supplemental quickstart note: `scripts/benchmarking/embeddings/README.md`
- Output root is user-supplied through `--outdir`.

## Outputs

Output layout:

- `<outdir>/embeddings/`
- `<outdir>/metadata/`
- `<outdir>/logs/`
- `<outdir>/manifest.json`

Embedding-part files:

- `--save-format pt`
  - each `part-*.pt` stores:
    - `embeddings`
    - `input_row_index`
    - `embedding_id`
- `--save-format parquet`
  - each `part-*.parquet` stores one row per embedding with:
    - `input_row_index`
    - `embedding_id`
    - `embedding`

Metadata outputs:

- `metadata/input_rows_snapshot.csv`
- `metadata/input_rows_snapshot.parquet`
- `metadata/embeddings_index.csv`
- `metadata/embeddings_index.parquet`

`embeddings_index.*` preserves all passthrough columns from the input CSV and adds:

- `input_row_index`
- `embedding_id`
- `resolved_image_path`
- `embedding_path`
- `embedding_format`
- `embedding_row_offset`
- `embedding_dim`
- `image_height`
- `image_width`

Failure outputs:

- `logs/run.log`
- `logs/failed_rows.csv`
- `logs/failed_rows.parquet`

Failure stages recorded in `failed_rows.*`:

- `path_resolution`
- `image_decode`
- `model_inference`

## Join Contract

- Preserve all input manifest columns in `embeddings_index.*`.
- For `flow_1` tile manifests, prefer joining embedding outputs back by `patch_id`.
- Always keep `input_row_index` because resume and per-run bookkeeping use it as the guaranteed row key.
- If no stable tile ID exists upstream, use `embedding_id` plus `input_row_index` for downstream joins.

## Resume Behavior

- Without `--overwrite`, the extractor resumes by reading processed `input_row_index` values from:
  - `metadata/embeddings_index.csv`
  - `logs/failed_rows.csv`
- Existing `manifest.json` must agree with:
  - `input_csv`
  - `model_id`
  - `save_format`
  - `image_column`
  - `id_column` if set
- If those fields differ, the run should fail fast instead of mixing incompatible outputs.

## Dependencies and Execution

- Run from the Pixi `default` environment.
- `timm>=1.0.3` is required by the official Prov-GigaPath loading path.
- If the model weights are not already cached, set `HF_TOKEN` after accepting the gated model terms on Hugging Face.
- Metadata and failure sidecars are always written as both CSV and parquet, so the runtime environment must include parquet support.

## Constraints

- The Pixi workspace is pinned to `linux-64`.
- Full extraction is expected on Linux or HPC rather than macOS.
- Mixed precision is optional through `--amp-dtype`; use `none` for full float32 inference when exact repeatability matters more than throughput.
- On 1 GPU / 24 GB RAM, tune `--batch-size` first if CUDA OOM occurs.
