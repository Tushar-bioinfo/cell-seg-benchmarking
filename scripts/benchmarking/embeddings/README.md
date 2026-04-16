# Prov-GigaPath Tile Embeddings

This directory adds a reproducible tile-level embedding step around the official Prov-GigaPath tile encoder.

## Expected Input CSV Columns

The extractor expects one row per image patch.

- Required:
  - an image-path column, usually `image_path`
- Recommended:
  - a stable tile ID column such as `patch_id`
- Common local columns:
  - `flow_1` tile manifests: `patch_id`, `sample_id`, `image_path`, `mask_path`, tile geometry columns
  - CoNIC raw manifest: `sample_id`, `image_path`, `mask_path`, `class_label_path`, class-count metadata

If the CSV uses different names, pass them explicitly with `--image-col` and `--id-col`.

## Official Model Loading

The script uses the official Prov-GigaPath tile-encoder path from the project README and Hugging Face model card:

- `timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)`
- preprocessing:
  - `Resize(256, interpolation=BICUBIC)`
  - `CenterCrop(224)`
  - `ToTensor()`
  - `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`

Notes:

- `timm>=1.0.3` is required by the official docs.
- You must accept the model terms on Hugging Face and provide `HF_TOKEN` if the weights are not already cached.

## Local Run

Use the Pixi default environment after syncing dependencies:

```bash
pixi run -e default python scripts/benchmarking/embeddings/gigapath_extract_embeddings.py \
  --input-csv data/conic_lizard/dataset_manifest.csv \
  --outdir inference/benchmarking/embeddings/conic_gigapath \
  --image-col image_path \
  --id-col sample_id \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --save-format pt
```

This repository's Pixi environments target `linux-64`. On macOS, local validation may be limited to CLI or syntax checks unless you have a compatible Linux runtime.

For stricter numerical reproducibility, disable mixed precision:

```bash
pixi run -e default python scripts/benchmarking/embeddings/gigapath_extract_embeddings.py \
  --input-csv <csv> \
  --outdir <outdir> \
  --amp-dtype none
```

## HPC Run

Suggested starting point for 1 GPU and 24 GB RAM:

```bash
pixi run -e default python scripts/benchmarking/embeddings/gigapath_extract_embeddings.py \
  --input-csv /abs/path/to/all_patches_dataset.csv \
  --outdir /abs/path/to/gigapath_embeddings \
  --image-col image_path \
  --id-col patch_id \
  --batch-size 128 \
  --num-workers 8 \
  --device cuda:0 \
  --amp-dtype float16 \
  --save-format pt
```

If you hit CUDA OOM, lower `--batch-size` first.

## Output Layout

```text
<outdir>/
  embeddings/
    part-000000.pt
    part-000001.pt
    ...
  metadata/
    input_rows_snapshot.csv
    input_rows_snapshot.parquet
    embeddings_index.csv
    embeddings_index.parquet
  logs/
    run.log
    failed_rows.csv
    failed_rows.parquet
  manifest.json
```

`embeddings_index.*` keeps the join back to the original CSV via `input_row_index`, the chosen ID column or derived `embedding_id`, and the original passthrough metadata from the input CSV.
