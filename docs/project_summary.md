# Project Summary

- Date: `2026-04-24`
- Version: `0.1.1`
- Status: `active`
- Purpose: compact project-level summary for the scientific framing, current artifact flow, and planned next step. Detailed input/output contracts stay in the lower-level docs.

## Core Idea

This project studies patch-level nucleus segmentation benchmarking and representation-based difficulty prediction in H&E pathology images. The central hypothesis is that segmentation failures are structured rather than random, and that pretrained pathology embeddings can capture image properties associated with segmentation reliability.

## Current Pipeline

### 1. Segmentation Benchmarking

- The reusable workflow is dataset-generic and is centered on `scripts/benchmarking/flow_1/`
- Typical workflow order:
  - paired image/mask dataset or manifest
  - optional rescaling
  - optional tiling
  - model inference
  - manifest-driven evaluation
- Source-image key is dataset-dependent, typically `unique_id`, `sample_id`, or another stable manifest ID
- Canonical patch key after tiling: `patch_id`
- Benchmarked model families:
  - `cellsam`
  - `cellpose_sam`
  - `cellvit_sam`
  - `stardist`
- Standard inference outputs live under `inference/benchmarking/<dataset_used>/<model_name>/`
- Each model run writes:
  - one instance mask per input patch
  - `predictions.csv`
  - `failed.csv`
- Standard evaluation reads each model's `predictions.csv` as the path source of truth and writes per-model CSVs under `inference/benchmarking/<dataset_used>/_evaluation/`
- The benchmark emphasis is patch-level variability rather than only one model-wide average
- The main patch-level quality signal of interest is panoptic quality (`pq`)
- Current repo example with the most concrete dataset-specific docs: MoNuSAC

### 2. Representation Analysis

- GigaPath tile embeddings are extracted from patch manifests with `scripts/benchmarking/embeddings/gigapath_extract_embeddings.py`
- Each patch receives one `1536`-dimensional embedding
- Embedding space is intended for exploratory analysis with PCA and UMAP before downstream prediction
- Embedding outputs preserve input manifest columns and add stable bookkeeping fields such as:
  - `input_row_index`
  - `embedding_id`
  - `embedding_path`
  - `embedding_dim`
- Preferred downstream join key for tiled patch workflows: `patch_id`
- Morphology summaries can be extracted from masks to provide interpretable patch descriptors such as:
  - `foreground_fraction`
  - `mean_area`
  - `mean_circularity`
  - `mean_eccentricity`
- When class-label metadata is available from CoNIC/Lizard manifests, patch-level cell-type composition can be used as an additional descriptor

### 3. Difficulty Modeling

- Patch difficulty is defined from the median `pq` across the four segmentation models
- The median is used as a consensus estimate of whether a patch is broadly easy, intermediate, or hard for current methods
- Planned predictive baselines on embeddings include:
  - logistic regression
  - SVM
  - random forest

## Why This Matters

If embedding space separates easy and hard patches in a stable way, pretrained pathology representations can support:

- segmentation quality control
- triage
- active learning
- failure-aware deployment

## Active Repo Contracts

- Preserve the upstream sample key across rescaling, tiling, inference, and evaluation
- Preserve `patch_id` and patch geometry fields on tile-level outputs
- Treat each model's `predictions.csv` as the path source of truth during evaluation
- Preserve manifest metadata columns in embedding and morphology outputs
- Keep this file high-level; update the detailed docs if any workflow contract changes

## Related Docs

- `docs/flow_1.md`: dataset-generic workflow order, path flow, manifests, inference, and evaluation
- `docs/monu_context.md`: current MoNuSAC-specific export, rescaling, tiling, naming, and manifests
- `docs/inference.md`: current standardized inference outputs and per-model directories
- `docs/evaluations.md`: current evaluation semantics and evaluation CSV behavior
- `docs/embedding.md`: GigaPath embedding extraction inputs, outputs, and resume rules
- `docs/morphology_labels_context.md`: morphology feature extraction inputs, outputs, and join behavior
- `docs/modeling_context.md`: sklearn-first difficulty-modeling workflow, canonical modeling table, failure-mode enrichment, and Slurm behavior
- `docs/conic_context.md`: CoNIC/Lizard class-label and count metadata when cell-type composition is needed

## Update Rule

- Bump the patch version for wording-only edits or clearer references
- Bump the minor version for new pipeline stages, datasets, or model families
- Bump the major version if the scientific framing or main objective changes
