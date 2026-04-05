# Repository Guide

## Read Strategy

- Do not read all docs by default.
- Start with heading-only lookup: `rg -n "^## " docs/*.md`.
- Read only the file and section needed for the task.
- Treat `docs/` as the context source of truth; read scripts directly only when implementing or verifying behavior.

## Doc Index

### Dataset, manifests, and preprocessing

- File: `docs/monu_context.md`
- Read when: working on export, rescaling, tiling, manifest schemas, dataset paths, join keys, or naming rules.
- Jump to:
  - `## Canonical Roots`
  - `## Original MoNuSAC Export`
  - `## Rescaled Export`
  - `## Tiled Export`
  - `## Organization Rules For Future Work`

### Inference pipeline

- File: `docs/inference.md`
- Read when: working on `scripts/02-inference/`, model execution, Pixi environments, input tile assumptions, outputs, or scheduler constraints.
- Jump to:
  - `## Inputs`
  - `## Paths`
  - `## Scripts`
  - `## Outputs`
  - `## Dependencies and Execution`
  - `## Config Knobs`
  - `## Constraints`

### Notebook and HPC bootstrap

- File: `scripts/benchmarking/README.md`
- Read when: you need notebook-era setup, Jupyter/HPC launch steps, or the original MoNuSAC download/QC workflow.
- Do not use this file as the main source of truth for the current inference contract.

## Stable Paths

- Benchmarking utilities and notebooks: `scripts/benchmarking/`
- Standardized inference entrypoints: `scripts/02-inference/`
- Original exported MoNuSAC data: `data/Monusac/`
- Rescaled MoNuSAC data: `data/Monusac/rescaled/`
- Tiled MoNuSAC data: `data/Monusac/tiles_<patch_size>/`
- Inference outputs: `inference/benchmarking/monusac/<modelname>/`

## High-Signal Rules

- Keep `unique_id` as the canonical source-image key across export, rescaling, tiling, and inference.
- Keep `patch_id` and patch geometry fields for tile-level outputs.
- Keep image/mask stems paired as `*_image.png` and `*_mask.png`.
- Preserve manifest columns and relative path semantics when writing derived outputs.
- Keep the top-of-file config blocks intact in `scripts/02-inference/*.py`.
- Expect full model execution on Linux/HPC; on macOS, validation may be limited to syntax or static checks.
