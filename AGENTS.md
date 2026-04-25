# Repository Guide

## Start Here

- Do not read all docs by default.
- Start with headings only: `rg -n "^## " docs/*.md`.
- For project-overview questions, read `docs/project_summary.md` first, then open only the lower-level docs you need.
- Read only the file and section needed for the task.
- Treat `docs/` as the repo source of truth. Read scripts directly only to implement or verify behavior.

## Use These Sources

- `docs/project_summary.md`
  Use for the current project-level framing, summary date/version, active milestones, and how benchmarking, embeddings, and difficulty prediction fit together.
- `docs/monu_context.md`
  Use for MoNuSAC export, rescaling, tiling, dataset roots, manifests, join keys, and naming rules.
- `docs/conic_context.md`
  Use for CoNIC/Lizard raw arrays, export layout, manifest schema, class-label paths, and `flow_1` usage notes.
- `docs/evaluations.md`
  Use for evaluation semantics, the standard `monusac_segmentation_evaluation.py` workflow, metric formulas, manifest assumptions, and output CSV behavior.
- `docs/inference.md`
  Use for `scripts/02-inference/`, model-output structure, `predictions.csv`, and Pixi execution details.
- `docs/embedding.md`
  Use for tile-level embedding extraction, Prov-GigaPath tile-encoder loading, input CSV expectations, output layout, and resume behavior.
- `docs/morphology_labels_context.md`
  Use for morphology feature extraction manifests, path resolution, mask/image format rules, output CSV layout, and failure semantics.
- `docs/flow_1.md`
  Use when asked questions related to `flow_1`.
- `scripts/benchmarking/README.md`
  Use only for notebook or HPC bootstrap context. Do not treat it as the current inference contract.

## Core Contracts

- Keep `unique_id` as the canonical source-image key across export, rescaling, tiling, inference, and evaluation.
- Keep `patch_id` and patch geometry fields on tile-level outputs.
- Keep paired stems as `*_image.png` and `*_mask.png`.
- Preserve manifest columns and relative path semantics when writing derived outputs.
- For embedding outputs, preserve all input CSV columns and keep `input_row_index` plus a stable tile ID such as `patch_id` whenever available.
- For morphology outputs, preserve manifest metadata columns, keep `input_row_index`, and treat recorded `image_path` and `mask_path` values as the path source of truth.
- For MoNuSAC evaluation, treat each model's `predictions.csv` as the path source of truth. Do not reconstruct GT or prediction paths from naming assumptions when the manifest already records them.
- Keep the top-of-file config blocks intact in `scripts/02-inference/*.py`.
- Expect full model execution on Linux or HPC. On macOS, validation may be limited to syntax or static checks.

## Stable Paths

- Benchmarking helpers: `scripts/benchmarking/`
- Embedding extractor: `scripts/benchmarking/embeddings/gigapath_extract_embeddings.py`
- CoNIC raw/export root: `data/conic_lizard/`
- CoNIC prep script: `scripts/benchmarking/flow_1/prepare_conic_lizard.py`
- Standardized inference entrypoints: `scripts/02-inference/`
- MoNuSAC original export: `data/Monusac/`
- MoNuSAC rescaled export: `data/Monusac/rescaled/`
- MoNuSAC tiled export: `data/Monusac/tiles_<patch_size>/`
- Inference outputs: `inference/benchmarking/monusac/<modelname>/`
- Standard batch evaluator: `scripts/benchmarking/monusac_segmentation_evaluation.py`

## Dependency Hygiene

- Before using a new tool, package, or library, check whether it already exists in `pixi.toml` and `pixi.lock`.
- If the dependency set changes, update the environment files before relying on the package.
- Verify compatibility with the target Pixi environment, especially feature-specific environments.
- Test the dependency in the environment where it will run, for example with `pixi run -e <env> ...`.
- Do not rely on undeclared dependencies or user-global packages.
