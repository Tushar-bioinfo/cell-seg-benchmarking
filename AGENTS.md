# Project Codex Instructions

- All work and files live in this repo.
- Scripts and notebooks must be HPC-compatible: use SLURM job scripts, avoid interactive or GUI dependencies, and keep I/O batch-friendly.
- Always consult `pixi.lock` before assuming any tool or package is available.
- If a required tool is missing or a version is incompatible with the HPC environment, suggest the exact `pixi.lock` change needed rather than silently skipping it.

## MoNuSAC Export Format

- Notebook: `scripts/benchmarking/monusac_download_extract.ipynb`
- Hugging Face source dataset: `RationAI/MoNuSAC`
- Source fields used by the notebook: `patient`, `image`, `instances`, `categories`, `tissue`
- Output root written by the notebook: `data/Monusac/`
- Output subdirectories:
  - `data/Monusac/all_merged/`
  - `data/Monusac/kidney_only/`
- Every exported sample uses one shared `unique_id` prefix for the paired image and mask files:
  - `{merged_index:04d}_{source_split}_{source_index:04d}_{patient_id}_{tissue_lower}_image.png`
  - `{merged_index:04d}_{source_split}_{source_index:04d}_{patient_id}_{tissue_lower}_mask.png`
- `patient_id` and tissue strings are sanitized to safe filename characters with non `[A-Za-z0-9._-]` characters replaced by `_`.
- Example pair:
  - `0007_train_0007_TCGA-AB-1234_kidney_image.png`
  - `0007_train_0007_TCGA-AB-1234_kidney_mask.png`
- Image format:
  - RGB PNG
- Mask format:
  - Single merged instance-label PNG per sample
  - Stored as `uint16`
  - Background is `0`
  - Each nucleus instance gets integer label `1..N` in the order provided by the dataset's `instances` list
  - This is not one mask file per nucleus; multiple nucleus instances are combined into one labeled mask image
  - If two instance masks overlap, later instances in the dataset order overwrite earlier labels at overlapping pixels
- `kidney_only/` contains copies of the already-exported image/mask pairs from `all_merged/` for rows where `tissue == Kidney`.
- Export summary CSV:
  - `data/Monusac/extraction_summary.csv`
  - Columns: `unique_id`, `patient`, `tissue`, `source_split`, `source_index`, `num_instances`, `overlap_pixels`, `image_path`, `mask_path`
