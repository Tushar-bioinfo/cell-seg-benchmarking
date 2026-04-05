# Benchmarking Notebook Quickstart

## Files in This Folder

- `monusac_download_extract.ipynb`: Downloads `RationAI/MoNuSAC` from Hugging Face, merges `train` and `test`, and exports paired image and mask PNG files into `../../data/Monusac/`.
- `monusac_annotation_qc.ipynb`: Loads exported MoNuSAC image/mask pairs, overlays the instance mask on the RGB patch, then rescales a 40x patch to 20x with a label-aware mask resize and shows the overlay again for QC.
- `monusac_visualization_utils.py`: Shared helper functions for MoNuSAC sample lookup, overlay rendering, and 40x to 20x image/mask rescaling.
- `monusac_tile_export.py`: Splits exported MoNuSAC image/mask pairs into fixed-size patches and writes both per-image `dataset.csv` files and global patch manifests.

## Pulling Latest Changes on HPC

From the repo root on the HPC system:

```bash
git fetch origin
git switch master
git pull --ff-only origin master
```

## Pixi Setup

If `pixi` is not installed yet, install it first on the HPC system, then reopen your shell so `pixi` is on `PATH`.

From the repo root:

```bash
pixi install
pixi run kernel-install
```

`pixi install` uses the checked-in `pixi.lock`, so the notebook environment will pick up the pinned `datasets` dependency that the MoNuSAC notebook needs.

## Starting Jupyter on HPC

Start Jupyter from a compute node, not the login node.

Example interactive allocation:

```bash
srun --pty --cpus-per-task=4 --mem=16G --time=02:00:00 bash
```

Then, from the repo root:

```bash
pixi run lab
```

That launches Jupyter Lab with `--no-browser --port 8888` from the repo task definition.

## Running the Notebook

Open:

```text
scripts/benchmarking/monusac_download_extract.ipynb
```

For annotation QC after export, open:

```text
scripts/benchmarking/monusac_annotation_qc.ipynb
```

For fixed-size tiling after export, run:

```bash
pixi run python scripts/benchmarking/monusac_tile_export.py --patch-size 256
```

That writes a new output tree under `data/Monusac/tiles_256/` by default, with:

- one folder per source `unique_id`
- patch image/mask PNGs for each full `256x256` crop
- `dataset.csv` inside each source-image folder
- `all_patches_dataset.csv` and `image_patch_summary.csv` at the tiling root

Important:

- Run the notebook in place from `scripts/benchmarking/`.
- The notebook writes data relative to its location and expects `../../data/Monusac/`.
- If Hugging Face requests authentication, run `hf auth login` before executing the download cell.
- The QC notebook also accepts `MONUSAC_ROOT` if your exported data lives outside the repo's default `data/Monusac/` path.
