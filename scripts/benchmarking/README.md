# Benchmarking Notebook Quickstart

## Files in This Folder

- `monusac_download_extract.ipynb`: Downloads `RationAI/MoNuSAC` from Hugging Face, merges `train` and `test`, and exports paired image and mask PNG files into `../../data/Monusac/`.

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

Important:

- Run the notebook in place from `scripts/benchmarking/`.
- The notebook writes data relative to its location and expects `../../data/Monusac/`.
- If Hugging Face requests authentication, run `hf auth login` before executing the download cell.
