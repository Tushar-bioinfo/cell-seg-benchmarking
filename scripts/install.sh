#!/bin/bash
#SBATCH --job-name=pixi-install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pixi-install-%j.out
#SBATCH --error=pixi-install-%j.err

set -euo pipefail

echo "Job started on: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_CPUS_PER_NODE: ${SLURM_JOB_CPUS_PER_NODE}"
echo "Working dir: ${SLURM_SUBMIT_DIR}"

# Go to your project directory
cd /share/lab_teng/trainee/tusharsingh/cell-seg

# Optional but very useful:
# keep Pixi cache out of small home quota if your cluster allows this
#export PIXI_CACHE_DIR=/share/lab_teng/trainee/tusharsingh/.pixi_cache 

# Make sure cache directory exists
#mkdir -p "${PIXI_CACHE_DIR}"

# If pixi is in your PATH already, this is enough
which pixi
pixi --version

pixi add python=3.10 jupyterlab ipykernel numpy pandas matplotlib pillow tqdm scikit-image openslide openslide-python opencv monai pytorch torchvision torchaudio

pixi run python -m ipykernel install --user --name cellseg-pixi --display-name "Python (cellseg-pixi)"
echo "Install finished successfully"
