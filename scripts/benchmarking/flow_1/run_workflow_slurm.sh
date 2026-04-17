#!/bin/bash
#SBATCH --job-name=flow1-workflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --output=flow1-workflow-%j.out
#SBATCH --error=flow1-workflow-%j.out

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-24}}"
RAM_LIMIT_GB="${RAM_LIMIT_GB:-180}"
GPU_SLOTS="${GPU_SLOTS:-1}"
DATA_ROOT="${DATA_ROOT:-data/conic_lizard}"
INPUT_MANIFEST="${INPUT_MANIFEST:-data/conic_lizard/dataset_manifest.csv}"
SKIP_RESCALE="${SKIP_RESCALE:-1}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${PROJECT_ROOT}/logs/flow_1"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run_workflow-${SLURM_JOB_ID:-manual}.log"

if [[ "${DRY_RUN}" != "1" ]]; then
  exec >>"${RUN_LOG}" 2>&1
fi

export OMP_NUM_THREADS="${WORKERS}"
export MKL_NUM_THREADS="${WORKERS}"
export OPENBLAS_NUM_THREADS="${WORKERS}"
export NUMEXPR_NUM_THREADS="${WORKERS}"
export VECLIB_MAXIMUM_THREADS="${WORKERS}"

echo "[run_workflow_slurm] project_root=${PROJECT_ROOT}"
echo "[run_workflow_slurm] data_root=${DATA_ROOT}"
echo "[run_workflow_slurm] input_manifest=${INPUT_MANIFEST}"
echo "[run_workflow_slurm] workers=${WORKERS} ram_limit_gb=${RAM_LIMIT_GB} gpu_slots=${GPU_SLOTS}"
echo "[run_workflow_slurm] run_log=${RUN_LOG}"
echo "[run_workflow_slurm] slurm_job_id=${SLURM_JOB_ID:-manual}"
echo "[run_workflow_slurm] host=$(hostname)"

if ! command -v pixi >/dev/null 2>&1; then
  echo "[run_workflow_slurm] pixi is not on PATH" >&2
  exit 1
fi

command=(
  pixi run python scripts/benchmarking/flow_1/run_workflow.py
  --in "${DATA_ROOT}"
  --manifest "${INPUT_MANIFEST}"
  --workers "${WORKERS}"
  --ram-limit-gb "${RAM_LIMIT_GB}"
  --gpu-slots "${GPU_SLOTS}"
)

if [[ "${SKIP_RESCALE}" == "1" ]]; then
  command+=(--skip-rescale)
fi

if [[ -n "${IMAGES_SUBDIR:-}" ]]; then
  command+=(--images-subdir "${IMAGES_SUBDIR}")
fi

if [[ -n "${MASKS_SUBDIR:-}" ]]; then
  command+=(--masks-subdir "${MASKS_SUBDIR}")
fi

command+=("$@")

printf '[run_workflow_slurm] command='
printf '%q ' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-${WORKERS}}" "${command[@]}"
else
  "${command[@]}"
fi
