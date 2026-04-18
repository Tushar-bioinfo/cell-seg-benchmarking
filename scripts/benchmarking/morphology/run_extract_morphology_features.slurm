#!/bin/bash
#SBATCH --job-name=morphology-features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=morphology-features-%j.out
#SBATCH --error=morphology-features-%j.out

set -euo pipefail

SCRIPT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_PROJECT_ROOT="/share/lab_teng/trainee/tusharsingh/cell-seg"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="${SCRIPT_PROJECT_ROOT}"
fi
cd "${PROJECT_ROOT}"

MANIFEST="${MANIFEST:-data/conic_lizard/dataset_manifest.csv}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/outputs/morphology/conic_lizard}"
LEVEL="${LEVEL:-both}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RAM_GB="${RAM_GB:-64}"
VERBOSITY="${VERBOSITY:-normal}"
MIN_AREA="${MIN_AREA:-}"
MAX_OBJECTS_PER_IMAGE="${MAX_OBJECTS_PER_IMAGE:-}"
INCLUDE_INTENSITY="${INCLUDE_INTENSITY:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

echo "[morphology_slurm] project_root=${PROJECT_ROOT}"
echo "[morphology_slurm] manifest=${MANIFEST}"
echo "[morphology_slurm] outdir=${OUTDIR}"
echo "[morphology_slurm] level=${LEVEL}"
echo "[morphology_slurm] workers=${WORKERS}"
echo "[morphology_slurm] batch_size=${BATCH_SIZE}"
echo "[morphology_slurm] ram_gb=${RAM_GB}"
echo "[morphology_slurm] verbosity=${VERBOSITY}"
echo "[morphology_slurm] include_intensity=${INCLUDE_INTENSITY}"
echo "[morphology_slurm] fail_fast=${FAIL_FAST}"
echo "[morphology_slurm] host=$(hostname)"
echo "[morphology_slurm] date=$(date)"
echo "[morphology_slurm] python=$(python --version 2>&1 || true)"
echo "[morphology_slurm] pixi=$(pixi --version 2>&1 || true)"
echo "[morphology_slurm] slurm_job_id=${SLURM_JOB_ID:-manual}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "[morphology_slurm] pixi is not on PATH" >&2
  exit 1
fi

command=(
  pixi run python scripts/benchmarking/morphology/extract_morphology_features.py
  --manifest "${MANIFEST}"
  --outdir "${OUTDIR}"
  --level "${LEVEL}"
  --workers "${WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --verbosity "${VERBOSITY}"
  --ram-gb "${RAM_GB}"
)

if [[ -n "${MIN_AREA}" ]]; then
  command+=(--min-area "${MIN_AREA}")
fi

if [[ -n "${MAX_OBJECTS_PER_IMAGE}" ]]; then
  command+=(--max-objects-per-image "${MAX_OBJECTS_PER_IMAGE}")
fi

if [[ "${INCLUDE_INTENSITY}" == "1" ]]; then
  command+=(--include-intensity)
fi

if [[ "${FAIL_FAST}" == "1" ]]; then
  command+=(--fail-fast)
fi

command+=("$@")

printf '[morphology_slurm] command='
printf '%q ' "${command[@]}"
printf '\n'

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-${WORKERS}}" "${command[@]}"
else
  "${command[@]}"
fi
