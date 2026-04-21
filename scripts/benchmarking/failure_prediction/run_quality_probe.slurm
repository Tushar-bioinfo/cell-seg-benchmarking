#!/bin/bash
#SBATCH --job-name=quality-probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=quality-probe-%j.out
#SBATCH --error=quality-probe-%j.out

set -euo pipefail

SCRIPT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_PROJECT_ROOT="/share/lab_teng/trainee/tusharsingh/cell-seg"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="${SCRIPT_PROJECT_ROOT}"
fi
cd "${PROJECT_ROOT}"

RUN_BUILD_MANIFEST="${RUN_BUILD_MANIFEST:-1}"
RUN_ATTACH_TARGETS="${RUN_ATTACH_TARGETS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

FAILURE_ROOT="${FAILURE_ROOT:-outputs/conic_liz/failure_prediction}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  DEFAULT_EXPERIMENT_DIR="${FAILURE_ROOT}/experiments/${EXPERIMENT_NAME}"
else
  DEFAULT_EXPERIMENT_DIR="${FAILURE_ROOT}"
fi
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${DEFAULT_EXPERIMENT_DIR}}"

PATCH_MANIFEST_INPUT="${PATCH_MANIFEST_INPUT:-outputs/embeddings/metadata/embeddings_index.parquet}"
PATCH_MANIFEST_OUTPUT="${PATCH_MANIFEST_OUTPUT:-${EXPERIMENT_DIR}/patch_manifest.parquet}"
PATCH_MANIFEST_PATCH_ID_COL="${PATCH_MANIFEST_PATCH_ID_COL:-sample_id}"
PATCH_MANIFEST_SLIDE_ID_COL="${PATCH_MANIFEST_SLIDE_ID_COL:-sample_id}"
PATCH_MANIFEST_SLIDE_ID_STRIP_REGEX="${PATCH_MANIFEST_SLIDE_ID_STRIP_REGEX:--[^-]+$}"
PATCH_MANIFEST_DATASET_VALUE="${PATCH_MANIFEST_DATASET_VALUE:-conic_liz}"
PATCH_MANIFEST_SPLIT_VALUE="${PATCH_MANIFEST_SPLIT_VALUE:-all}"
PATCH_MANIFEST_EXTRA_COLS="${PATCH_MANIFEST_EXTRA_COLS:-conic_index image_path mask_path class_label_path resolved_image_path embedding_row_offset embedding_format embedding_dim}"

EVAL_DIR="${EVAL_DIR:-outputs/conic_liz}"
EVAL_GLOB="${EVAL_GLOB:-*_evaluation.csv}"
JOINED_MANIFEST_OUTPUT="${JOINED_MANIFEST_OUTPUT:-${EXPERIMENT_DIR}/patch_manifest_with_eval_targets.parquet}"

TRAIN_INPUT="${TRAIN_INPUT:-${JOINED_MANIFEST_OUTPUT}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${EXPERIMENT_DIR}/quality_probe}"
TRAIN_PROBLEM_TYPE="${TRAIN_PROBLEM_TYPE:-classification}"
TRAIN_FEATURE_MODE="${TRAIN_FEATURE_MODE:-embedding_only}"
TRAIN_TARGET_COL="${TRAIN_TARGET_COL:-instance_pq}"
TRAIN_TARGET_AGGREGATION="${TRAIN_TARGET_AGGREGATION:-median}"
TRAIN_GROUP_COL="${TRAIN_GROUP_COL:-slide_id}"
TRAIN_CLASSIFICATION_THRESHOLD_STRATEGY="${TRAIN_CLASSIFICATION_THRESHOLD_STRATEGY:-train_median}"
TRAIN_CLASSIFICATION_THRESHOLD="${TRAIN_CLASSIFICATION_THRESHOLD:-}"
TRAIN_POSITIVE_CLASS_NAME="${TRAIN_POSITIVE_CLASS_NAME:-failure}"
TRAIN_CLASSIFIER_FAMILIES="${TRAIN_CLASSIFIER_FAMILIES:-logistic_regression random_forest svm}"
TRAIN_OPTUNA_TRIALS="${TRAIN_OPTUNA_TRIALS:-40}"
TRAIN_CV_FOLDS="${TRAIN_CV_FOLDS:-5}"
TRAIN_N_JOBS="${TRAIN_N_JOBS:-1}"
TRAIN_REQUIRE_COMPLETE_MODEL_COVERAGE="${TRAIN_REQUIRE_COMPLETE_MODEL_COVERAGE:-1}"
TRAIN_AUTO_METADATA_COLS="${TRAIN_AUTO_METADATA_COLS:-0}"
TRAIN_METADATA_COLS="${TRAIN_METADATA_COLS:-}"
TRAIN_REQUIRE_EXISTING_EMBEDDINGS="${TRAIN_REQUIRE_EXISTING_EMBEDDINGS:-0}"

INFER_INPUT="${INFER_INPUT:-${PATCH_MANIFEST_OUTPUT}}"
INFER_OUTPUT="${INFER_OUTPUT:-${TRAIN_OUTPUT_DIR}/inference_predictions.parquet}"

SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${EXPERIMENT_DIR}/quality_probe_summary.csv}"

echo "[quality_probe_slurm] project_root=${PROJECT_ROOT}"
echo "[quality_probe_slurm] run_build_manifest=${RUN_BUILD_MANIFEST}"
echo "[quality_probe_slurm] run_attach_targets=${RUN_ATTACH_TARGETS}"
echo "[quality_probe_slurm] run_train=${RUN_TRAIN}"
echo "[quality_probe_slurm] run_infer=${RUN_INFER}"
echo "[quality_probe_slurm] run_summary=${RUN_SUMMARY}"
echo "[quality_probe_slurm] failure_root=${FAILURE_ROOT}"
echo "[quality_probe_slurm] experiment_name=${EXPERIMENT_NAME}"
echo "[quality_probe_slurm] experiment_dir=${EXPERIMENT_DIR}"
echo "[quality_probe_slurm] patch_manifest_input=${PATCH_MANIFEST_INPUT}"
echo "[quality_probe_slurm] patch_manifest_output=${PATCH_MANIFEST_OUTPUT}"
echo "[quality_probe_slurm] patch_manifest_patch_id_col=${PATCH_MANIFEST_PATCH_ID_COL}"
echo "[quality_probe_slurm] patch_manifest_slide_id_col=${PATCH_MANIFEST_SLIDE_ID_COL}"
echo "[quality_probe_slurm] patch_manifest_slide_id_strip_regex=${PATCH_MANIFEST_SLIDE_ID_STRIP_REGEX}"
echo "[quality_probe_slurm] patch_manifest_dataset_value=${PATCH_MANIFEST_DATASET_VALUE}"
echo "[quality_probe_slurm] patch_manifest_split_value=${PATCH_MANIFEST_SPLIT_VALUE}"
echo "[quality_probe_slurm] eval_dir=${EVAL_DIR}"
echo "[quality_probe_slurm] eval_glob=${EVAL_GLOB}"
echo "[quality_probe_slurm] joined_manifest_output=${JOINED_MANIFEST_OUTPUT}"
echo "[quality_probe_slurm] train_input=${TRAIN_INPUT}"
echo "[quality_probe_slurm] train_output_dir=${TRAIN_OUTPUT_DIR}"
echo "[quality_probe_slurm] train_problem_type=${TRAIN_PROBLEM_TYPE}"
echo "[quality_probe_slurm] train_feature_mode=${TRAIN_FEATURE_MODE}"
echo "[quality_probe_slurm] train_target_col=${TRAIN_TARGET_COL}"
echo "[quality_probe_slurm] train_target_aggregation=${TRAIN_TARGET_AGGREGATION}"
echo "[quality_probe_slurm] train_group_col=${TRAIN_GROUP_COL}"
echo "[quality_probe_slurm] train_classification_threshold_strategy=${TRAIN_CLASSIFICATION_THRESHOLD_STRATEGY}"
echo "[quality_probe_slurm] train_positive_class_name=${TRAIN_POSITIVE_CLASS_NAME}"
echo "[quality_probe_slurm] train_classifier_families=${TRAIN_CLASSIFIER_FAMILIES}"
echo "[quality_probe_slurm] train_optuna_trials=${TRAIN_OPTUNA_TRIALS}"
echo "[quality_probe_slurm] train_cv_folds=${TRAIN_CV_FOLDS}"
echo "[quality_probe_slurm] train_n_jobs=${TRAIN_N_JOBS}"
echo "[quality_probe_slurm] infer_input=${INFER_INPUT}"
echo "[quality_probe_slurm] infer_output=${INFER_OUTPUT}"
echo "[quality_probe_slurm] summary_output=${SUMMARY_OUTPUT}"
echo "[quality_probe_slurm] host=$(hostname)"
echo "[quality_probe_slurm] date=$(date)"
echo "[quality_probe_slurm] python=$(python --version 2>&1 || true)"
echo "[quality_probe_slurm] pixi=$(pixi --version 2>&1 || true)"
echo "[quality_probe_slurm] slurm_job_id=${SLURM_JOB_ID:-manual}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "[quality_probe_slurm] pixi is not on PATH" >&2
  exit 1
fi

run_command() {
  local -a command=("$@")
  printf '[quality_probe_slurm] command='
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" "${command[@]}"
  else
    "${command[@]}"
  fi
}

if [[ "${RUN_BUILD_MANIFEST}" == "1" ]]; then
  read -r -a patch_manifest_extra_cols <<< "${PATCH_MANIFEST_EXTRA_COLS}"
  run_command \
    pixi run -e default python scripts/benchmarking/failure_prediction/01_build_patch_manifest.py \
    --input "${PATCH_MANIFEST_INPUT}" \
    --patch-id-col "${PATCH_MANIFEST_PATCH_ID_COL}" \
    --slide-id-col "${PATCH_MANIFEST_SLIDE_ID_COL}" \
    --slide-id-strip-regex "${PATCH_MANIFEST_SLIDE_ID_STRIP_REGEX}" \
    --dataset-value "${PATCH_MANIFEST_DATASET_VALUE}" \
    --split-value "${PATCH_MANIFEST_SPLIT_VALUE}" \
    --extra-cols "${patch_manifest_extra_cols[@]}" \
    --output "${PATCH_MANIFEST_OUTPUT}"
fi

if [[ "${RUN_ATTACH_TARGETS}" == "1" ]]; then
  run_command \
    pixi run -e default python scripts/benchmarking/failure_prediction/02_attach_eval_targets.py \
    --manifest "${PATCH_MANIFEST_OUTPUT}" \
    --eval-dir "${EVAL_DIR}" \
    --eval-glob "${EVAL_GLOB}" \
    --output "${JOINED_MANIFEST_OUTPUT}"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  read -r -a classifier_families <<< "${TRAIN_CLASSIFIER_FAMILIES}"
  train_command=(
    pixi run -e default python scripts/benchmarking/failure_prediction/03_train_quality_probe.py
    --input "${TRAIN_INPUT}"
    --output-dir "${TRAIN_OUTPUT_DIR}"
    --problem-type "${TRAIN_PROBLEM_TYPE}"
    --feature-mode "${TRAIN_FEATURE_MODE}"
    --target-col "${TRAIN_TARGET_COL}"
    --target-aggregation "${TRAIN_TARGET_AGGREGATION}"
    --group-col "${TRAIN_GROUP_COL}"
    --classification-threshold-strategy "${TRAIN_CLASSIFICATION_THRESHOLD_STRATEGY}"
    --positive-class-name "${TRAIN_POSITIVE_CLASS_NAME}"
    --classifier-families "${classifier_families[@]}"
    --optuna-trials "${TRAIN_OPTUNA_TRIALS}"
    --cv-folds "${TRAIN_CV_FOLDS}"
    --n-jobs "${TRAIN_N_JOBS}"
  )

  if [[ "${TRAIN_AUTO_METADATA_COLS}" == "1" ]]; then
    train_command+=(--auto-metadata-cols)
  fi

  if [[ "${TRAIN_REQUIRE_EXISTING_EMBEDDINGS}" == "1" ]]; then
    train_command+=(--require-existing-embeddings)
  fi

  if [[ "${TRAIN_REQUIRE_COMPLETE_MODEL_COVERAGE}" == "1" ]]; then
    train_command+=(--require-complete-model-coverage)
  else
    train_command+=(--no-require-complete-model-coverage)
  fi

  if [[ -n "${TRAIN_CLASSIFICATION_THRESHOLD}" ]]; then
    train_command+=(--classification-threshold "${TRAIN_CLASSIFICATION_THRESHOLD}")
  fi

  if [[ -n "${TRAIN_METADATA_COLS}" ]]; then
    read -r -a metadata_cols <<< "${TRAIN_METADATA_COLS}"
    train_command+=(--metadata-cols "${metadata_cols[@]}")
  fi

  run_command "${train_command[@]}"
fi

if [[ "${RUN_INFER}" == "1" ]]; then
  run_command \
    pixi run -e default python scripts/benchmarking/failure_prediction/04_infer_quality_probe.py \
    --input "${INFER_INPUT}" \
    --artifact-dir "${TRAIN_OUTPUT_DIR}" \
    --output "${INFER_OUTPUT}"
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  run_command \
    pixi run -e default python scripts/benchmarking/failure_prediction/05_summarize_quality_probe.py \
    --artifact-dirs "${TRAIN_OUTPUT_DIR}" \
    --inference-tables "${INFER_OUTPUT}" \
    --output "${SUMMARY_OUTPUT}"
fi
