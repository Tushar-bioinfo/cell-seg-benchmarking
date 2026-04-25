#!/bin/bash
#SBATCH --job-name=model-workflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --output=model-workflow-%j.out
#SBATCH --error=model-workflow-%j.out

set -euo pipefail

SCRIPT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_PROJECT_ROOT="/share/lab_teng/trainee/tusharsingh/cell-seg"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="${SCRIPT_PROJECT_ROOT}"
fi
cd "${PROJECT_ROOT}"

RUN_PREP="${RUN_PREP:-1}"
RUN_MAIN="${RUN_MAIN:-1}"
RUN_REPORT="${RUN_REPORT:-1}"
RUN_FAILURE="${RUN_FAILURE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

MODEL_ROOT="${MODEL_ROOT:-outputs/conic_liz/model}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-baseline}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${MODEL_ROOT}/experiments/${EXPERIMENT_NAME}}"

PREP_JOINED_TARGET_TABLE="${PREP_JOINED_TARGET_TABLE:-outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet}"
PREP_EMBEDDINGS_INDEX="${PREP_EMBEDDINGS_INDEX:-}"
PREP_EMBED_MORPH="${PREP_EMBED_MORPH:-outputs/conic_liz/embed_morph.csv}"
PREP_PATCH_FEATURES="${PREP_PATCH_FEATURES:-}"
PREP_PATCH_ID_COL="${PREP_PATCH_ID_COL:-patch_id}"
PREP_SLIDE_ID_COL="${PREP_SLIDE_ID_COL:-slide_id}"
MODEL_TABLE_OUTPUT="${MODEL_TABLE_OUTPUT:-${EXPERIMENT_DIR}/model_table/modeling_table.csv.gz}"

COMMON_GROUP_COL="${COMMON_GROUP_COL:-slide_id}"
COMMON_TEST_SIZE="${COMMON_TEST_SIZE:-0.2}"
COMMON_CV_FOLDS="${COMMON_CV_FOLDS:-5}"
COMMON_OPTUNA_TRIALS="${COMMON_OPTUNA_TRIALS:-20}"
COMMON_OPTUNA_TIMEOUT="${COMMON_OPTUNA_TIMEOUT:-600}"
COMMON_N_JOBS="${COMMON_N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
COMMON_RANDOM_SEED="${COMMON_RANDOM_SEED:-42}"

MAIN_INPUT="${MAIN_INPUT:-${MODEL_TABLE_OUTPUT}}"
MAIN_OUTPUT_DIR="${MAIN_OUTPUT_DIR:-${EXPERIMENT_DIR}/main_model}"
MAIN_TARGET_COL="${MAIN_TARGET_COL:-pq_median}"
MAIN_FEATURE_SET="${MAIN_FEATURE_SET:-embedding_only}"
MAIN_METADATA_COLS="${MAIN_METADATA_COLS:-}"
MAIN_MODEL_FAMILIES="${MAIN_MODEL_FAMILIES:-ridge svr random_forest}"

REPORT_INPUT="${REPORT_INPUT:-${MODEL_TABLE_OUTPUT}}"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-${EXPERIMENT_DIR}/report_model}"
REPORT_TARGET_COL="${REPORT_TARGET_COL:-pq_median}"
REPORT_FEATURE_SET="${REPORT_FEATURE_SET:-embedding_only}"
REPORT_METADATA_COLS="${REPORT_METADATA_COLS:-}"
REPORT_MODEL_FAMILIES="${REPORT_MODEL_FAMILIES:-logistic_regression svm random_forest}"
REPORT_CLASS_QUANTILES="${REPORT_CLASS_QUANTILES:-0.333333 0.666667}"
REPORT_CLASS_LABELS="${REPORT_CLASS_LABELS:-hard medium easy}"

FAILURE_INPUT="${FAILURE_INPUT:-${MODEL_TABLE_OUTPUT}}"
FAILURE_OUTPUT_DIR="${FAILURE_OUTPUT_DIR:-${EXPERIMENT_DIR}/failure_mode}"
FAILURE_TARGET_COL="${FAILURE_TARGET_COL:-pq_median}"
FAILURE_FEATURE_SET="${FAILURE_FEATURE_SET:-embedding_only}"
FAILURE_METADATA_COLS="${FAILURE_METADATA_COLS:-}"
FAILURE_MODEL_FAMILIES="${FAILURE_MODEL_FAMILIES:-logistic_regression svm random_forest}"
FAILURE_HARD_QUANTILE="${FAILURE_HARD_QUANTILE:-0.333333}"
FAILURE_DOMINANCE_MARGIN="${FAILURE_DOMINANCE_MARGIN:-0.05}"
FAILURE_MIN_CLASS_COUNT="${FAILURE_MIN_CLASS_COUNT:-25}"

SUMMARY_MAIN_DIR="${SUMMARY_MAIN_DIR:-${MAIN_OUTPUT_DIR}}"
SUMMARY_REPORT_DIR="${SUMMARY_REPORT_DIR:-${REPORT_OUTPUT_DIR}}"
SUMMARY_FAILURE_DIR="${SUMMARY_FAILURE_DIR:-${FAILURE_OUTPUT_DIR}}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-${EXPERIMENT_DIR}/summary}"

echo "[model_workflow_slurm] project_root=${PROJECT_ROOT}"
echo "[model_workflow_slurm] run_prep=${RUN_PREP}"
echo "[model_workflow_slurm] run_main=${RUN_MAIN}"
echo "[model_workflow_slurm] run_report=${RUN_REPORT}"
echo "[model_workflow_slurm] run_failure=${RUN_FAILURE}"
echo "[model_workflow_slurm] run_summary=${RUN_SUMMARY}"
echo "[model_workflow_slurm] model_root=${MODEL_ROOT}"
echo "[model_workflow_slurm] experiment_name=${EXPERIMENT_NAME}"
echo "[model_workflow_slurm] experiment_dir=${EXPERIMENT_DIR}"
echo "[model_workflow_slurm] model_table_output=${MODEL_TABLE_OUTPUT}"
echo "[model_workflow_slurm] main_output_dir=${MAIN_OUTPUT_DIR}"
echo "[model_workflow_slurm] report_output_dir=${REPORT_OUTPUT_DIR}"
echo "[model_workflow_slurm] failure_output_dir=${FAILURE_OUTPUT_DIR}"
echo "[model_workflow_slurm] summary_output_dir=${SUMMARY_OUTPUT_DIR}"
echo "[model_workflow_slurm] common_n_jobs=${COMMON_N_JOBS}"
echo "[model_workflow_slurm] host=$(hostname)"
echo "[model_workflow_slurm] date=$(date)"
echo "[model_workflow_slurm] python=$(python --version 2>&1 || true)"
echo "[model_workflow_slurm] pixi=$(pixi --version 2>&1 || true)"
echo "[model_workflow_slurm] slurm_job_id=${SLURM_JOB_ID:-manual}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "[model_workflow_slurm] pixi is not on PATH" >&2
  exit 1
fi

run_command() {
  local -a command=("$@")
  printf '[model_workflow_slurm] command='
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" "${command[@]}"
  else
    "${command[@]}"
  fi
}

if [[ "${RUN_PREP}" == "1" ]]; then
  prep_command=(
    pixi run -e default python scripts/benchmarking/model/01_build_model_table.py
    --joined-target-table "${PREP_JOINED_TARGET_TABLE}"
    --patch-id-col "${PREP_PATCH_ID_COL}"
    --slide-id-col "${PREP_SLIDE_ID_COL}"
    --output "${MODEL_TABLE_OUTPUT}"
  )
  if [[ -n "${PREP_EMBEDDINGS_INDEX}" ]]; then
    prep_command+=(--embeddings-index "${PREP_EMBEDDINGS_INDEX}")
  fi
  if [[ -n "${PREP_EMBED_MORPH}" ]]; then
    prep_command+=(--embed-morph "${PREP_EMBED_MORPH}")
  fi
  if [[ -n "${PREP_PATCH_FEATURES}" ]]; then
    prep_command+=(--patch-features "${PREP_PATCH_FEATURES}")
  fi
  run_command "${prep_command[@]}"
fi

if [[ "${RUN_MAIN}" == "1" ]]; then
  read -r -a main_model_families <<< "${MAIN_MODEL_FAMILIES}"
  main_command=(
    pixi run -e default python scripts/benchmarking/model/02_train_main_model.py
    --input-table "${MAIN_INPUT}"
    --output-dir "${MAIN_OUTPUT_DIR}"
    --target-col "${MAIN_TARGET_COL}"
    --feature-set "${MAIN_FEATURE_SET}"
    --model-families "${main_model_families[@]}"
    --group-col "${COMMON_GROUP_COL}"
    --test-size "${COMMON_TEST_SIZE}"
    --cv-folds "${COMMON_CV_FOLDS}"
    --optuna-trials "${COMMON_OPTUNA_TRIALS}"
    --optuna-timeout "${COMMON_OPTUNA_TIMEOUT}"
    --n-jobs "${COMMON_N_JOBS}"
    --random-seed "${COMMON_RANDOM_SEED}"
  )
  if [[ -n "${MAIN_METADATA_COLS}" ]]; then
    read -r -a main_metadata_cols <<< "${MAIN_METADATA_COLS}"
    main_command+=(--metadata-cols "${main_metadata_cols[@]}")
  fi
  run_command "${main_command[@]}"
fi

if [[ "${RUN_REPORT}" == "1" ]]; then
  read -r -a report_model_families <<< "${REPORT_MODEL_FAMILIES}"
  read -r -a report_class_quantiles <<< "${REPORT_CLASS_QUANTILES}"
  read -r -a report_class_labels <<< "${REPORT_CLASS_LABELS}"
  report_command=(
    pixi run -e default python scripts/benchmarking/model/03_train_report_model.py
    --input-table "${REPORT_INPUT}"
    --output-dir "${REPORT_OUTPUT_DIR}"
    --target-col "${REPORT_TARGET_COL}"
    --feature-set "${REPORT_FEATURE_SET}"
    --model-families "${report_model_families[@]}"
    --group-col "${COMMON_GROUP_COL}"
    --test-size "${COMMON_TEST_SIZE}"
    --cv-folds "${COMMON_CV_FOLDS}"
    --optuna-trials "${COMMON_OPTUNA_TRIALS}"
    --optuna-timeout "${COMMON_OPTUNA_TIMEOUT}"
    --n-jobs "${COMMON_N_JOBS}"
    --random-seed "${COMMON_RANDOM_SEED}"
    --class-quantiles "${report_class_quantiles[@]}"
    --class-labels "${report_class_labels[@]}"
  )
  if [[ -n "${REPORT_METADATA_COLS}" ]]; then
    read -r -a report_metadata_cols <<< "${REPORT_METADATA_COLS}"
    report_command+=(--metadata-cols "${report_metadata_cols[@]}")
  fi
  run_command "${report_command[@]}"
fi

if [[ "${RUN_FAILURE}" == "1" ]]; then
  read -r -a failure_model_families <<< "${FAILURE_MODEL_FAMILIES}"
  failure_command=(
    pixi run -e default python scripts/benchmarking/model/04_train_failure_mode_model.py
    --input-table "${FAILURE_INPUT}"
    --output-dir "${FAILURE_OUTPUT_DIR}"
    --target-col "${FAILURE_TARGET_COL}"
    --feature-set "${FAILURE_FEATURE_SET}"
    --model-families "${failure_model_families[@]}"
    --group-col "${COMMON_GROUP_COL}"
    --test-size "${COMMON_TEST_SIZE}"
    --cv-folds "${COMMON_CV_FOLDS}"
    --optuna-trials "${COMMON_OPTUNA_TRIALS}"
    --optuna-timeout "${COMMON_OPTUNA_TIMEOUT}"
    --n-jobs "${COMMON_N_JOBS}"
    --random-seed "${COMMON_RANDOM_SEED}"
    --hard-quantile "${FAILURE_HARD_QUANTILE}"
    --dominance-margin "${FAILURE_DOMINANCE_MARGIN}"
    --min-class-count "${FAILURE_MIN_CLASS_COUNT}"
  )
  if [[ -n "${FAILURE_METADATA_COLS}" ]]; then
    read -r -a failure_metadata_cols <<< "${FAILURE_METADATA_COLS}"
    failure_command+=(--metadata-cols "${failure_metadata_cols[@]}")
  fi
  run_command "${failure_command[@]}"
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  run_command \
    pixi run -e default python scripts/benchmarking/model/05_summarize_model_runs.py \
    --main-dir "${SUMMARY_MAIN_DIR}" \
    --report-dir "${SUMMARY_REPORT_DIR}" \
    --failure-dir "${SUMMARY_FAILURE_DIR}" \
    --output-dir "${SUMMARY_OUTPUT_DIR}"
fi
