# Modeling Workflow

This directory contains the next-phase sklearn-first modeling workflow for the cell-segmentation difficulty project.

The workflow has three modeling stages built on one canonical prepared table:

1. `main_model`: regression on continuous consensus patch quality, default target `pq_median`
2. `report_model`: 3-class easy/medium/hard classification from train-only quantiles of `pq_median`
3. `failure_mode`: hard-patch failure-type classification from dominant consensus deficits in `rq_median`, `sq_median`, `pixel_precision_median`, and `pixel_recall_median`

## Why this workflow exists

- Keep the modeling stack simple, inspectable, and reproducible
- Reuse one canonical prepared table across all stages
- Keep split-dependent label derivation inside the training scripts, not in the prep step
- Prefer sklearn baselines with small focused Optuna searches over a heavier framework
- Write inspectable artifacts as CSV, CSV.GZ, JSON, Markdown, and plots only

## Downstream analysis support

The workflow now writes both ready-to-use plots and the structured tables needed to recreate additional figures later.

Key reusable artifacts across training stages:

- `predictions.csv`
- `study_trials.csv`
- `metrics.json`
- `best_params.json`
- `family_search_results.json`
- `optuna_family_scores.csv`
- `optuna_param_importances.json`
- stage-specific plot files under `plots/`

The summary step also writes a recursive `report_asset_index.csv` and `report_asset_index.md` so downstream notebooks can discover plot and table assets without hard-coding filenames.

## Default artifact root

Base root:

```text
outputs/conic_liz/model/
```

Named experiment root:

```text
outputs/conic_liz/model/experiments/<experiment_name>/
```

Typical stage layout inside an experiment:

```text
outputs/conic_liz/model/experiments/<experiment_name>/
  model_table/
    modeling_table.csv.gz
    config.json
    validation.json
    validation.md
    run.log
    timing.json
  main_model/
  report_model/
  failure_mode/
  summary/
```

## Inputs

Primary target source:

```text
tmp/extra/outputs_22_04_26/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet
```

Optional metadata enrichments:

```text
outputs/conic_liz/embed_morph.csv
outputs/conic_liz/embeddings/metadata/embeddings_index.csv
outputs/conic_liz/patch_features.csv
```

Important current contract:

- `patch_id` is the canonical join key
- `slide_id` is the grouped split key
- `modeling_table.csv.gz` is the canonical prepared table
- New workflow outputs do not use parquet

Important data caveat:

- The failure-mode stage requires consensus columns for `rq_median`, `sq_median`, `pixel_precision_median`, and `pixel_recall_median`
- If the upstream joined target table does not expose those metrics, `04_train_failure_mode_model.py` will stop with a clear missing-column error instead of guessing

## Step 01: Build Model Table

Purpose:
- Collapse patch-by-model evaluation rows into one patch-level modeling table

Why it exists in the project:
- All downstream stages should consume the same prepared inputs and target columns

Reads:
- `--joined-target-table`
- Optional `--embeddings-index`
- Optional `--embed-morph`
- Optional `--patch-features`

Writes:
- `modeling_table.csv.gz`
- `config.json`
- `validation.json`
- `validation.md`
- `run.log`
- `timing.json`

Important CLI arguments:
- `--joined-target-table`
- `--embeddings-index`
- `--embed-morph`
- `--patch-features`
- `--patch-id-col`
- `--slide-id-col`
- `--output`

What it derives:
- Continuous consensus targets such as `pq_median`
- Per-model metric columns such as `pq__cellsam`
- Metric coverage counts such as `pq_model_count`
- Joined patch-level metadata columns from optional inputs

What it does not derive:
- Train/test splits
- Easy/medium/hard labels
- Failure-mode labels

Example command:

```bash
pixi run -e default python scripts/benchmarking/model/01_build_model_table.py \
  --joined-target-table tmp/extra/outputs_22_04_26/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet \
  --embed-morph outputs/conic_liz/embed_morph.csv \
  --output outputs/conic_liz/model/experiments/baseline/model_table/modeling_table.csv.gz
```

Expected artifacts:
- `modeling_table.csv.gz`
- validation reports describing row counts, discovered metrics, optional joins, duplicate checks, and missingness

## Step 02: Train Main Model

Purpose:
- Predict continuous consensus patch quality

Why it exists in the project:
- Continuous regression is the cleanest first test of whether embeddings or metadata capture structured difficulty

Reads:
- `--input-table`
- Embedding files referenced in `modeling_table.csv.gz` when `--feature-set` includes embeddings

Writes:
- `model.joblib`
- `metrics.json`
- `best_params.json`
- `family_search_results.json`
- `optuna_family_scores.csv`
- `optuna_param_importances.json`
- `predictions.csv`
- `study_trials.csv`
- `plots/predicted_vs_observed.png`
- `plots/residuals.png`
- `plots/residual_histogram.png`
- `plots/optuna_family_cv_scores.png`
- `plots/optuna_history_<family>.png`
- `plots/optuna_param_importance_<family>.png`
- validation artifacts plus logs/timing

Important CLI arguments:
- `--input-table`
- `--output-dir`
- `--target-col`
- `--feature-set`
- `--metadata-cols`
- `--model-families`
- `--group-col`
- `--test-size`
- `--cv-folds`
- `--optuna-trials`
- `--optuna-timeout`
- `--n-jobs`
- `--random-seed`

Supported model families:
- `ridge`
- `svr`
- `random_forest`
- `xgboost` only when importable in the environment

Metrics and plots:
- MAE
- RMSE
- R2
- Pearson
- Spearman
- predicted-vs-observed scatter
- residual plot
- residual histogram
- Optuna family comparison
- Optuna optimization history per family
- Optuna parameter-importance plots per family when available

Example command:

```bash
pixi run -e default python scripts/benchmarking/model/02_train_main_model.py \
  --input-table outputs/conic_liz/model/experiments/baseline/model_table/modeling_table.csv.gz \
  --output-dir outputs/conic_liz/model/experiments/baseline/main_model \
  --target-col pq_median \
  --feature-set embedding_only \
  --model-families ridge svr random_forest \
  --group-col slide_id \
  --optuna-trials 20 \
  --cv-folds 5
```

Expected artifacts:
- A fitted regression model plus inspectable metrics, predictions, search results, plots, and validation reports

## Step 03: Train Report Model

Purpose:
- Predict ordinal easy/medium/hard patch classes

Why it exists in the project:
- This is the report-friendly classification view for communication and downstream triage

Reads:
- `--input-table`
- Embedding files when `--feature-set` includes embeddings

Writes:
- `model.joblib`
- `metrics.json`
- `best_params.json`
- `family_search_results.json`
- `optuna_family_scores.csv`
- `optuna_param_importances.json`
- `predictions.csv`
- `study_trials.csv`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `plots/confusion_matrix.png`
- `plots/per_class_metrics.png`
- `plots/label_distribution.png`
- `plots/prediction_confidence.png` when class probabilities are available
- `plots/optuna_family_cv_scores.png`
- `plots/optuna_history_<family>.png`
- `plots/optuna_param_importance_<family>.png`
- validation artifacts plus logs/timing

Important CLI arguments:
- Common modeling flags listed in Step 02
- `--class-quantiles`
- `--class-labels`

Important label contract:
- Thresholds are computed on the training split only
- Labels are ordered from low target values to high target values
- Default labels are `hard medium easy`

Supported model families:
- `logistic_regression`
- `svm`
- `random_forest`
- `xgboost` only when importable in the environment

Metrics:
- macro-F1
- balanced accuracy
- per-class metrics
- quadratic weighted kappa
- confusion matrix
- per-class precision/recall/F1 plot
- true-vs-predicted label distribution plot
- prediction-confidence histogram
- Optuna family comparison, history, and parameter-importance plots

Example command:

```bash
pixi run -e default python scripts/benchmarking/model/03_train_report_model.py \
  --input-table outputs/conic_liz/model/experiments/baseline/model_table/modeling_table.csv.gz \
  --output-dir outputs/conic_liz/model/experiments/baseline/report_model \
  --target-col pq_median \
  --feature-set embedding_only \
  --class-quantiles 0.333333 0.666667 \
  --class-labels hard medium easy
```

Expected artifacts:
- A fitted ordinal classifier, test predictions, confusion-matrix assets, and validation reports

## Step 04: Train Failure-Mode Model

Purpose:
- Predict the dominant failure type among hard patches only

Why it exists in the project:
- Distinguish overall difficulty from failure mechanism, so the workflow can ask both "is this patch hard?" and "what kind of failure is likely?"

Reads:
- `--input-table`
- Embedding files when `--feature-set` includes embeddings

Writes:
- `model.joblib`
- `metrics.json`
- `best_params.json`
- `family_search_results.json`
- `optuna_family_scores.csv`
- `optuna_param_importances.json`
- `predictions.csv`
- `hard_patch_labels.csv`
- `study_trials.csv`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `plots/confusion_matrix.png`
- `plots/per_class_metrics.png`
- `plots/label_distribution.png`
- `plots/prediction_confidence.png` when class probabilities are available
- `plots/optuna_family_cv_scores.png`
- `plots/optuna_history_<family>.png`
- `plots/optuna_param_importance_<family>.png`
- validation artifacts plus logs/timing

Important CLI arguments:
- Common modeling flags listed in Step 02
- `--hard-quantile`
- `--dominance-margin`
- `--min-class-count`

Failure-label logic:
- The hard threshold comes from the training split only
- A patch is considered hard when `pq_median <= train_quantile`
- Failure labels are based on the largest deficit among:
  - `rq_median`
  - `sq_median`
  - `pixel_precision_median`
  - `pixel_recall_median`
- A label is assigned only when the largest deficit exceeds the second-largest deficit by at least `--dominance-margin`
- Classes below `--min-class-count` in train are dropped explicitly and recorded

Supported model families:
- `logistic_regression`
- `svm`
- `random_forest`
- `xgboost` only when importable in the environment

Metrics:
- macro-F1
- balanced accuracy
- per-class metrics
- confusion matrix
- per-class precision/recall/F1 plot
- true-vs-predicted label distribution plot
- prediction-confidence histogram
- Optuna family comparison, history, and parameter-importance plots

Example command:

```bash
pixi run -e default python scripts/benchmarking/model/04_train_failure_mode_model.py \
  --input-table outputs/conic_liz/model/experiments/baseline/model_table/modeling_table.csv.gz \
  --output-dir outputs/conic_liz/model/experiments/baseline/failure_mode \
  --target-col pq_median \
  --feature-set embedding_only \
  --hard-quantile 0.333333 \
  --dominance-margin 0.05 \
  --min-class-count 25
```

Expected artifacts:
- A fitted failure-mode classifier, hard-patch label table, confusion-matrix assets, and validation reports

## Step 05: Summarize Model Runs

Purpose:
- Verify that the requested stage artifacts exist and flatten the core outputs into compact report tables

Why it exists in the project:
- Downstream analysis and reporting need one stable place to compare the three modeling stages

Reads:
- `--main-dir`
- `--report-dir`
- `--failure-dir`

Writes:
- `stage_metrics.csv`
- `stage_metrics.md`
- `report_asset_index.csv`
- `report_asset_index.md`
- `summary.json`
- validation artifacts plus logs/timing

Important CLI arguments:
- `--main-dir`
- `--report-dir`
- `--failure-dir`
- `--output-dir`

Example command:

```bash
pixi run -e default python scripts/benchmarking/model/05_summarize_model_runs.py \
  --main-dir outputs/conic_liz/model/experiments/baseline/main_model \
  --report-dir outputs/conic_liz/model/experiments/baseline/report_model \
  --failure-dir outputs/conic_liz/model/experiments/baseline/failure_mode \
  --output-dir outputs/conic_liz/model/experiments/baseline/summary
```

Expected artifacts:
- compact CSV/Markdown stage comparison tables
- a recursive report asset index that includes plot files and auxiliary tables
- validation reports confirming the required upstream artifacts existed

## Feature sets

All training stages support:

- `embedding_only`
- `metadata_only`
- `embedding_plus_metadata`

Notes:

- Embedding modes require readable embedding metadata in the canonical table
- Metadata columns can be provided explicitly with `--metadata-cols`
- If `--metadata-cols` is omitted, columns are inferred while excluding target/per-model metric columns

## Validation behavior

Every script writes:

- `validation.json`
- `validation.md`
- `run.log`
- `timing.json`

Training-stage validation also checks:

- train/test group disjointness
- metrics file creation
- predictions file creation
- model artifact creation

## SLURM wrapper

The wrapper `run_model_workflow.slurm` mirrors the existing failure-prediction style and supports environment toggles:

- `RUN_PREP`
- `RUN_MAIN`
- `RUN_REPORT`
- `RUN_FAILURE`
- `RUN_SUMMARY`
- `EXPERIMENT_NAME`

Default named run:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=baseline scripts/benchmarking/model/run_model_workflow.slurm
```

The wrapper writes under:

```text
outputs/conic_liz/model/experiments/<experiment_name>/
```
