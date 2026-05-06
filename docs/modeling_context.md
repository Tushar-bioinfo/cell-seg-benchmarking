# Modeling Workflow Context

## Scope

- This file is the compact source of truth for `scripts/benchmarking/model/`.
- Use it when working on the sklearn-first difficulty-modeling workflow, the canonical modeling table, grouped train/test splitting, failure-mode labeling, or the Slurm entrypoint.
- Prefer this file over reopening the full modeling scripts unless implementation details must be verified.

## Goals

- Connect the segmentation benchmarking outputs to the project goal of predicting patch-level difficulty from pretrained pathology representations.
- Keep the modeling stack simple, inspectable, and reproducible.
- Reuse one canonical patch-level table across all downstream modeling stages.
- Separate three related questions:
  - how well can we predict continuous consensus quality?
  - how well can we predict report-friendly easy/medium/hard difficulty classes?
  - among the hardest patches, what failure pattern is most likely?

## Project Fit

- The workflow supports the main project claim that segmentation failures are structured rather than random.
- It operationalizes the project goals in `docs/goals.md` by turning consensus patch quality into reusable modeling targets.
- It is intended to answer these practical questions:
  - do embeddings predict continuous patch quality?
  - do embeddings predict easy/medium/hard difficulty labels?
  - can hard patches be separated by dominant failure type?

## Workflow Overview

- Runtime root:
  - `scripts/benchmarking/model/`
- Canonical output root:
  - `outputs/conic_liz/model/`
- Named experiment root:
  - `outputs/conic_liz/model/experiments/<experiment_name>/`
- Canonical prepared table:
  - `modeling_table.csv.gz`
- New workflow outputs are inspectable only:
  - `.csv`
  - `.csv.gz`
  - `.json`
  - `.md`
  - plot files
- New workflow outputs do not use parquet.

## Stage Order

- `01_build_model_table.py`
  - build one canonical patch-level modeling table
- `02_train_main_model.py`
  - regress continuous consensus quality, default target `pq_median`
- `03_train_report_model.py`
  - derive train-only quantile labels and train an easy/medium/hard classifier
- `04_train_failure_mode_model.py`
  - restrict to hard patches and classify dominant failure type
- `05_summarize_model_runs.py`
  - write compact comparison tables and a recursive asset index
- `run_model_workflow.slurm`
  - wrapper for full or partial HPC runs

## Inputs

- Primary joined target table:
  - typically `outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet`
- Optional metadata joins:
  - `outputs/conic_liz/embed_morph.csv`
  - `outputs/conic_liz/embeddings/metadata/embeddings_index.csv`
  - `outputs/conic_liz/patch_features.csv`
- Optional evaluation CSVs used for metric enrichment:
  - `outputs/conic_liz/*_evaluation.csv`

Required joined-target columns:

- `patch_id`
- `slide_id`
- `model_name`
- at least one PQ alias:
  - `pq`
  - `instance_pq`
  - `panoptic_quality`

When failure-mode training is required, the workflow also needs consensus inputs for:

- `rq_median`
- `sq_median`
- `pixel_precision_median`
- `pixel_recall_median`

Those may come either from the joined target table directly or from automatic enrichment against the evaluation CSVs during prep.

## Key Contracts

- `patch_id` is the canonical patch-level join key.
- `slide_id` is the grouped split key.
- The prep stage writes one row per `patch_id`.
- Split-dependent labels are not derived in prep.
- The same prepared table is reused by all three modeling stages.
- Models are saved with `joblib`.
- Random seeds are fixed and explicit.
- Grouped splitting is mandatory; train and test groups must be disjoint.

## Metric Alias Contract

- Canonical PQ aliases:
  - `pq`
  - `instance_pq`
  - `panoptic_quality`
- Canonical RQ aliases:
  - `rq`
  - `instance_rq`
  - `recognition_quality`
- Canonical SQ aliases:
  - `sq`
  - `instance_sq`
  - `segmentation_quality`
- Canonical pixel-precision aliases:
  - `pixel_precision`
  - `precision`
- Canonical pixel-recall aliases:
  - `pixel_recall`
  - `recall`

Prep collapses patch-by-model rows into:

- per-model columns such as `pq__cellsam`
- consensus medians such as `pq_median`
- model-count columns such as `pq_model_count`

## Missing-Metric Enrichment

- When failure-mode training is requested, prep can auto-enrich missing metrics from evaluation CSVs.
- Current default enrichment source:
  - `outputs/conic_liz/*_evaluation.csv`
- Enrichment is patch-safe:
  - evaluation rows are matched back to `patch_id` through normalized filename-based keys
  - merged rows are still validated for duplicate `(patch_id, model_name)` pairs
- If required consensus metrics still cannot be derived after enrichment, prep fails early.
- This prevents long jobs from reaching `04_train_failure_mode_model.py` and failing late with missing-column errors.

## Canonical Prepared Table

Typical leading columns in `modeling_table.csv.gz`:

- `patch_id`
- `slide_id`
- `dataset`
- `split`
- embedding metadata when available:
  - `embedding_path`
  - `embedding_format`
  - `embedding_row_offset`
  - `embedding_dim`
- consensus targets when available:
  - `pq_median`
  - `rq_median`
  - `sq_median`
  - `pixel_precision_median`
  - `pixel_recall_median`

The prepared table may also include:

- per-model metric columns such as `pq__cellpose_sam`
- metric coverage columns such as `pq_model_count`
- morphology features such as `foreground_fraction` and `mean_area`
- class-composition metadata when available

Prep validation checks:

- input file existence
- required input columns
- duplicate `patch_id`
- duplicate `(patch_id, model_name)`
- patch-level consistency of shared metadata
- output readability
- required output columns
- missingness summary
- optional-source join status
- enrichment status when evaluation CSV repair was used

## Feature Sets

All three training stages support:

- `embedding_only`
- `metadata_only`
- `embedding_plus_metadata`

Feature behavior:

- embedding modes require readable embedding metadata in the canonical table
- metadata columns can be passed explicitly with `--metadata-cols`
- if omitted, metadata columns are inferred while excluding target columns, failure labels, and per-model metric columns

## Main Model

Purpose:

- Predict continuous consensus patch quality, default target `pq_median`

Supported model families:

- `ridge`
- `svr`
- `random_forest`
- `xgboost` only when importable

Primary metrics:

- MAE
- RMSE
- R2
- Pearson
- Spearman

Primary plots:

- predicted-vs-observed scatter
- residual plot
- residual histogram
- Optuna family comparison
- Optuna optimization history
- Optuna parameter-importance plots when available

## Report Model

Purpose:

- Predict three report-friendly difficulty classes: easy, medium, hard

Label contract:

- thresholds are computed on the training split only
- default quantiles are one-third and two-thirds
- default label order is low-to-high quality:
  - `hard`
  - `medium`
  - `easy`

Supported model families:

- `logistic_regression`
- `svm`
- `random_forest`
- `xgboost` only when importable

Primary metrics:

- macro-F1
- balanced accuracy
- per-class metrics
- quadratic weighted kappa

Primary plots:

- confusion matrix
- per-class metrics plot
- label distribution plot
- prediction-confidence plot when probabilities are available
- Optuna family comparison
- Optuna optimization history
- Optuna parameter-importance plots when available

## Failure-Mode Model

Purpose:

- Restrict to hard patches and predict the dominant failure type

Hard-patch contract:

- hard threshold is derived from the training split only
- a patch is hard when `pq_median <= train_quantile`

Failure-label contract:

- failure type is based on the largest deficit among:
  - `rq_median`
  - `sq_median`
  - `pixel_precision_median`
  - `pixel_recall_median`
- deficits are computed as `1 - metric`
- a dominant label is assigned only when the largest deficit exceeds the second-largest deficit by at least `--dominance-margin`
- classes below `--min-class-count` in train are dropped explicitly and recorded

Supported model families:

- `logistic_regression`
- `svm`
- `random_forest`
- `xgboost` only when importable

Primary metrics:

- macro-F1
- balanced accuracy
- per-class metrics
- confusion matrix

Primary plots:

- confusion matrix
- per-class metrics plot
- label distribution plot
- prediction-confidence plot when probabilities are available
- Optuna family comparison
- Optuna optimization history
- Optuna parameter-importance plots when available

## Summary Stage

- Reads stage artifacts from `main_model`, `report_model`, and `failure_mode`.
- Writes:
  - `stage_metrics.csv`
  - `stage_metrics.md`
  - `report_asset_index.csv`
  - `report_asset_index.md`
  - `summary.json`
  - validation artifacts
- Summary now supports partial reporting:
  - available stage directories are summarized
  - missing stage directories are recorded as skipped
- The recursive asset index is intended for downstream notebook and report generation.

## Standard Artifacts

Every stage writes:

- `config.json`
- `validation.json`
- `validation.md`
- `run.log`
- `timing.json`

Training stages also write:

- `model.joblib`
- `metrics.json`
- `best_params.json`
- `family_search_results.json`
- `predictions.csv`
- `study_trials.csv`
- Optuna summary tables and plots

Classification stages also write:

- `confusion_matrix.csv`
- `per_class_metrics.csv`

Failure-mode additionally writes:

- `hard_patch_labels.csv`

## Validation Behavior

All stages validate their own outputs before exiting successfully.

Common checks include:

- file existence
- required columns
- row counts
- duplicate `patch_id` checks where relevant
- missingness summaries where relevant

Training-stage checks also include:

- train/test group disjointness under `slide_id`
- metrics file creation
- predictions file creation
- model artifact creation
- stage-specific plot creation

Summary-stage checks include:

- requested or available stage artifact verification before summary writing
- final summary artifact existence
- stage and asset counts

## Slurm Wrapper

Primary entrypoint:

- `scripts/benchmarking/model/run_model_workflow.slurm`

Supported environment toggles:

- `RUN_PREP`
- `RUN_MAIN`
- `RUN_REPORT`
- `RUN_FAILURE`
- `RUN_SUMMARY`
- `EXPERIMENT_NAME`

Important wrapper behavior:

- outputs are derived under `outputs/conic_liz/model/experiments/<experiment_name>/`
- `${SLURM_CPUS_PER_TASK}` is forwarded to Python `--n-jobs`
- memory stays a Slurm concern rather than a Python argument
- the wrapper stops on failed checks because it runs with `set -euo pipefail`

Failure-mode-related wrapper defaults:

- when `RUN_FAILURE=1`, the wrapper automatically requests:
  - `PREP_REQUIRED_CONSENSUS_METRICS="pq rq sq pixel_precision pixel_recall"`
- when `RUN_FAILURE=1`, the wrapper automatically enables:
  - `PREP_AUTO_ENRICH_MISSING_METRICS=1`
- evaluation CSV discovery defaults to:
  - `PREP_EVAL_DIR=outputs/conic_liz`
  - `PREP_EVAL_GLOB=*_evaluation.csv`

## Practical Defaults

- Treat `pq_median` as the main continuous target unless there is a study-specific override.
- Treat `patch_id` as the primary patch-level join key.
- Treat `slide_id` as the only supported grouped split key unless the workflow is explicitly changed.
- Use small Optuna budgets for fast iteration and higher budgets only after the data contract is stable.
- For reruns after successful prep or training stages, disable completed stages with the wrapper toggles rather than repeating work.
- Prefer partial summary generation over rerunning finished stages when only reporting artifacts are missing.

## Stable Paths

- Workflow scripts:
  - `scripts/benchmarking/model/`
- Canonical prepared table inside an experiment:
  - `outputs/conic_liz/model/experiments/<experiment_name>/model_table/modeling_table.csv.gz`
- Main model output:
  - `outputs/conic_liz/model/experiments/<experiment_name>/main_model/`
- Report model output:
  - `outputs/conic_liz/model/experiments/<experiment_name>/report_model/`
- Failure-mode output:
  - `outputs/conic_liz/model/experiments/<experiment_name>/failure_mode/`
- Final summary output:
  - `outputs/conic_liz/model/experiments/<experiment_name>/summary/`
