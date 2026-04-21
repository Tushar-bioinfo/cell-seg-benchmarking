# Quality Probe Workflow

This directory contains the patch-level failure-prediction workflow for segmentation quality.

Scripts:
- `01_build_patch_manifest.py`: build a canonical patch manifest from an embeddings index table
- `02_attach_eval_targets.py`: join the patch manifest with evaluation CSVs to create training targets
- `03_train_quality_probe.py`: aggregate patch-level targets, tune multiple classifiers with Optuna, and train the final failure predictor
- `04_infer_quality_probe.py`: run the trained probe on new patches without ground truth
- `05_summarize_quality_probe.py`: flatten artifact metrics/config plus optional inference tables into one report

## Quickstart

Default output root:

```text
outputs/conic_liz/failure_prediction/
```

Named experiment output root:

```text
outputs/conic_liz/failure_prediction/experiments/<experiment_name>/
```

Default full HPC run:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=median_pq_classifier scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

That single job will run `01` -> `02` -> `03` -> `04` -> `05` and write under:

```text
outputs/conic_liz/failure_prediction/experiments/median_pq_classifier/
```

## Typical local flow

Full local run, step by step:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/01_build_patch_manifest.py \
  --input outputs/embeddings/metadata/embeddings_index.parquet \
  --patch-id-col sample_id \
  --slide-id-col sample_id \
  --slide-id-strip-regex '-[^-]+$' \
  --dataset-value conic_liz \
  --split-value all \
  --extra-cols conic_index image_path mask_path class_label_path resolved_image_path embedding_row_offset embedding_format embedding_dim \
  --output outputs/conic_liz/failure_prediction/patch_manifest.parquet

pixi run -e default python scripts/benchmarking/failure_prediction/02_attach_eval_targets.py \
  --manifest outputs/conic_liz/failure_prediction/patch_manifest.parquet \
  --eval-dir outputs/conic_liz \
  --output outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet

pixi run -e default python scripts/benchmarking/failure_prediction/03_train_quality_probe.py \
  --input outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet \
  --output-dir outputs/conic_liz/failure_prediction/quality_probe \
  --problem-type classification \
  --feature-mode embedding_only \
  --target-col instance_pq \
  --target-aggregation median \
  --group-col slide_id \
  --classifier-families logistic_regression random_forest svm \
  --optuna-trials 40 \
  --cv-folds 5 \
  --require-complete-model-coverage \
  --require-existing-embeddings

pixi run -e default python scripts/benchmarking/failure_prediction/04_infer_quality_probe.py \
  --input outputs/conic_liz/failure_prediction/patch_manifest.parquet \
  --artifact-dir outputs/conic_liz/failure_prediction/quality_probe \
  --output outputs/conic_liz/failure_prediction/quality_probe/inference_predictions.parquet

pixi run -e default python scripts/benchmarking/failure_prediction/05_summarize_quality_probe.py \
  --artifact-dirs outputs/conic_liz/failure_prediction/quality_probe \
  --inference-tables outputs/conic_liz/failure_prediction/quality_probe/inference_predictions.parquet \
  --output outputs/conic_liz/failure_prediction/quality_probe_summary.csv
```

Build a canonical patch manifest from the embedding index:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/01_build_patch_manifest.py \
  --input outputs/embeddings/metadata/embeddings_index.parquet \
  --patch-id-col sample_id \
  --slide-id-col sample_id \
  --slide-id-strip-regex '-[^-]+$' \
  --dataset-value conic_liz \
  --split-value all \
  --extra-cols conic_index image_path mask_path class_label_path resolved_image_path embedding_row_offset embedding_format embedding_dim \
  --output outputs/conic_liz/failure_prediction/patch_manifest.parquet
```

Attach evaluation targets:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/02_attach_eval_targets.py \
  --manifest outputs/conic_liz/failure_prediction/patch_manifest.parquet \
  --eval-dir outputs/conic_liz \
  --output outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet
```

Train the failure predictor. This aggregates `instance_pq` to one row per patch by taking the median across models, derives a binary failure label from the outer-train median, tunes logistic regression, random forest, and SVM with grouped CV, then refits the best family for inference:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/03_train_quality_probe.py \
  --input outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet \
  --output-dir outputs/conic_liz/failure_prediction/quality_probe \
  --problem-type classification \
  --feature-mode embedding_only \
  --target-col instance_pq \
  --target-aggregation median \
  --group-col slide_id \
  --classifier-families logistic_regression random_forest svm \
  --optuna-trials 40 \
  --cv-folds 5 \
  --require-complete-model-coverage \
  --require-existing-embeddings
```

Run inference on unlabeled patches:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/04_infer_quality_probe.py \
  --input outputs/conic_liz/failure_prediction/patch_manifest.parquet \
  --artifact-dir outputs/conic_liz/failure_prediction/quality_probe \
  --output outputs/conic_liz/failure_prediction/quality_probe/inference_predictions.parquet
```

Summarize the saved artifacts:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/05_summarize_quality_probe.py \
  --artifact-dirs outputs/conic_liz/failure_prediction/quality_probe \
  --inference-tables outputs/conic_liz/failure_prediction/quality_probe/inference_predictions.parquet \
  --output outputs/conic_liz/failure_prediction/quality_probe_summary.csv
```

## Named Slurm runs

The Slurm wrapper can run the full workflow in one job. The cleanest way to keep multiple experiments separate is to set `EXPERIMENT_NAME`.

Default full run:

```bash
sbatch scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

Named run with derived outputs under `outputs/conic_liz/failure_prediction/experiments/<name>/`:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=median_pq_embed_only scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

Named run with a different training setup:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=logistic_instance_pq_embed_plus_meta,TRAIN_PROBLEM_TYPE=classification,TRAIN_FEATURE_MODE=embedding_plus_metadata,TRAIN_TARGET_COL=instance_pq,TRAIN_AUTO_METADATA_COLS=1 scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

Named run with a smaller Optuna budget for debugging:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=median_pq_debug,TRAIN_OPTUNA_TRIALS=5,TRAIN_CV_FOLDS=3 scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

Run only training, inference, and summary on an already prepared joined manifest:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=rerun_train_only,RUN_BUILD_MANIFEST=0,RUN_ATTACH_TARGETS=0,RUN_TRAIN=1,RUN_INFER=1,RUN_SUMMARY=1,TRAIN_INPUT=outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

Run only the summary step on an existing experiment:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=median_pq_classifier,RUN_BUILD_MANIFEST=0,RUN_ATTACH_TARGETS=0,RUN_TRAIN=0,RUN_INFER=0,RUN_SUMMARY=1,TRAIN_OUTPUT_DIR=outputs/conic_liz/failure_prediction/experiments/median_pq_classifier/quality_probe,INFER_OUTPUT=outputs/conic_liz/failure_prediction/experiments/median_pq_classifier/quality_probe/inference_predictions.parquet,SUMMARY_OUTPUT=outputs/conic_liz/failure_prediction/experiments/median_pq_classifier/quality_probe_summary.csv scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

You can still override individual paths directly. The wrapper derives these from `EXPERIMENT_NAME` unless you override them:
- `PATCH_MANIFEST_OUTPUT`
- `JOINED_MANIFEST_OUTPUT`
- `TRAIN_OUTPUT_DIR`
- `INFER_OUTPUT`
- `SUMMARY_OUTPUT`

## Artifact layout

`03_train_quality_probe.py` writes into `--output-dir`:
- `aggregated_patch_targets.parquet`
- `best_params.json`
- `candidate_family_metrics.json`
- `config.json`
- `metrics.json`
- `predictions.csv`
- `study_trials.csv`
- `predictor.pkl`
- `scaler.pkl`
- `plots/`

`04_infer_quality_probe.py` writes a clean prediction table:
- regression artifacts from older runs still load
- classification runs write `predicted_failure_label`, `predicted_failure_probability`, `predicted_quality_label`, and `predicted_quality_probability`

## Notes

- Grouped splitting is mandatory; `slide_id` is the default group column.
- The current classifier search covers logistic regression, random forest, and SVM.
- The training plots are CV/learning-curve diagnostics for these sklearn models; they are not neural-network epoch logs.
- If your manifest or evaluation schema differs from the defaults, use the CLI column-mapping flags in `01` and `02`.
- `embedding_only` and `embedding_plus_metadata` require readable `embedding_path` values.
- In the Slurm wrapper, `EXPERIMENT_NAME` is the easiest way to keep separate runs from overwriting each other.
