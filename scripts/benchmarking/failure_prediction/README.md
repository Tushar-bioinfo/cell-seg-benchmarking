# Quality Probe Workflow

This directory contains a simple failure-prediction workflow for patch-level segmentation quality.

Scripts:
- `01_build_patch_manifest.py`: build a canonical patch manifest from an embeddings index table
- `02_attach_eval_targets.py`: join the patch manifest with evaluation CSVs to create training targets
- `03_train_quality_probe.py`: train a simple regression or classification baseline
- `04_infer_quality_probe.py`: run the trained probe on new patches without ground truth
- `05_summarize_quality_probe.py`: flatten artifact metrics/config plus optional inference tables into one report

## Typical local flow

Build a canonical patch manifest:

```bash
`01_build_patch_manifest.py` accepts the following arguments:
- `--input`: path to the embeddings index parquet file
- `--output`: path to write the patch manifest parquet file
- `--patch-id-col`: column to use as patch ID (e.g. `sample_id`)
- `--slide-id-col`: column to use as slide ID (e.g. `sample_id`)
- `--dataset-value`: dataset label to assign (e.g. `conic_liz`)
- `--split-value`: split label to assign (e.g. `all`)
- `--extra-cols`: additional columns to carry through (e.g. `sample_id conic_index image_path mask_path class_label_path resolved_image_path embedding_row_offset embedding_format embedding_dim`)
```

Attach evaluation targets:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/02_attach_eval_targets.py \
  --manifest outputs/conic_liz/failure_prediction/patch_manifest.parquet \
  --eval-dir outputs/conic_liz \
  --output outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet
```

Train a baseline quality probe:

```bash
pixi run -e default python scripts/benchmarking/failure_prediction/03_train_quality_probe.py \
  --input outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet \
  --output-dir outputs/conic_liz/failure_prediction/quality_probe \
  --problem-type regression \
  --feature-mode embedding_only \
  --target-col instance_pq \
  --group-col slide_id
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
sbatch --export=ALL,EXPERIMENT_NAME=ridge_instance_pq_embed_only scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

Named run with a different training setup:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=logistic_instance_pq_embed_plus_meta,TRAIN_PROBLEM_TYPE=classification,TRAIN_FEATURE_MODE=embedding_plus_metadata,TRAIN_TARGET_COL=instance_pq,TRAIN_AUTO_METADATA_COLS=1 scripts/benchmarking/failure_prediction/run_quality_probe.slurm
```

You can still override individual paths directly. The wrapper derives these from `EXPERIMENT_NAME` unless you override them:
- `PATCH_MANIFEST_OUTPUT`
- `JOINED_MANIFEST_OUTPUT`
- `TRAIN_OUTPUT_DIR`
- `INFER_OUTPUT`
- `SUMMARY_OUTPUT`

## Artifact layout

`03_train_quality_probe.py` writes into `--output-dir`:
- `config.json`
- `metrics.json`
- `predictions.csv`
- `predictor.pkl`
- `scaler.pkl`

`04_infer_quality_probe.py` writes a clean prediction table:
- regression: `predicted_quality_score`
- classification: `predicted_quality_label`, `predicted_quality_probability`, `predicted_failure_probability`

## Notes

- The training script only supports simple baselines right now.
- Grouped splitting is mandatory; `slide_id` is the default group column.
- If your manifest or evaluation schema differs from the defaults, use the CLI column-mapping flags in `01` and `02`.
- `embedding_only` and `embedding_plus_metadata` require readable `embedding_path` values.
- In the Slurm wrapper, `EXPERIMENT_NAME` is the easiest way to keep separate runs from overwriting each other.
