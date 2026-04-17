# Raw Evaluation

This file is the evaluation-specific source of truth for MoNuSAC benchmarking in this repo.

Current scope:

- The standard batch workflow in `scripts/benchmarking/monusac_segmentation_evaluation.py`
- Pixel-level foreground evaluation semantics from `scripts/benchmarking/monusac_pixel_confusion.py`
- Instance-level matching and panoptic-style metrics from `scripts/benchmarking/monusac_instance_comparison.py`
- Evaluation assumptions for tiled MoNuSAC masks and standardized inference outputs

## Scope

- `monusac_segmentation_evaluation.py` is the standard evaluation entrypoint for MoNuSAC inference outputs in this repo.
- The lower-level helpers still define the metric semantics reused by the unified workflow.
- Evaluation starts after `scripts/02-inference/` has written predicted masks and each model's `predictions.csv`.
- Evaluation does not run model inference.
- Evaluation does not redo dataset export, rescaling, or tiling.

## Unified Evaluation Workflow

Standard pipeline position:

- `docs/monu_context.md` defines the export, rescale, and tiling contracts.
- `scripts/02-inference/` runs model inference and writes one model directory per method.
- Each model directory contains predicted masks plus `predictions.csv`.
- `scripts/benchmarking/monusac_segmentation_evaluation.py` reads those manifests, resolves GT and prediction paths, loads masks, computes pixel and instance metrics, and writes evaluation CSVs.

Main entrypoints in `monusac_segmentation_evaluation.py`:

- `evaluate_segmentation(...)`
  Computes both pixel-level and instance-level metrics for one in-memory GT and prediction mask pair.
- `evaluate_segmentation_files(...)`
  Loads one GT mask file and one prediction mask file, then runs the standard evaluation logic.
- `evaluate_folder(...)`
  Generic folder-level evaluator that pairs files by filename stem. Use this for ad hoc non-manifest directories, not as the default MoNuSAC workflow.
- `evaluate_monusac_models(...)`
  Repo-standard batch evaluator for model outputs under `inference/benchmarking/monusac/`.

## Inputs and Shared Assumptions

- Ground-truth and predicted masks must have identical shapes.
- Masks are treated as 2D instance-label arrays.
- Background label is `0`.
- Any positive integer label is treated as foreground.
- Predicted label values do not need to match GT label values numerically.
- Default instance-match threshold is `0.5`.
- Standard mask files are PNG or TIFF.

Manifest-driven MoNuSAC evaluation expects each model directory to contain `predictions.csv` with:

- Required columns:
  - `predicted_mask_path`
  - `mask_path`
- Needed when `mask_path` is relative:
  - `source_image_path`
  - `relative_image_path` or `image_path`
- Helpful metadata used in outputs when available:
  - `patch_id`
  - `patient_id` or `patient`
  - `predicted_mask_relative_path`
  - `predicted_mask_name`

Important path assumption:

- For the standard MoNuSAC workflow, `predictions.csv` is the path source of truth.
- The evaluator does not reconstruct GT or prediction paths from directory naming assumptions when the manifest already records them.

## Predicted Mask Reading

`monusac_segmentation_evaluation.py` provides the standard mask loader for this workflow.

Important behavior:

- `load_mask(...)` accepts `.png`, `.tif`, and `.tiff`.
- Loaded masks must resolve to a 2D array.
- The loader preserves the on-disk numeric dtype when practical so instance labels such as `uint16` survive the round trip.
- Extra dimensions are accepted only when reduction is unambiguous:
  - singleton dimensions are squeezed
  - replicated grayscale RGB or RGBA channels are reduced to one channel
- Ambiguous multi-channel masks are rejected.
- Missing files, unsupported suffixes, empty files, and malformed masks fail with explicit errors.

Important implication:

- The standard MoNuSAC evaluator no longer depends on reconstructing prediction paths with `read_predicted_masks(...)` from `monusac_pixel_confusion.py`.
- Manifest-recorded paths are preferred because they preserve the exact files used by inference.

## Manifest-Driven Model Evaluation

`evaluate_monusac_models(...)` is the batch workflow future contributors should use for MoNuSAC benchmarking.

Path resolution details:

- Default prediction root is `inference/benchmarking/monusac/`.
- The implementation also tolerates an existing `inference/benchamking/monusac/` directory for backward compatibility.
- If `predicted_mask_path` is absolute, it is used directly.
- If `predicted_mask_path` is relative, it is resolved relative to the manifest directory.
- If `mask_path` is absolute, it is used directly.
- If `mask_path` is relative and already points to a repo-relative file such as `data/conic_lizard/...`, the evaluator uses that recorded repo-relative path directly.
- Otherwise, if `mask_path` is relative, the evaluator derives the tiled input root by removing `relative_image_path` or `image_path` from the recorded `source_image_path`.
- This avoids hardcoded dataset roots and keeps evaluation tied to the exact inference manifest row.

Row identification details:

- `image_id` is chosen from the first available field in this order:
  - `patch_id`
  - `relative_image_path`
  - `image_path`
  - `predicted_mask_relative_path`
  - `predicted_mask_name`
  - `predicted_mask_path`
- If none are available, the evaluator falls back to `row_<line_number>`.
- `patient_id` is read from `patient_id` or `patient` when present.

Outputs:

- One per-model CSV is written by default to `inference/benchmarking/monusac/_evaluation/<model_name>_evaluation.csv`.
- An optional combined CSV can also be written, default name `all_models_evaluation.csv`.
- Output rows include:
  - model and patient identifiers
  - GT and prediction paths
  - GT and prediction relative paths when available
  - flattened pixel metrics
  - flattened instance metrics
  - `status`
  - `error_message`

Status behavior:

- Manifest-driven MoNuSAC rows are marked `ok` or `error`.
- Errors are recorded per row so one bad file does not abort the full model evaluation.
- Console summaries report manifest row count, successful evaluations, evaluation errors, manifest path, and output CSV path.
- This workflow evaluates successful inference rows from `predictions.csv`; it does not merge in `failed.csv`.

## Pixel-Level Evaluation

`monusac_segmentation_evaluation.py` uses the same foreground-vs-background semantics as `monusac_pixel_confusion.py`.

Core logic:

- `binarize_mask(mask)` converts any instance mask to binary using `mask > 0`
- Instance IDs are ignored after binarization
- `compute_pixel_confusion(...)` returns:
  - `tp`
  - `tn`
  - `fp`
  - `fn`
- `compute_pixel_metrics(...)` derives:
  - `precision = tp / (tp + fp)`
  - `recall = tp / (tp + fn)`
  - `f1 = 2 * tp / (2 * tp + fp + fn)`
  - `dice = 2 * tp / (2 * tp + fp + fn)`

Important implications:

- Pixel-level `f1` and `dice` are identical in this implementation.
- Label identity does not matter. Only foreground occupancy matters.
- A split or merged instance can still score well here if the foreground pixels overlap.
- `safe_divide(...)` returns `0.0` when the denominator is not positive.

Edge-case behavior documented by the synthetic tests and example logic:

- GT has objects and prediction is empty:
  - precision `0.0`
  - recall `0.0`
  - f1/dice `0.0`
- GT is empty and prediction has objects:
  - precision `0.0`
  - recall `0.0`
  - f1/dice `0.0`
- Both masks are empty or all background:
  - all pixels are true negatives
  - precision `0.0`
  - recall `0.0`
  - f1/dice `0.0`

## Instance-Level Evaluation

`monusac_segmentation_evaluation.py` uses the same object-matching semantics as `monusac_instance_comparison.py`.

Core logic:

- `extract_instance_labels(mask)` collects sorted unique non-zero labels
- `compute_pair_iou(...)` computes IoU for one GT object and one predicted object
- `compute_iou_matrix(...)` builds the full GT-by-prediction IoU matrix
- `match_instances(...)` performs one-to-one assignment with the Hungarian algorithm
- Matching uses cost `1 - IoU`, so maximizing total IoU becomes a minimum-cost assignment problem
- After assignment, any pair with `iou < threshold` is discarded

Why matching is implemented this way:

- Matching is global, not greedy
- A greedy highest-IoU-first strategy can produce a worse overall assignment
- One predicted object can match at most one GT object, and vice versa

Returned match structure:

- `matched_pairs`
  - each entry contains `gt_label`, `pred_label`, and `iou`
- `unmatched_gt_labels`
- `unmatched_pred_labels`

## Instance Metrics

`compute_instance_metrics(...)` turns the matching result into panoptic-style instance metrics.

Counts:

- `tp = number of matched pairs`
- `fp = number of unmatched predicted labels`
- `fn = number of unmatched GT labels`

Derived metrics:

- `object_precision = tp / (tp + fp)`
- `object_recall = tp / (tp + fn)`
- `rq = tp / (tp + 0.5 * fp + 0.5 * fn)`
- `sq = sum(matched IoUs) / tp`
- `pq = sum(matched IoUs) / (tp + 0.5 * fp + 0.5 * fn)`

Interpretation:

- `object_precision` penalizes extra predicted instances
- `object_recall` penalizes missed GT instances
- `rq` measures detection quality after one-to-one matching
- `sq` measures average overlap quality of matched objects only
- `pq` combines detection quality and overlap quality into one score

Important implications:

- Instance metrics are sensitive to splits and merges in a way pixel metrics are not.
- Numeric label identity still does not matter. Overlap after matching is what matters.
- If there are no matches, `sq`, `rq`, and `pq` safely fall back to `0.0`.

## Practical Use In This Repo

- Use `evaluate_monusac_models(...)` or the module CLI for the standard MoNuSAC batch evaluation workflow.
- Use `evaluate_segmentation_files(...)` for one-off validation of a GT and prediction mask pair.
- Use `evaluate_folder(...)` only when you have directories that should be matched by filename stem and there is no inference manifest to trust.
- Use `docs/inference.md` when the question is about how `predictions.csv` is produced.
- Use `docs/monu_context.md` when the question is about MoNuSAC export, rescaling, tiling, or manifest provenance.
