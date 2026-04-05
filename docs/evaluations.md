# Raw Evaluation

This file is the evaluation-specific source of truth for MoNuSAC benchmarking helpers in this repo.

Current scope:

- Pixel-level foreground evaluation logic in `scripts/benchmarking/monusac_pixel_confusion.py`
- Instance-level matching and panoptic-style metrics in `scripts/benchmarking/monusac_instance_comparison.py`
- Evaluation assumptions for tiled MoNuSAC masks and standardized inference outputs

## Scope

- These utilities evaluate one ground-truth mask and one predicted mask at a time.
- They do not run model inference.
- They do not write aggregate reports or CSV outputs by themselves.
- They assume mask-loading and dataset path context from `docs/monu_context.md` and `docs/inference.md`.

## Inputs and Shared Assumptions

- Ground-truth and predicted masks must have identical shapes.
- Masks are treated as 2D instance-label arrays.
- Background label is `0`.
- Any positive integer label is treated as foreground.
- Predicted label values do not need to match GT label values numerically.
- For tiled data, the expected prediction path preserves the source tile's relative path beneath:
  - `inference/benchmarking/monusac/<model_name>/`
- Typical relative tile path example:
  - `all_merged/<unique_id>/00000x_00000y_image.png`

## Predicted Mask Reading

`scripts/benchmarking/monusac_pixel_confusion.py` contains the mask-loading helper used as the path contract for evaluation.

Important behavior:

- `_default_inference_root()` resolves to `inference/benchmarking/monusac/`
- `read_predicted_masks(...)` reads one tile's predicted mask for one or more model directories
- Default model names are:
  - `cellpose_sam`
  - `cellsam`
  - `cellvit_sam`
  - `stardist`
- `relative_tile_path` must be relative, not absolute
- Each loaded predicted mask must be 2D
- Loaded arrays are cast to `uint16`

## Pixel-Level Evaluation

`scripts/benchmarking/monusac_pixel_confusion.py` evaluates foreground-vs-background agreement only.

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

- Pixel-level `f1` and `dice` are identical in this implementation
- Label identity does not matter; only foreground occupancy matters
- A split or merged instance can still score well here if the foreground pixels overlap
- `safe_divide(...)` returns `0.0` when the denominator is not positive

Edge-case behavior documented by the script's synthetic tests:

- GT has objects and prediction is empty:
  - precision `0.0`
  - recall `0.0`
  - f1/dice `0.0`
- GT is empty and prediction has objects:
  - precision `0.0`
  - recall `0.0`
  - f1/dice `0.0`
- Both masks are empty / all background:
  - all pixels are true negatives
  - precision `0.0`
  - recall `0.0`
  - f1/dice `0.0`

## Instance-Level Evaluation

`scripts/benchmarking/monusac_instance_comparison.py` evaluates object matching quality using pairwise IoU plus global one-to-one assignment.

Core logic:

- `extract_instance_labels(mask)` collects sorted unique non-zero labels
- `compute_pair_iou(...)` computes IoU for one GT object and one predicted object
- `compute_iou_matrix(...)` builds the full GT-by-prediction IoU matrix
- `match_instances(...)` performs one-to-one assignment with the Hungarian algorithm
- Matching uses cost `1 - IoU`, so maximizing total IoU becomes a minimum-cost assignment problem
- After assignment, any pair with `iou < threshold` is discarded
- Default instance-match threshold is `0.5`

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

- Instance metrics are sensitive to splits and merges in a way pixel metrics are not
- Numeric label identity still does not matter; overlap after matching is what matters
- If there are no matches, `sq`, `rq`, and `pq` safely fall back to `0.0`

## Practical Use In This Repo

- Use `docs/monu_context.md` for MoNuSAC export, rescale, tiling, and manifest context
- Use `docs/inference.md` for model-output directory structure and prediction path expectations
- Use this file when the question is about evaluation semantics, matching behavior, metric formulas, or how to interpret `monusac_pixel_confusion.py` and `monusac_instance_comparison.py`
