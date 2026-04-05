from __future__ import annotations

"""Unified pixel-level and instance-level evaluation utilities for segmentation masks.

This module combines the verified evaluation semantics documented in
`docs/evaluations.md` with the implementation patterns from:

- `scripts/benchmarking/monusac_pixel_confusion.py`
- `scripts/benchmarking/monusac_instance_comparison.py`

Design notes
------------
- Pixel metrics treat any positive label as foreground (`mask > 0`).
- Instance metrics treat masks as 2D labeled-instance arrays with background `0`.
- Instance IDs do not need to match numerically between GT and predictions.
- One-to-one instance matching is performed globally with the Hungarian algorithm.
- Assigned pairs below the IoU threshold are discarded.

This module intentionally does not include file loading, folder traversal,
or CSV export. It focuses only on in-memory evaluation logic.
"""

from pprint import pprint
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment


class PixelConfusion(TypedDict):
    """Foreground-vs-background confusion counts."""

    tp: int
    tn: int
    fp: int
    fn: int


class PixelMetrics(PixelConfusion):
    """Pixel-level counts plus overlap metrics."""

    precision: float
    recall: float
    f1: float
    dice: float


class MatchedPair(TypedDict):
    """One accepted GT/prediction instance match."""

    gt_label: int
    pred_label: int
    iou: float


class MatchResult(TypedDict):
    """Instance matching summary."""

    matched_pairs: list[MatchedPair]
    unmatched_gt_labels: list[int]
    unmatched_pred_labels: list[int]


class InstanceMetrics(TypedDict):
    """Panoptic-style instance metrics plus match bookkeeping."""

    tp: int
    fp: int
    fn: int
    object_precision: float
    object_recall: float
    rq: float
    sq: float
    pq: float
    matched_pairs: list[MatchedPair]
    unmatched_gt_labels: list[int]
    unmatched_pred_labels: list[int]


class SegmentationEvaluation(TypedDict):
    """Top-level nested evaluation result."""

    pixel_metrics: PixelMetrics
    instance_metrics: InstanceMetrics


ArrayLike = npt.ArrayLike
IntArray = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _as_2d_array(mask: ArrayLike, name: str) -> npt.NDArray[Any]:
    """Convert an input mask to a NumPy array and ensure it is 2D."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    return array


def _normalize_instance_mask(mask: ArrayLike, name: str) -> IntArray:
    """Return a validated 2D instance mask as `int64`.

    The documented instance-mask contract is:
    - background label is `0`
    - foreground instances use positive integer labels

    This helper accepts integer, boolean, or integer-valued floating inputs and
    rejects negative labels or non-integer floating labels.
    """
    array = _as_2d_array(mask, name)

    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.int64, copy=False)

    if np.issubdtype(array.dtype, np.integer):
        normalized = array.astype(np.int64, copy=False)
    elif np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain integer-valued labels.")
        normalized = array.astype(np.int64)
    else:
        raise ValueError(
            f"{name} must contain numeric labels, got dtype {array.dtype}"
        )

    if np.any(normalized < 0):
        raise ValueError(f"{name} must not contain negative labels.")

    return normalized


def _validate_binary_mask(mask: ArrayLike, name: str) -> UInt8Array:
    """Return a validated binary mask containing only 0/1 values."""
    array = _as_2d_array(mask, name)
    unique_values = np.unique(array)

    if not np.all(np.isin(unique_values, (0, 1))):
        raise ValueError(
            f"{name} must contain only binary values 0 or 1, got {unique_values.tolist()}"
        )

    return array.astype(np.uint8, copy=False)


def _sum_matched_pair_ious(matched_pairs: list[MatchedPair]) -> float:
    """Return the total IoU across all accepted instance matches."""
    return float(sum(pair["iou"] for pair in matched_pairs))


def validate_same_shape(mask1: ArrayLike, mask2: ArrayLike) -> None:
    """Raise `ValueError` when two masks do not have identical shapes.

    Parameters
    ----------
    mask1:
        First mask-like input.
    mask2:
        Second mask-like input.

    Raises
    ------
    ValueError
        If the two inputs do not share the same shape.
    """
    shape1 = np.asarray(mask1).shape
    shape2 = np.asarray(mask2).shape

    if shape1 != shape2:
        raise ValueError(
            "Masks must have identical shapes: "
            f"got mask1_shape={shape1}, mask2_shape={shape2}"
        )


def binarize_mask(mask: ArrayLike) -> UInt8Array:
    """Convert an instance-labeled mask into a binary foreground mask.

    Any value strictly greater than zero is treated as foreground.
    """
    instance_mask = _normalize_instance_mask(mask, "mask")
    return (instance_mask > 0).astype(np.uint8)


def compute_pixel_confusion(gt_binary: ArrayLike, pred_binary: ArrayLike) -> PixelConfusion:
    """Compute TP, TN, FP, and FN counts from two binary masks."""
    validate_same_shape(gt_binary, pred_binary)
    gt_array = _validate_binary_mask(gt_binary, "gt_binary")
    pred_array = _validate_binary_mask(pred_binary, "pred_binary")

    tp = int(np.count_nonzero((gt_array == 1) & (pred_array == 1)))
    tn = int(np.count_nonzero((gt_array == 0) & (pred_array == 0)))
    fp = int(np.count_nonzero((gt_array == 0) & (pred_array == 1)))
    fn = int(np.count_nonzero((gt_array == 1) & (pred_array == 0)))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers.

    Following the documented evaluation behavior, the `default` value is
    returned when the denominator is not strictly positive.
    """
    if denominator > 0:
        return float(numerator / denominator)
    return float(default)


def compute_pixel_metrics(gt_mask: ArrayLike, pred_mask: ArrayLike) -> PixelMetrics:
    """Compute binary pixel-level confusion counts and overlap metrics.

    Both masks are binarized using `mask > 0`, which means numeric label identity
    is ignored at the pixel level and only foreground occupancy matters.
    """
    validate_same_shape(gt_mask, pred_mask)
    gt_binary = binarize_mask(gt_mask)
    pred_binary = binarize_mask(pred_mask)
    confusion = compute_pixel_confusion(gt_binary, pred_binary)

    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
    }


def extract_instance_labels(mask: ArrayLike) -> IntArray:
    """Return sorted unique non-zero instance labels from a mask."""
    instance_mask = _normalize_instance_mask(mask, "mask")
    labels = np.unique(instance_mask)
    return labels[labels != 0].astype(np.int64, copy=False)


def compute_pair_iou(gt_instance: ArrayLike, pred_instance: ArrayLike) -> float:
    """Compute IoU between one GT instance mask and one predicted instance mask."""
    validate_same_shape(gt_instance, pred_instance)
    gt_array = _as_2d_array(gt_instance, "gt_instance").astype(bool, copy=False)
    pred_array = _as_2d_array(pred_instance, "pred_instance").astype(bool, copy=False)

    intersection = int(np.count_nonzero(gt_array & pred_array))
    union = int(np.count_nonzero(gt_array | pred_array))

    return safe_divide(intersection, union)


def compute_iou_matrix(
    gt_mask: ArrayLike,
    pred_mask: ArrayLike,
) -> tuple[FloatArray, IntArray, IntArray]:
    """Compute pairwise IoU for every GT and predicted instance.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        `(iou_matrix, gt_labels, pred_labels)` where:
        - `iou_matrix` has shape `(num_gt_instances, num_pred_instances)`
        - `gt_labels` are the row labels
        - `pred_labels` are the column labels
    """
    validate_same_shape(gt_mask, pred_mask)
    gt_array = _normalize_instance_mask(gt_mask, "gt_mask")
    pred_array = _normalize_instance_mask(pred_mask, "pred_mask")

    gt_labels = extract_instance_labels(gt_array)
    pred_labels = extract_instance_labels(pred_array)
    iou_matrix = np.zeros((len(gt_labels), len(pred_labels)), dtype=np.float64)

    for gt_index, gt_label in enumerate(gt_labels):
        gt_instance = gt_array == gt_label
        for pred_index, pred_label in enumerate(pred_labels):
            pred_instance = pred_array == pred_label
            iou_matrix[gt_index, pred_index] = compute_pair_iou(
                gt_instance,
                pred_instance,
            )

    return iou_matrix, gt_labels, pred_labels


def match_instances(
    iou_matrix: ArrayLike,
    gt_labels: ArrayLike,
    pred_labels: ArrayLike,
    threshold: float = 0.5,
) -> MatchResult:
    """Perform global one-to-one instance matching from a pairwise IoU matrix.

    Matching is solved with the Hungarian algorithm on cost `1 - IoU`, which
    maximizes the total IoU across all assignments. Assigned pairs with IoU
    below `threshold` are discarded.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be between 0.0 and 1.0 inclusive, got {threshold}"
        )

    gt_label_array = np.asarray(gt_labels, dtype=np.int64)
    pred_label_array = np.asarray(pred_labels, dtype=np.int64)
    iou_array = np.asarray(iou_matrix, dtype=np.float64)

    expected_shape = (len(gt_label_array), len(pred_label_array))
    if iou_array.size == 0 and iou_array.ndim == 1:
        iou_array = iou_array.reshape(expected_shape)

    if iou_array.ndim != 2:
        raise ValueError(
            "iou_matrix must be 2D with shape (num_gt, num_pred), "
            f"got shape {iou_array.shape}"
        )

    if iou_array.shape != expected_shape:
        raise ValueError(
            "iou_matrix shape must match gt_labels and pred_labels: "
            f"expected {expected_shape}, got {iou_array.shape}"
        )

    if len(gt_label_array) == 0 or len(pred_label_array) == 0:
        return {
            "matched_pairs": [],
            "unmatched_gt_labels": [int(label) for label in gt_label_array.tolist()],
            "unmatched_pred_labels": [int(label) for label in pred_label_array.tolist()],
        }

    # Minimize (1 - IoU) to maximize total IoU under a one-to-one constraint.
    cost_matrix = 1.0 - iou_array
    assigned_gt_indices, assigned_pred_indices = linear_sum_assignment(cost_matrix)

    matched_pairs: list[MatchedPair] = []
    matched_gt_labels: set[int] = set()
    matched_pred_labels: set[int] = set()

    for gt_index, pred_index in zip(assigned_gt_indices, assigned_pred_indices):
        pair_iou = float(iou_array[gt_index, pred_index])
        if pair_iou < threshold:
            continue

        gt_label = int(gt_label_array[gt_index])
        pred_label = int(pred_label_array[pred_index])

        matched_pairs.append(
            {
                "gt_label": gt_label,
                "pred_label": pred_label,
                "iou": pair_iou,
            }
        )
        matched_gt_labels.add(gt_label)
        matched_pred_labels.add(pred_label)

    return {
        "matched_pairs": matched_pairs,
        "unmatched_gt_labels": [
            int(label)
            for label in gt_label_array.tolist()
            if int(label) not in matched_gt_labels
        ],
        "unmatched_pred_labels": [
            int(label)
            for label in pred_label_array.tolist()
            if int(label) not in matched_pred_labels
        ],
    }


def compute_instance_metrics(
    gt_mask: ArrayLike,
    pred_mask: ArrayLike,
    threshold: float = 0.5,
) -> InstanceMetrics:
    """Compute panoptic-style instance metrics from two labeled masks.

    Definitions
    -----------
    - `tp`: number of accepted matched pairs
    - `fp`: number of unmatched predicted instances
    - `fn`: number of unmatched GT instances
    - `object_precision = tp / (tp + fp)`
    - `object_recall = tp / (tp + fn)`
    - `rq = tp / (tp + 0.5 * fp + 0.5 * fn)`
    - `sq = sum(matched IoUs) / tp`
    - `pq = sum(matched IoUs) / (tp + 0.5 * fp + 0.5 * fn)`
    """
    iou_matrix, gt_labels, pred_labels = compute_iou_matrix(gt_mask, pred_mask)
    match_result = match_instances(
        iou_matrix=iou_matrix,
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        threshold=threshold,
    )

    matched_pairs = list(match_result["matched_pairs"])
    unmatched_gt_labels = list(match_result["unmatched_gt_labels"])
    unmatched_pred_labels = list(match_result["unmatched_pred_labels"])

    tp = len(matched_pairs)
    fp = len(unmatched_pred_labels)
    fn = len(unmatched_gt_labels)
    matched_iou_sum = _sum_matched_pair_ious(matched_pairs)

    object_precision = safe_divide(tp, tp + fp)
    object_recall = safe_divide(tp, tp + fn)
    panoptic_denominator = tp + (0.5 * fp) + (0.5 * fn)
    rq = safe_divide(tp, panoptic_denominator)
    sq = safe_divide(matched_iou_sum, tp)
    pq = safe_divide(matched_iou_sum, panoptic_denominator)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "object_precision": object_precision,
        "object_recall": object_recall,
        "rq": rq,
        "sq": sq,
        "pq": pq,
        "matched_pairs": matched_pairs,
        "unmatched_gt_labels": [int(label) for label in unmatched_gt_labels],
        "unmatched_pred_labels": [int(label) for label in unmatched_pred_labels],
    }


def evaluate_segmentation(
    gt_mask: ArrayLike,
    pred_mask: ArrayLike,
    threshold: float = 0.5,
) -> SegmentationEvaluation:
    """Evaluate a predicted instance mask at both pixel and instance levels."""
    validate_same_shape(gt_mask, pred_mask)

    return {
        "pixel_metrics": compute_pixel_metrics(gt_mask, pred_mask),
        "instance_metrics": compute_instance_metrics(
            gt_mask,
            pred_mask,
            threshold=threshold,
        ),
    }


def _synthetic_example() -> tuple[IntArray, IntArray]:
    """Return a small example with two correct matches, one miss, and one extra object.

    Expected behavior with `threshold=0.5`:
    - Pixel metrics should show strong but imperfect overlap.
    - Instance metrics should produce:
      - 2 matched pairs
      - 1 missed GT instance
      - 1 extra predicted instance
      - `sq == 1.0` because both accepted matches are exact
    """
    gt_mask = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 2],
            [0, 0, 0, 2],
            [0, 3, 3, 0],
        ],
        dtype=np.uint16,
    )
    pred_mask = np.array(
        [
            [5, 5, 0, 0],
            [5, 5, 0, 7],
            [0, 0, 0, 7],
            [0, 0, 0, 9],
        ],
        dtype=np.uint16,
    )
    return gt_mask.astype(np.int64), pred_mask.astype(np.int64)


if __name__ == "__main__":
    gt_example, pred_example = _synthetic_example()
    result = evaluate_segmentation(gt_example, pred_example, threshold=0.5)

    print("Synthetic GT mask:")
    print(gt_example)
    print("\nSynthetic prediction mask:")
    print(pred_example)
    print("\nPretty-printed evaluation result:")
    pprint(result, sort_dicts=False)
