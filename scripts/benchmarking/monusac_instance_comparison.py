from __future__ import annotations

"""Foundation utilities for instance-level IoU comparisons on MoNuSAC masks.

Mask-reading context from `scripts/benchmarking/monusac_pixel_confusion.py`:
- predicted masks are stored as 2D PNG instance masks
- arrays are treated as instance-labeled masks
- label `0` is background

This module provides pairwise IoU matrix construction, one-to-one instance
matching, and instance-level evaluation metrics such as RQ, SQ, and PQ.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


MatchPair = dict[str, int | float]
MatchResult = dict[str, list[MatchPair] | list[int]]
InstanceMetrics = dict[str, int | float | list[MatchPair] | list[int]]


def extract_instance_labels(mask: np.ndarray) -> np.ndarray:
    """Return sorted unique non-zero instance labels from an instance mask.

    Parameters
    ----------
    mask:
        Instance-labeled mask where `0` denotes background.

    Returns
    -------
    np.ndarray
        Sorted unique labels present in the mask, excluding background.
    """
    mask_array = np.asarray(mask)
    labels = np.unique(mask_array)
    return labels[labels != 0]


def compute_pair_iou(gt_instance: np.ndarray, pred_instance: np.ndarray) -> float:
    """Compute IoU between one ground-truth object mask and one predicted object mask.

    Parameters
    ----------
    gt_instance:
        Boolean mask for one ground-truth instance.
    pred_instance:
        Boolean mask for one predicted instance.

    Returns
    -------
    float
        Intersection-over-union between the two boolean masks.

    Raises
    ------
    ValueError
        If the two instance masks do not share the same shape.
    """
    gt_bool = np.asarray(gt_instance, dtype=bool)
    pred_bool = np.asarray(pred_instance, dtype=bool)

    if gt_bool.shape != pred_bool.shape:
        raise ValueError(
            "Ground-truth and prediction instance masks must have identical shapes: "
            f"got gt_shape={gt_bool.shape}, pred_shape={pred_bool.shape}"
        )

    intersection = np.count_nonzero(gt_bool & pred_bool)
    union = np.count_nonzero(gt_bool | pred_bool)

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_iou_matrix(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the pairwise IoU matrix for all GT and predicted instances.

    Parameters
    ----------
    gt_mask:
        Ground-truth instance mask where each non-zero integer denotes one object.
    pred_mask:
        Predicted instance mask where each non-zero integer denotes one object.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - IoU matrix with shape `(num_gt, num_pred)`
        - sorted GT labels excluding background
        - sorted predicted labels excluding background

    Raises
    ------
    ValueError
        If the two masks do not share the same shape.
    """
    gt_array = np.asarray(gt_mask)
    pred_array = np.asarray(pred_mask)

    if gt_array.shape != pred_array.shape:
        raise ValueError(
            "Ground-truth and prediction masks must have identical shapes: "
            f"got gt_shape={gt_array.shape}, pred_shape={pred_array.shape}"
        )

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
    iou_matrix: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    threshold: float = 0.5,
) -> MatchResult:
    """Match GT and predicted instances with global one-to-one assignment.

    Parameters
    ----------
    iou_matrix:
        Pairwise IoU matrix with shape `(num_gt, num_pred)`.
    gt_labels:
        GT instance labels corresponding to the rows of `iou_matrix`.
    pred_labels:
        Predicted instance labels corresponding to the columns of `iou_matrix`.
    threshold:
        Minimum IoU required for an assigned pair to be kept as a match.

    Returns
    -------
    MatchResult
        Dictionary with matched pairs, unmatched GT labels, and unmatched
        predicted labels.

    Raises
    ------
    ValueError
        If `threshold` is outside `[0.0, 1.0]`, if `iou_matrix` is not 2D, or
        if its shape does not agree with `gt_labels` and `pred_labels`.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0.0 and 1.0 inclusive: got {threshold}")

    gt_label_array = np.asarray(gt_labels)
    pred_label_array = np.asarray(pred_labels)
    iou_array = np.asarray(iou_matrix, dtype=np.float64)

    expected_shape = (len(gt_label_array), len(pred_label_array))

    if iou_array.size == 0 and iou_array.ndim == 1:
        iou_array = iou_array.reshape(expected_shape)

    if iou_array.ndim != 2:
        raise ValueError(
            "iou_matrix must be a 2D array with shape (num_gt, num_pred): "
            f"got ndim={iou_array.ndim}, shape={iou_array.shape}"
        )

    if iou_array.shape != expected_shape:
        raise ValueError(
            "iou_matrix shape must match the number of GT and predicted labels: "
            f"expected {expected_shape}, got {iou_array.shape}"
        )

    if len(gt_label_array) == 0 or len(pred_label_array) == 0:
        return {
            "matched_pairs": [],
            "unmatched_gt_labels": [int(label) for label in gt_label_array.tolist()],
            "unmatched_pred_labels": [int(label) for label in pred_label_array.tolist()],
        }

    # Hungarian matching minimizes a cost, so convert IoU to (1 - IoU) so that
    # minimizing cost is equivalent to maximizing total IoU.
    #
    # A global assignment is preferred over greedy matching because grabbing the
    # largest local IoU first can block a better overall one-to-one solution.
    cost_matrix = 1.0 - iou_array
    assigned_gt_indices, assigned_pred_indices = linear_sum_assignment(cost_matrix)

    matched_pairs: list[MatchPair] = []
    for gt_index, pred_index in zip(assigned_gt_indices, assigned_pred_indices):
        pair_iou = float(iou_array[gt_index, pred_index])
        if pair_iou < threshold:
            continue
        matched_pairs.append(
            {
                "gt_label": int(gt_label_array[gt_index]),
                "pred_label": int(pred_label_array[pred_index]),
                "iou": pair_iou,
            }
        )

    matched_gt_labels = {int(pair["gt_label"]) for pair in matched_pairs}
    matched_pred_labels = {int(pair["pred_label"]) for pair in matched_pairs}

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


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return `numerator / denominator`, or `0.0` when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _sum_matched_pair_ious(matched_pairs: list[MatchPair]) -> float:
    """Return the sum of IoUs over all matched GT/prediction pairs."""
    return float(sum(float(pair["iou"]) for pair in matched_pairs))


def compute_instance_metrics(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float = 0.5,
) -> InstanceMetrics:
    """Compute instance-level detection and segmentation metrics from two masks.

    Parameters
    ----------
    gt_mask:
        Ground-truth instance mask where each non-zero integer denotes one object.
    pred_mask:
        Predicted instance mask where each non-zero integer denotes one object.
    threshold:
        Minimum IoU required for an assigned pair to count as a match.

    Returns
    -------
    InstanceMetrics
        Dictionary containing counts, object-level precision/recall, RQ, SQ, PQ,
        matched pairs, and unmatched GT/predicted labels.

    Raises
    ------
    ValueError
        If the masks do not share the same shape, or if `threshold` is outside
        `[0.0, 1.0]`.

    Notes
    -----
    - SQ asks: among the objects that were matched correctly, how good was the
      mask overlap on average?
    - RQ asks: how well did the method find the right objects, penalizing both
      misses and extra predictions?
    - PQ combines both ideas into one score, so a method needs good detection
      and good overlap quality to score well.
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

    object_precision = _safe_divide(tp, tp + fp)
    object_recall = _safe_divide(tp, tp + fn)

    # RQ measures instance detection quality after matching.
    panoptic_denominator = tp + (0.5 * fp) + (0.5 * fn)
    rq = _safe_divide(tp, panoptic_denominator)

    # SQ measures the average overlap quality of only the matched objects.
    sq = _safe_divide(matched_iou_sum, tp)

    # PQ is the combined instance score: find the right objects and segment them well.
    pq = _safe_divide(matched_iou_sum, panoptic_denominator)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "object_precision": float(object_precision),
        "object_recall": float(object_recall),
        "rq": float(rq),
        "sq": float(sq),
        "pq": float(pq),
        "matched_pairs": matched_pairs,
        "unmatched_gt_labels": [int(label) for label in unmatched_gt_labels],
        "unmatched_pred_labels": [int(label) for label in unmatched_pred_labels],
    }


def _assert_matching_result(
    actual_result: MatchResult,
    expected_pairs: list[MatchPair],
    expected_unmatched_gt: list[int],
    expected_unmatched_pred: list[int],
) -> None:
    """Assert that a matching result matches the expected labels and IoUs."""
    actual_pairs = actual_result["matched_pairs"]

    assert len(actual_pairs) == len(expected_pairs)
    for actual_pair, expected_pair in zip(actual_pairs, expected_pairs):
        assert actual_pair["gt_label"] == expected_pair["gt_label"]
        assert actual_pair["pred_label"] == expected_pair["pred_label"]
        assert np.isclose(actual_pair["iou"], expected_pair["iou"])

    assert actual_result["unmatched_gt_labels"] == expected_unmatched_gt
    assert actual_result["unmatched_pred_labels"] == expected_unmatched_pred


def _run_matching_test_from_masks(
    name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    expected_iou_matrix: np.ndarray,
    expected_pairs: list[MatchPair],
    expected_unmatched_gt: list[int],
    expected_unmatched_pred: list[int],
    threshold: float = 0.5,
) -> None:
    """Compute IoU from masks, run matching, print results, and assert them."""
    iou_matrix, gt_labels, pred_labels = compute_iou_matrix(gt_mask, pred_mask)
    result = match_instances(
        iou_matrix=iou_matrix,
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        threshold=threshold,
    )

    print(f"\n=== {name} ===")
    print("gt_labels:", gt_labels)
    print("pred_labels:", pred_labels)
    print("IoU matrix:")
    print(iou_matrix)
    print("matched_pairs:", result["matched_pairs"])
    print("unmatched_gt_labels:", result["unmatched_gt_labels"])
    print("unmatched_pred_labels:", result["unmatched_pred_labels"])

    assert np.array_equal(iou_matrix.shape, expected_iou_matrix.shape)
    assert np.allclose(iou_matrix, expected_iou_matrix)
    _assert_matching_result(
        actual_result=result,
        expected_pairs=expected_pairs,
        expected_unmatched_gt=expected_unmatched_gt,
        expected_unmatched_pred=expected_unmatched_pred,
    )


def _run_matching_test_from_matrix(
    name: str,
    iou_matrix: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    expected_pairs: list[MatchPair],
    expected_unmatched_gt: list[int],
    expected_unmatched_pred: list[int],
    threshold: float = 0.5,
) -> None:
    """Run matching from a precomputed IoU matrix, print results, and assert them."""
    result = match_instances(
        iou_matrix=iou_matrix,
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        threshold=threshold,
    )

    print(f"\n=== {name} ===")
    print("gt_labels:", gt_labels)
    print("pred_labels:", pred_labels)
    print("IoU matrix:")
    print(iou_matrix)
    print("matched_pairs:", result["matched_pairs"])
    print("unmatched_gt_labels:", result["unmatched_gt_labels"])
    print("unmatched_pred_labels:", result["unmatched_pred_labels"])

    _assert_matching_result(
        actual_result=result,
        expected_pairs=expected_pairs,
        expected_unmatched_gt=expected_unmatched_gt,
        expected_unmatched_pred=expected_unmatched_pred,
    )


def _assert_instance_metric_result(
    actual_result: InstanceMetrics,
    expected_result: InstanceMetrics,
) -> None:
    """Assert that the computed instance metrics match the expected values."""
    assert actual_result["tp"] == expected_result["tp"]
    assert actual_result["fp"] == expected_result["fp"]
    assert actual_result["fn"] == expected_result["fn"]
    assert np.isclose(actual_result["object_precision"], expected_result["object_precision"])
    assert np.isclose(actual_result["object_recall"], expected_result["object_recall"])
    assert np.isclose(actual_result["rq"], expected_result["rq"])
    assert np.isclose(actual_result["sq"], expected_result["sq"])
    assert np.isclose(actual_result["pq"], expected_result["pq"])

    _assert_matching_result(
        actual_result={
            "matched_pairs": list(actual_result["matched_pairs"]),
            "unmatched_gt_labels": list(actual_result["unmatched_gt_labels"]),
            "unmatched_pred_labels": list(actual_result["unmatched_pred_labels"]),
        },
        expected_pairs=list(expected_result["matched_pairs"]),
        expected_unmatched_gt=list(expected_result["unmatched_gt_labels"]),
        expected_unmatched_pred=list(expected_result["unmatched_pred_labels"]),
    )


def _run_instance_metric_test(
    name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    expected_result: InstanceMetrics,
    threshold: float = 0.5,
) -> None:
    """Run one instance-metric test, print the result, and assert correctness."""
    gt_labels = extract_instance_labels(gt_mask)
    pred_labels = extract_instance_labels(pred_mask)
    result = compute_instance_metrics(
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        threshold=threshold,
    )

    print(f"\n=== {name} ===")
    print("number of GT objects:", len(gt_labels))
    print("number of predicted objects:", len(pred_labels))
    print("matched pairs:", result["matched_pairs"])
    print("TP, FP, FN:", result["tp"], result["fp"], result["fn"])
    print("object precision:", result["object_precision"])
    print("object recall:", result["object_recall"])
    print("RQ:", result["rq"])
    print("SQ:", result["sq"])
    print("PQ:", result["pq"])

    _assert_instance_metric_result(
        actual_result=result,
        expected_result=expected_result,
    )


if __name__ == "__main__":
    # Test 1: simple one-to-one perfect match.
    perfect_gt = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )
    perfect_pred = np.array(
        [
            [0, 9, 9],
            [0, 9, 9],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )
    _run_matching_test_from_masks(
        name="1. one GT object, one perfect predicted object",
        gt_mask=perfect_gt,
        pred_mask=perfect_pred,
        expected_iou_matrix=np.array([[1.0]], dtype=np.float64),
        expected_pairs=[{"gt_label": 1, "pred_label": 9, "iou": 1.0}],
        expected_unmatched_gt=[],
        expected_unmatched_pred=[],
    )

    # Test 2: one GT object and two predictions. Only the better-overlapping
    # prediction should survive the IoU threshold.
    one_gt_two_pred_gt = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint16,
    )
    one_gt_two_pred_pred = np.array(
        [
            [10, 20, 0],
            [10, 0, 0],
        ],
        dtype=np.uint16,
    )
    _run_matching_test_from_masks(
        name="2. one GT object, two predicted objects",
        gt_mask=one_gt_two_pred_gt,
        pred_mask=one_gt_two_pred_pred,
        expected_iou_matrix=np.array([[0.5, 0.25]], dtype=np.float64),
        expected_pairs=[{"gt_label": 1, "pred_label": 10, "iou": 0.5}],
        expected_unmatched_gt=[],
        expected_unmatched_pred=[20],
    )

    # Test 3: two GT objects and one prediction. Only one GT can be matched in
    # a one-to-one assignment.
    _run_matching_test_from_matrix(
        name="3. two GT objects, one predicted object",
        iou_matrix=np.array([[0.7], [0.4]], dtype=np.float64),
        gt_labels=np.array([1, 2], dtype=np.uint16),
        pred_labels=np.array([9], dtype=np.uint16),
        expected_pairs=[{"gt_label": 1, "pred_label": 9, "iou": 0.7}],
        expected_unmatched_gt=[2],
        expected_unmatched_pred=[],
    )

    # Test 4: global matching is better than greedy here. Greedy would grab
    # IoU 0.90 first and force a poor 0.10 second match, while Hungarian
    # matching chooses 0.80 and 0.85 for the best total IoU.
    _run_matching_test_from_matrix(
        name="4. global assignment preferred over greedy matching",
        iou_matrix=np.array(
            [
                [0.90, 0.80],
                [0.85, 0.10],
            ],
            dtype=np.float64,
        ),
        gt_labels=np.array([1, 2], dtype=np.uint16),
        pred_labels=np.array([10, 20], dtype=np.uint16),
        expected_pairs=[
            {"gt_label": 1, "pred_label": 20, "iou": 0.80},
            {"gt_label": 2, "pred_label": 10, "iou": 0.85},
        ],
        expected_unmatched_gt=[],
        expected_unmatched_pred=[],
    )

    # Test 5: no objects on either side. This also checks that an empty 1D IoU
    # array can be reshaped cleanly when both label lists are empty.
    _run_matching_test_from_matrix(
        name="5. no GT objects and no predicted objects",
        iou_matrix=np.array([], dtype=np.float64),
        gt_labels=np.array([], dtype=np.uint16),
        pred_labels=np.array([], dtype=np.uint16),
        expected_pairs=[],
        expected_unmatched_gt=[],
        expected_unmatched_pred=[],
    )

    # Test 6: GT objects exist but no predictions do.
    _run_matching_test_from_matrix(
        name="6. GT objects exist but no predicted objects",
        iou_matrix=np.array([], dtype=np.float64).reshape((2, 0)),
        gt_labels=np.array([1, 2], dtype=np.uint16),
        pred_labels=np.array([], dtype=np.uint16),
        expected_pairs=[],
        expected_unmatched_gt=[1, 2],
        expected_unmatched_pred=[],
    )

    # Test 7: predicted objects exist but no GT objects do.
    _run_matching_test_from_matrix(
        name="7. predicted objects exist but no GT objects",
        iou_matrix=np.array([], dtype=np.float64).reshape((0, 2)),
        gt_labels=np.array([], dtype=np.uint16),
        pred_labels=np.array([10, 20], dtype=np.uint16),
        expected_pairs=[],
        expected_unmatched_gt=[],
        expected_unmatched_pred=[10, 20],
    )

    background_only = np.zeros((2, 2), dtype=np.uint16)
    gt_only = np.array(
        [
            [1, 1],
            [0, 0],
        ],
        dtype=np.uint16,
    )
    pred_only = np.array(
        [
            [7, 7],
            [0, 0],
        ],
        dtype=np.uint16,
    )
    two_object_gt = np.array(
        [
            [1, 1, 0, 2],
            [1, 1, 0, 2],
        ],
        dtype=np.uint16,
    )
    two_object_pred = np.array(
        [
            [5, 5, 0, 6],
            [5, 0, 0, 6],
        ],
        dtype=np.uint16,
    )

    _run_instance_metric_test(
        name="8. metrics with background-only masks",
        gt_mask=background_only,
        pred_mask=background_only,
        expected_result={
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "object_precision": 0.0,
            "object_recall": 0.0,
            "rq": 0.0,
            "sq": 0.0,
            "pq": 0.0,
            "matched_pairs": [],
            "unmatched_gt_labels": [],
            "unmatched_pred_labels": [],
        },
    )

    _run_instance_metric_test(
        name="9. metrics when GT has objects but prediction is empty",
        gt_mask=gt_only,
        pred_mask=background_only,
        expected_result={
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "object_precision": 0.0,
            "object_recall": 0.0,
            "rq": 0.0,
            "sq": 0.0,
            "pq": 0.0,
            "matched_pairs": [],
            "unmatched_gt_labels": [1],
            "unmatched_pred_labels": [],
        },
    )

    _run_instance_metric_test(
        name="10. metrics when prediction has objects but GT is empty",
        gt_mask=background_only,
        pred_mask=pred_only,
        expected_result={
            "tp": 0,
            "fp": 1,
            "fn": 0,
            "object_precision": 0.0,
            "object_recall": 0.0,
            "rq": 0.0,
            "sq": 0.0,
            "pq": 0.0,
            "matched_pairs": [],
            "unmatched_gt_labels": [],
            "unmatched_pred_labels": [7],
        },
    )

    _run_instance_metric_test(
        name="11. metrics for one perfect instance match",
        gt_mask=perfect_gt,
        pred_mask=perfect_pred,
        expected_result={
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "object_precision": 1.0,
            "object_recall": 1.0,
            "rq": 1.0,
            "sq": 1.0,
            "pq": 1.0,
            "matched_pairs": [{"gt_label": 1, "pred_label": 9, "iou": 1.0}],
            "unmatched_gt_labels": [],
            "unmatched_pred_labels": [],
        },
    )

    _run_instance_metric_test(
        name="12. metrics with one matched object and one extra prediction",
        gt_mask=one_gt_two_pred_gt,
        pred_mask=one_gt_two_pred_pred,
        expected_result={
            "tp": 1,
            "fp": 1,
            "fn": 0,
            "object_precision": 0.5,
            "object_recall": 1.0,
            "rq": 2.0 / 3.0,
            "sq": 0.5,
            "pq": 1.0 / 3.0,
            "matched_pairs": [{"gt_label": 1, "pred_label": 10, "iou": 0.5}],
            "unmatched_gt_labels": [],
            "unmatched_pred_labels": [20],
        },
    )

    _run_instance_metric_test(
        name="13. metrics with two matched objects and imperfect overlap",
        gt_mask=two_object_gt,
        pred_mask=two_object_pred,
        expected_result={
            "tp": 2,
            "fp": 0,
            "fn": 0,
            "object_precision": 1.0,
            "object_recall": 1.0,
            "rq": 1.0,
            "sq": 0.875,
            "pq": 0.875,
            "matched_pairs": [
                {"gt_label": 1, "pred_label": 5, "iou": 0.75},
                {"gt_label": 2, "pred_label": 6, "iou": 1.0},
            ],
            "unmatched_gt_labels": [],
            "unmatched_pred_labels": [],
        },
    )
