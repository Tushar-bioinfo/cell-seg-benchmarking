from __future__ import annotations

"""Foundation utilities for instance-level IoU comparisons on MoNuSAC masks.

Mask-reading context from `scripts/benchmarking/monusac_pixel_confusion.py`:
- predicted masks are stored as 2D PNG instance masks
- arrays are treated as instance-labeled masks
- label `0` is background

This module intentionally stops at per-instance IoU matrix construction.
It does not implement matching, PQ, RQ, or SQ yet.
"""

import numpy as np


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


def _run_synthetic_test(
    name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    expected_iou_matrix: np.ndarray,
) -> None:
    """Print one small synthetic IoU test case and verify its output."""
    iou_matrix, gt_labels, pred_labels = compute_iou_matrix(gt_mask, pred_mask)

    print(f"\n=== {name} ===")
    print("GT mask:")
    print(gt_mask)
    print("Predicted mask:")
    print(pred_mask)
    print("gt_labels:", gt_labels)
    print("pred_labels:", pred_labels)
    print("IoU matrix:")
    print(iou_matrix)

    assert np.array_equal(iou_matrix.shape, expected_iou_matrix.shape)
    assert np.allclose(iou_matrix, expected_iou_matrix)


if __name__ == "__main__":
    # Test 1: one GT object and one perfect matching predicted object.
    # GT label 1 occupies a 2x2 block.
    # Predicted label 9 occupies the exact same 2x2 block.
    # Intersection = 4, union = 4, so IoU = 1.0.
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
    _run_synthetic_test(
        name="1. one GT object, one perfect predicted object",
        gt_mask=perfect_gt,
        pred_mask=perfect_pred,
        expected_iou_matrix=np.array([[1.0]], dtype=np.float64),
    )

    # Test 2: one GT object and one partial-overlap predicted object.
    # GT label 1 is a 2x2 block in the upper-left.
    # Predicted label 7 is the same 2x2 shape shifted one column right.
    # Overlap pixels = 2, total union pixels = 6, so IoU = 2 / 6 = 1 / 3.
    partial_gt = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )
    partial_pred = np.array(
        [
            [0, 7, 7],
            [0, 7, 7],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )
    _run_synthetic_test(
        name="2. one GT object, one partial-overlap predicted object",
        gt_mask=partial_gt,
        pred_mask=partial_pred,
        expected_iou_matrix=np.array([[1.0 / 3.0]], dtype=np.float64),
    )

    # Test 3: multiple GT objects and multiple predicted objects.
    # GT label 1 is the upper-left 2x2 block.
    # GT label 2 is the lower-right 2x2 block.
    # Predicted label 10 covers only the left column of GT label 1:
    #   intersection = 2, union = 4, so IoU(1, 10) = 0.5.
    # Predicted label 20 matches GT label 2 exactly:
    #   intersection = 4, union = 4, so IoU(2, 20) = 1.0.
    # The off-diagonal entries are 0.0 because those objects do not overlap at all.
    multi_gt = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.uint16,
    )
    multi_pred = np.array(
        [
            [10, 0, 0, 0],
            [10, 0, 0, 0],
            [0, 0, 20, 20],
            [0, 0, 20, 20],
        ],
        dtype=np.uint16,
    )
    _run_synthetic_test(
        name="3. multiple GT objects, multiple predicted objects",
        gt_mask=multi_gt,
        pred_mask=multi_pred,
        expected_iou_matrix=np.array(
            [
                [0.5, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    # Test 4: empty masks / background only.
    # There are no non-zero labels in either mask.
    # Therefore gt_labels and pred_labels are empty,
    # and the IoU matrix has shape (0, 0).
    empty_gt = np.zeros((2, 3), dtype=np.uint16)
    empty_pred = np.zeros((2, 3), dtype=np.uint16)
    _run_synthetic_test(
        name="4. empty masks / background only",
        gt_mask=empty_gt,
        pred_mask=empty_pred,
        expected_iou_matrix=np.zeros((0, 0), dtype=np.float64),
    )
