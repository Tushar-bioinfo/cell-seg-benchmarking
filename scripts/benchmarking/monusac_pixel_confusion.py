from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

DEFAULT_MODEL_NAMES: tuple[str, ...] = (
    "cellpose_sam",
    "cellsam",
    "cellvit_sam",
    "stardist",
)


def _default_inference_root() -> Path:
    """Return the documented MoNuSAC inference output root."""
    return Path(__file__).resolve().parents[2] / "inference" / "benchmarking" / "monusac"


def read_predicted_masks(
    relative_tile_path: str | Path,
    model_names: Sequence[str] | None = None,
    inference_root: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Read predicted instance masks for one tile from one or more model output directories.

    The inference pipeline writes one `uint16` PNG per input tile under:
    `inference/benchmarking/monusac/<model_name>/`
    while preserving the source tile's relative path. For example:
    `all_merged/<unique_id>/00000x_00000y_image.png`

    Parameters
    ----------
    relative_tile_path:
        Relative tile path beneath a model directory.
    model_names:
        Model directory names to read. If omitted, all documented model
        directories are used.
    inference_root:
        Override for the inference root. Defaults to the repository's
        `inference/benchmarking/monusac` directory.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of model name to its loaded instance mask.

    Raises
    ------
    FileNotFoundError
        If any requested mask file does not exist.
    ValueError
        If a requested path is absolute or a loaded mask is not 2D.
    """
    relative_path = Path(relative_tile_path)
    if relative_path.is_absolute():
        raise ValueError(
            "relative_tile_path must be relative to a model directory, "
            f"got absolute path: {relative_path}"
        )

    requested_models = tuple(model_names or DEFAULT_MODEL_NAMES)
    if not requested_models:
        raise ValueError("model_names must contain at least one model name.")

    root = Path(inference_root).expanduser().resolve() if inference_root is not None else _default_inference_root()

    masks: dict[str, np.ndarray] = {}
    for model_name in requested_models:
        mask_path = root / model_name / relative_path
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Predicted mask not found for model '{model_name}': {mask_path}"
            )

        with Image.open(mask_path) as image:
            mask = np.asarray(image)

        if mask.ndim != 2:
            raise ValueError(
                f"Predicted mask for model '{model_name}' must be 2D, got shape {mask.shape}"
            )

        masks[model_name] = mask.astype(np.uint16, copy=False)

    return masks


def validate_same_shape(gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
    """Validate that a ground-truth mask and predicted mask share the same shape."""
    gt_shape = np.asarray(gt_mask).shape
    pred_shape = np.asarray(pred_mask).shape
    if gt_shape != pred_shape:
        raise ValueError(
            "Ground-truth and prediction masks must have identical shapes: "
            f"got gt_shape={gt_shape}, pred_shape={pred_shape}"
        )


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert an instance-labeled mask into a binary `uint8` foreground mask."""
    mask_array = np.asarray(mask)
    return (mask_array > 0).astype(np.uint8)


def _validate_binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Return a validated binary `uint8` mask with values restricted to 0 or 1."""
    mask_array = np.asarray(mask)
    unique_values = np.unique(mask_array)
    if not np.all(np.isin(unique_values, (0, 1))):
        raise ValueError(
            f"{name} must contain only binary values 0 or 1, got values {unique_values.tolist()}"
        )
    return mask_array.astype(np.uint8, copy=False)


def compute_pixel_confusion(gt_binary: np.ndarray, pred_binary: np.ndarray) -> dict[str, int]:
    """Compute TP, TN, FP, and FN counts for two binary masks."""
    validate_same_shape(gt_binary, pred_binary)
    gt_array = _validate_binary_mask(gt_binary, "gt_binary")
    pred_array = _validate_binary_mask(pred_binary, "pred_binary")

    tp = int(np.sum((gt_array == 1) & (pred_array == 1)))
    tn = int(np.sum((gt_array == 0) & (pred_array == 0)))
    fp = int(np.sum((gt_array == 0) & (pred_array == 1)))
    fn = int(np.sum((gt_array == 1) & (pred_array == 0)))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return a safe floating-point division result.

    Parameters
    ----------
    numerator:
        Numerator of the division.
    denominator:
        Denominator of the division.
    default:
        Value returned when the denominator is not strictly positive.

    Returns
    -------
    float
        `numerator / denominator` when `denominator > 0`, otherwise `default`.
    """
    if denominator > 0:
        return float(numerator / denominator)
    return float(default)


def compute_pixel_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict[str, int | float]:
    """Compute binary pixel-level confusion counts and overlap metrics.

    Both inputs are first validated to have identical shapes and then binarized
    using the convention `mask > 0` for foreground.

    Parameters
    ----------
    gt_mask:
        Ground-truth mask, with any non-zero value treated as foreground.
    pred_mask:
        Predicted mask, with any non-zero value treated as foreground.

    Returns
    -------
    dict[str, int | float]
        Dictionary containing TP, TN, FP, FN, precision, recall, F1, and Dice.
        F1 and Dice are reported separately even though they are identical for
        this binary pixel-level formulation.
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


def _run_synthetic_test(
    name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    expectation_comment: str,
) -> None:
    """Print a small synthetic test case with confusion counts and metrics."""
    print(f"\n=== {name} ===")
    print("GT mask:")
    print(gt_mask)
    print("Prediction mask:")
    print(pred_mask)
    print(f"Why these counts are expected: {expectation_comment}")

    gt_binary = binarize_mask(gt_mask)
    pred_binary = binarize_mask(pred_mask)
    metrics = compute_pixel_metrics(gt_mask, pred_mask)
    confusion = {key: metrics[key] for key in ("tp", "tn", "fp", "fn")}
    summary_metrics = {key: metrics[key] for key in ("precision", "recall", "f1", "dice")}

    print("GT binary:")
    print(gt_binary)
    print("Prediction binary:")
    print(pred_binary)
    print("Confusion counts:")
    print(confusion)
    print("Metrics:")
    print(summary_metrics)


if __name__ == "__main__":
    # Perfect foreground match:
    # Precision should be 1.0 because every predicted foreground pixel is correct.
    # Recall should be 1.0 because every ground-truth foreground pixel is recovered.
    perfect_gt = np.array([[0, 1], [1, 0]], dtype=np.uint16)
    perfect_pred = np.array([[0, 5], [9, 0]], dtype=np.uint16)
    _run_synthetic_test(
        name="1. perfect foreground match",
        gt_mask=perfect_gt,
        pred_mask=perfect_pred,
        expectation_comment="Two foreground pixels align exactly and the remaining two pixels are background in both masks.",
    )
    perfect_metrics = compute_pixel_metrics(perfect_gt, perfect_pred)
    assert perfect_metrics["tp"] == 2
    assert perfect_metrics["tn"] == 2
    assert perfect_metrics["fp"] == 0
    assert perfect_metrics["fn"] == 0
    assert np.isclose(perfect_metrics["precision"], 1.0)
    assert np.isclose(perfect_metrics["recall"], 1.0)
    assert np.isclose(perfect_metrics["f1"], 1.0)
    assert np.isclose(perfect_metrics["dice"], 1.0)

    # Partial overlap:
    # Precision should be 0.5 because only half of the predicted foreground pixels are correct.
    # Recall should be 0.5 because only half of the true foreground pixels are recovered.
    partial_gt = np.array([[1, 1], [0, 0]], dtype=np.uint16)
    partial_pred = np.array([[1, 0], [1, 0]], dtype=np.uint16)
    _run_synthetic_test(
        name="2. partial overlap",
        gt_mask=partial_gt,
        pred_mask=partial_pred,
        expectation_comment="There is one TP, one TN, one FP, and one FN, so overlap is only partial.",
    )
    partial_metrics = compute_pixel_metrics(partial_gt, partial_pred)
    assert partial_metrics["tp"] == 1
    assert partial_metrics["tn"] == 1
    assert partial_metrics["fp"] == 1
    assert partial_metrics["fn"] == 1
    assert np.isclose(partial_metrics["precision"], 0.5)
    assert np.isclose(partial_metrics["recall"], 0.5)
    assert np.isclose(partial_metrics["f1"], 0.5)
    assert np.isclose(partial_metrics["dice"], 0.5)

    # GT has objects, prediction empty:
    # Precision should fall back to 0.0 because there are no predicted positives to score.
    # Recall should be 0.0 because none of the true foreground pixels are recovered.
    missed_gt = np.array([[1, 0], [1, 0]], dtype=np.uint16)
    missed_pred = np.zeros((2, 2), dtype=np.uint16)
    _run_synthetic_test(
        name="3. GT has objects, prediction empty",
        gt_mask=missed_gt,
        pred_mask=missed_pred,
        expectation_comment="All true foreground pixels become false negatives because the prediction contains only background.",
    )
    missed_metrics = compute_pixel_metrics(missed_gt, missed_pred)
    assert missed_metrics["tp"] == 0
    assert missed_metrics["tn"] == 2
    assert missed_metrics["fp"] == 0
    assert missed_metrics["fn"] == 2
    assert np.isclose(missed_metrics["precision"], 0.0)
    assert np.isclose(missed_metrics["recall"], 0.0)
    assert np.isclose(missed_metrics["f1"], 0.0)
    assert np.isclose(missed_metrics["dice"], 0.0)

    # GT empty, prediction has objects:
    # Precision should be 0.0 because every predicted foreground pixel is a false positive.
    # Recall should fall back to 0.0 because there is no ground-truth foreground to recover.
    extra_gt = np.zeros((2, 2), dtype=np.uint16)
    extra_pred = np.array([[1, 0], [1, 0]], dtype=np.uint16)
    _run_synthetic_test(
        name="4. GT empty, prediction has objects",
        gt_mask=extra_gt,
        pred_mask=extra_pred,
        expectation_comment="All predicted foreground pixels are false positives because the ground truth is entirely background.",
    )
    extra_metrics = compute_pixel_metrics(extra_gt, extra_pred)
    assert extra_metrics["tp"] == 0
    assert extra_metrics["tn"] == 2
    assert extra_metrics["fp"] == 2
    assert extra_metrics["fn"] == 0
    assert np.isclose(extra_metrics["precision"], 0.0)
    assert np.isclose(extra_metrics["recall"], 0.0)
    assert np.isclose(extra_metrics["f1"], 0.0)
    assert np.isclose(extra_metrics["dice"], 0.0)

    # Both empty / all background:
    # Precision should fall back to 0.0 because there are no predicted positives.
    # Recall should fall back to 0.0 because there are no true positives to recover.
    empty_gt = np.zeros((2, 3), dtype=np.uint16)
    empty_pred = np.zeros((2, 3), dtype=np.uint16)
    _run_synthetic_test(
        name="5. both empty / all background",
        gt_mask=empty_gt,
        pred_mask=empty_pred,
        expectation_comment="Every pixel is a true negative because both masks are entirely background.",
    )
    empty_metrics = compute_pixel_metrics(empty_gt, empty_pred)
    assert empty_metrics["tp"] == 0
    assert empty_metrics["tn"] == 6
    assert empty_metrics["fp"] == 0
    assert empty_metrics["fn"] == 0
    assert np.isclose(empty_metrics["precision"], 0.0)
    assert np.isclose(empty_metrics["recall"], 0.0)
    assert np.isclose(empty_metrics["f1"], 0.0)
    assert np.isclose(empty_metrics["dice"], 0.0)
