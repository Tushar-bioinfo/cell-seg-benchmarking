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
- Single-mask file loading is supported for PNG and TIFF inputs.
- Folder-level batch evaluation and CSV export are supported for simple
  filename-stem based mask pairing.
"""

import csv
from pathlib import Path
from pprint import pprint
from typing import Any, TypedDict

import imageio.v3 as iio
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
CSVRow = dict[str, Any]

FLAT_METRIC_FIELDNAMES: tuple[str, ...] = (
    "pixel_tp",
    "pixel_tn",
    "pixel_fp",
    "pixel_fn",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_dice",
    "instance_tp",
    "instance_fp",
    "instance_fn",
    "instance_object_precision",
    "instance_object_recall",
    "instance_rq",
    "instance_sq",
    "instance_pq",
    "instance_match_count",
    "instance_unmatched_gt_count",
    "instance_unmatched_pred_count",
)

CSV_FIELDNAMES: tuple[str, ...] = (
    "image_id",
    "status",
    "error_message",
    "gt_path",
    "pred_path",
    *FLAT_METRIC_FIELDNAMES,
)


def _as_2d_array(mask: ArrayLike, name: str) -> npt.NDArray[Any]:
    """Convert an input mask to a NumPy array and ensure it is 2D."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    return array


def _safely_reduce_loaded_mask(array: npt.NDArray[Any], path: Path) -> npt.NDArray[Any]:
    """Return a 2D mask array from a loaded image when reduction is unambiguous."""
    if array.ndim == 2:
        return array

    if array.ndim != 3:
        raise ValueError(
            "Mask file must decode to a 2D array. "
            f"Loaded {path} with shape {array.shape}."
        )

    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        return np.asarray(squeezed)

    if array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(
            rgb[..., 0], rgb[..., 2]
        ):
            if array.shape[-1] == 4:
                alpha_channel = array[..., 3]
                if not np.all(alpha_channel == alpha_channel.flat[0]):
                    raise ValueError(
                        "Mask file has an RGBA layout with a non-constant alpha channel, "
                        f"which is ambiguous for label masks: {path}"
                    )
            return np.asarray(rgb[..., 0])

    if array.shape[0] in (3, 4):
        rgb = array[:3, ...]
        if np.array_equal(rgb[0, ...], rgb[1, ...]) and np.array_equal(
            rgb[0, ...], rgb[2, ...]
        ):
            if array.shape[0] == 4:
                alpha_channel = array[3, ...]
                if not np.all(alpha_channel == alpha_channel.flat[0]):
                    raise ValueError(
                        "Mask file has a channel-first RGBA layout with a non-constant "
                        f"alpha channel, which is ambiguous for label masks: {path}"
                    )
            return np.asarray(rgb[0, ...])

    raise ValueError(
        "Mask file must be 2D. Extra channels are only accepted when they are "
        "singleton dimensions or replicated grayscale channels. "
        f"Loaded {path} with shape {array.shape}."
    )


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


def load_mask(path: str) -> npt.NDArray[Any]:
    """Load one PNG or TIFF segmentation mask as a 2D NumPy array.

    The loader preserves the on-disk numeric dtype whenever practical so integer
    instance labels such as `uint16` survive the round trip. Ambiguous
    multi-channel inputs are rejected with a clear error.

    Parameters
    ----------
    path:
        Filesystem path to a `.png`, `.tif`, or `.tiff` mask file.

    Returns
    -------
    np.ndarray
        A 2D mask array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the suffix is unsupported, the file cannot be interpreted as a 2D
        mask, or the file contents are otherwise ambiguous for label masks.
    """
    mask_path = Path(path).expanduser()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    suffix = mask_path.suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"}:
        raise ValueError(
            "Mask file must be a PNG or TIFF image, "
            f"got '{suffix or '<no suffix>'}' for {mask_path}"
        )

    try:
        loaded = iio.imread(mask_path)
    except Exception as exc:  # pragma: no cover - depends on image backend/runtime.
        raise ValueError(f"Failed to read mask file {mask_path}: {exc}") from exc

    array = np.asarray(loaded)
    if array.size == 0:
        raise ValueError(f"Mask file is empty: {mask_path}")

    return _safely_reduce_loaded_mask(array, mask_path)


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


def evaluate_segmentation_files(
    gt_path: str,
    pred_path: str,
    threshold: float = 0.5,
) -> SegmentationEvaluation:
    """Load two mask files and evaluate them with the standard segmentation API.

    Parameters
    ----------
    gt_path:
        Path to the ground-truth PNG or TIFF mask.
    pred_path:
        Path to the predicted PNG or TIFF mask.
    threshold:
        IoU threshold used for one-to-one instance matching.

    Returns
    -------
    SegmentationEvaluation
        The same nested result dictionary returned by `evaluate_segmentation`.

    Examples
    --------
    PNG files:
        `result = evaluate_segmentation_files("gt_mask.png", "pred_mask.png")`

    TIFF files:
        `result = evaluate_segmentation_files("gt_mask.tiff", "pred_mask.tif", threshold=0.5)`
    """
    gt_mask = load_mask(gt_path)
    pred_mask = load_mask(pred_path)

    try:
        return evaluate_segmentation(gt_mask, pred_mask, threshold=threshold)
    except ValueError as exc:
        raise ValueError(
            "Failed to evaluate segmentation files "
            f"gt_path={Path(gt_path).expanduser()}, pred_path={Path(pred_path).expanduser()}: "
            f"{exc}"
        ) from exc


def flatten_metrics_for_csv(result: dict[str, Any], image_id: str) -> CSVRow:
    """Flatten a nested segmentation result into one CSV-friendly row.

    Parameters
    ----------
    result:
        Nested segmentation result in the format returned by
        `evaluate_segmentation(...)` or `evaluate_segmentation_files(...)`.
    image_id:
        Identifier to store alongside the flattened metrics, typically the
        filename stem used to pair GT and prediction masks.

    Returns
    -------
    dict[str, Any]
        Flat mapping of scalar summary metrics from both the pixel-level and
        instance-level results. Verbose list fields such as `matched_pairs` are
        intentionally omitted.
    """
    pixel_metrics = result["pixel_metrics"]
    instance_metrics = result["instance_metrics"]

    return {
        "image_id": image_id,
        "pixel_tp": pixel_metrics["tp"],
        "pixel_tn": pixel_metrics["tn"],
        "pixel_fp": pixel_metrics["fp"],
        "pixel_fn": pixel_metrics["fn"],
        "pixel_precision": pixel_metrics["precision"],
        "pixel_recall": pixel_metrics["recall"],
        "pixel_f1": pixel_metrics["f1"],
        "pixel_dice": pixel_metrics["dice"],
        "instance_tp": instance_metrics["tp"],
        "instance_fp": instance_metrics["fp"],
        "instance_fn": instance_metrics["fn"],
        "instance_object_precision": instance_metrics["object_precision"],
        "instance_object_recall": instance_metrics["object_recall"],
        "instance_rq": instance_metrics["rq"],
        "instance_sq": instance_metrics["sq"],
        "instance_pq": instance_metrics["pq"],
        "instance_match_count": len(instance_metrics["matched_pairs"]),
        "instance_unmatched_gt_count": len(instance_metrics["unmatched_gt_labels"]),
        "instance_unmatched_pred_count": len(
            instance_metrics["unmatched_pred_labels"]
        ),
    }


def _empty_csv_row(
    image_id: str,
    *,
    status: str,
    error_message: str = "",
    gt_path: Path | None = None,
    pred_path: Path | None = None,
) -> CSVRow:
    """Create a CSV row shell with consistent columns across all statuses."""
    row: CSVRow = {field: "" for field in CSV_FIELDNAMES}
    row["image_id"] = image_id
    row["status"] = status
    row["error_message"] = error_message
    row["gt_path"] = str(gt_path) if gt_path is not None else ""
    row["pred_path"] = str(pred_path) if pred_path is not None else ""
    return row


def _find_mask_files(mask_dir: Path, pattern: str) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Index mask files under one directory by filename stem.

    Notes
    -----
    - `pattern` is passed directly to `Path.glob(...)`
    - use `"**/*.png"` if recursive matching is needed
    - stems must be unique within a directory to avoid ambiguous pairing
    """
    files_by_stem: dict[str, Path] = {}
    duplicate_paths_by_stem: dict[str, list[Path]] = {}

    for path in sorted(mask_dir.glob(pattern)):
        if not path.is_file():
            continue

        stem = path.stem
        if stem in duplicate_paths_by_stem:
            duplicate_paths_by_stem[stem].append(path)
            continue

        if stem in files_by_stem:
            duplicate_paths_by_stem[stem] = [files_by_stem.pop(stem), path]
            continue

        files_by_stem[stem] = path

    return files_by_stem, duplicate_paths_by_stem


def _write_rows_to_csv(rows: list[CSVRow], output_csv: Path) -> None:
    """Write evaluation rows to CSV using a stable column order."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def _print_evaluation_summary(
    *,
    matched_pair_count: int,
    failure_count: int,
    gt_only_stems: list[str],
    pred_only_stems: list[str],
) -> None:
    """Print a concise summary for one folder-level evaluation run."""
    print(f"Matched file pairs processed: {matched_pair_count}")
    print(f"Failures: {failure_count}")
    print(f"GT-only files: {len(gt_only_stems)}")
    print(f"Prediction-only files: {len(pred_only_stems)}")

    if gt_only_stems:
        print("Unmatched GT stems:", ", ".join(gt_only_stems))

    if pred_only_stems:
        print("Unmatched prediction stems:", ", ".join(pred_only_stems))


def _duplicate_stem_rows(
    duplicate_paths_by_stem: dict[str, list[Path]],
    *,
    is_gt: bool,
) -> list[CSVRow]:
    """Create error rows for ambiguous duplicate stems within one folder."""
    rows: list[CSVRow] = []
    side = "ground-truth" if is_gt else "prediction"
    path_field = "gt_path" if is_gt else "pred_path"

    for image_id, paths in sorted(duplicate_paths_by_stem.items()):
        row = _empty_csv_row(
            image_id,
            status="error",
            error_message=(
                f"Duplicate {side} files found for filename stem '{image_id}'. "
                "Rename files so stems are unique before evaluation."
            ),
        )
        row[path_field] = " | ".join(str(path) for path in paths)
        rows.append(row)

    return rows


def evaluate_folder(
    gt_dir: str,
    pred_dir: str,
    pattern: str = "*.png",
    threshold: float = 0.5,
    output_csv: str | None = None,
) -> list[CSVRow]:
    """Evaluate many GT/prediction mask pairs matched by filename stem.

    Parameters
    ----------
    gt_dir:
        Directory containing ground-truth mask files.
    pred_dir:
        Directory containing predicted mask files.
    pattern:
        Glob pattern passed to `Path.glob(...)` in each directory. The default
        matches PNG files in the top level only. Use `"**/*.png"` for recursive
        matching.
    threshold:
        IoU threshold used for one-to-one instance matching.
    output_csv:
        Optional path where the flattened evaluation rows should be saved.

    Returns
    -------
    list[dict]
        One row per matched pair or unmatched file. Successful rows contain
        flattened metrics with `status="ok"`. Failed evaluations and unmatched
        files receive an explanatory `status` and `error_message`.

    Expected folder structure
    -------------------------
    ```text
    gt_masks/
      image_001.png
      image_002.png
      image_003.png

    pred_masks/
      image_001.png
      image_002.png
      image_004.png
    ```

    Example
    -------
    ```python
    rows = evaluate_folder(
        gt_dir="gt_masks",
        pred_dir="pred_masks",
        output_csv="results/monusac_eval.csv",
    )
    ```
    """
    gt_root = Path(gt_dir).expanduser()
    pred_root = Path(pred_dir).expanduser()

    if not gt_root.is_dir():
        raise NotADirectoryError(f"Ground-truth directory not found: {gt_root}")
    if not pred_root.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {pred_root}")

    gt_files, gt_duplicates = _find_mask_files(gt_root, pattern)
    pred_files, pred_duplicates = _find_mask_files(pred_root, pattern)

    gt_stems = set(gt_files)
    pred_stems = set(pred_files)
    gt_duplicate_stems = set(gt_duplicates)
    pred_duplicate_stems = set(pred_duplicates)
    matched_stems = sorted(gt_stems & pred_stems)
    gt_only_stems = sorted((gt_stems - pred_stems) - pred_duplicate_stems)
    pred_only_stems = sorted((pred_stems - gt_stems) - gt_duplicate_stems)

    rows: list[CSVRow] = []
    rows.extend(_duplicate_stem_rows(gt_duplicates, is_gt=True))
    rows.extend(_duplicate_stem_rows(pred_duplicates, is_gt=False))
    failure_count = len(rows)

    for image_id in matched_stems:
        gt_path = gt_files[image_id]
        pred_path = pred_files[image_id]
        row = _empty_csv_row(
            image_id,
            status="ok",
            gt_path=gt_path,
            pred_path=pred_path,
        )

        try:
            result = evaluate_segmentation_files(
                str(gt_path),
                str(pred_path),
                threshold=threshold,
            )
            row.update(flatten_metrics_for_csv(result, image_id=image_id))
        except Exception as exc:
            row["status"] = "error"
            row["error_message"] = str(exc)
            failure_count += 1

        rows.append(row)

    for image_id in gt_only_stems:
        gt_path = gt_files[image_id]
        rows.append(
            _empty_csv_row(
                image_id,
                status="missing_prediction",
                error_message=(
                    f"No prediction mask matched GT filename stem '{image_id}'."
                ),
                gt_path=gt_path,
            )
        )

    for image_id in pred_only_stems:
        pred_path = pred_files[image_id]
        rows.append(
            _empty_csv_row(
                image_id,
                status="missing_ground_truth",
                error_message=(
                    f"No ground-truth mask matched prediction filename stem '{image_id}'."
                ),
                pred_path=pred_path,
            )
        )

    if output_csv is not None:
        output_path = Path(output_csv).expanduser()
        _write_rows_to_csv(rows, output_path)
        print(f"Saved CSV results to: {output_path}")

    _print_evaluation_summary(
        matched_pair_count=len(matched_stems),
        failure_count=failure_count,
        gt_only_stems=gt_only_stems,
        pred_only_stems=pred_only_stems,
    )

    return rows


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
