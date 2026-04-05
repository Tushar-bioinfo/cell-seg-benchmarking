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


def _run_synthetic_test(
    name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    expectation_comment: str,
) -> None:
    """Print a small synthetic test case and either its confusion counts or the raised error."""
    print(f"\n=== {name} ===")
    print("GT mask:")
    print(gt_mask)
    print("Prediction mask:")
    print(pred_mask)
    print(f"Why these counts are expected: {expectation_comment}")

    try:
        validate_same_shape(gt_mask, pred_mask)
        gt_binary = binarize_mask(gt_mask)
        pred_binary = binarize_mask(pred_mask)
        confusion = compute_pixel_confusion(gt_binary, pred_binary)

        print("GT binary:")
        print(gt_binary)
        print("Prediction binary:")
        print(pred_binary)
        print("Confusion counts:")
        print(confusion)
    except ValueError as error:
        print("GT binary:")
        print("not computed")
        print("Prediction binary:")
        print("not computed")
        print("Confusion counts:")
        print(f"not computed: {error}")


if __name__ == "__main__":
    # Perfect binary match:
    # foreground pixels overlap exactly, so TP=2 and TN=2 with no FP/FN.
    _run_synthetic_test(
        name="1. perfect binary match",
        gt_mask=np.array([[0, 1], [1, 0]], dtype=np.uint16),
        pred_mask=np.array([[0, 5], [9, 0]], dtype=np.uint16),
        expectation_comment="Two foreground pixels align exactly and the remaining two pixels are background in both masks.",
    )

    # One FP and one FN:
    # one true foreground pixel is missed (FN=1) and one background pixel is predicted as foreground (FP=1).
    # The other two pixels are one TP and one TN.
    _run_synthetic_test(
        name="2. one FP and one FN",
        gt_mask=np.array([[1, 0], [1, 0]], dtype=np.uint16),
        pred_mask=np.array([[1, 1], [0, 0]], dtype=np.uint16),
        expectation_comment="Top-left is TP, top-right is FP, bottom-left is FN, and bottom-right is TN.",
    )

    # All background:
    # every pixel is background in both masks, so TN equals the full pixel count.
    _run_synthetic_test(
        name="3. all background",
        gt_mask=np.zeros((2, 3), dtype=np.uint16),
        pred_mask=np.zeros((2, 3), dtype=np.uint16),
        expectation_comment="All six pixels are background in both masks, so every pixel is a true negative.",
    )

    # Shape mismatch:
    # confusion counts are undefined because pixelwise comparison requires identical shapes.
    _run_synthetic_test(
        name="4. shape mismatch",
        gt_mask=np.array([[0, 1], [1, 0]], dtype=np.uint16),
        pred_mask=np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint16),
        expectation_comment="Pixelwise TP/TN/FP/FN cannot be computed when the two masks do not have the same shape.",
    )
