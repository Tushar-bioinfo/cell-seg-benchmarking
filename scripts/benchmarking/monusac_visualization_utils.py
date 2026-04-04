from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SUFFIX = "_image.png"
MASK_SUFFIX = "_mask.png"
DEFAULT_OVERLAY_ALPHA = 0.35
DEFAULT_MIN_INSTANCE_FRACTION = 0.25


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not find the repository root from the provided start path.")


def default_monusac_root(search_from: Path | None = None) -> Path:
    return find_repo_root(search_from or Path(__file__).resolve()) / "data" / "Monusac"


def resolve_monusac_root(data_root: str | Path | None = None, search_from: Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()

    env_root = os.environ.get("MONUSAC_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return default_monusac_root(search_from=search_from)


def _scan_export_pairs(data_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for folder in ("all_merged", "kidney_only"):
        folder_path = data_root / folder
        if not folder_path.exists():
            continue

        for image_path in sorted(folder_path.glob(f"*{IMAGE_SUFFIX}")):
            unique_id = image_path.name[: -len(IMAGE_SUFFIX)]
            mask_path = folder_path / f"{unique_id}{MASK_SUFFIX}"
            if not mask_path.exists():
                continue

            records.append(
                {
                    "unique_id": unique_id,
                    "folder": folder,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )

    if not records:
        raise FileNotFoundError(
            f"No exported MoNuSAC image/mask pairs were found under {data_root}."
        )

    return pd.DataFrame.from_records(records)


def load_export_index(data_root: str | Path | None = None) -> pd.DataFrame:
    monusac_root = resolve_monusac_root(data_root=data_root, search_from=Path(__file__).resolve())
    summary_path = monusac_root / "extraction_summary.csv"

    if summary_path.exists():
        export_index = pd.read_csv(summary_path)
        if "folder" not in export_index.columns:
            export_index["folder"] = "all_merged"

            kidney_dir = monusac_root / "kidney_only"
            if kidney_dir.exists() and "tissue" in export_index.columns:
                kidney_rows = export_index.loc[
                    export_index["tissue"].astype(str).str.casefold().eq("kidney")
                ].copy()
                if not kidney_rows.empty:
                    kidney_rows["folder"] = "kidney_only"
                    export_index = pd.concat([export_index, kidney_rows], ignore_index=True)
        return export_index

    return _scan_export_pairs(monusac_root)


def _resolve_export_path(
    path_like: object,
    data_root: Path,
    unique_id: str,
    suffix: str,
    preferred_folder: str | None = None,
) -> Path:
    candidates: list[Path] = []
    folder_candidates: list[str] = []

    if preferred_folder in {"all_merged", "kidney_only"}:
        folder_candidates.append(preferred_folder)
    folder_candidates.extend(folder for folder in ("all_merged", "kidney_only") if folder not in folder_candidates)

    if isinstance(path_like, str) and path_like.strip():
        raw_path = Path(path_like).expanduser()
        candidates.append(raw_path)
        candidates.append(data_root / raw_path.name)
        if raw_path.name:
            for folder in folder_candidates:
                candidates.append(data_root / folder / raw_path.name)

    filename = f"{unique_id}{suffix}"
    candidates.extend(data_root / folder / filename for folder in folder_candidates)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Could not resolve {suffix} for sample {unique_id!r} under {data_root}."
    )


def select_sample(
    export_index: pd.DataFrame,
    *,
    unique_id: str | None = None,
    tissue: str | None = None,
    folder: str = "all_merged",
    row_index: int = 0,
) -> pd.Series:
    if export_index.empty:
        raise ValueError("The export index is empty.")

    filtered = export_index.copy()

    if "folder" in filtered.columns and folder:
        filtered = filtered.loc[filtered["folder"].eq(folder)]

    if tissue is not None and "tissue" in filtered.columns:
        filtered = filtered.loc[filtered["tissue"].astype(str).str.casefold().eq(tissue.casefold())]

    if unique_id is not None:
        filtered = filtered.loc[filtered["unique_id"].astype(str).eq(unique_id)]

    if filtered.empty:
        raise ValueError("No MoNuSAC samples matched the requested filters.")

    if row_index < 0 or row_index >= len(filtered):
        raise IndexError(f"row_index={row_index} is out of range for {len(filtered)} matching samples.")

    return filtered.iloc[row_index]


def load_sample_arrays(
    sample_row: pd.Series,
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    monusac_root = resolve_monusac_root(data_root=data_root, search_from=Path(__file__).resolve())
    unique_id = str(sample_row["unique_id"])
    preferred_folder = sample_row.get("folder")
    image_path = _resolve_export_path(
        sample_row.get("image_path"),
        monusac_root,
        unique_id,
        IMAGE_SUFFIX,
        preferred_folder=preferred_folder,
    )
    mask_path = _resolve_export_path(
        sample_row.get("mask_path"),
        monusac_root,
        unique_id,
        MASK_SUFFIX,
        preferred_folder=preferred_folder,
    )

    with Image.open(image_path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"), dtype=np.uint8).copy()

    with Image.open(mask_path) as mask_handle:
        mask = np.asarray(mask_handle, dtype=np.uint16).copy()

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {unique_id}: image={image.shape[:2]}, mask={mask.shape}."
        )

    return image, mask


def colorize_instance_mask(instance_mask: np.ndarray, seed: int = 7) -> np.ndarray:
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]
    color_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)

    if labels.size == 0:
        return color_mask

    for label in labels:
        rng = np.random.default_rng(seed + int(label))
        color = rng.integers(32, 224, size=3, dtype=np.uint8)
        color_mask[instance_mask == label] = color

    return color_mask


def find_instance_boundaries(instance_mask: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(instance_mask.shape, dtype=bool)

    boundaries[1:, :] |= instance_mask[1:, :] != instance_mask[:-1, :]
    boundaries[:-1, :] |= instance_mask[:-1, :] != instance_mask[1:, :]
    boundaries[:, 1:] |= instance_mask[:, 1:] != instance_mask[:, :-1]
    boundaries[:, :-1] |= instance_mask[:, :-1] != instance_mask[:, 1:]

    boundaries &= instance_mask > 0
    return boundaries


def overlay_instance_mask(
    image: np.ndarray,
    instance_mask: np.ndarray,
    *,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    seed: int = 7,
) -> np.ndarray:
    if image.shape[:2] != instance_mask.shape:
        raise ValueError(
            f"Overlay expects matching image/mask shapes, got {image.shape[:2]} and {instance_mask.shape}."
        )

    overlay = image.astype(np.float32).copy()
    color_mask = colorize_instance_mask(instance_mask, seed=seed)
    foreground = instance_mask > 0
    overlay[foreground] = (1.0 - alpha) * overlay[foreground] + alpha * color_mask[foreground]
    overlay[find_instance_boundaries(instance_mask)] = 255
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _target_size(shape: tuple[int, int], scale_factor: float) -> tuple[int, int]:
    height, width = shape
    target_height = max(1, int(round(height * scale_factor)))
    target_width = max(1, int(round(width * scale_factor)))
    return target_height, target_width


def _resize_rgb_image(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    resized = Image.fromarray(image, mode="RGB").resize(
        (target_width, target_height),
        resample=Image.Resampling.BOX,
    )
    return np.asarray(resized, dtype=np.uint8)


def resize_instance_mask(
    instance_mask: np.ndarray,
    *,
    scale_factor: float,
    min_instance_fraction: float = DEFAULT_MIN_INSTANCE_FRACTION,
) -> np.ndarray:
    if not 0 < scale_factor:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}.")
    if not 0 <= min_instance_fraction <= 1:
        raise ValueError(
            f"min_instance_fraction must be between 0 and 1, got {min_instance_fraction}."
        )

    target_shape = _target_size(instance_mask.shape, scale_factor)
    target_height, target_width = target_shape
    resized_mask = np.zeros(target_shape, dtype=np.uint16)
    best_score = np.zeros(target_shape, dtype=np.float32)

    labels = np.unique(instance_mask)
    labels = labels[labels > 0]

    for label in labels:
        binary = (instance_mask == label).astype(np.uint8) * 255
        resized_binary = Image.fromarray(binary, mode="L").resize(
            (target_width, target_height),
            resample=Image.Resampling.BOX if scale_factor < 1 else Image.Resampling.NEAREST,
        )
        score = np.asarray(resized_binary, dtype=np.float32) / 255.0
        better = (score >= min_instance_fraction) & (score > best_score)
        resized_mask[better] = np.uint16(label)
        best_score[better] = score[better]

    return resized_mask


def rescale_patch_and_mask(
    image: np.ndarray,
    instance_mask: np.ndarray,
    *,
    source_magnification: float,
    target_magnification: float,
    min_instance_fraction: float = DEFAULT_MIN_INSTANCE_FRACTION,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if source_magnification <= 0 or target_magnification <= 0:
        raise ValueError("Magnifications must be positive.")
    if image.shape[:2] != instance_mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: image={image.shape[:2]}, mask={instance_mask.shape}."
        )

    scale_factor = float(target_magnification) / float(source_magnification)
    target_shape = _target_size(instance_mask.shape, scale_factor)
    resized_image = _resize_rgb_image(image, target_shape)
    resized_mask = resize_instance_mask(
        instance_mask,
        scale_factor=scale_factor,
        min_instance_fraction=min_instance_fraction,
    )

    original_labels = np.unique(instance_mask)
    original_labels = original_labels[original_labels > 0]
    resized_labels = np.unique(resized_mask)
    resized_labels = resized_labels[resized_labels > 0]
    dropped_labels = sorted(set(original_labels.tolist()) - set(resized_labels.tolist()))

    metadata = {
        "source_shape": tuple(int(value) for value in instance_mask.shape),
        "target_shape": tuple(int(value) for value in resized_mask.shape),
        "source_magnification": float(source_magnification),
        "target_magnification": float(target_magnification),
        "scale_factor": scale_factor,
        "min_instance_fraction": float(min_instance_fraction),
        "original_instance_count": int(original_labels.size),
        "resized_instance_count": int(resized_labels.size),
        "dropped_instance_count": int(len(dropped_labels)),
        "dropped_instance_labels": dropped_labels,
    }
    return resized_image, resized_mask, metadata


def plot_sample_triptych(
    image: np.ndarray,
    instance_mask: np.ndarray,
    *,
    title: str | None = None,
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
    seed: int = 7,
) -> tuple[plt.Figure, np.ndarray]:
    color_mask = colorize_instance_mask(instance_mask, seed=seed)
    overlay = overlay_instance_mask(image, instance_mask, alpha=overlay_alpha, seed=seed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("RGB patch")
    axes[1].imshow(color_mask)
    axes[1].set_title("Instance mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    return fig, axes


def plot_resize_comparison(
    original_image: np.ndarray,
    original_mask: np.ndarray,
    resized_image: np.ndarray,
    resized_mask: np.ndarray,
    *,
    title: str | None = None,
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
    seed: int = 7,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    panels = (
        (axes[0, 0], original_image, "Original RGB patch"),
        (axes[0, 1], colorize_instance_mask(original_mask, seed=seed), "Original instance mask"),
        (axes[0, 2], overlay_instance_mask(original_image, original_mask, alpha=overlay_alpha, seed=seed), "Original overlay"),
        (axes[1, 0], resized_image, "Rescaled RGB patch"),
        (axes[1, 1], colorize_instance_mask(resized_mask, seed=seed), "Rescaled instance mask"),
        (axes[1, 2], overlay_instance_mask(resized_image, resized_mask, alpha=overlay_alpha, seed=seed), "Rescaled overlay"),
    )

    for axis, array, axis_title in panels:
        axis.imshow(array)
        axis.set_title(axis_title)
        axis.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    return fig, axes
