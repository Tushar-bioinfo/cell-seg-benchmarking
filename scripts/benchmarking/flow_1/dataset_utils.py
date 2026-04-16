from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from common import PROJECT_ROOT

DEFAULT_IMAGE_EXTENSIONS = (".png",)
DEFAULT_MASK_EXTENSIONS = (".png", ".tif", ".tiff")
DEFAULT_IMAGE_SUFFIX_TOKEN = "_image"
DEFAULT_MASK_SUFFIX_TOKEN = "_mask"
DEFAULT_PAIR_MODE = "suffix"
DEFAULT_SUMMARY_NAMES = ("dataset_manifest.csv", "extraction_summary.csv")

_SAFE_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SamplePair:
    sample_id: str
    image_path: Path
    mask_path: Path
    relative_image_path: Path
    relative_mask_path: Path
    metadata: dict[str, Any]


def normalize_extensions(values: Sequence[str] | None, *, defaults: Sequence[str]) -> tuple[str, ...]:
    if not values:
        values = defaults
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized.append(text if text.startswith(".") else f".{text}")
    deduped: list[str] = []
    for value in normalized:
        lowered = value.lower()
        if lowered not in deduped:
            deduped.append(lowered)
    return tuple(deduped)


def infer_dataset_manifest_path(input_root: Path) -> Path | None:
    for name in DEFAULT_SUMMARY_NAMES:
        candidate = input_root / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _safe_token(text: str) -> str:
    sanitized = _SAFE_TOKEN_PATTERN.sub("_", text).strip("_")
    return sanitized or "sample"


def sample_id_from_relative_path(relative_path: Path, *, suffix_token: str | None = None) -> str:
    without_suffix = relative_path.with_suffix("")
    parts = list(without_suffix.parts)
    if parts:
        filename = parts[-1]
        if suffix_token and filename.endswith(suffix_token):
            stripped = filename[: -len(suffix_token)]
            if stripped:
                filename = stripped
        parts[-1] = filename
    return "__".join(_safe_token(part) for part in parts if part)


def prepare_passthrough_metadata(
    metadata: dict[str, Any],
    *,
    reserved_names: set[str],
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for raw_key, raw_value in metadata.items():
        key = str(raw_key)
        if key in reserved_names:
            key = f"input_{key}"
        prepared[key] = _normalize_scalar(raw_value)
    return prepared


def _resolve_recorded_path(raw_value: Any, *, base_dirs: Sequence[Path]) -> Path:
    raw_text = str(raw_value).strip()
    candidate = Path(raw_text).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [base_dir / candidate for base_dir in base_dirs]
    if not candidate.is_absolute():
        candidates.append(PROJECT_ROOT / candidate)
        candidates.append(candidate)
    checked: set[Path] = set()
    for candidate_path in candidates:
        resolved = candidate_path.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Could not resolve a file path from {raw_text!r}.")


def _relative_to_any(path: Path, roots: Sequence[Path]) -> Path:
    for root in roots:
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return Path(path.name)


def _candidate_column(fieldnames: Sequence[str], preferred: str | None, fallbacks: Sequence[str]) -> str | None:
    if preferred and preferred in fieldnames:
        return preferred
    for name in fallbacks:
        if name in fieldnames:
            return name
    return None


def _derive_sample_id(
    *,
    row: dict[str, Any],
    relative_image_path: Path,
    sample_id_column: str | None,
    pair_mode: str,
    image_suffix_token: str,
) -> str:
    candidates = []
    if sample_id_column:
        candidates.append(sample_id_column)
    candidates.extend(("sample_id", "unique_id", "image_id", "id"))
    for column in candidates:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return _safe_token(text)
    suffix_token = image_suffix_token if pair_mode == "suffix" else None
    return sample_id_from_relative_path(relative_image_path, suffix_token=suffix_token)


def _samples_from_manifest(
    *,
    input_root: Path,
    manifest_path: Path,
    images_subdir: str | None,
    masks_subdir: str | None,
    pair_mode: str,
    image_suffix_token: str,
    image_column: str | None,
    mask_column: str | None,
    sample_id_column: str | None,
    sample_ids: set[str] | None,
) -> list[SamplePair]:
    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        return []

    fieldnames = tuple(str(column) for column in manifest_df.columns)
    resolved_image_column = _candidate_column(fieldnames, image_column, ("image_path", "source_image_path", "path"))
    resolved_mask_column = _candidate_column(fieldnames, mask_column, ("mask_path", "source_mask_path"))
    if resolved_image_column is None or resolved_mask_column is None:
        raise ValueError(
            f"Manifest {manifest_path} must include image and mask columns. "
            f"Found columns: {', '.join(fieldnames)}"
        )

    images_root = (input_root / images_subdir).resolve() if images_subdir else input_root
    masks_root = (input_root / masks_subdir).resolve() if masks_subdir else input_root

    pairs: list[SamplePair] = []
    for row in manifest_df.to_dict(orient="records"):
        image_path = _resolve_recorded_path(
            row[resolved_image_column],
            base_dirs=(manifest_path.parent, input_root, images_root),
        )
        mask_path = _resolve_recorded_path(
            row[resolved_mask_column],
            base_dirs=(manifest_path.parent, input_root, masks_root),
        )
        relative_image_path = _relative_to_any(image_path, (images_root, input_root))
        relative_mask_path = _relative_to_any(mask_path, (masks_root, input_root))
        sample_id = _derive_sample_id(
            row=row,
            relative_image_path=relative_image_path,
            sample_id_column=sample_id_column,
            pair_mode=pair_mode,
            image_suffix_token=image_suffix_token,
        )
        if sample_ids and sample_id not in sample_ids:
            continue
        pairs.append(
            SamplePair(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
                relative_image_path=relative_image_path,
                relative_mask_path=relative_mask_path,
                metadata={str(key): _normalize_scalar(value) for key, value in row.items()},
            )
        )
    return pairs


def _iter_files(root: Path, *, recursive: bool) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(iterator):
        if path.is_file():
            yield path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _samples_from_scan(
    *,
    input_root: Path,
    images_subdir: str | None,
    masks_subdir: str | None,
    pair_mode: str,
    image_suffix_token: str,
    mask_suffix_token: str,
    image_extensions: Sequence[str],
    mask_extensions: Sequence[str],
    recursive: bool,
    sample_ids: set[str] | None,
) -> list[SamplePair]:
    images_root = (input_root / images_subdir).resolve() if images_subdir else input_root
    masks_root = (input_root / masks_subdir).resolve() if masks_subdir else input_root
    if not images_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {images_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Mask root does not exist: {masks_root}")

    mask_lookup: dict[str, Path] = {}
    for mask_path in _iter_files(masks_root, recursive=recursive):
        if mask_path.suffix.lower() not in mask_extensions:
            continue
        relative_mask_path = mask_path.relative_to(masks_root)
        mask_lookup[relative_mask_path.as_posix().lower()] = mask_path.resolve()

    pairs: list[SamplePair] = []
    missing_masks: list[str] = []

    for image_path in _iter_files(images_root, recursive=recursive):
        if image_path.suffix.lower() not in image_extensions:
            continue
        if masks_root != images_root and _is_relative_to(image_path.resolve(), masks_root):
            continue

        relative_image_path = image_path.relative_to(images_root)
        image_stem = relative_image_path.stem
        if pair_mode == "suffix":
            if mask_suffix_token and image_stem.endswith(mask_suffix_token):
                continue
            stem_base = image_stem
            if image_suffix_token and stem_base.endswith(image_suffix_token):
                stripped = stem_base[: -len(image_suffix_token)]
                if stripped:
                    stem_base = stripped
            candidate_stem = f"{stem_base}{mask_suffix_token}"
            sample_id = sample_id_from_relative_path(relative_image_path, suffix_token=image_suffix_token)
        elif pair_mode == "stem":
            candidate_stem = image_stem
            sample_id = sample_id_from_relative_path(relative_image_path)
        else:
            raise ValueError(f"Unsupported pair mode: {pair_mode}")

        if sample_ids and sample_id not in sample_ids:
            continue

        candidate_mask_path: Path | None = None
        for extension in mask_extensions:
            relative_candidate = relative_image_path.with_name(f"{candidate_stem}{extension}")
            candidate_mask_path = mask_lookup.get(relative_candidate.as_posix().lower())
            if candidate_mask_path is not None:
                break

        if candidate_mask_path is None:
            missing_masks.append(str(relative_image_path))
            continue

        pairs.append(
            SamplePair(
                sample_id=sample_id,
                image_path=image_path.resolve(),
                mask_path=candidate_mask_path.resolve(),
                relative_image_path=relative_image_path,
                relative_mask_path=_relative_to_any(candidate_mask_path.resolve(), (masks_root, input_root)),
                metadata={},
            )
        )

    if missing_masks:
        examples = ", ".join(missing_masks[:5])
        raise FileNotFoundError(
            "Could not find matching mask files for one or more images. "
            f"Examples: {examples}"
        )

    return pairs


def collect_dataset_samples(
    input_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    images_subdir: str | None = None,
    masks_subdir: str | None = None,
    pair_mode: str = DEFAULT_PAIR_MODE,
    image_suffix_token: str = DEFAULT_IMAGE_SUFFIX_TOKEN,
    mask_suffix_token: str = DEFAULT_MASK_SUFFIX_TOKEN,
    image_extensions: Sequence[str] | None = None,
    mask_extensions: Sequence[str] | None = None,
    recursive: bool = True,
    image_column: str | None = None,
    mask_column: str | None = None,
    sample_id_column: str | None = None,
    sample_ids: Sequence[str] | None = None,
) -> list[SamplePair]:
    resolved_input_root = Path(input_root).expanduser().resolve()
    resolved_manifest_path = (
        Path(manifest_path).expanduser().resolve() if manifest_path is not None else infer_dataset_manifest_path(resolved_input_root)
    )
    normalized_image_extensions = normalize_extensions(image_extensions, defaults=DEFAULT_IMAGE_EXTENSIONS)
    normalized_mask_extensions = normalize_extensions(mask_extensions, defaults=DEFAULT_MASK_EXTENSIONS)
    sample_id_filter = {_safe_token(str(value)) for value in sample_ids} if sample_ids else None

    if resolved_manifest_path is not None and resolved_manifest_path.is_file():
        pairs = _samples_from_manifest(
            input_root=resolved_input_root,
            manifest_path=resolved_manifest_path,
            images_subdir=images_subdir,
            masks_subdir=masks_subdir,
            pair_mode=pair_mode,
            image_suffix_token=image_suffix_token,
            image_column=image_column,
            mask_column=mask_column,
            sample_id_column=sample_id_column,
            sample_ids=sample_id_filter,
        )
    else:
        pairs = _samples_from_scan(
            input_root=resolved_input_root,
            images_subdir=images_subdir,
            masks_subdir=masks_subdir,
            pair_mode=pair_mode,
            image_suffix_token=image_suffix_token,
            mask_suffix_token=mask_suffix_token,
            image_extensions=normalized_image_extensions,
            mask_extensions=normalized_mask_extensions,
            recursive=recursive,
            sample_ids=sample_id_filter,
        )

    deduped_pairs: list[SamplePair] = []
    seen_image_paths: set[Path] = set()
    for pair in pairs:
        if pair.image_path in seen_image_paths:
            continue
        seen_image_paths.add(pair.image_path)
        deduped_pairs.append(pair)

    if not deduped_pairs:
        raise FileNotFoundError(
            f"No paired image/mask samples were found under {resolved_input_root}."
        )

    deduped_pairs.sort(key=lambda item: (item.sample_id, item.relative_image_path.as_posix()))
    return deduped_pairs


def _reduce_loaded_mask(array: np.ndarray, path: Path) -> np.ndarray:
    if array.ndim == 2:
        return array
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        return np.asarray(squeezed)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            return np.asarray(rgb[..., 0])
    raise ValueError(f"Mask {path} must decode to a 2D label array, got shape {array.shape}.")


def load_rgb_image(path: str | Path) -> np.ndarray:
    resolved_path = Path(path).expanduser().resolve()
    with Image.open(resolved_path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()


def load_instance_mask(path: str | Path) -> np.ndarray:
    resolved_path = Path(path).expanduser().resolve()
    with Image.open(resolved_path) as handle:
        array = np.asarray(handle)
    reduced = _reduce_loaded_mask(np.asarray(array), resolved_path)
    return np.asarray(reduced, dtype=np.uint16).copy()


def write_rgb_png(image: np.ndarray, path: str | Path) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(resolved_path)
    return resolved_path


def write_mask_png(mask: np.ndarray, path: str | Path) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint16), mode="I;16").save(resolved_path)
    return resolved_path


def _target_size(shape: tuple[int, int], scale_factor: float) -> tuple[int, int]:
    height, width = shape
    target_height = max(1, int(round(height * scale_factor)))
    target_width = max(1, int(round(width * scale_factor)))
    return target_height, target_width


def _resize_rgb_image(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    resized = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").resize(
        (target_width, target_height),
        resample=Image.Resampling.BOX,
    )
    return np.asarray(resized, dtype=np.uint8)


def resize_instance_mask(
    instance_mask: np.ndarray,
    *,
    scale_factor: float,
    min_instance_fraction: float,
) -> np.ndarray:
    if scale_factor <= 0:
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


def rescale_image_and_mask(
    image: np.ndarray,
    instance_mask: np.ndarray,
    *,
    source_magnification: float,
    target_magnification: float,
    min_instance_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
        "target_shape": tuple(int(value) for value in target_shape),
        "source_magnification": float(source_magnification),
        "target_magnification": float(target_magnification),
        "scale_factor": float(scale_factor),
        "min_instance_fraction": float(min_instance_fraction),
        "original_instance_count": int(len(original_labels)),
        "resized_instance_count": int(len(resized_labels)),
        "dropped_instance_count": int(len(dropped_labels)),
        "dropped_instance_labels": dropped_labels,
    }
    return resized_image, resized_mask, metadata
