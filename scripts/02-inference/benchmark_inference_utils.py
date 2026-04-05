from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXCLUDED_NAME_TOKENS = ("_mask", "_masks", "_flows")


@dataclass(frozen=True)
class TileRecord:
    source_path: Path
    relative_path: Path
    image_name: str
    image_stem: str
    manifest_metadata: dict[str, Any]


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str
    format: str | None
    metadata: dict[str, Any]


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not resolve the repository root from the current script location.")


def default_input_dir() -> Path:
    return find_repo_root() / "data" / "Monusac" / "tiles_256"


def default_output_root() -> Path:
    return find_repo_root() / "inference" / "benchmarking" / "monusac"


def resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_cpu_threads(worker_count: int) -> None:
    normalized = str(max(1, int(worker_count)))
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(env_name, normalized)


def is_supported_image(path: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False
    lowered_stem = path.stem.lower()
    return not any(lowered_stem.endswith(token) for token in EXCLUDED_NAME_TOKENS)


def _candidate_manifest_paths(input_dir: Path) -> list[Path]:
    return [
        input_dir / "all_patches_dataset.csv",
        input_dir / "dataset.csv",
        input_dir / "patches.csv",
        input_dir.parent / "all_patches_dataset.csv",
    ]


def infer_manifest_path(input_dir: Path) -> Path | None:
    for candidate in _candidate_manifest_paths(input_dir):
        if candidate.exists():
            return candidate
    return None


def _normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _serialize_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_serialize_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return str(value)


def _resolve_manifest_image_path(raw_value: Any, *, manifest_path: Path, input_dir: Path) -> Path | None:
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if not raw_text:
        return None

    candidate = Path(raw_text).expanduser()
    candidate_paths = [
        candidate,
        manifest_path.parent / candidate,
        input_dir / candidate,
        input_dir / candidate.name,
    ]
    for item in candidate_paths:
        resolved = item.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _scan_records(input_dir: Path, *, recursive: bool) -> list[TileRecord]:
    iterator: Iterable[Path]
    if recursive:
        iterator = sorted(path for path in input_dir.rglob("*") if path.is_file())
    else:
        iterator = sorted(path for path in input_dir.glob("*") if path.is_file())

    records: list[TileRecord] = []
    for path in iterator:
        if not is_supported_image(path):
            continue
        relative_path = path.relative_to(input_dir)
        records.append(
            TileRecord(
                source_path=path,
                relative_path=relative_path,
                image_name=path.name,
                image_stem=path.stem,
                manifest_metadata={},
            )
        )
    return records


def _records_from_manifest(
    input_dir: Path,
    manifest_path: Path,
) -> list[TileRecord]:
    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        return []

    image_column = None
    for candidate in ("image_path", "source_image_path", "patch_image_path", "path"):
        if candidate in manifest_df.columns:
            image_column = candidate
            break
    if image_column is None:
        return _scan_records(input_dir, recursive=True)

    records: list[TileRecord] = []
    for row in manifest_df.to_dict(orient="records"):
        source_path = _resolve_manifest_image_path(
            row.get(image_column),
            manifest_path=manifest_path,
            input_dir=input_dir,
        )
        if source_path is None or not is_supported_image(source_path):
            continue
        try:
            relative_path = source_path.relative_to(input_dir)
        except ValueError:
            raw_path = Path(str(row.get(image_column))).expanduser()
            relative_path = raw_path if not raw_path.is_absolute() else Path(source_path.name)

        metadata = {column: _normalize_scalar(value) for column, value in row.items()}
        records.append(
            TileRecord(
                source_path=source_path,
                relative_path=relative_path,
                image_name=source_path.name,
                image_stem=source_path.stem,
                manifest_metadata=metadata,
            )
        )
    return records


def load_tile_records(
    input_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    recursive: bool = True,
) -> tuple[list[TileRecord], Path | None]:
    resolved_input_dir = resolve_path(input_dir)
    if not resolved_input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved_input_dir}")

    resolved_manifest_path = (
        resolve_path(manifest_path)
        if manifest_path is not None
        else infer_manifest_path(resolved_input_dir)
    )

    if resolved_manifest_path is not None and resolved_manifest_path.exists():
        records = _records_from_manifest(resolved_input_dir, resolved_manifest_path)
    else:
        records = _scan_records(resolved_input_dir, recursive=recursive)

    deduped_records: list[TileRecord] = []
    seen_paths: set[Path] = set()
    for record in records:
        if record.source_path in seen_paths:
            continue
        seen_paths.add(record.source_path)
        deduped_records.append(record)

    if not deduped_records:
        raise FileNotFoundError(f"No input image tiles were found under {resolved_input_dir}")

    deduped_records.sort(key=lambda item: str(item.relative_path))
    return deduped_records, resolved_manifest_path


def read_rgb_image(path: str | Path) -> tuple[np.ndarray, ImageInfo]:
    resolved_path = resolve_path(path)
    with Image.open(resolved_path) as handle:
        metadata = {key: _serialize_jsonable(value) for key, value in handle.info.items()}
        image = handle.convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
        info = ImageInfo(
            width=int(image.width),
            height=int(image.height),
            mode=str(handle.mode),
            format=handle.format,
            metadata=metadata,
        )
    return array, info


def ensure_rgb_array(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=-1).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        return np.repeat(array, 3, axis=-1).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] >= 3:
        return array[..., :3].astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {array.shape}")


def count_instances(mask: np.ndarray) -> int:
    mask_array = np.asarray(mask)
    labels = np.unique(mask_array)
    return int(np.count_nonzero(labels))


def output_mask_path(
    output_root: str | Path,
    record: TileRecord,
    *,
    output_extension: str = ".png",
) -> Path:
    resolved_output_root = resolve_path(output_root)
    normalized_extension = output_extension if output_extension.startswith(".") else f".{output_extension}"
    relative_output = record.relative_path.with_suffix(normalized_extension)
    return resolved_output_root / relative_output


def save_instance_mask(mask: np.ndarray, path: str | Path) -> Path:
    resolved_path = resolve_path(path)
    ensure_directory(resolved_path.parent)
    Image.fromarray(np.asarray(mask, dtype=np.uint16), mode="I;16").save(resolved_path)
    return resolved_path


def manifest_row(
    *,
    model_name: str,
    record: TileRecord,
    image_info: ImageInfo,
    mask_path: Path,
    instance_count: int,
    runtime_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(record.manifest_metadata)
    row.update(
        {
            "model_name": model_name,
            "status": "ok",
            "source_image_path": str(record.source_path),
            "relative_image_path": str(record.relative_path),
            "source_image_name": record.image_name,
            "source_image_stem": record.image_stem,
            "predicted_mask_path": str(mask_path),
            "predicted_mask_name": mask_path.name,
            "predicted_mask_relative_path": str(record.relative_path.with_suffix(mask_path.suffix)),
            "image_width": image_info.width,
            "image_height": image_info.height,
            "image_mode": image_info.mode,
            "image_format": image_info.format,
            "image_metadata_json": json.dumps(_serialize_jsonable(image_info.metadata), sort_keys=True),
            "instance_count": int(instance_count),
            "runtime_seconds": float(runtime_seconds),
        }
    )
    if extra:
        row.update({key: _serialize_jsonable(value) for key, value in extra.items()})
    return row


def failure_row(
    *,
    model_name: str,
    record: TileRecord,
    error: Exception | str,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        **record.manifest_metadata,
        "model_name": model_name,
        "status": "failed",
        "source_image_path": str(record.source_path),
        "relative_image_path": str(record.relative_path),
        "source_image_name": record.image_name,
        "source_image_stem": record.image_stem,
        "predicted_mask_path": "",
        "instance_count": -1,
        "runtime_seconds": float(runtime_seconds),
        "error": str(error),
    }


def write_table(rows: list[dict[str, Any]], path: str | Path) -> Path:
    resolved_path = resolve_path(path)
    ensure_directory(resolved_path.parent)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(resolved_path, index=False)
    return resolved_path


def write_model_reports(
    *,
    output_dir: str | Path,
    success_rows: list[dict[str, Any]],
    failure_rows_list: list[dict[str, Any]],
    manifest_name: str = "predictions.csv",
    failures_name: str = "failed.csv",
) -> tuple[Path, Path]:
    resolved_output_dir = ensure_directory(resolve_path(output_dir))
    manifest_path = write_table(success_rows, resolved_output_dir / manifest_name)
    failures_path = write_table(failure_rows_list, resolved_output_dir / failures_name)
    return manifest_path, failures_path


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    normalized_size = max(1, int(size))
    for index in range(0, len(items), normalized_size):
        yield items[index : index + normalized_size]


def centered_pad_to_square(image: np.ndarray, target_size: int, *, fill_value: int = 255) -> tuple[np.ndarray, tuple[int, int]]:
    array = ensure_rgb_array(image)
    height, width = array.shape[:2]
    if height > target_size or width > target_size:
        raise ValueError(
            f"Image shape {array.shape[:2]} exceeds the requested padded size {target_size}."
        )
    canvas = np.full((target_size, target_size, 3), fill_value, dtype=np.uint8)
    y0 = (target_size - height) // 2
    x0 = (target_size - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = array
    return canvas, (y0, x0)


def crop_to_original_extent(mask: np.ndarray, *, offset: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    y0, x0 = offset
    height, width = shape
    return np.asarray(mask)[y0 : y0 + height, x0 : x0 + width]


def select_records(records: list[TileRecord], *, limit: int | None = None) -> list[TileRecord]:
    if limit is None or limit <= 0:
        return list(records)
    return list(records[:limit])


def monotonic_seconds() -> float:
    return time.perf_counter()
