from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label as connected_components
from skimage.measure import regionprops
from tqdm import tqdm

REQUIRED_COLUMNS: tuple[str, ...] = ("image_path", "mask_path")
SUPPORTED_MASK_SUFFIXES: set[str] = {".png", ".tif", ".tiff"}
INSTANCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "label_id",
    "area_pixels",
    "perimeter_pixels",
    "equivalent_diameter",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "solidity",
    "extent",
    "orientation_deg",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
    "centroid_row",
    "centroid_col",
    "aspect_ratio",
    "circularity",
    "perimeter_area_ratio",
    "bbox_area",
    "fill_ratio",
)
PATCH_FEATURE_COLUMNS: tuple[str, ...] = (
    "num_objects",
    "total_mask_area",
    "foreground_fraction",
    "mean_area",
    "median_area",
    "std_area",
    "mean_eccentricity",
    "mean_solidity",
    "mean_circularity",
)
PATCH_INTENSITY_COLUMNS: tuple[str, ...] = (
    "image_mean_r",
    "image_mean_g",
    "image_mean_b",
    "image_std_r",
    "image_std_g",
    "image_std_b",
    "foreground_mean_r",
    "foreground_mean_g",
    "foreground_mean_b",
    "foreground_std_r",
    "foreground_std_g",
    "foreground_std_b",
)
FAILURE_COLUMNS: tuple[str, ...] = ("stage", "error_message")


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not resolve the repository root from the current script location.")


PROJECT_ROOT = find_repo_root(Path(__file__).resolve())


class RowProcessingError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message


@dataclass(frozen=True)
class WorkerConfig:
    resolution_roots: tuple[Path, ...]
    include_intensity: bool
    min_area: int | None
    max_objects_per_image: int | None


@dataclass(frozen=True)
class RowProcessingResult:
    ok: bool
    metadata: dict[str, Any]
    input_row_index: int
    instance_rows: list[dict[str, Any]]
    patch_row: dict[str, Any] | None
    object_count: int
    stage: str | None = None
    error_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract instance-level and patch-level morphology features from segmentation masks "
            "listed in a dataset manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to dataset_manifest.csv.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for outputs and logs.")
    parser.add_argument(
        "--level",
        choices=("instance", "patch", "both"),
        default="both",
        help="Which output level(s) to write.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to use.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Multiprocessing chunk size for imap_unordered(). Ignored when workers <= 1.",
    )
    parser.add_argument(
        "--verbosity",
        choices=("quiet", "normal", "debug"),
        default="normal",
        help="Console log verbosity.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately instead of skipping bad rows.",
    )
    parser.add_argument(
        "--include-intensity",
        action="store_true",
        help="Compute patch-level RGB mean/std summaries from image_path in addition to mask features.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help="Optional minimum object area in pixels. Smaller objects are dropped before summaries.",
    )
    parser.add_argument(
        "--max-objects-per-image",
        type=int,
        default=None,
        help="Optional guardrail on retained object count per row after min-area filtering.",
    )
    parser.add_argument(
        "--ram-gb",
        type=float,
        default=None,
        help="Optional memory hint used for logging and HPC guidance only.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.min_area is not None and args.min_area < 1:
        parser.error("--min-area must be >= 1 when provided")
    if args.max_objects_per_image is not None and args.max_objects_per_image < 1:
        parser.error("--max-objects-per-image must be >= 1 when provided")
    return args


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def configure_logger(log_path: Path, verbosity: str) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger(f"morphology.{log_path}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_level = {
        "quiet": logging.WARNING,
        "normal": logging.INFO,
        "debug": logging.DEBUG,
    }[verbosity]
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


class CsvAppender:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        prepared_rows = list(rows)
        if not prepared_rows:
            return
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            for row in prepared_rows:
                writer.writerow({field: row.get(field) for field in self.fieldnames})


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing required columns: {', '.join(missing)}"
        )
    return frame.reset_index(drop=False).rename(columns={"index": "input_row_index"})


def build_resolution_roots(manifest_path: Path) -> tuple[Path, ...]:
    candidates = (manifest_path.parent, PROJECT_ROOT, Path.cwd())
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        roots.append(resolved)
        seen.add(resolved)
    return tuple(roots)


def resolve_existing_file(raw_value: Any, *, roots: Sequence[Path]) -> Path | None:
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if not raw_text:
        return None

    candidate = Path(raw_text).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        return resolved if resolved.is_file() else None

    checked: set[Path] = set()
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _as_2d_array(mask: Any, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    return array


def _safely_reduce_loaded_mask(array: np.ndarray, path: Path) -> np.ndarray:
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
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
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
        if np.array_equal(rgb[0, ...], rgb[1, ...]) and np.array_equal(rgb[0, ...], rgb[2, ...]):
            if array.shape[0] == 4:
                alpha_channel = array[3, ...]
                if not np.all(alpha_channel == alpha_channel.flat[0]):
                    raise ValueError(
                        "Mask file has a channel-first RGBA layout with a non-constant alpha channel, "
                        f"which is ambiguous for label masks: {path}"
                    )
            return np.asarray(rgb[0, ...])

    raise ValueError(
        "Mask file must be 2D. Extra channels are only accepted when they are singleton "
        f"dimensions or replicated grayscale channels. Loaded {path} with shape {array.shape}."
    )


def load_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MASK_SUFFIXES:
        raise ValueError(
            "Mask file must be a PNG or TIFF image, "
            f"got '{suffix or '<no suffix>'}' for {path}"
        )

    try:
        loaded = iio.imread(path)
    except Exception as exc:
        raise ValueError(f"Failed to read mask file {path}: {exc}") from exc

    array = np.asarray(loaded)
    if array.size == 0:
        raise ValueError(f"Mask file is empty: {path}")
    return _safely_reduce_loaded_mask(array, path)


def normalize_instance_mask(mask: np.ndarray) -> np.ndarray:
    array = _as_2d_array(mask, "mask")

    if np.issubdtype(array.dtype, np.bool_):
        normalized = array.astype(np.int64, copy=False)
    elif np.issubdtype(array.dtype, np.integer):
        normalized = array.astype(np.int64, copy=False)
    elif np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("mask must contain only finite values")
        if not np.all(array == np.floor(array)):
            raise ValueError("mask must contain integer-valued labels")
        normalized = array.astype(np.int64)
    else:
        raise ValueError(f"mask must contain numeric labels, got dtype {array.dtype}")

    if np.any(normalized < 0):
        raise ValueError("mask must not contain negative labels")
    return normalized


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            return np.asarray(image, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed to read image file {path}: {exc}") from exc


def infer_labeled_mask(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    positive_labels = np.unique(mask[mask > 0])
    if positive_labels.size <= 1:
        return connected_components(mask > 0, connectivity=2).astype(np.int64, copy=False), True
    return mask.astype(np.int64, copy=False), False


def nan_summary(values: Sequence[float], reducer: str) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return float("nan")
    if reducer == "mean":
        return float(np.mean(finite_values))
    if reducer == "median":
        return float(np.median(finite_values))
    if reducer == "std":
        return float(np.std(finite_values))
    raise ValueError(f"Unsupported reducer: {reducer}")


def compute_patch_intensity_features(image_rgb: np.ndarray, retained_mask: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    channel_names = ("r", "g", "b")
    for channel_index, channel_name in enumerate(channel_names):
        channel = image_rgb[..., channel_index]
        features[f"image_mean_{channel_name}"] = float(np.mean(channel))
        features[f"image_std_{channel_name}"] = float(np.std(channel))

        foreground_pixels = channel[retained_mask]
        if foreground_pixels.size == 0:
            features[f"foreground_mean_{channel_name}"] = float("nan")
            features[f"foreground_std_{channel_name}"] = float("nan")
        else:
            features[f"foreground_mean_{channel_name}"] = float(np.mean(foreground_pixels))
            features[f"foreground_std_{channel_name}"] = float(np.std(foreground_pixels))
    return features


def compute_instance_rows(
    labeled_mask: np.ndarray,
    metadata: dict[str, Any],
    *,
    min_area: int | None,
    max_objects_per_image: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    retained_rows: list[dict[str, Any]] = []
    kept_labels: list[int] = []
    region_list = list(regionprops(labeled_mask))

    for region in region_list:
        area = int(region.area)
        if min_area is not None and area < min_area:
            continue

        perimeter = float(region.perimeter)
        equivalent_diameter = float(math.sqrt((4.0 * area) / math.pi)) if area > 0 else float("nan")
        major_axis_length = float(region.major_axis_length)
        minor_axis_length = float(region.minor_axis_length)
        min_row, min_col, max_row, max_col = region.bbox
        bbox_area = int((max_row - min_row) * (max_col - min_col))
        aspect_ratio = float(major_axis_length / minor_axis_length) if minor_axis_length > 0 else float("nan")
        circularity = float((4.0 * math.pi * area) / (perimeter * perimeter)) if perimeter > 0 else float("nan")
        perimeter_area_ratio = float(perimeter / area) if area > 0 else float("nan")
        fill_ratio = float(area / bbox_area) if bbox_area > 0 else float("nan")

        row = {
            **metadata,
            "label_id": int(region.label),
            "area_pixels": area,
            "perimeter_pixels": perimeter,
            "equivalent_diameter": equivalent_diameter,
            "major_axis_length": major_axis_length,
            "minor_axis_length": minor_axis_length,
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "extent": float(region.extent),
            "orientation_deg": float(math.degrees(region.orientation)),
            "bbox_min_row": int(min_row),
            "bbox_min_col": int(min_col),
            "bbox_max_row": int(max_row),
            "bbox_max_col": int(max_col),
            "centroid_row": float(region.centroid[0]),
            "centroid_col": float(region.centroid[1]),
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "perimeter_area_ratio": perimeter_area_ratio,
            "bbox_area": bbox_area,
            "fill_ratio": fill_ratio,
        }
        retained_rows.append(row)
        kept_labels.append(int(region.label))

    if max_objects_per_image is not None and len(retained_rows) > max_objects_per_image:
        raise RowProcessingError(
            "feature_extraction",
            "retained object count "
            f"{len(retained_rows)} exceeds --max-objects-per-image={max_objects_per_image}",
        )

    retained_mask = np.isin(labeled_mask, kept_labels) if kept_labels else np.zeros_like(labeled_mask, dtype=bool)
    patch_row = build_patch_summary(metadata, retained_rows, labeled_mask.shape)
    return retained_rows, patch_row, retained_mask


def build_patch_summary(
    metadata: dict[str, Any],
    instance_rows: Sequence[dict[str, Any]],
    mask_shape: tuple[int, int],
) -> dict[str, Any]:
    areas = [float(row["area_pixels"]) for row in instance_rows]
    eccentricities = [float(row["eccentricity"]) for row in instance_rows]
    solidities = [float(row["solidity"]) for row in instance_rows]
    circularities = [float(row["circularity"]) for row in instance_rows]
    total_mask_area = int(sum(int(row["area_pixels"]) for row in instance_rows))
    total_pixels = int(mask_shape[0] * mask_shape[1])
    patch_row = {
        **metadata,
        "num_objects": int(len(instance_rows)),
        "total_mask_area": total_mask_area,
        "foreground_fraction": float(total_mask_area / total_pixels) if total_pixels > 0 else float("nan"),
        "mean_area": nan_summary(areas, "mean"),
        "median_area": nan_summary(areas, "median"),
        "std_area": nan_summary(areas, "std"),
        "mean_eccentricity": nan_summary(eccentricities, "mean"),
        "mean_solidity": nan_summary(solidities, "mean"),
        "mean_circularity": nan_summary(circularities, "mean"),
    }
    return patch_row


def extract_row_features(row_data: dict[str, Any], config: WorkerConfig) -> RowProcessingResult:
    input_row_index = int(row_data["input_row_index"])
    metadata = {key: normalize_scalar(value) for key, value in row_data.items()}

    try:
        mask_path = resolve_existing_file(row_data.get("mask_path"), roots=config.resolution_roots)
        if mask_path is None:
            raise RowProcessingError(
                "path_resolution",
                f"Could not resolve mask_path={row_data.get('mask_path')!r}",
            )

        try:
            raw_mask = load_mask(mask_path)
        except Exception as exc:
            raise RowProcessingError("mask_decode", str(exc)) from exc

        try:
            normalized_mask = normalize_instance_mask(raw_mask)
            labeled_mask, _ = infer_labeled_mask(normalized_mask)
        except RowProcessingError:
            raise
        except Exception as exc:
            raise RowProcessingError("mask_validation", str(exc)) from exc

        try:
            instance_rows, patch_row, retained_mask = compute_instance_rows(
                labeled_mask,
                metadata,
                min_area=config.min_area,
                max_objects_per_image=config.max_objects_per_image,
            )
        except RowProcessingError:
            raise
        except Exception as exc:
            raise RowProcessingError("feature_extraction", str(exc)) from exc

        if config.include_intensity:
            image_path = resolve_existing_file(row_data.get("image_path"), roots=config.resolution_roots)
            if image_path is None:
                raise RowProcessingError(
                    "path_resolution",
                    f"Could not resolve image_path={row_data.get('image_path')!r}",
                )
            try:
                image_rgb = load_rgb_image(image_path)
            except Exception as exc:
                raise RowProcessingError("image_decode", str(exc)) from exc

            if image_rgb.shape[:2] != labeled_mask.shape:
                raise RowProcessingError(
                    "feature_extraction",
                    "image and mask shapes do not match: "
                    f"image={image_rgb.shape[:2]} mask={labeled_mask.shape}",
                )

            patch_row.update(compute_patch_intensity_features(image_rgb, retained_mask))

        return RowProcessingResult(
            ok=True,
            metadata=metadata,
            input_row_index=input_row_index,
            instance_rows=instance_rows,
            patch_row=patch_row,
            object_count=len(instance_rows),
        )
    except RowProcessingError as exc:
        return RowProcessingResult(
            ok=False,
            metadata=metadata,
            input_row_index=input_row_index,
            instance_rows=[],
            patch_row=None,
            object_count=0,
            stage=exc.stage,
            error_message=exc.message,
        )


def iter_results(
    row_records: list[dict[str, Any]],
    *,
    config: WorkerConfig,
    workers: int,
    batch_size: int,
) -> Iterable[RowProcessingResult]:
    if workers <= 1:
        for row_data in row_records:
            yield extract_row_features(row_data, config)
        return

    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(
            _extract_row_features_for_pool,
            ((row_data, config) for row_data in row_records),
            chunksize=batch_size,
        )
        yield from iterator


def _extract_row_features_for_pool(payload: tuple[dict[str, Any], WorkerConfig]) -> RowProcessingResult:
    row_data, config = payload
    return extract_row_features(row_data, config)


def build_failure_row(result: RowProcessingResult) -> dict[str, Any]:
    return {
        **result.metadata,
        "stage": result.stage,
        "error_message": result.error_message,
    }


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:d}m {remaining_seconds:0.1f}s"
    if minutes:
        return f"{minutes:d}m {remaining_seconds:0.1f}s"
    return f"{remaining_seconds:0.1f}s"


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    manifest_path = resolve_path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    outdir = ensure_directory(resolve_path(args.outdir))
    logger = configure_logger(outdir / "run.log", args.verbosity)
    logger.info("starting manifest=%s outdir=%s", manifest_path, outdir)
    logger.info(
        "options level=%s workers=%s batch_size=%s include_intensity=%s min_area=%s "
        "max_objects_per_image=%s ram_gb=%s fail_fast=%s",
        args.level,
        args.workers,
        args.batch_size,
        args.include_intensity,
        args.min_area,
        args.max_objects_per_image,
        args.ram_gb,
        args.fail_fast,
    )

    load_started_at = time.perf_counter()
    manifest_df = load_manifest(manifest_path)
    manifest_columns = [column for column in manifest_df.columns if column != "input_row_index"]
    row_records = manifest_df.to_dict(orient="records")
    resolution_roots = build_resolution_roots(manifest_path)
    logger.info(
        "manifest_loaded rows=%s elapsed=%s resolution_roots=%s",
        len(manifest_df),
        format_elapsed(time.perf_counter() - load_started_at),
        ", ".join(str(path) for path in resolution_roots),
    )

    instance_writer = None
    patch_writer = None
    if args.level in {"instance", "both"}:
        instance_writer = CsvAppender(
            outdir / "instance_features.csv",
            [*manifest_columns, "input_row_index", *INSTANCE_FEATURE_COLUMNS],
        )
    if args.level in {"patch", "both"}:
        patch_columns = [*PATCH_FEATURE_COLUMNS]
        if args.include_intensity:
            patch_columns.extend(PATCH_INTENSITY_COLUMNS)
        patch_writer = CsvAppender(
            outdir / "patch_features.csv",
            [*manifest_columns, "input_row_index", *patch_columns],
        )
    failure_writer = CsvAppender(
        outdir / "failed_rows.csv",
        [*manifest_columns, "input_row_index", *FAILURE_COLUMNS],
    )

    summary_path = outdir / "processing_summary.json"
    summary: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "outdir": str(outdir),
        "rows_total": int(len(row_records)),
        "rows_processed": 0,
        "rows_skipped": 0,
        "total_objects_found": 0,
        "runtime_seconds": 0.0,
        "average_time_per_row_seconds": 0.0,
        "level": args.level,
        "include_intensity": bool(args.include_intensity),
        "workers": int(args.workers),
        "batch_size": int(args.batch_size),
        "min_area": args.min_area,
        "max_objects_per_image": args.max_objects_per_image,
        "ram_gb_hint": args.ram_gb,
        "status": "running",
    }

    processing_started_at = time.perf_counter()
    try:
        results = iter_results(
            row_records,
            config=WorkerConfig(
                resolution_roots=resolution_roots,
                include_intensity=bool(args.include_intensity),
                min_area=args.min_area,
                max_objects_per_image=args.max_objects_per_image,
            ),
            workers=args.workers,
            batch_size=args.batch_size,
        )

        for result in tqdm(
            results,
            total=len(row_records),
            disable=args.verbosity == "quiet",
            desc="morphology",
        ):
            if result.ok:
                summary["rows_processed"] += 1
                summary["total_objects_found"] += int(result.object_count)
                if instance_writer is not None:
                    instance_writer.write_rows(result.instance_rows)
                if patch_writer is not None and result.patch_row is not None:
                    patch_writer.write_rows([result.patch_row])
                if args.verbosity == "debug":
                    logger.debug(
                        "processed input_row_index=%s objects=%s",
                        result.input_row_index,
                        result.object_count,
                    )
                continue

            summary["rows_skipped"] += 1
            failure_writer.write_rows([build_failure_row(result)])
            logger.warning(
                "skipping input_row_index=%s stage=%s error=%s",
                result.input_row_index,
                result.stage,
                result.error_message,
            )
            if args.fail_fast:
                raise RuntimeError(
                    f"Fail-fast triggered at row {result.input_row_index} during {result.stage}: "
                    f"{result.error_message}"
                )

        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["fatal_error"] = str(exc)
        logger.exception("workflow failed: %s", exc)
        raise
    finally:
        runtime_seconds = time.perf_counter() - started_at
        processing_seconds = time.perf_counter() - processing_started_at
        total_rows = int(summary["rows_total"])
        summary["runtime_seconds"] = float(runtime_seconds)
        summary["average_time_per_row_seconds"] = float(runtime_seconds / total_rows) if total_rows else 0.0
        summary["processing_elapsed_seconds"] = float(processing_seconds)
        atomic_write_json(summary_path, summary)
        logger.info(
            "finished status=%s rows_processed=%s rows_skipped=%s total_objects=%s runtime=%s",
            summary["status"],
            summary["rows_processed"],
            summary["rows_skipped"],
            summary["total_objects_found"],
            format_elapsed(runtime_seconds),
        )


if __name__ == "__main__":
    main()
