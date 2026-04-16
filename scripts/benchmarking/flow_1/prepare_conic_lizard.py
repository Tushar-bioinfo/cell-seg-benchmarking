from __future__ import annotations

from pathlib import Path

import argparse
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import PROJECT_ROOT, ensure_directory, format_elapsed, log, resolve_path
from dataset_utils import write_mask_png, write_rgb_png

FLOW_NAME = "flow_1.prepare_conic_lizard"
INPUT_DIR = PROJECT_ROOT / "data" / "conic_lizard"
OUTPUT_DIR = None
MANIFEST_NAME = "dataset_manifest.csv"
IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
CLASS_LABELS_SUBDIR = "class_labels"
OVERWRITE = False

_IMAGE_NPY_CANDIDATES = ("images.npy", "Copy of images.npy")
_LABELS_NPY_CANDIDATES = ("labels.npy", "Copy of labels.npy")
_COUNTS_CSV_CANDIDATES = ("counts.csv", "Copy of counts.csv")
_PATCH_INFO_CSV_CANDIDATES = ("patch_info.csv", "Copy of patch_info.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the CoNIC/Lizard array release into paired PNG files plus "
            "a flow_1-compatible dataset_manifest.csv."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--images-npy", type=Path, default=None)
    parser.add_argument("--labels-npy", type=Path, default=None)
    parser.add_argument("--counts-csv", type=Path, default=None)
    parser.add_argument("--patch-info-csv", type=Path, default=None)
    parser.add_argument("--manifest-name", default=MANIFEST_NAME)
    parser.add_argument("--images-subdir", default=IMAGES_SUBDIR)
    parser.add_argument("--masks-subdir", default=MASKS_SUBDIR)
    parser.add_argument("--class-labels-subdir", default=CLASS_LABELS_SUBDIR)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    return parser.parse_args()


def _default_output_dir(input_dir: Path) -> Path:
    return input_dir


def _resolve_existing_file(base_dir: Path, explicit_path: Path | None, candidates: tuple[str, ...], label: str) -> Path:
    if explicit_path is not None:
        resolved = resolve_path(explicit_path)
        if not resolved or not resolved.is_file():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
        return resolved

    for name in candidates:
        candidate = base_dir / name
        if candidate.is_file():
            return candidate.resolve()

    tried = ", ".join(str(base_dir / name) for name in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried: {tried}")


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _read_patch_info(path: Path) -> pd.Series:
    patch_info_df = pd.read_csv(path)
    if patch_info_df.empty:
        raise ValueError(f"Patch info CSV is empty: {path}")
    if "patch_info" in patch_info_df.columns:
        column = "patch_info"
    elif len(patch_info_df.columns) == 1:
        column = str(patch_info_df.columns[0])
    else:
        raise ValueError(
            f"Could not infer patch_info column from {path}. "
            f"Columns: {list(patch_info_df.columns)}"
        )

    patch_info = patch_info_df[column].astype(str).str.strip()
    if (patch_info == "").any():
        raise ValueError(f"Patch info CSV contains empty sample identifiers: {path}")
    if not patch_info.is_unique:
        duplicates = patch_info[patch_info.duplicated()].unique()[:5]
        raise ValueError(f"Patch info values must be unique. Examples: {list(duplicates)}")
    return patch_info


def _validate_counts(path: Path, expected_rows: int) -> pd.DataFrame:
    counts_df = pd.read_csv(path)
    if len(counts_df) != expected_rows:
        raise ValueError(
            f"Counts CSV row count ({len(counts_df)}) does not match expected rows ({expected_rows}): {path}"
        )
    return counts_df


def _validate_arrays(images: np.memmap, labels: np.memmap) -> tuple[int, int, int]:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images.npy must have shape (N, H, W, 3). Got {images.shape}")
    if labels.ndim != 4 or labels.shape[-1] != 2:
        raise ValueError(f"labels.npy must have shape (N, H, W, 2). Got {labels.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images.npy and labels.npy must have the same number of samples. "
            f"Got {images.shape[0]} and {labels.shape[0]}"
        )
    if images.shape[1:3] != labels.shape[1:3]:
        raise ValueError(
            f"images.npy and labels.npy must share HxW. Got {images.shape[1:3]} and {labels.shape[1:3]}"
        )
    return int(images.shape[0]), int(images.shape[1]), int(images.shape[2])


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir) or _default_output_dir(input_dir)
    images_dir = ensure_directory(output_dir / args.images_subdir)
    masks_dir = ensure_directory(output_dir / args.masks_subdir)
    class_labels_dir = ensure_directory(output_dir / args.class_labels_subdir)
    manifest_path = output_dir / args.manifest_name

    images_npy = _resolve_existing_file(input_dir, args.images_npy, _IMAGE_NPY_CANDIDATES, "images.npy")
    labels_npy = _resolve_existing_file(input_dir, args.labels_npy, _LABELS_NPY_CANDIDATES, "labels.npy")
    counts_csv = _resolve_existing_file(input_dir, args.counts_csv, _COUNTS_CSV_CANDIDATES, "counts.csv")
    patch_info_csv = _resolve_existing_file(
        input_dir, args.patch_info_csv, _PATCH_INFO_CSV_CANDIDATES, "patch_info.csv"
    )

    log(FLOW_NAME, "start")
    log(FLOW_NAME, f"input_dir={input_dir}")
    log(FLOW_NAME, f"output_dir={output_dir}")
    log(FLOW_NAME, f"images_npy={images_npy}")
    log(FLOW_NAME, f"labels_npy={labels_npy}")
    log(FLOW_NAME, f"counts_csv={counts_csv}")
    log(FLOW_NAME, f"patch_info_csv={patch_info_csv}")

    images = np.load(images_npy, mmap_mode="r")
    labels = np.load(labels_npy, mmap_mode="r")
    sample_count, image_height, image_width = _validate_arrays(images, labels)
    patch_info = _read_patch_info(patch_info_csv)
    if len(patch_info) != sample_count:
        raise ValueError(
            f"patch_info.csv row count ({len(patch_info)}) does not match array rows ({sample_count})"
        )
    counts_df = _validate_counts(counts_csv, sample_count)
    count_columns = [str(column) for column in counts_df.columns]

    manifest_rows: list[dict[str, object]] = []
    written_count = 0
    skipped_count = 0

    for index in tqdm(range(sample_count), desc="Exporting CoNIC patches"):
        sample_id = patch_info.iat[index]
        image_path = images_dir / f"{sample_id}_image.png"
        mask_path = masks_dir / f"{sample_id}_mask.png"
        class_label_path = class_labels_dir / f"{sample_id}_class_labels.png"

        if image_path.exists() and mask_path.exists() and class_label_path.exists() and not args.overwrite:
            skipped_count += 1
        else:
            write_rgb_png(np.asarray(images[index], dtype=np.uint8), image_path)
            write_mask_png(np.asarray(labels[index, :, :, 0], dtype=np.uint16), mask_path)
            write_mask_png(np.asarray(labels[index, :, :, 1], dtype=np.uint16), class_label_path)
            written_count += 1

        count_values = {
            column: counts_df.iloc[index][column].item()
            if hasattr(counts_df.iloc[index][column], "item")
            else counts_df.iloc[index][column]
            for column in count_columns
        }
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "conic_index": index,
                "image_path": _project_relative(image_path),
                "mask_path": _project_relative(mask_path),
                "class_label_path": _project_relative(class_label_path),
                "image_height": image_height,
                "image_width": image_width,
                "count_total": int(sum(int(count_values[column]) for column in count_columns)),
                **count_values,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    elapsed = time.perf_counter() - started_at
    log(FLOW_NAME, f"samples={sample_count}")
    log(FLOW_NAME, f"written={written_count} skipped={skipped_count}")
    log(FLOW_NAME, f"manifest={manifest_path}")
    log(FLOW_NAME, f"done elapsed={format_elapsed(elapsed)}")


if __name__ == "__main__":
    main()
