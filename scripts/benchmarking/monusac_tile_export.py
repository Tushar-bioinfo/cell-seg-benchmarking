from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from monusac_visualization_utils import (
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    load_export_index,
    load_sample_arrays,
    resolve_monusac_root,
)

SUPPORTED_FOLDERS = ("all_merged", "kidney_only")
DEFAULT_PATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile exported MoNuSAC RGB/mask pairs into fixed-size patches and "
            "write per-image plus global manifests."
        ),
        epilog=(
            "Example:\n"
            "  pixi run python scripts/benchmarking/monusac_tile_export.py\n\n"
            "Default behavior keeps only full non-overlapping patches. An image "
            "smaller than the patch size will yield zero patches."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root containing the exported MoNuSAC files. Defaults to MONUSAC_ROOT "
            "or the repo's data/Monusac directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <data_root>/tiles_<patch_size>.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size in pixels. Default: {DEFAULT_PATCH_SIZE}.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Patch stride in pixels. Defaults to the patch size.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        choices=SUPPORTED_FOLDERS,
        default=["all_merged"],
        help="Which exported folders to tile. Default: all_merged.",
    )
    parser.add_argument(
        "--unique-id",
        action="append",
        default=None,
        help="Restrict processing to one unique_id. Repeat this flag to process more than one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing patch files instead of skipping them.",
    )
    return parser.parse_args()


def _ensure_folder_column(export_index: pd.DataFrame) -> pd.DataFrame:
    normalized = export_index.copy()
    if "folder" not in normalized.columns:
        normalized["folder"] = "all_merged"
    return normalized


def _filter_export_index(
    export_index: pd.DataFrame,
    *,
    folders: list[str],
    unique_ids: list[str] | None,
) -> pd.DataFrame:
    filtered = _ensure_folder_column(export_index)
    filtered = filtered.loc[filtered["folder"].isin(folders)].copy()

    if unique_ids:
        unique_id_set = set(unique_ids)
        filtered = filtered.loc[filtered["unique_id"].astype(str).isin(unique_id_set)].copy()

    if filtered.empty:
        raise ValueError("No MoNuSAC samples matched the requested filters.")

    return filtered.sort_values(["folder", "unique_id"]).reset_index(drop=True)


def _full_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    return list(range(0, length - patch_size + 1, stride))


def _write_rgb_png(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask_png(instance_mask: np.ndarray, path: Path) -> None:
    Image.fromarray(instance_mask.astype(np.uint16, copy=False)).save(path)


def _patch_descriptor(x0: int, y0: int) -> str:
    return f"{x0:05d}x_{y0:05d}y"


def _build_patch_row(
    sample_row: pd.Series,
    *,
    folder: str,
    rel_sample_dir: Path,
    descriptor: str,
    patch_index: int,
    patch_row: int,
    patch_col: int,
    x0: int,
    y0: int,
    patch_size: int,
    stride: int,
    source_height: int,
    source_width: int,
) -> dict[str, object]:
    row = dict(sample_row)
    unique_id = str(sample_row["unique_id"])

    row["folder"] = folder
    row["source_image_path"] = row.pop("image_path", None)
    row["source_mask_path"] = row.pop("mask_path", None)
    row["patch_id"] = f"{unique_id}.{descriptor}"
    row["patch_index"] = int(patch_index)
    row["patch_row"] = int(patch_row)
    row["patch_col"] = int(patch_col)
    row["x0"] = int(x0)
    row["y0"] = int(y0)
    row["x1"] = int(x0 + patch_size)
    row["y1"] = int(y0 + patch_size)
    row["patch_size"] = int(patch_size)
    row["stride"] = int(stride)
    row["source_height"] = int(source_height)
    row["source_width"] = int(source_width)
    row["image_path"] = str(rel_sample_dir / f"{descriptor}{IMAGE_SUFFIX}")
    row["mask_path"] = str(rel_sample_dir / f"{descriptor}{MASK_SUFFIX}")
    return row


def _build_summary_row(
    sample_row: pd.Series,
    *,
    folder: str,
    rel_sample_dir: Path,
    patch_size: int,
    stride: int,
    source_height: int,
    source_width: int,
    patch_rows: int,
    patch_cols: int,
    patch_count: int,
) -> dict[str, object]:
    row = dict(sample_row)
    row["folder"] = folder
    row["source_image_path"] = row.pop("image_path", None)
    row["source_mask_path"] = row.pop("mask_path", None)
    row["patches_dir"] = str(rel_sample_dir)
    row["dataset_csv"] = str(rel_sample_dir / "dataset.csv")
    row["patch_size"] = int(patch_size)
    row["stride"] = int(stride)
    row["source_height"] = int(source_height)
    row["source_width"] = int(source_width)
    row["patch_rows"] = int(patch_rows)
    row["patch_cols"] = int(patch_cols)
    row["patch_count"] = int(patch_count)
    return row


def _patch_manifest_columns(source_columns: list[str]) -> list[str]:
    passthrough = [column for column in source_columns if column not in {"image_path", "mask_path"}]
    return [
        *passthrough,
        "source_image_path",
        "source_mask_path",
        "patch_id",
        "patch_index",
        "patch_row",
        "patch_col",
        "x0",
        "y0",
        "x1",
        "y1",
        "patch_size",
        "stride",
        "source_height",
        "source_width",
        "image_path",
        "mask_path",
    ]


def _summary_columns(source_columns: list[str]) -> list[str]:
    passthrough = [column for column in source_columns if column not in {"image_path", "mask_path"}]
    return [
        *passthrough,
        "source_image_path",
        "source_mask_path",
        "patches_dir",
        "dataset_csv",
        "patch_size",
        "stride",
        "source_height",
        "source_width",
        "patch_rows",
        "patch_cols",
        "patch_count",
    ]


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be > 0.")

    stride = args.patch_size if args.stride is None else args.stride
    if stride <= 0:
        raise ValueError("--stride must be > 0.")

    data_root = resolve_monusac_root(data_root=args.data_root, search_from=Path(__file__).resolve())
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (data_root / f"tiles_{args.patch_size}").resolve()
    )

    export_index = load_export_index(data_root=data_root)
    filtered_index = _filter_export_index(
        export_index,
        folders=args.folders,
        unique_ids=args.unique_id,
    )
    patch_columns = _patch_manifest_columns(list(filtered_index.columns))
    summary_columns = _summary_columns(list(filtered_index.columns))

    for folder in args.folders:
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    print(f"Input root:   {data_root}")
    print(f"Output root:  {output_root}")
    print(f"Patch size:   {args.patch_size}")
    print(f"Stride:       {stride}")
    print(f"Selected rows:{len(filtered_index)}")

    all_patch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    written_patch_count = 0
    skipped_patch_count = 0

    for _, sample_row in tqdm(
        filtered_index.iterrows(),
        total=len(filtered_index),
        desc="Tiling MoNuSAC exports",
    ):
        unique_id = str(sample_row["unique_id"])
        folder = str(sample_row["folder"])
        rel_sample_dir = Path(folder) / unique_id
        sample_output_dir = output_root / rel_sample_dir
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        image, instance_mask = load_sample_arrays(sample_row, data_root=data_root)
        source_height, source_width = image.shape[:2]
        y_starts = _full_patch_starts(source_height, args.patch_size, stride)
        x_starts = _full_patch_starts(source_width, args.patch_size, stride)

        patch_rows_for_sample: list[dict[str, object]] = []
        patch_index = 0

        for patch_row, y0 in enumerate(y_starts):
            for patch_col, x0 in enumerate(x_starts):
                descriptor = _patch_descriptor(x0, y0)
                patch_image = image[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
                patch_mask = instance_mask[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]

                patch_row_record = _build_patch_row(
                    sample_row,
                    folder=folder,
                    rel_sample_dir=rel_sample_dir,
                    descriptor=descriptor,
                    patch_index=patch_index,
                    patch_row=patch_row,
                    patch_col=patch_col,
                    x0=x0,
                    y0=y0,
                    patch_size=args.patch_size,
                    stride=stride,
                    source_height=source_height,
                    source_width=source_width,
                )
                patch_rows_for_sample.append(patch_row_record)
                all_patch_rows.append(patch_row_record)

                patch_image_path = output_root / patch_row_record["image_path"]
                patch_mask_path = output_root / patch_row_record["mask_path"]
                if not args.overwrite and patch_image_path.exists() and patch_mask_path.exists():
                    skipped_patch_count += 1
                else:
                    _write_rgb_png(patch_image, patch_image_path)
                    _write_mask_png(patch_mask, patch_mask_path)
                    written_patch_count += 1

                patch_index += 1

        sample_dataset_path = sample_output_dir / "dataset.csv"
        pd.DataFrame(patch_rows_for_sample, columns=patch_columns).to_csv(sample_dataset_path, index=False)

        summary_rows.append(
            _build_summary_row(
                sample_row,
                folder=folder,
                rel_sample_dir=rel_sample_dir,
                patch_size=args.patch_size,
                stride=stride,
                source_height=source_height,
                source_width=source_width,
                patch_rows=len(y_starts),
                patch_cols=len(x_starts),
                patch_count=len(patch_rows_for_sample),
            )
        )

    all_patches_path = output_root / "all_patches_dataset.csv"
    image_summary_path = output_root / "image_patch_summary.csv"
    pd.DataFrame(all_patch_rows, columns=patch_columns).to_csv(all_patches_path, index=False)
    pd.DataFrame(summary_rows, columns=summary_columns).to_csv(image_summary_path, index=False)

    print(f"Patch files written: {written_patch_count}")
    print(f"Patch files skipped: {skipped_patch_count}")
    print(f"Patch manifest:      {all_patches_path}")
    print(f"Image summary:       {image_summary_path}")


if __name__ == "__main__":
    main()
