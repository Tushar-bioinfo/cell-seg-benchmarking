from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from monusac_visualization_utils import (
    DEFAULT_MIN_INSTANCE_FRACTION,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    load_export_index,
    load_sample_arrays,
    rescale_patch_and_mask,
    resolve_monusac_root,
)

SUPPORTED_FOLDERS = ("all_merged", "kidney_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-rescale exported MoNuSAC image/mask pairs using the same "
            "label-aware logic as scripts/benchmarking/monusac_visualization_utils.py."
        ),
        epilog=(
            "Example:\n"
            "  pixi run python scripts/benchmarking/monusac_rescale_export.py\n\n"
            "By default this reads from MONUSAC_ROOT or data/Monusac and writes a "
            "new MoNuSAC-style export under <data_root>/rescaled/."
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
        help="Output root. Defaults to <data_root>/rescaled.",
    )
    parser.add_argument(
        "--source-magnification",
        type=float,
        default=40.0,
        help="Source magnification for the exported patches. Default: 40.",
    )
    parser.add_argument(
        "--target-magnification",
        type=float,
        default=20.0,
        help="Target magnification for the rescaled patches. Default: 20.",
    )
    parser.add_argument(
        "--min-instance-fraction",
        type=float,
        default=DEFAULT_MIN_INSTANCE_FRACTION,
        help=(
            "Minimum resized binary-mask occupancy required to keep an instance label. "
            f"Default: {DEFAULT_MIN_INSTANCE_FRACTION}."
        ),
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        choices=SUPPORTED_FOLDERS,
        default=list(SUPPORTED_FOLDERS),
        help="Which output folders to generate. Default: all_merged kidney_only.",
    )
    parser.add_argument(
        "--unique-id",
        action="append",
        default=None,
        help=(
            "Restrict processing to one unique_id. Repeat this flag to process more than one."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rescaled files instead of skipping them.",
    )
    return parser.parse_args()


def _ensure_folder_column(export_index: pd.DataFrame) -> pd.DataFrame:
    normalized = export_index.copy()
    if "folder" not in normalized.columns:
        normalized["folder"] = "all_merged"
    return normalized


def _ensure_kidney_only_rows(export_index: pd.DataFrame) -> pd.DataFrame:
    normalized = _ensure_folder_column(export_index)
    if "tissue" not in normalized.columns:
        return normalized

    existing_kidney_only_ids = set(
        normalized.loc[normalized["folder"].eq("kidney_only"), "unique_id"].astype(str)
    )

    kidney_rows = normalized.loc[
        normalized["tissue"].astype(str).str.casefold().eq("kidney")
        & normalized["folder"].eq("all_merged")
    ].copy()
    kidney_rows = kidney_rows.loc[
        ~kidney_rows["unique_id"].astype(str).isin(existing_kidney_only_ids)
    ]

    if kidney_rows.empty:
        return normalized

    kidney_rows["folder"] = "kidney_only"
    return pd.concat([normalized, kidney_rows], ignore_index=True)


def _filter_export_index(
    export_index: pd.DataFrame,
    *,
    folders: list[str],
    unique_ids: list[str] | None,
) -> pd.DataFrame:
    filtered = _ensure_kidney_only_rows(export_index)
    filtered = filtered.loc[filtered["folder"].isin(folders)].copy()

    if unique_ids:
        unique_id_set = set(unique_ids)
        filtered = filtered.loc[filtered["unique_id"].astype(str).isin(unique_id_set)].copy()

    if filtered.empty:
        raise ValueError("No MoNuSAC samples matched the requested filters.")

    return filtered.sort_values(["folder", "unique_id"]).reset_index(drop=True)


def _write_rgb_png(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask_png(instance_mask: np.ndarray, path: Path) -> None:
    Image.fromarray(instance_mask.astype(np.uint16, copy=False)).save(path)


def _build_summary_row(
    sample_row: pd.Series,
    *,
    output_image_path: Path,
    output_mask_path: Path,
    resize_metadata: dict[str, object],
) -> dict[str, object]:
    summary_row = dict(sample_row)
    source_height, source_width = resize_metadata["source_shape"]
    target_height, target_width = resize_metadata["target_shape"]

    summary_row["source_image_path"] = summary_row.get("image_path")
    summary_row["source_mask_path"] = summary_row.get("mask_path")
    summary_row["image_path"] = str(output_image_path)
    summary_row["mask_path"] = str(output_mask_path)
    summary_row["source_height"] = int(source_height)
    summary_row["source_width"] = int(source_width)
    summary_row["target_height"] = int(target_height)
    summary_row["target_width"] = int(target_width)
    summary_row["source_magnification"] = float(resize_metadata["source_magnification"])
    summary_row["target_magnification"] = float(resize_metadata["target_magnification"])
    summary_row["scale_factor"] = float(resize_metadata["scale_factor"])
    summary_row["min_instance_fraction"] = float(resize_metadata["min_instance_fraction"])
    summary_row["original_instance_count"] = int(resize_metadata["original_instance_count"])
    summary_row["resized_instance_count"] = int(resize_metadata["resized_instance_count"])
    summary_row["dropped_instance_count"] = int(resize_metadata["dropped_instance_count"])
    summary_row["dropped_instance_labels"] = ",".join(
        str(label) for label in resize_metadata["dropped_instance_labels"]
    )
    return summary_row


def main() -> None:
    args = parse_args()
    data_root = resolve_monusac_root(data_root=args.data_root, search_from=Path(__file__).resolve())
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (data_root / "rescaled").resolve()
    )

    export_index = load_export_index(data_root=data_root)
    filtered_index = _filter_export_index(
        export_index,
        folders=args.folders,
        unique_ids=args.unique_id,
    )

    for folder in args.folders:
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    print(f"Input root:  {data_root}")
    print(f"Output root: {output_root}")
    print(
        "Rescaling with "
        f"source={args.source_magnification:g}x, "
        f"target={args.target_magnification:g}x, "
        f"min_instance_fraction={args.min_instance_fraction:.3f}"
    )
    print(f"Selected samples: {len(filtered_index)}")

    summary_rows: list[dict[str, object]] = []
    processed_count = 0
    skipped_count = 0

    for _, sample_row in tqdm(
        filtered_index.iterrows(),
        total=len(filtered_index),
        desc="Rescaling MoNuSAC exports",
    ):
        unique_id = str(sample_row["unique_id"])
        folder = str(sample_row["folder"])
        output_folder = output_root / folder
        output_image_path = output_folder / f"{unique_id}{IMAGE_SUFFIX}"
        output_mask_path = output_folder / f"{unique_id}{MASK_SUFFIX}"

        if not args.overwrite and output_image_path.exists() and output_mask_path.exists():
            skipped_count += 1
            summary_row = dict(sample_row)
            summary_row["source_image_path"] = summary_row.get("image_path")
            summary_row["source_mask_path"] = summary_row.get("mask_path")
            summary_row["image_path"] = str(output_image_path)
            summary_row["mask_path"] = str(output_mask_path)
            summary_rows.append(summary_row)
            continue

        image, instance_mask = load_sample_arrays(sample_row, data_root=data_root)
        resized_image, resized_mask, resize_metadata = rescale_patch_and_mask(
            image=image,
            instance_mask=instance_mask,
            source_magnification=args.source_magnification,
            target_magnification=args.target_magnification,
            min_instance_fraction=args.min_instance_fraction,
        )

        _write_rgb_png(resized_image, output_image_path)
        _write_mask_png(resized_mask, output_mask_path)

        summary_rows.append(
            _build_summary_row(
                sample_row,
                output_image_path=output_image_path,
                output_mask_path=output_mask_path,
                resize_metadata=resize_metadata,
            )
        )
        processed_count += 1

    summary_path = output_root / "extraction_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
