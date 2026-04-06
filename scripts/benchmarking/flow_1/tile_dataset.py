from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = PROJECT_ROOT / "data" / "benchmark_input"
INPUT_MANIFEST = None
OUTPUT_ROOT = None
IMAGES_SUBDIR = None
MASKS_SUBDIR = None
PAIR_MODE = "suffix"
IMAGE_SUFFIX_TOKEN = "_image"
MASK_SUFFIX_TOKEN = "_mask"
IMAGE_EXTENSIONS = (".png",)
MASK_EXTENSIONS = (".png", ".tif", ".tiff")
RECURSIVE_INPUT = True
PATCH_SIZE = 256
STRIDE = None
OVERWRITE = False
OUTPUT_IMAGE_SUFFIX_TOKEN = "_image"
OUTPUT_MASK_SUFFIX_TOKEN = "_mask"

import argparse
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import ensure_directory, format_elapsed, log, resolve_path
from dataset_utils import (
    collect_dataset_samples,
    load_instance_mask,
    load_rgb_image,
    normalize_extensions,
    prepare_passthrough_metadata,
    write_mask_png,
    write_rgb_png,
)

FLOW_NAME = "flow_1.tile_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile paired image/mask samples from a generic dataset root and "
            "write per-sample plus global manifests."
        )
    )
    parser.add_argument("--in", "--input-root", dest="input_root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--out", "--output-root", dest="output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", "--input-manifest", dest="input_manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--images-subdir", default=IMAGES_SUBDIR)
    parser.add_argument("--masks-subdir", default=MASKS_SUBDIR)
    parser.add_argument("--pair-mode", choices=("suffix", "stem"), default=PAIR_MODE)
    parser.add_argument("--image-suffix-token", default=IMAGE_SUFFIX_TOKEN)
    parser.add_argument("--mask-suffix-token", default=MASK_SUFFIX_TOKEN)
    parser.add_argument("--image-exts", nargs="+", default=list(IMAGE_EXTENSIONS))
    parser.add_argument("--mask-exts", nargs="+", default=list(MASK_EXTENSIONS))
    parser.add_argument("--non-recursive", action="store_true", default=not RECURSIVE_INPUT)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    return parser.parse_args()


def _default_output_root(input_root: Path, patch_size: int) -> Path:
    return input_root / f"tiles_{patch_size}"


def _full_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    return list(range(0, length - patch_size + 1, stride))


def _patch_descriptor(x0: int, y0: int) -> str:
    return f"{x0:05d}x_{y0:05d}y"


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be > 0.")
    stride = args.patch_size if args.stride is None else args.stride
    if stride <= 0:
        raise ValueError("--stride must be > 0.")

    started_at = time.perf_counter()
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root) or _default_output_root(input_root, args.patch_size)
    image_extensions = normalize_extensions(args.image_exts, defaults=IMAGE_EXTENSIONS)
    mask_extensions = normalize_extensions(args.mask_exts, defaults=MASK_EXTENSIONS)
    samples = collect_dataset_samples(
        input_root,
        manifest_path=resolve_path(args.input_manifest),
        images_subdir=args.images_subdir,
        masks_subdir=args.masks_subdir,
        pair_mode=args.pair_mode,
        image_suffix_token=args.image_suffix_token,
        mask_suffix_token=args.mask_suffix_token,
        image_extensions=image_extensions,
        mask_extensions=mask_extensions,
        recursive=not args.non_recursive,
        sample_ids=args.sample_id,
    )

    ensure_directory(output_root)
    log(FLOW_NAME, "start")
    log(FLOW_NAME, f"input_root={input_root}")
    log(FLOW_NAME, f"output_root={output_root}")
    log(FLOW_NAME, f"manifest={resolve_path(args.input_manifest) if args.input_manifest else 'scan/auto'}")
    log(FLOW_NAME, f"samples={len(samples)} patch_size={args.patch_size} stride={stride}")

    patch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    written_patch_count = 0
    skipped_patch_count = 0

    patch_reserved = {
        "sample_id",
        "sample_image_path",
        "sample_mask_path",
        "sample_image_relative_path",
        "sample_mask_relative_path",
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
    }
    summary_reserved = {
        "sample_id",
        "sample_image_path",
        "sample_mask_path",
        "sample_image_relative_path",
        "sample_mask_relative_path",
        "patches_dir",
        "dataset_csv",
        "patch_size",
        "stride",
        "source_height",
        "source_width",
        "patch_rows",
        "patch_cols",
        "patch_count",
    }
    patch_columns: list[str] = [
        "sample_id",
        "sample_image_path",
        "sample_mask_path",
        "sample_image_relative_path",
        "sample_mask_relative_path",
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
    summary_columns: list[str] = [
        "sample_id",
        "sample_image_path",
        "sample_mask_path",
        "sample_image_relative_path",
        "sample_mask_relative_path",
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

    def _extend_columns(columns: list[str], extra_keys: list[str]) -> None:
        for key in extra_keys:
            if key not in columns:
                columns.append(key)

    for sample in tqdm(samples, desc="Tiling samples"):
        image = load_rgb_image(sample.image_path)
        mask = load_instance_mask(sample.mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {sample.sample_id}: "
                f"image={image.shape[:2]} mask={mask.shape}"
            )

        source_height, source_width = image.shape[:2]
        y_starts = _full_patch_starts(source_height, args.patch_size, stride)
        x_starts = _full_patch_starts(source_width, args.patch_size, stride)
        sample_output_rel_dir = Path("samples") / sample.sample_id
        sample_output_dir = output_root / sample_output_rel_dir
        ensure_directory(sample_output_dir)
        sample_patch_rows: list[dict[str, object]] = []
        patch_index = 0

        patch_passthrough = prepare_passthrough_metadata(sample.metadata, reserved_names=patch_reserved)
        summary_passthrough = prepare_passthrough_metadata(sample.metadata, reserved_names=summary_reserved)
        _extend_columns(patch_columns, list(patch_passthrough))
        _extend_columns(summary_columns, list(summary_passthrough))

        for patch_row_index, y0 in enumerate(y_starts):
            for patch_col_index, x0 in enumerate(x_starts):
                descriptor = _patch_descriptor(x0, y0)
                patch_image = image[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
                patch_mask = mask[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
                patch_image_rel = sample_output_rel_dir / f"{descriptor}{OUTPUT_IMAGE_SUFFIX_TOKEN}.png"
                patch_mask_rel = sample_output_rel_dir / f"{descriptor}{OUTPUT_MASK_SUFFIX_TOKEN}.png"
                patch_image_path = output_root / patch_image_rel
                patch_mask_path = output_root / patch_mask_rel

                if patch_image_path.exists() and patch_mask_path.exists() and not args.overwrite:
                    skipped_patch_count += 1
                else:
                    write_rgb_png(patch_image, patch_image_path)
                    write_mask_png(patch_mask.astype(np.uint16, copy=False), patch_mask_path)
                    written_patch_count += 1

                patch_row = {
                    **patch_passthrough,
                    "sample_id": sample.sample_id,
                    "sample_image_path": str(sample.image_path),
                    "sample_mask_path": str(sample.mask_path),
                    "sample_image_relative_path": sample.relative_image_path.as_posix(),
                    "sample_mask_relative_path": sample.relative_mask_path.as_posix(),
                    "patch_id": f"{sample.sample_id}.{descriptor}",
                    "patch_index": int(patch_index),
                    "patch_row": int(patch_row_index),
                    "patch_col": int(patch_col_index),
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x0 + args.patch_size),
                    "y1": int(y0 + args.patch_size),
                    "patch_size": int(args.patch_size),
                    "stride": int(stride),
                    "source_height": int(source_height),
                    "source_width": int(source_width),
                    "image_path": patch_image_rel.as_posix(),
                    "mask_path": patch_mask_rel.as_posix(),
                }
                patch_rows.append(patch_row)
                sample_patch_rows.append(patch_row)
                patch_index += 1

        sample_dataset_path = sample_output_dir / "dataset.csv"
        pd.DataFrame(sample_patch_rows, columns=patch_columns).to_csv(sample_dataset_path, index=False)
        summary_rows.append(
            {
                **summary_passthrough,
                "sample_id": sample.sample_id,
                "sample_image_path": str(sample.image_path),
                "sample_mask_path": str(sample.mask_path),
                "sample_image_relative_path": sample.relative_image_path.as_posix(),
                "sample_mask_relative_path": sample.relative_mask_path.as_posix(),
                "patches_dir": sample_output_rel_dir.as_posix(),
                "dataset_csv": (sample_output_rel_dir / "dataset.csv").as_posix(),
                "patch_size": int(args.patch_size),
                "stride": int(stride),
                "source_height": int(source_height),
                "source_width": int(source_width),
                "patch_rows": int(len(y_starts)),
                "patch_cols": int(len(x_starts)),
                "patch_count": int(len(sample_patch_rows)),
            }
        )

    all_patches_path = output_root / "all_patches_dataset.csv"
    summary_path = output_root / "sample_patch_summary.csv"
    pd.DataFrame(patch_rows, columns=patch_columns).to_csv(all_patches_path, index=False)
    pd.DataFrame(summary_rows, columns=summary_columns).to_csv(summary_path, index=False)

    elapsed = time.perf_counter() - started_at
    log(FLOW_NAME, f"patches_written={written_patch_count} patches_skipped={skipped_patch_count}")
    log(FLOW_NAME, f"patch_manifest={all_patches_path}")
    log(FLOW_NAME, f"sample_summary={summary_path}")
    log(FLOW_NAME, f"done elapsed={format_elapsed(elapsed)}")


if __name__ == "__main__":
    main()
