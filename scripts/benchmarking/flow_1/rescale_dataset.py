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
SOURCE_MAGNIFICATION = 40.0
TARGET_MAGNIFICATION = 20.0
MIN_INSTANCE_FRACTION = 0.25
OVERWRITE = False
SUMMARY_NAME = "dataset_manifest.csv"

import argparse
import time

import pandas as pd
from tqdm import tqdm

from common import ensure_directory, format_elapsed, log, resolve_path
from dataset_utils import (
    collect_dataset_samples,
    load_instance_mask,
    load_rgb_image,
    normalize_extensions,
    prepare_passthrough_metadata,
    rescale_image_and_mask,
    write_mask_png,
    write_rgb_png,
)

FLOW_NAME = "flow_1.rescale_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescale paired image/mask samples from a generic dataset root and "
            "write a reusable manifest for downstream tiling and inference."
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
    parser.add_argument("--source-magnification", type=float, default=SOURCE_MAGNIFICATION)
    parser.add_argument("--target-magnification", type=float, default=TARGET_MAGNIFICATION)
    parser.add_argument("--min-instance-fraction", type=float, default=MIN_INSTANCE_FRACTION)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    return parser.parse_args()


def _default_output_root(input_root: Path) -> Path:
    return input_root / "rescaled"


def _relative_output_path(relative_path: Path, subdir: str) -> Path:
    return Path(subdir) / relative_path.with_suffix(".png")


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root) or _default_output_root(input_root)
    image_extensions = normalize_extensions(args.image_exts, defaults=IMAGE_EXTENSIONS)
    mask_extensions = normalize_extensions(args.mask_exts, defaults=MASK_EXTENSIONS)
    summary_path = output_root / SUMMARY_NAME

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
    log(
        FLOW_NAME,
        "rescale="
        f"{args.source_magnification:g}x->{args.target_magnification:g}x "
        f"min_instance_fraction={args.min_instance_fraction:.3f} "
        f"pair_mode={args.pair_mode}",
    )
    log(FLOW_NAME, f"samples={len(samples)}")

    summary_rows: list[dict[str, object]] = []
    written_count = 0
    skipped_count = 0

    reserved_names = {
        "sample_id",
        "sample_image_path",
        "sample_mask_path",
        "sample_image_relative_path",
        "sample_mask_relative_path",
        "image_path",
        "mask_path",
        "source_height",
        "source_width",
        "target_height",
        "target_width",
        "source_magnification",
        "target_magnification",
        "scale_factor",
        "min_instance_fraction",
        "original_instance_count",
        "resized_instance_count",
        "dropped_instance_count",
        "dropped_instance_labels",
        "status",
    }

    for sample in tqdm(samples, desc="Rescaling samples"):
        image = load_rgb_image(sample.image_path)
        mask = load_instance_mask(sample.mask_path)
        resized_image, resized_mask, resize_metadata = rescale_image_and_mask(
            image,
            mask,
            source_magnification=args.source_magnification,
            target_magnification=args.target_magnification,
            min_instance_fraction=args.min_instance_fraction,
        )

        output_image_rel = _relative_output_path(sample.relative_image_path, "images")
        output_mask_rel = _relative_output_path(sample.relative_mask_path, "masks")
        output_image_path = output_root / output_image_rel
        output_mask_path = output_root / output_mask_rel
        status = "skipped_existing"

        if output_image_path.exists() and output_mask_path.exists() and not args.overwrite:
            skipped_count += 1
        else:
            write_rgb_png(resized_image, output_image_path)
            write_mask_png(resized_mask, output_mask_path)
            written_count += 1
            status = "written"

        passthrough = prepare_passthrough_metadata(sample.metadata, reserved_names=reserved_names)
        source_height, source_width = resize_metadata["source_shape"]
        target_height, target_width = resize_metadata["target_shape"]
        summary_rows.append(
            {
                **passthrough,
                "sample_id": sample.sample_id,
                "sample_image_path": str(sample.image_path),
                "sample_mask_path": str(sample.mask_path),
                "sample_image_relative_path": sample.relative_image_path.as_posix(),
                "sample_mask_relative_path": sample.relative_mask_path.as_posix(),
                "image_path": output_image_rel.as_posix(),
                "mask_path": output_mask_rel.as_posix(),
                "source_height": int(source_height),
                "source_width": int(source_width),
                "target_height": int(target_height),
                "target_width": int(target_width),
                "source_magnification": float(resize_metadata["source_magnification"]),
                "target_magnification": float(resize_metadata["target_magnification"]),
                "scale_factor": float(resize_metadata["scale_factor"]),
                "min_instance_fraction": float(resize_metadata["min_instance_fraction"]),
                "original_instance_count": int(resize_metadata["original_instance_count"]),
                "resized_instance_count": int(resize_metadata["resized_instance_count"]),
                "dropped_instance_count": int(resize_metadata["dropped_instance_count"]),
                "dropped_instance_labels": ",".join(
                    str(label) for label in resize_metadata["dropped_instance_labels"]
                ),
                "status": status,
            }
        )

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    elapsed = time.perf_counter() - started_at
    log(FLOW_NAME, f"written={written_count} skipped={skipped_count}")
    log(FLOW_NAME, f"summary={summary_path}")
    log(FLOW_NAME, f"done elapsed={format_elapsed(elapsed)}")


if __name__ == "__main__":
    main()
