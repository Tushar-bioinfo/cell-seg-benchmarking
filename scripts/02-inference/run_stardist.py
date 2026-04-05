from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "Monusac" / "tiles_256"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "benchmarking" / "monusac" / "stardist"
INPUT_MANIFEST = None
OUTPUT_MASK_EXTENSION = ".png"
WORKERS = 8
RAM_LIMIT_GB = 24
GPU_INDEX = 0
OVERWRITE = False
RECURSIVE_INPUT = True
LIMIT = None
MODEL_NAME_PRETRAINED = "2D_versatile_he"
PNORM = (1, 99.8)
N_TILES = None
PROB_THRESH = None
NMS_THRESH = None

import argparse

import numpy as np
from tqdm.auto import tqdm

from benchmark_inference_utils import (
    configure_cpu_threads,
    count_instances,
    ensure_rgb_array,
    failure_row,
    load_tile_records,
    manifest_row,
    monotonic_seconds,
    output_mask_path,
    read_rgb_image,
    save_instance_mask,
    select_records,
    write_model_reports,
)

MODEL_NAME = "stardist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StarDist inference on tiled image patches.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--input-manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--ram-limit-gb", type=float, default=RAM_LIMIT_GB)
    parser.add_argument("--gpu-index", type=int, default=GPU_INDEX)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--non-recursive", action="store_true")
    return parser.parse_args()


def _ensure_yx_or_yxc(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[-1] in (1, 3):
            return img
        if img.shape[-1] == 4:
            return img[..., :3]
        if img.shape[0] in (1, 3):
            return np.moveaxis(img, 0, -1)
        if img.shape[0] == 4:
            return np.moveaxis(img[:3], 0, -1)
    raise ValueError(f"Unsupported image shape {img.shape}. Expected (Y, X) or (Y, X, C).")


def run_inference(args: argparse.Namespace) -> None:
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    configure_cpu_threads(args.workers)
    records, manifest_path = load_tile_records(
        args.input_dir,
        manifest_path=args.input_manifest,
        recursive=not args.non_recursive,
    )
    records = select_records(records, limit=args.limit)

    model = StarDist2D.from_pretrained(MODEL_NAME_PRETRAINED)

    print(f"[{MODEL_NAME}] input_dir={Path(args.input_dir).resolve()}")
    print(f"[{MODEL_NAME}] output_dir={Path(args.output_dir).resolve()}")
    print(f"[{MODEL_NAME}] detected_records={len(records)}")
    print(f"[{MODEL_NAME}] manifest={manifest_path if manifest_path else 'none'}")
    print(f"[{MODEL_NAME}] pretrained_model={MODEL_NAME_PRETRAINED} workers={args.workers} ram_limit_gb={args.ram_limit_gb:g}")

    success_rows: list[dict[str, object]] = []
    failure_rows_list: list[dict[str, object]] = []

    for record in tqdm(records, desc="StarDist"):
        started_at = monotonic_seconds()
        mask_path = output_mask_path(args.output_dir, record, output_extension=OUTPUT_MASK_EXTENSION)
        if mask_path.exists() and not args.overwrite:
            image_array, image_info = read_rgb_image(record.source_path)
            success_rows.append(
                manifest_row(
                    model_name=MODEL_NAME,
                    record=record,
                    image_info=image_info,
                    mask_path=mask_path,
                    instance_count=-1,
                    runtime_seconds=0.0,
                    extra={"skipped_existing": True},
                )
            )
            continue

        try:
            image_array, image_info = read_rgb_image(record.source_path)
            model_input = _ensure_yx_or_yxc(ensure_rgb_array(image_array))
            axis_norm = (0, 1) if model_input.ndim == 3 else None
            normalized = normalize(model_input, *PNORM, axis=axis_norm)
            labels, _ = model.predict_instances(
                normalized,
                n_tiles=N_TILES,
                prob_thresh=PROB_THRESH,
                nms_thresh=NMS_THRESH,
            )
            save_instance_mask(labels, mask_path)
            success_rows.append(
                manifest_row(
                    model_name=MODEL_NAME,
                    record=record,
                    image_info=image_info,
                    mask_path=mask_path,
                    instance_count=count_instances(labels),
                    runtime_seconds=monotonic_seconds() - started_at,
                    extra={"mask_dtype": str(labels.dtype)},
                )
            )
        except Exception as error:
            failure_rows_list.append(
                failure_row(
                    model_name=MODEL_NAME,
                    record=record,
                    error=error,
                    runtime_seconds=monotonic_seconds() - started_at,
                )
            )

    manifest_csv, failed_csv = write_model_reports(
        output_dir=args.output_dir,
        success_rows=success_rows,
        failure_rows_list=failure_rows_list,
    )
    print(f"[{MODEL_NAME}] predictions={manifest_csv}")
    print(f"[{MODEL_NAME}] failures={failed_csv}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
