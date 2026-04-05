from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "Monusac" / "tiles_256"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "benchmarking" / "monusac" / "cellpose_sam"
INPUT_MANIFEST = None
OUTPUT_MASK_EXTENSION = ".png"
WORKERS = 8
RAM_LIMIT_GB = 24
GPU_INDEX = 0
BATCH_SIZE = 16
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
TILE_NORM_BLOCKSIZE = 0
OVERWRITE = False
RECURSIVE_INPUT = True
LIMIT = None

import argparse

from tqdm.auto import tqdm

from benchmark_inference_utils import (
    configure_cpu_threads,
    count_instances,
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

MODEL_NAME = "cellpose_sam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cellpose SAM inference on tiled image patches.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--input-manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--ram-limit-gb", type=float, default=RAM_LIMIT_GB)
    parser.add_argument("--gpu-index", type=int, default=GPU_INDEX)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--non-recursive", action="store_true")
    return parser.parse_args()


def _load_model(gpu_enabled: bool):
    from cellpose import models

    return models.CellposeModel(gpu=gpu_enabled)


def run_inference(args: argparse.Namespace) -> None:
    from cellpose import core

    configure_cpu_threads(args.workers)
    records, manifest_path = load_tile_records(
        args.input_dir,
        manifest_path=args.input_manifest,
        recursive=not args.non_recursive,
    )
    records = select_records(records, limit=args.limit)

    gpu_enabled = bool(core.use_gpu())
    if args.gpu_index != 0:
        print(
            f"[{MODEL_NAME}] Cellpose SAM uses the default visible GPU. "
            "Set CUDA_VISIBLE_DEVICES before launching if you need a different device."
        )

    print(f"[{MODEL_NAME}] input_dir={Path(args.input_dir).resolve()}")
    print(f"[{MODEL_NAME}] output_dir={Path(args.output_dir).resolve()}")
    print(f"[{MODEL_NAME}] detected_records={len(records)}")
    print(f"[{MODEL_NAME}] manifest={manifest_path if manifest_path else 'none'}")
    print(f"[{MODEL_NAME}] gpu_enabled={gpu_enabled}")
    print(f"[{MODEL_NAME}] batch_size={args.batch_size} workers={args.workers} ram_limit_gb={args.ram_limit_gb:g}")

    model = _load_model(gpu_enabled=gpu_enabled)
    success_rows: list[dict[str, object]] = []
    failure_rows_list: list[dict[str, object]] = []

    for batch_start in tqdm(range(0, len(records), max(1, args.batch_size)), desc="Cellpose SAM"):
        batch_records = records[batch_start : batch_start + max(1, args.batch_size)]
        batch_images = []
        batch_infos = []
        batch_run_times = []

        for record in batch_records:
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
            image_array, image_info = read_rgb_image(record.source_path)
            batch_images.append(image_array)
            batch_infos.append((record, image_info, mask_path))
            batch_run_times.append(monotonic_seconds())

        if not batch_images:
            continue

        masks, _, _ = model.eval(
            batch_images,
            batch_size=max(1, args.batch_size),
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
            normalize={"tile_norm_blocksize": TILE_NORM_BLOCKSIZE},
        )

        for mask, (record, image_info, mask_path), started_at in zip(masks, batch_infos, batch_run_times):
            try:
                save_instance_mask(mask, mask_path)
                success_rows.append(
                    manifest_row(
                        model_name=MODEL_NAME,
                        record=record,
                        image_info=image_info,
                        mask_path=mask_path,
                        instance_count=count_instances(mask),
                        runtime_seconds=monotonic_seconds() - started_at,
                        extra={"mask_dtype": str(mask.dtype)},
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
