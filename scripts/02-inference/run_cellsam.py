from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "Monusac" / "tiles_256"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "benchmarking" / "monusac" / "cellsam"
INPUT_MANIFEST = None
OUTPUT_MASK_EXTENSION = ".png"
WORKERS = 8
RAM_LIMIT_GB = 24
GPU_INDEX = 0
OVERWRITE = False
RECURSIVE_INPUT = True
LIMIT = None
DEEPCELL_ACCESS_TOKEN = ""
USE_WSI_MODE = False
LOW_CONTRAST_ENHANCEMENT = False
GAUGE_CELL_SIZE = False

import argparse
import os
import warnings
from contextlib import nullcontext

import numpy as np
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

MODEL_NAME = "cellsam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CellSAM inference on tiled image patches.")
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


def _prepare_environment() -> None:
    token = DEEPCELL_ACCESS_TOKEN.strip() or os.environ.get("DEEPCELL_ACCESS_TOKEN", "").strip()
    if token:
        os.environ["DEEPCELL_ACCESS_TOKEN"] = token
    else:
        print(
            f"[{MODEL_NAME}] DEEPCELL_ACCESS_TOKEN is not set. "
            "If CellSAM weights are not already cached locally, the run will fail."
        )

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*torch.load.*weights_only.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Low IOU threshold.*",
    )


def run_inference(args: argparse.Namespace) -> None:
    import torch
    from cellSAM import cellsam_pipeline

    configure_cpu_threads(args.workers)
    _prepare_environment()

    records, manifest_path = load_tile_records(
        args.input_dir,
        manifest_path=args.input_manifest,
        recursive=not args.non_recursive,
    )
    records = select_records(records, limit=args.limit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    print(f"[{MODEL_NAME}] input_dir={Path(args.input_dir).resolve()}")
    print(f"[{MODEL_NAME}] output_dir={Path(args.output_dir).resolve()}")
    print(f"[{MODEL_NAME}] detected_records={len(records)}")
    print(f"[{MODEL_NAME}] manifest={manifest_path if manifest_path else 'none'}")
    print(f"[{MODEL_NAME}] device={device} workers={args.workers} ram_limit_gb={args.ram_limit_gb:g}")

    success_rows: list[dict[str, object]] = []
    failure_rows_list: list[dict[str, object]] = []

    for record in tqdm(records, desc="CellSAM"):
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
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode():
                with amp_context:
                    mask = cellsam_pipeline(
                        image_array,
                        use_wsi=USE_WSI_MODE,
                        low_contrast_enhancement=LOW_CONTRAST_ENHANCEMENT,
                        gauge_cell_size=GAUGE_CELL_SIZE,
                    )

            if mask is None or not hasattr(mask, "shape"):
                raise RuntimeError("CellSAM returned an empty or invalid mask.")

            save_instance_mask(np.asarray(mask, dtype=np.uint16), mask_path)
            success_rows.append(
                manifest_row(
                    model_name=MODEL_NAME,
                    record=record,
                    image_info=image_info,
                    mask_path=mask_path,
                    instance_count=count_instances(mask),
                    runtime_seconds=monotonic_seconds() - started_at,
                    extra={"mask_dtype": str(np.asarray(mask).dtype)},
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
