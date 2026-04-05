from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "Monusac" / "tiles_256"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "benchmarking" / "monusac" / "cellvit_sam"
INPUT_MANIFEST = None
OUTPUT_MASK_EXTENSION = ".png"
WORKERS = 8
RAM_LIMIT_GB = 24
GPU_INDEX = 0
BATCH_SIZE = 2
OVERWRITE = False
RECURSIVE_INPUT = True
LIMIT = None
NUCLEI_TAXONOMY = "binary"
PATCH_SIZE = 1024
PADDING_FILL_VALUE = 255
ENFORCE_AMP = True

import argparse
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from benchmark_inference_utils import (
    centered_pad_to_square,
    configure_cpu_threads,
    count_instances,
    crop_to_original_extent,
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

MODEL_NAME = "cellvit_sam"


@dataclass(frozen=True)
class PreparedTile:
    record: object
    image_info: object
    mask_path: Path
    original_shape: tuple[int, int]
    offset: tuple[int, int]
    tensor: object
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CellViT SAM inference on tiled image patches.")
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


def _build_system_configuration(args: argparse.Namespace):
    from cellvit.utils.ressource_manager import SystemConfiguration

    system_configuration = SystemConfiguration(gpu=args.gpu_index)
    system_configuration.overwrite_available_cpus(args.workers)
    system_configuration.overwrite_memory(int(args.ram_limit_gb * 1024))
    return system_configuration


def _prepare_batch(pipeline, batch_records: list, output_dir: Path, overwrite: bool) -> tuple[list[PreparedTile], list[dict[str, object]]]:
    import torch

    prepared_tiles: list[PreparedTile] = []
    skipped_rows: list[dict[str, object]] = []
    for record in batch_records:
        mask_path = output_mask_path(output_dir, record, output_extension=OUTPUT_MASK_EXTENSION)
        image_array, image_info = read_rgb_image(record.source_path)
        if mask_path.exists() and not overwrite:
            skipped_rows.append(
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

        padded_image, offset = centered_pad_to_square(
            image_array,
            target_size=PATCH_SIZE,
            fill_value=PADDING_FILL_VALUE,
        )
        tensor = pipeline.inference_transforms(padded_image)
        prepared_tiles.append(
            PreparedTile(
                record=record,
                image_info=image_info,
                mask_path=mask_path,
                original_shape=image_array.shape[:2],
                offset=offset,
                tensor=tensor,
                started_at=monotonic_seconds(),
            )
        )
    return prepared_tiles, skipped_rows


def run_inference(args: argparse.Namespace) -> None:
    import ray
    import torch
    from cellvit.data.dataclass.wsi import WSIMetadata
    from cellvit.inference.inference import CellViTInference

    configure_cpu_threads(args.workers)
    records, manifest_path = load_tile_records(
        args.input_dir,
        manifest_path=args.input_manifest,
        recursive=not args.non_recursive,
    )
    records = select_records(records, limit=args.limit)
    if not torch.cuda.is_available():
        raise RuntimeError("CellViT SAM requires a CUDA-capable environment for this pipeline.")

    print(f"[{MODEL_NAME}] input_dir={Path(args.input_dir).resolve()}")
    print(f"[{MODEL_NAME}] output_dir={Path(args.output_dir).resolve()}")
    print(f"[{MODEL_NAME}] detected_records={len(records)}")
    print(f"[{MODEL_NAME}] manifest={manifest_path if manifest_path else 'none'}")
    print(
        f"[{MODEL_NAME}] batch_size={args.batch_size} workers={args.workers} "
        f"ram_limit_gb={args.ram_limit_gb:g} gpu_index={args.gpu_index}"
    )

    system_configuration = _build_system_configuration(args)
    pipeline = CellViTInference(
        model_name="SAM",
        outdir=args.output_dir,
        system_configuration=system_configuration,
        nuclei_taxonomy=NUCLEI_TAXONOMY,
        batch_size=max(1, args.batch_size),
        patch_size=PATCH_SIZE,
        overlap=0,
        geojson=False,
        graph=False,
        compression=False,
        enforce_amp=ENFORCE_AMP,
        debug=False,
    )

    postprocessor_cls, _ = pipeline._import_postprocessing()
    dummy_wsi = WSIMetadata(name="tile_batch", slide_path="tile_batch", metadata={})
    postprocessor = postprocessor_cls(
        dummy_wsi,
        nr_types=pipeline.run_conf["data"]["num_nuclei_classes"],
        classifier=pipeline.classifier,
        binary=pipeline.binary,
    )

    success_rows: list[dict[str, object]] = []
    failure_rows_list: list[dict[str, object]] = []

    try:
        for batch_start in tqdm(range(0, len(records), max(1, args.batch_size)), desc="CellViT SAM"):
            batch_records = records[batch_start : batch_start + max(1, args.batch_size)]
            prepared_tiles, skipped_rows = _prepare_batch(
                pipeline,
                batch_records,
                output_dir=Path(args.output_dir),
                overwrite=args.overwrite,
            )
            success_rows.extend(skipped_rows)
            if not prepared_tiles:
                continue

            batch_tensor = torch.stack([tile.tensor for tile in prepared_tiles]).to(
                pipeline.device,
                non_blocking=True,
            )

            try:
                use_amp = bool(str(pipeline.device).startswith("cuda") and pipeline.mixed_precision)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp
                    else nullcontext()
                )
                with torch.inference_mode():
                    with amp_context:
                        predictions = pipeline.model.forward(batch_tensor, retrieve_tokens=True)
                predictions = pipeline.apply_softmax_reorder(predictions)
                instance_maps, _ = postprocessor.post_process_batch(predictions)
                instance_maps_np = instance_maps.detach().cpu().numpy().astype(np.uint16, copy=False)
            except Exception as error:
                for tile in prepared_tiles:
                    failure_rows_list.append(
                        failure_row(
                            model_name=MODEL_NAME,
                            record=tile.record,
                            error=error,
                            runtime_seconds=monotonic_seconds() - tile.started_at,
                        )
                    )
                continue

            for tile, instance_map in zip(prepared_tiles, instance_maps_np):
                try:
                    cropped_mask = crop_to_original_extent(
                        instance_map,
                        offset=tile.offset,
                        shape=tile.original_shape,
                    )
                    save_instance_mask(cropped_mask, tile.mask_path)
                    success_rows.append(
                        manifest_row(
                            model_name=MODEL_NAME,
                            record=tile.record,
                            image_info=tile.image_info,
                            mask_path=tile.mask_path,
                            instance_count=count_instances(cropped_mask),
                            runtime_seconds=monotonic_seconds() - tile.started_at,
                            extra={
                                "mask_dtype": str(cropped_mask.dtype),
                                "padded_patch_size": PATCH_SIZE,
                                "padding_offset_y": int(tile.offset[0]),
                                "padding_offset_x": int(tile.offset[1]),
                            },
                        )
                    )
                except Exception as error:
                    failure_rows_list.append(
                        failure_row(
                            model_name=MODEL_NAME,
                            record=tile.record,
                            error=error,
                            runtime_seconds=monotonic_seconds() - tile.started_at,
                        )
                    )
    finally:
        if ray.is_initialized():
            ray.shutdown()

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
