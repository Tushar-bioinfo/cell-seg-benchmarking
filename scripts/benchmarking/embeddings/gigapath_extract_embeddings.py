from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from gigapath_embedding_utils import (
    DEFAULT_ID_COLUMN_CANDIDATES,
    DEFAULT_IMAGE_COLUMN_CANDIDATES,
    DEFAULT_MODEL_ID,
    OFFICIAL_TRANSFORM_CONFIG,
    RowRecord,
    append_rows_csv,
    atomic_write_json,
    build_failure_row,
    build_resolution_roots,
    configure_logger,
    ensure_directory,
    finalize_sidecar_tables,
    find_repo_root,
    infer_column_name,
    load_input_frame,
    next_part_index,
    now_utc_iso,
    prepare_records,
    read_processed_row_indices,
    relative_to,
    reset_output_tree,
    resolve_path,
)


@dataclass(frozen=True)
class ImageSample:
    record: RowRecord
    image_tensor: Any
    image_height: int
    image_width: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Prov-GigaPath tile embeddings from a flow_1-compatible patch manifest or another CSV that "
            "contains patch image paths."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV file with patch image paths and metadata.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for embeddings, metadata, and logs.")
    parser.add_argument("--image-col", default=None, help="Column that contains the image patch path.")
    parser.add_argument("--id-col", default=None, help="Column to use as the stable row-level embedding identifier.")
    parser.add_argument(
        "--path-base-dir",
        type=Path,
        default=None,
        help="Optional extra directory for resolving relative image paths.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="PyTorch DataLoader worker count.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cuda, cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16", "none"),
        default="float16",
        help="Mixed precision dtype to use on CUDA. Use none for full float32 inference.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on input rows after CSV load.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing outputs under outdir before running instead of attempting resume.",
    )
    parser.add_argument(
        "--save-format",
        choices=("pt", "parquet"),
        default="pt",
        help="Chunk format used under outdir/embeddings.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Prov-GigaPath tile encoder identifier. Keep the official default unless you intentionally override it.",
    )
    return parser.parse_args()


def build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                OFFICIAL_TRANSFORM_CONFIG["resize"],
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(OFFICIAL_TRANSFORM_CONFIG["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=tuple(OFFICIAL_TRANSFORM_CONFIG["normalize_mean"]),
                std=tuple(OFFICIAL_TRANSFORM_CONFIG["normalize_std"]),
            ),
        ]
    )


def load_tile_encoder(model_id: str, *, device: Any):
    import timm
    import torch

    try:
        model = timm.create_model(model_id, pretrained=True)
    except Exception as exc:
        hint = ""
        if not os.environ.get("HF_TOKEN"):
            hint = (
                " Set HF_TOKEN after accepting the Prov-GigaPath model terms on Hugging Face if the weights are not cached."
            )
        raise RuntimeError(
            f"Failed to load Prov-GigaPath tile encoder via timm.create_model({model_id!r}, pretrained=True).{hint}"
        ) from exc

    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model


class TileImageDataset:
    """Dataset that converts image-read failures into row-level errors instead of crashing workers."""

    def __init__(self, records: list[RowRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            with Image.open(record.image_path) as handle:
                image = handle.convert("RGB")
                image_width, image_height = image.size
                image_tensor = self.transform(image)
        except Exception as exc:
            return {
                "ok": False,
                "record": record,
                "error": f"{type(exc).__name__}: {exc}",
            }

        return {
            "ok": True,
            "sample": ImageSample(
                record=record,
                image_tensor=image_tensor,
                image_height=int(image_height),
                image_width=int(image_width),
            ),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    successes: list[ImageSample] = []
    failures: list[dict[str, Any]] = []

    for item in batch:
        if item.get("ok"):
            successes.append(item["sample"])
        else:
            failures.append(item)

    image_tensor = None
    if successes:
        image_tensor = torch.stack([sample.image_tensor for sample in successes], dim=0)

    return {
        "successes": successes,
        "failures": failures,
        "image_tensor": image_tensor,
    }


def resolve_device(device_text: str):
    import torch

    normalized = device_text.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def resolve_amp_dtype(*, device: Any, amp_dtype_text: str):
    import torch

    if device.type != "cuda" or amp_dtype_text == "none":
        return None

    dtype_lookup = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_lookup[amp_dtype_text]


def autocast_context(*, device: Any, amp_dtype: Any):
    import torch

    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def normalize_model_output(model_output: Any):
    import torch

    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (list, tuple)) and model_output:
        first_item = model_output[0]
        if isinstance(first_item, torch.Tensor):
            return first_item
    if isinstance(model_output, dict):
        for key in ("embedding", "embeddings", "x", "features"):
            value = model_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported model output type: {type(model_output)!r}")


def save_embedding_part(
    *,
    embeddings,
    samples: list[ImageSample],
    part_path: Path,
    save_format: str,
) -> list[dict[str, Any]]:
    import torch

    ensure_directory(part_path.parent)
    embedding_tensor = embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()
    embedding_dim = int(embedding_tensor.shape[-1])

    if save_format == "pt":
        payload = {
            "embeddings": embedding_tensor,
            "input_row_index": [sample.record.input_row_index for sample in samples],
            "embedding_id": [sample.record.embedding_id for sample in samples],
        }
        torch.save(payload, part_path)
    elif save_format == "parquet":
        part_df = pd.DataFrame(
            {
                "input_row_index": [sample.record.input_row_index for sample in samples],
                "embedding_id": [sample.record.embedding_id for sample in samples],
                "embedding": embedding_tensor.numpy().tolist(),
            }
        )
        part_df.to_parquet(part_path, index=False)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")

    rows: list[dict[str, Any]] = []
    for row_offset, sample in enumerate(samples):
        row = dict(sample.record.row_data)
        row.update(
            {
                "input_row_index": int(sample.record.input_row_index),
                "embedding_id": sample.record.embedding_id,
                "resolved_image_path": str(sample.record.image_path),
                "embedding_path": str(part_path),
                "embedding_format": save_format,
                "embedding_row_offset": int(row_offset),
                "embedding_dim": embedding_dim,
                "image_height": int(sample.image_height),
                "image_width": int(sample.image_width),
            }
        )
        rows.append(row)
    return rows


def maybe_resume_guard(
    manifest_path: Path,
    *,
    input_csv: Path,
    outdir: Path,
    overwrite: bool,
    model_id: str,
    save_format: str,
    image_column: str | None,
    id_column: str | None,
) -> dict[str, Any] | None:
    if overwrite or not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    comparisons = {
        "input_csv": str(input_csv.resolve()),
        "model_id": model_id,
        "save_format": save_format,
    }
    if image_column is not None:
        comparisons["image_column"] = image_column
    if id_column is not None:
        comparisons["id_column"] = id_column

    for key, expected_value in comparisons.items():
        previous_value = manifest.get(key)
        if previous_value is None:
            continue
        if str(previous_value) != str(expected_value):
            raise ValueError(
                f"Existing outdir {outdir} was created with {key}={previous_value!r}, "
                f"but the current run expects {expected_value!r}. Use --overwrite or a new --outdir."
            )
    return manifest


def count_rows_if_exists(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path)))


def write_manifest(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    repo_root: Path,
    image_column: str | None,
    id_column: str | None,
    device_text: str | None,
    amp_dtype_text: str | None,
    status: str,
    counts: dict[str, Any],
    started_at: str,
    finished_at: str | None,
    note: str | None = None,
) -> None:
    payload = {
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "repo_root": str(repo_root),
        "input_csv": str(args.input_csv.resolve()),
        "outdir": str(args.outdir.resolve()),
        "image_column": image_column,
        "id_column": id_column,
        "path_base_dir": None if args.path_base_dir is None else str(args.path_base_dir.resolve()),
        "model_id": args.model_id,
        "official_transform": OFFICIAL_TRANSFORM_CONFIG,
        "device": device_text,
        "amp_dtype": amp_dtype_text,
        "save_format": args.save_format,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "limit": args.limit,
        "overwrite": bool(args.overwrite),
        "counts": counts,
        "note": note,
    }
    atomic_write_json(manifest_path, payload)


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    repo_root = find_repo_root(Path(__file__).resolve())
    input_csv = resolve_path(args.input_csv)
    outdir = resolve_path(args.outdir)
    path_base_dir = resolve_path(args.path_base_dir)
    if input_csv is None or outdir is None:
        raise ValueError("input_csv and outdir must resolve to concrete filesystem paths.")

    embeddings_dir = outdir / "embeddings"
    metadata_dir = outdir / "metadata"
    logs_dir = outdir / "logs"
    manifest_path = outdir / "manifest.json"
    metadata_csv = metadata_dir / "embeddings_index.csv"
    metadata_parquet = metadata_dir / "embeddings_index.parquet"
    snapshot_csv = metadata_dir / "input_rows_snapshot.csv"
    snapshot_parquet = metadata_dir / "input_rows_snapshot.parquet"
    failures_csv = logs_dir / "failed_rows.csv"
    failures_parquet = logs_dir / "failed_rows.parquet"
    run_log_path = logs_dir / "run.log"

    if args.overwrite:
        reset_output_tree(outdir)

    ensure_directory(embeddings_dir)
    ensure_directory(metadata_dir)
    ensure_directory(logs_dir)

    logger = configure_logger(run_log_path)
    started_at = now_utc_iso()

    input_df = load_input_frame(input_csv, limit=args.limit)
    input_columns = [str(column) for column in input_df.columns]
    image_column = infer_column_name(
        input_columns,
        args.image_col,
        DEFAULT_IMAGE_COLUMN_CANDIDATES,
        required=True,
    )
    id_column = infer_column_name(
        input_columns,
        args.id_col,
        DEFAULT_ID_COLUMN_CANDIDATES,
        required=False,
    )

    maybe_resume_guard(
        manifest_path,
        input_csv=input_csv,
        outdir=outdir,
        overwrite=args.overwrite,
        model_id=args.model_id,
        save_format=args.save_format,
        image_column=image_column,
        id_column=id_column,
    )

    input_df.to_csv(snapshot_csv, index=False)
    input_df.to_parquet(snapshot_parquet, index=False)

    write_manifest(
        manifest_path,
        args=args,
        repo_root=repo_root,
        image_column=image_column,
        id_column=id_column,
        device_text=None,
        amp_dtype_text=None,
        status="running",
        counts={
            "input_rows_total": int(len(input_df)),
            "succeeded_rows": 0,
            "failed_rows": 0,
            "skipped_rows_from_resume": 0,
            "embedding_parts": 0,
        },
        started_at=started_at,
        finished_at=None,
        note="Run initialized.",
    )

    processed_row_indices = set()
    if not args.overwrite:
        processed_row_indices = read_processed_row_indices(metadata_csv, failures_csv)

    skipped_rows_from_resume = len(processed_row_indices)
    if processed_row_indices:
        input_df = input_df.loc[~input_df["input_row_index"].isin(processed_row_indices)].copy()
        logger.info("Resume mode: skipping %d previously processed rows.", skipped_rows_from_resume)

    resolution_roots = build_resolution_roots(
        input_csv=input_csv,
        repo_root=repo_root,
        path_base_dir=path_base_dir,
    )

    logger.info("Input rows after resume filter: %d", len(input_df))
    logger.info("Using image column: %s", image_column)
    logger.info("Using id column: %s", id_column if id_column else "<derived>")
    logger.info("Path resolution roots: %s", [str(root) for root in resolution_roots])

    records, path_failures = prepare_records(
        input_df,
        image_column=image_column,
        id_column=id_column,
        roots=resolution_roots,
    )
    append_rows_csv(failures_csv, path_failures)

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device=device, amp_dtype_text=args.amp_dtype)
    logger.info("Resolved device: %s", device)
    logger.info("AMP dtype: %s", args.amp_dtype if amp_dtype is not None else "none")

    write_manifest(
        manifest_path,
        args=args,
        repo_root=repo_root,
        image_column=image_column,
        id_column=id_column,
        device_text=str(device),
        amp_dtype_text=args.amp_dtype if amp_dtype is not None else "none",
        status="running",
        counts={
            "input_rows_total": int(len(input_df) + skipped_rows_from_resume),
            "prepared_rows": int(len(records)),
            "path_resolution_failures": int(len(path_failures)),
            "succeeded_rows": 0,
            "failed_rows": int(len(path_failures)),
            "skipped_rows_from_resume": int(skipped_rows_from_resume),
            "embedding_parts": 0,
        },
        started_at=started_at,
        finished_at=None,
        note="Manifest columns and path resolution completed.",
    )

    if not records:
        finalize_sidecar_tables(
            metadata_csv=metadata_csv,
            metadata_parquet=metadata_parquet,
            failures_csv=failures_csv,
            failures_parquet=failures_parquet,
            logger=logger,
        )
        write_manifest(
            manifest_path,
            args=args,
            repo_root=repo_root,
            image_column=image_column,
            id_column=id_column,
            device_text=str(device),
            amp_dtype_text=args.amp_dtype if amp_dtype is not None else "none",
            status="completed",
            counts={
                "input_rows_total": int(len(input_df) + skipped_rows_from_resume),
                "prepared_rows": 0,
                "path_resolution_failures": int(len(path_failures)),
                "succeeded_rows": 0,
                "failed_rows": int(len(path_failures)),
                "skipped_rows_from_resume": int(skipped_rows_from_resume),
                "embedding_parts": 0,
            },
            started_at=started_at,
            finished_at=now_utc_iso(),
            note="No valid rows remained after path resolution and resume filtering.",
        )
        logger.info("No valid rows remained after path resolution and resume filtering.")
        return 0

    transform = build_transform()
    dataset = TileImageDataset(records, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=collate_batch,
    )

    model = load_tile_encoder(args.model_id, device=device)

    success_count = 0
    failure_count = len(path_failures)
    part_index = next_part_index(embeddings_dir)
    batch_count = 0

    try:
        with torch.inference_mode():
            for batch in dataloader:
                batch_count += 1
                batch_failures: list[dict[str, Any]] = []
                for failed_item in batch["failures"]:
                    record: RowRecord = failed_item["record"]
                    batch_failures.append(
                        build_failure_row(
                            row_data=record.row_data,
                            input_row_index=record.input_row_index,
                            embedding_id=record.embedding_id,
                            image_column=image_column,
                            raw_image_value=record.row_data.get(image_column),
                            resolved_image_path=record.image_path,
                            failure_stage="image_decode",
                            error=failed_item["error"],
                        )
                    )

                if batch_failures:
                    append_rows_csv(failures_csv, batch_failures)
                    failure_count += len(batch_failures)

                samples: list[ImageSample] = batch["successes"]
                image_tensor = batch["image_tensor"]
                if not samples or image_tensor is None:
                    logger.info("Batch %d contained only failed image-decode rows.", batch_count)
                    continue

                try:
                    image_tensor = image_tensor.to(device, non_blocking=device.type == "cuda")
                    with autocast_context(device=device, amp_dtype=amp_dtype):
                        model_output = model(image_tensor)
                    embeddings = normalize_model_output(model_output)
                    if embeddings.ndim != 2 or embeddings.shape[0] != len(samples):
                        raise RuntimeError(
                            f"Unexpected embedding tensor shape {tuple(embeddings.shape)} for batch with {len(samples)} rows."
                        )
                except Exception as exc:
                    inference_failures = [
                        build_failure_row(
                            row_data=sample.record.row_data,
                            input_row_index=sample.record.input_row_index,
                            embedding_id=sample.record.embedding_id,
                            image_column=image_column,
                            raw_image_value=sample.record.row_data.get(image_column),
                            resolved_image_path=sample.record.image_path,
                            failure_stage="model_inference",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        for sample in samples
                    ]
                    append_rows_csv(failures_csv, inference_failures)
                    failure_count += len(inference_failures)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    logger.exception("Batch %d failed during model inference.", batch_count)
                    continue

                part_suffix = ".pt" if args.save_format == "pt" else ".parquet"
                part_path = embeddings_dir / f"part-{part_index:06d}{part_suffix}"
                metadata_rows = save_embedding_part(
                    embeddings=embeddings,
                    samples=samples,
                    part_path=part_path,
                    save_format=args.save_format,
                )
                append_rows_csv(metadata_csv, metadata_rows)

                success_count += len(samples)
                logger.info(
                    "Batch %d saved %d embeddings to %s",
                    batch_count,
                    len(samples),
                    relative_to(outdir, part_path),
                )
                part_index += 1

    except Exception as exc:
        finished_at = now_utc_iso()
        write_manifest(
            manifest_path,
            args=args,
            repo_root=repo_root,
            image_column=image_column,
            id_column=id_column,
            device_text=str(device),
            amp_dtype_text=args.amp_dtype if amp_dtype is not None else "none",
            status="failed",
            counts={
                "input_rows_total": int(len(input_df) + skipped_rows_from_resume),
                "prepared_rows": int(len(records)),
                "path_resolution_failures": int(len(path_failures)),
                "succeeded_rows": int(success_count),
                "failed_rows": int(failure_count),
                "skipped_rows_from_resume": int(skipped_rows_from_resume),
                "embedding_parts": int(len(list(embeddings_dir.glob("part-*.*")))),
            },
            started_at=started_at,
            finished_at=finished_at,
            note=f"Fatal error: {type(exc).__name__}: {exc}",
        )
        raise

    finalize_sidecar_tables(
        metadata_csv=metadata_csv,
        metadata_parquet=metadata_parquet,
        failures_csv=failures_csv,
        failures_parquet=failures_parquet,
        logger=logger,
    )

    finished_at = now_utc_iso()
    embedding_parts = len(list(embeddings_dir.glob("part-*.*")))
    counts = {
        "input_rows_total": int(len(load_input_frame(input_csv, limit=args.limit))),
        "prepared_rows": int(len(records)),
        "path_resolution_failures": int(len(path_failures)),
        "succeeded_rows": count_rows_if_exists(metadata_csv),
        "failed_rows": count_rows_if_exists(failures_csv),
        "skipped_rows_from_resume": int(skipped_rows_from_resume),
        "embedding_parts": int(embedding_parts),
    }
    write_manifest(
        manifest_path,
        args=args,
        repo_root=repo_root,
        image_column=image_column,
        id_column=id_column,
        device_text=str(device),
        amp_dtype_text=args.amp_dtype if amp_dtype is not None else "none",
        status="completed",
        counts=counts,
        started_at=started_at,
        finished_at=finished_at,
        note="Run completed successfully.",
    )

    logger.info("Completed extraction: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
