"""Build the canonical modeling table for the difficulty workflow.

Why this script exists:
- Convert the upstream patch-by-model target table into one canonical
  patch-level table that every downstream modeling stage can reuse.
- Keep split-dependent labels out of the prep step so train/test logic stays in
  the training scripts.

What it reads:
- The joined target table that contains patch IDs, slide IDs, model names, and
- per-model evaluation metrics.
- Optional metadata tables such as embedding indexes, morphology tables, and
  patch-feature exports.

What it writes:
- `modeling_table.csv.gz` with one row per `patch_id`
- `config.json`, `validation.json`, `validation.md`, `run.log`, and
  `timing.json` in the output directory

What validation it performs:
- Input file existence and required-column checks
- Duplicate `patch_id` and duplicate `(patch_id, model_name)` checks
- Required consensus target checks, row counts, and missingness summaries
- Output readability and required output-column checks
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import (
    EMBEDDING_METADATA_COLUMNS,
    METRIC_COLUMN_ALIASES,
    file_existence_rows,
    infer_join_key,
    infer_metric_source_columns,
    load_table,
    missingness_summary,
    normalize_model_name,
    now_utc_iso,
    require_columns,
    resolve_path,
    save_csv_table,
    setup_stage_logging,
    verify_no_duplicate_patch_ids,
    write_json,
    write_timing_json,
    write_validation_reports,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the model-table build step."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joined-target-table",
        default="outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet",
        help="Patch-by-model target table with evaluation metrics.",
    )
    parser.add_argument(
        "--embeddings-index",
        default="",
        help="Optional embeddings index table to join by patch ID.",
    )
    parser.add_argument(
        "--embed-morph",
        default="outputs/conic_liz/embed_morph.csv",
        help="Optional embedding+morphology table to join by patch ID.",
    )
    parser.add_argument(
        "--patch-features",
        default="",
        help="Optional patch-feature table to join by patch ID.",
    )
    parser.add_argument(
        "--patch-id-col",
        default="patch_id",
        help="Patch ID column in the joined target table and optional joins.",
    )
    parser.add_argument(
        "--slide-id-col",
        default="slide_id",
        help="Slide/group column in the joined target table.",
    )
    parser.add_argument(
        "--output",
        default="outputs/conic_liz/model/model_table/modeling_table.csv.gz",
        help="Output path for the canonical modeling table.",
    )
    return parser.parse_args()


def first_non_null(series: pd.Series) -> Any:
    """Return the first non-null value in a series, or NaN when absent."""

    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def verify_patch_level_consistency(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Count patch-level columns that vary within the same `patch_id`."""

    inconsistent: dict[str, int] = {}
    grouped = frame.groupby("patch_id", dropna=False)
    for column in columns:
        if column not in frame.columns:
            continue
        varying_patch_count = int(grouped[column].nunique(dropna=True).gt(1).sum())
        if varying_patch_count:
            inconsistent[column] = varying_patch_count
    return inconsistent


def build_patch_level_base(frame: pd.DataFrame, excluded_columns: set[str]) -> pd.DataFrame:
    """Collapse repeated patch rows into one patch-level table."""

    patch_level_columns = [
        column for column in frame.columns if column not in excluded_columns and column != "patch_id"
    ]
    grouped = frame.groupby("patch_id", dropna=False)
    patch_base = grouped[patch_level_columns].agg(first_non_null).reset_index(drop=True)
    patch_base.insert(0, "patch_id", grouped.size().index.to_series().values)
    return patch_base


def join_optional_source(
    model_table: pd.DataFrame,
    source_path: str,
    source_name: str,
    preferred_patch_id_col: str,
    logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join an optional patch-level metadata source by `patch_id`."""

    info: dict[str, Any] = {
        "source_name": source_name,
        "source_path": source_path,
        "used": False,
        "joined_columns": [],
    }
    if not source_path:
        info["note"] = "No path provided."
        return model_table, info

    resolved_path = resolve_path(source_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Optional source `{source_name}` does not exist: {resolved_path}")

    source_frame = load_table(resolved_path)
    source_patch_id_col = infer_join_key(source_frame, preferred_patch_id_col)
    source_frame = source_frame.rename(columns={source_patch_id_col: "patch_id"}).copy()
    duplicate_patch_id_count = verify_no_duplicate_patch_ids(source_frame)
    if duplicate_patch_id_count:
        raise ValueError(
            f"Optional source `{source_name}` contains {duplicate_patch_id_count} duplicated `patch_id` values."
        )

    overlap_columns = [
        column for column in source_frame.columns if column != "patch_id" and column in model_table.columns
    ]
    new_columns = [
        column for column in source_frame.columns if column != "patch_id" and column not in model_table.columns
    ]

    if overlap_columns:
        overlap_frame = source_frame.loc[:, ["patch_id", *overlap_columns]].copy()
        renamed_overlap_columns = {
            column: f"{column}__{source_name}" for column in overlap_columns
        }
        overlap_frame = overlap_frame.rename(columns=renamed_overlap_columns)
        model_table = model_table.merge(overlap_frame, on="patch_id", how="left")
        for column in overlap_columns:
            fill_column = f"{column}__{source_name}"
            model_table[column] = model_table[column].where(model_table[column].notna(), model_table[fill_column])
            model_table = model_table.drop(columns=[fill_column])

    if new_columns:
        model_table = model_table.merge(
            source_frame.loc[:, ["patch_id", *new_columns]],
            on="patch_id",
            how="left",
        )

    info["used"] = True
    info["rows"] = int(len(source_frame))
    info["columns"] = int(source_frame.shape[1])
    info["joined_columns"] = new_columns
    info["overlap_columns_filled_when_missing"] = overlap_columns
    logger.info(
        "joined source=%s rows=%s new_columns=%s overlap_columns=%s",
        source_name,
        len(source_frame),
        len(new_columns),
        len(overlap_columns),
    )
    return model_table, info


def main() -> None:
    """Run the model-table build step."""

    args = parse_args()
    output_path = resolve_path(args.output)
    output_dir = output_path.parent
    logger = setup_stage_logging(output_dir, stage_name="build_model_table")
    started_time = time.time()

    config = {
        "stage_name": "build_model_table",
        "joined_target_table": str(resolve_path(args.joined_target_table)),
        "embeddings_index": str(resolve_path(args.embeddings_index)) if args.embeddings_index else "",
        "embed_morph": str(resolve_path(args.embed_morph)) if args.embed_morph else "",
        "patch_features": str(resolve_path(args.patch_features)) if args.patch_features else "",
        "patch_id_col": args.patch_id_col,
        "slide_id_col": args.slide_id_col,
        "output": str(output_path),
        "generated_at_utc": now_utc_iso(),
    }
    write_json(output_dir / "config.json", config)

    joined_frame = load_table(args.joined_target_table).copy()
    require_columns(joined_frame, [args.patch_id_col, args.slide_id_col, "model_name"], "joined target table")
    joined_frame = joined_frame.rename(
        columns={args.patch_id_col: "patch_id", args.slide_id_col: "slide_id"}
    )

    metric_source_columns = infer_metric_source_columns(joined_frame.columns)
    if "pq" not in metric_source_columns:
        available = ", ".join(sorted(joined_frame.columns))
        known_aliases = ", ".join(METRIC_COLUMN_ALIASES["pq"])
        raise ValueError(
            "Could not derive `pq_median` because none of the known PQ source columns were present. "
            f"Tried aliases: {known_aliases}. Available columns: {available}"
        )

    duplicate_patch_model_count = int(joined_frame.duplicated(subset=["patch_id", "model_name"]).sum())
    if duplicate_patch_model_count:
        raise ValueError(
            f"Joined target table contains {duplicate_patch_model_count} duplicated `(patch_id, model_name)` rows."
        )

    consistency_columns = ["slide_id", "dataset", "split", *EMBEDDING_METADATA_COLUMNS]
    inconsistent_patch_columns = verify_patch_level_consistency(joined_frame, consistency_columns)
    if inconsistent_patch_columns:
        raise ValueError(
            "Patch-level metadata varies within the same `patch_id`: "
            f"{inconsistent_patch_columns}"
        )

    excluded_base_columns = set(metric_source_columns.values()) | {"model_name", "patch_id"}
    patch_base = build_patch_level_base(joined_frame, excluded_columns=excluded_base_columns)

    metric_tables: list[pd.DataFrame] = []
    consensus_notes: list[str] = []
    for canonical_metric, source_column in metric_source_columns.items():
        pivot = joined_frame.pivot_table(
            index="patch_id",
            columns="model_name",
            values=source_column,
            aggfunc="first",
        )
        renamed_columns = {
            column: f"{canonical_metric}__{normalize_model_name(column)}"
            for column in pivot.columns
        }
        pivot = pivot.rename(columns=renamed_columns)
        per_model_columns = list(pivot.columns)
        pivot[f"{canonical_metric}_median"] = pivot[per_model_columns].median(axis=1, skipna=True)
        pivot[f"{canonical_metric}_model_count"] = pivot[per_model_columns].notna().sum(axis=1)
        metric_tables.append(pivot.reset_index())
        consensus_notes.append(
            f"Derived `{canonical_metric}_median` from source column `{source_column}` across {len(per_model_columns)} model columns."
        )

    model_table = patch_base.copy()
    for metric_table in metric_tables:
        model_table = model_table.merge(metric_table, on="patch_id", how="left")

    optional_join_infos: list[dict[str, Any]] = []
    for source_name, source_path in (
        ("embeddings_index", args.embeddings_index),
        ("embed_morph", args.embed_morph),
        ("patch_features", args.patch_features),
    ):
        model_table, join_info = join_optional_source(
            model_table=model_table,
            source_path=source_path,
            source_name=source_name,
            preferred_patch_id_col=args.patch_id_col,
            logger=logger,
        )
        optional_join_infos.append(join_info)

    ordered_columns = [
        column
        for column in (
            "patch_id",
            "slide_id",
            "dataset",
            "split",
            *EMBEDDING_METADATA_COLUMNS,
            "pq_median",
            "rq_median",
            "sq_median",
            "pixel_precision_median",
            "pixel_recall_median",
        )
        if column in model_table.columns
    ]
    ordered_columns.extend(
        [
            column
            for column in model_table.columns
            if column not in ordered_columns
        ]
    )
    model_table = model_table.loc[:, ordered_columns]

    duplicate_patch_id_count = verify_no_duplicate_patch_ids(model_table)
    if duplicate_patch_id_count:
        raise ValueError(
            f"Canonical modeling table still contains {duplicate_patch_id_count} duplicated `patch_id` values."
        )

    save_csv_table(model_table, output_path, index=False)
    reloaded_output = load_table(output_path)
    require_columns(reloaded_output, ["patch_id", "slide_id", "pq_median"], "reloaded modeling table")
    if len(reloaded_output) != len(model_table):
        raise AssertionError(
            f"Reloaded output row count {len(reloaded_output)} did not match in-memory row count {len(model_table)}."
        )

    artifact_checks = file_existence_rows(
        {
            "modeling_table": output_path,
            "config_json": output_dir / "config.json",
        }
    )
    report = {
        "stage_name": "build_model_table",
        "success": True,
        "input_path": str(resolve_path(args.joined_target_table)),
        "output_dir": str(output_dir),
        "summary": [
            {"name": "input_rows", "value": int(len(joined_frame))},
            {"name": "input_unique_patch_ids", "value": int(joined_frame["patch_id"].nunique())},
            {"name": "output_rows", "value": int(len(model_table))},
            {"name": "output_columns", "value": int(model_table.shape[1])},
            {"name": "duplicate_patch_id_count", "value": duplicate_patch_id_count},
            {"name": "metric_source_count", "value": int(len(metric_source_columns))},
            {"name": "missing_optional_sources", "value": int(sum(not info["used"] for info in optional_join_infos))},
        ],
        "artifact_checks": artifact_checks,
        "missingness_summary": missingness_summary(model_table),
        "notes": [
            *consensus_notes,
            *[
                f"Optional source `{info['source_name']}` used={info['used']} joined_columns={len(info.get('joined_columns', []))}"
                for info in optional_join_infos
            ],
            *[
                f"Metric `{canonical}` sourced from `{source}`."
                for canonical, source in metric_source_columns.items()
            ],
        ],
        "metric_source_columns": metric_source_columns,
        "optional_joins": optional_join_infos,
        "inconsistent_patch_columns": inconsistent_patch_columns,
    }
    write_validation_reports(output_dir, report)
    write_timing_json(
        output_dir,
        started_time,
        stage_name="build_model_table",
        extra={
            "output_rows": int(len(model_table)),
            "output_columns": int(model_table.shape[1]),
        },
    )
    logger.info("wrote modeling table rows=%s columns=%s path=%s", len(model_table), model_table.shape[1], output_path)


if __name__ == "__main__":
    main()
