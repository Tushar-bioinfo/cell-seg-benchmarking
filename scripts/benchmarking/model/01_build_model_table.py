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
- Optional evaluation CSVs when missing metrics must be reattached before the
  canonical modeling table can be built.

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

AUTO_EVAL_JOIN_COLUMN_CANDIDATES = (
    "patch_id",
    "image_id",
    "match_key",
    "relative_image_path",
    "image_path",
    "pred_relative_path",
    "pred_file_name",
    "relative_mask_path",
    "gt_relative_path",
    "gt_file_name",
    "pred_path",
    "gt_path",
)

ENRICHED_EVAL_SOURCE_COLUMNS = {
    "pq": "instance_pq",
    "rq": "instance_rq",
    "sq": "instance_sq",
    "pixel_precision": "pixel_precision",
    "pixel_recall": "pixel_recall",
}


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
        "--required-consensus-metrics",
        nargs="*",
        default=["pq"],
        choices=sorted(METRIC_COLUMN_ALIASES),
        help=(
            "Consensus metrics that must be present in the final modeling table. "
            "Pass `pq rq sq pixel_precision pixel_recall` when failure-mode training is required."
        ),
    )
    parser.add_argument(
        "--auto-enrich-missing-metrics",
        action="store_true",
        help=(
            "If required metrics are missing from the joined target table, reattach them from "
            "evaluation CSVs before building the canonical modeling table."
        ),
    )
    parser.add_argument(
        "--eval-dir",
        default="outputs/conic_liz",
        help="Directory searched for evaluation CSVs used to enrich missing metrics.",
    )
    parser.add_argument(
        "--eval-glob",
        default="*_evaluation.csv",
        help="Glob used under --eval-dir to discover evaluation CSVs.",
    )
    parser.add_argument(
        "--eval-files",
        nargs="*",
        default=[],
        help="Optional explicit evaluation CSVs included in addition to --eval-dir/--eval-glob.",
    )
    parser.add_argument(
        "--status-col",
        default="status",
        help="Optional evaluation status column used with --status-values while enriching metrics.",
    )
    parser.add_argument(
        "--status-values",
        nargs="*",
        default=["ok"],
        help="If the status column exists, keep only these evaluation rows while enriching metrics.",
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


def resolve_eval_paths(eval_dir: str, eval_glob: str, eval_files: list[str]) -> list[Path]:
    """Resolve and deduplicate evaluation CSV paths."""

    discovered: list[Path] = []
    if eval_dir:
        resolved_dir = resolve_path(eval_dir)
        if not resolved_dir.exists():
            raise FileNotFoundError(f"Evaluation directory does not exist: {resolved_dir}")
        discovered.extend(sorted(resolved_dir.glob(eval_glob)))
    for source_path in eval_files:
        resolved_path = resolve_path(source_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Evaluation file does not exist: {resolved_path}")
        discovered.append(resolved_path)

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path in discovered:
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        unique_paths.append(resolved_path)
    return unique_paths


def build_auto_join_variants(series: pd.Series) -> list[tuple[str, pd.Series]]:
    """Build normalized variants for matching evaluation rows back to `patch_id`."""

    normalized = series.astype("string").fillna("").str.strip()
    basename = normalized.map(lambda value: Path(value).name if value else value)
    stem = basename.str.replace(r"\.[^.]+$", "", regex=True)
    stripped_role = stem.str.replace(r"_(image|mask|class_labels)$", "", regex=True)
    return [
        ("raw", normalized),
        ("basename", basename),
        ("basename_stem", stem),
        ("basename_strip_role", stripped_role),
    ]


def select_eval_join_series(
    eval_frame: pd.DataFrame,
    manifest_keys: set[str],
    eval_path: Path,
) -> tuple[str, pd.Series, str, int]:
    """Choose the evaluation join column/normalization with the most matches."""

    candidate_columns = [
        column for column in AUTO_EVAL_JOIN_COLUMN_CANDIDATES if column in eval_frame.columns
    ]
    if not candidate_columns:
        raise ValueError(
            f"Evaluation file {eval_path.name} does not contain any supported join columns: "
            f"{list(AUTO_EVAL_JOIN_COLUMN_CANDIDATES)}"
        )

    best_column: str | None = None
    best_variant_name = ""
    best_series: pd.Series | None = None
    best_match_count = -1
    best_non_blank_count = -1
    for column in candidate_columns:
        for variant_name, variant_series in build_auto_join_variants(eval_frame[column]):
            non_blank = variant_series.loc[variant_series.ne("")]
            unique_values = pd.Index(non_blank.unique())
            match_count = int(unique_values.isin(manifest_keys).sum())
            non_blank_count = int(len(unique_values))
            if match_count > best_match_count or (
                match_count == best_match_count and non_blank_count > best_non_blank_count
            ):
                best_column = column
                best_variant_name = variant_name
                best_series = variant_series
                best_match_count = match_count
                best_non_blank_count = non_blank_count

    if best_column is None or best_series is None or best_match_count <= 0:
        raise ValueError(
            f"Could not auto-detect a usable evaluation join column for {eval_path.name}. "
            "The joined target table patch IDs did not match any supported evaluation identifiers."
        )
    return best_column, best_series, best_variant_name, best_match_count


def derive_model_name_from_path(path: Path) -> str:
    """Derive a fallback model name from an evaluation filename."""

    name = path.stem
    if name.endswith("_evaluation"):
        name = name[: -len("_evaluation")]
    return name


def split_eval_groups(eval_frame: pd.DataFrame, eval_path: Path) -> list[tuple[str, pd.DataFrame]]:
    """Split an evaluation CSV by model name when available."""

    if "model_name" in eval_frame.columns:
        model_series = eval_frame["model_name"].astype("string").fillna("").str.strip()
        non_blank = model_series.ne("")
        if non_blank.any():
            groups: list[tuple[str, pd.DataFrame]] = []
            for model_name, group in eval_frame.loc[non_blank].groupby(model_series.loc[non_blank], sort=True):
                groups.append((str(model_name), group.copy()))
            if (~non_blank).any():
                groups.append((derive_model_name_from_path(eval_path), eval_frame.loc[~non_blank].copy()))
            return groups
    return [(derive_model_name_from_path(eval_path), eval_frame.copy())]


def enrich_joined_frame_with_eval_metrics(
    joined_frame: pd.DataFrame,
    required_metrics: list[str],
    args: argparse.Namespace,
    logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reattach missing per-model metrics from evaluation CSVs when requested."""

    info: dict[str, Any] = {
        "requested": bool(args.auto_enrich_missing_metrics),
        "used": False,
        "required_metrics": required_metrics,
        "missing_metrics_before": [],
        "missing_metrics_after": [],
        "eval_paths": [],
        "added_source_columns": [],
        "per_file_matches": [],
    }
    if not required_metrics:
        return joined_frame, info

    missing_before = [
        metric for metric in required_metrics if metric not in infer_metric_source_columns(joined_frame.columns)
    ]
    info["missing_metrics_before"] = missing_before
    if not missing_before:
        return joined_frame, info
    if not args.auto_enrich_missing_metrics:
        return joined_frame, info

    eval_paths = resolve_eval_paths(args.eval_dir, args.eval_glob, list(args.eval_files))
    if not eval_paths:
        raise FileNotFoundError(
            "Required consensus metrics were missing from the joined target table and no evaluation CSVs "
            f"were found from eval_dir={args.eval_dir!r}, eval_glob={args.eval_glob!r}, eval_files={args.eval_files!r}."
        )
    info["eval_paths"] = [str(path) for path in eval_paths]

    manifest_frame = joined_frame.loc[:, ["patch_id"]].drop_duplicates().copy()
    manifest_frame["_join_key"] = manifest_frame["patch_id"].astype("string").fillna("").str.strip()
    manifest_keys = set(manifest_frame["_join_key"])
    enriched_frames: list[pd.DataFrame] = []

    for eval_path in eval_paths:
        eval_frame = load_table(eval_path)
        for model_name, eval_group in split_eval_groups(eval_frame, eval_path):
            filtered_group = eval_group.copy()
            if args.status_values and args.status_col in filtered_group.columns:
                filtered_group = filtered_group.loc[
                    filtered_group[args.status_col].isin(args.status_values)
                ].copy()
            join_column, join_series, join_strategy, match_count = select_eval_join_series(
                filtered_group,
                manifest_keys,
                eval_path,
            )
            filtered_group["_join_key"] = join_series
            filtered_group = filtered_group.loc[filtered_group["_join_key"].ne("")].copy()
            duplicate_count = int(filtered_group.duplicated(subset=["_join_key"]).sum())
            if duplicate_count:
                logger.warning(
                    "Evaluation file %s model=%s had %s duplicate join keys on column %s. Keeping the first row per key.",
                    eval_path.name,
                    model_name,
                    duplicate_count,
                    join_column,
                )
                filtered_group = filtered_group.drop_duplicates(subset=["_join_key"], keep="first").copy()

            rename_map: dict[str, str] = {}
            keep_columns = ["_join_key"]
            used_source_columns: dict[str, str] = {}
            for canonical_metric in missing_before:
                for candidate in METRIC_COLUMN_ALIASES[canonical_metric]:
                    if candidate in filtered_group.columns:
                        standard_column = ENRICHED_EVAL_SOURCE_COLUMNS[canonical_metric]
                        rename_map[candidate] = standard_column
                        keep_columns.append(candidate)
                        used_source_columns[canonical_metric] = candidate
                        break
            if len(keep_columns) == 1:
                logger.warning(
                    "Evaluation file %s model=%s did not expose any of the missing metrics %s.",
                    eval_path.name,
                    model_name,
                    missing_before,
                )
                continue

            metric_subset = filtered_group.loc[:, keep_columns].rename(columns=rename_map).copy()
            merged = manifest_frame.merge(metric_subset, on="_join_key", how="left", validate="one_to_one")
            merged["model_name"] = model_name
            enriched_frames.append(merged.drop(columns=["_join_key"]))
            info["per_file_matches"].append(
                {
                    "eval_file": str(eval_path),
                    "model_name": model_name,
                    "join_column": join_column,
                    "join_strategy": join_strategy,
                    "match_count": match_count,
                    "used_source_columns": used_source_columns,
                }
            )

    if not enriched_frames:
        raise ValueError(
            "Auto-enrichment was requested, but no evaluation rows yielded any of the missing metrics "
            f"{missing_before}. Check the evaluation directory and metric column names."
        )

    enriched_frame = pd.concat(enriched_frames, ignore_index=True)
    duplicate_patch_model_count = int(
        enriched_frame.duplicated(subset=["patch_id", "model_name"]).sum()
    )
    if duplicate_patch_model_count:
        raise ValueError(
            "Auto-enriched evaluation metrics produced duplicated `(patch_id, model_name)` rows: "
            f"{duplicate_patch_model_count}"
        )

    merge_source_columns = sorted(
        {
            ENRICHED_EVAL_SOURCE_COLUMNS[metric]
            for metric in missing_before
            if ENRICHED_EVAL_SOURCE_COLUMNS[metric] in enriched_frame.columns
        }
    )
    joined_frame = joined_frame.merge(
        enriched_frame,
        on=["patch_id", "model_name"],
        how="left",
        suffixes=("", "__enriched"),
    )
    added_source_columns: list[str] = []
    for column in merge_source_columns:
        enriched_column = f"{column}__enriched"
        if enriched_column in joined_frame.columns:
            if column in joined_frame.columns:
                joined_frame[column] = joined_frame[column].where(
                    joined_frame[column].notna(),
                    joined_frame[enriched_column],
                )
                joined_frame = joined_frame.drop(columns=[enriched_column])
            else:
                joined_frame = joined_frame.rename(columns={enriched_column: column})
        if column in joined_frame.columns:
            added_source_columns.append(column)

    info["used"] = True
    info["added_source_columns"] = added_source_columns
    info["missing_metrics_after"] = [
        metric for metric in required_metrics if metric not in infer_metric_source_columns(joined_frame.columns)
    ]
    return joined_frame, info


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
        "required_consensus_metrics": list(args.required_consensus_metrics),
        "auto_enrich_missing_metrics": bool(args.auto_enrich_missing_metrics),
        "eval_dir": str(resolve_path(args.eval_dir)) if args.eval_dir else "",
        "eval_glob": args.eval_glob,
        "eval_files": [str(resolve_path(path)) for path in args.eval_files],
        "status_col": args.status_col,
        "status_values": list(args.status_values),
        "output": str(output_path),
        "generated_at_utc": now_utc_iso(),
    }
    write_json(output_dir / "config.json", config)

    joined_frame = load_table(args.joined_target_table).copy()
    require_columns(joined_frame, [args.patch_id_col, args.slide_id_col, "model_name"], "joined target table")
    joined_frame = joined_frame.rename(
        columns={args.patch_id_col: "patch_id", args.slide_id_col: "slide_id"}
    )

    enrichment_info: dict[str, Any] = {
        "requested": False,
        "used": False,
        "required_metrics": list(args.required_consensus_metrics),
        "missing_metrics_before": [],
        "missing_metrics_after": [],
        "eval_paths": [],
        "added_source_columns": [],
        "per_file_matches": [],
    }
    if args.required_consensus_metrics:
        joined_frame, enrichment_info = enrich_joined_frame_with_eval_metrics(
            joined_frame=joined_frame,
            required_metrics=list(args.required_consensus_metrics),
            args=args,
            logger=logger,
        )

    metric_source_columns = infer_metric_source_columns(joined_frame.columns)
    missing_required_metrics = [
        metric for metric in args.required_consensus_metrics if metric not in metric_source_columns
    ]
    if missing_required_metrics:
        available = ", ".join(sorted(joined_frame.columns))
        alias_text = {
            metric: list(METRIC_COLUMN_ALIASES[metric]) for metric in missing_required_metrics
        }
        raise ValueError(
            "Could not derive all required consensus metrics for the modeling table. "
            f"Missing metrics: {missing_required_metrics}. "
            f"Tried aliases: {alias_text}. "
            f"Available columns: {available}"
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

    excluded_base_columns = {
        candidate
        for candidates in METRIC_COLUMN_ALIASES.values()
        for candidate in candidates
    } | {"model_name", "patch_id", "pixel_dice"}
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
    required_output_columns = [
        "patch_id",
        "slide_id",
        *[f"{metric}_median" for metric in args.required_consensus_metrics],
    ]
    require_columns(reloaded_output, required_output_columns, "reloaded modeling table")
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
            {"name": "required_metric_count", "value": int(len(args.required_consensus_metrics))},
            {"name": "missing_optional_sources", "value": int(sum(not info["used"] for info in optional_join_infos))},
        ],
        "artifact_checks": artifact_checks,
        "missingness_summary": missingness_summary(model_table),
        "notes": [
            *consensus_notes,
            *(
                [
                    "Auto-enriched missing metrics from evaluation CSVs before collapsing to one row per patch.",
                    f"Evaluation CSVs used for enrichment: {enrichment_info['eval_paths']}",
                    f"Added source metric columns: {enrichment_info['added_source_columns']}",
                ]
                if enrichment_info.get("used")
                else []
            ),
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
        "enrichment_info": enrichment_info,
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
