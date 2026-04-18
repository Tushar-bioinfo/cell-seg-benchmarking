#!/usr/bin/env python3
"""Merge CoNIC patch morphology summaries with embedding metadata.

This script joins patch-level morphology features with tile embedding metadata
using the stable CoNIC patch identifier columns already present in both tables.
It preserves useful downstream metadata, keeps the existing patch morphology
features, and adds class-fraction and patch-richness summaries derived from the
per-patch CoNIC class-count columns.

Default behavior uses an ``inner`` join on ``sample_id`` because:

- the inspected CoNIC exports use ``sample_id`` as the stable raw-patch key
- both input tables already carry the same patch-level rows
- downstream analyses usually require both morphology and embedding metadata

If needed, callers can override the join columns and join mode explicitly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

DEFAULT_JOIN_COLUMNS: tuple[str, ...] = ("sample_id",)
CLASS_COUNT_COLUMNS: tuple[str, ...] = (
    "neutrophil",
    "epithelial",
    "lymphocyte",
    "plasma",
    "eosinophil",
    "connective",
)
EMBEDDING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "embedding_id",
    "embedding_path",
    "embedding_format",
    "embedding_row_offset",
    "embedding_dim",
)
FEATURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "count_total",
    "num_objects",
    "total_mask_area",
    "foreground_fraction",
)
FIXED_DERIVED_COLUMNS: tuple[str, ...] = (
    "total_nuclei",
    "dominant_class",
    "dominant_fraction",
    "richness_label",
)
EMBEDDING_INPUT_ROW_INDEX_COLUMN = "embedding_input_row_index"


class ScriptError(RuntimeError):
    """Raised when the CLI cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Merge CoNIC patch morphology features with embedding metadata and "
            "add per-class fraction and patch richness labels."
        ),
        epilog=(
            "Example:\n"
            "  python scripts/benchmarking/morphology/embed_morph.py "
            "--features-csv outputs/conic_liz/patch_features.csv "
            "--embeddings-csv outputs/conic_liz/embeddings/metadata/embeddings_index.csv "
            "--out-csv outputs/conic_liz/embed_morph.csv"
        ),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to patch_features.csv produced by the morphology workflow.",
    )
    parser.add_argument(
        "--embeddings-csv",
        type=Path,
        required=True,
        help="Path to embeddings_index.csv produced by the embedding workflow.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Path to write the merged output CSV.",
    )
    parser.add_argument(
        "--join-how",
        choices=("inner", "left", "right", "outer"),
        default="inner",
        help=(
            "Pandas merge mode. Default: inner, which keeps only rows present "
            "in both inputs."
        ),
    )
    parser.add_argument(
        "--join-columns",
        nargs="+",
        default=list(DEFAULT_JOIN_COLUMNS),
        help=(
            "Column(s) used to join the two tables. Default: sample_id. "
            "Override only if your exported schema differs."
        ),
    )
    parser.add_argument(
        "--rich-threshold",
        type=float,
        default=0.60,
        help=(
            "Minimum dominant-class fraction required for an enriched label. "
            "Default: 0.60."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more detailed logging.",
    )
    args = parser.parse_args()
    if not 0.0 < args.rich_threshold <= 1.0:
        parser.error("--rich-threshold must be > 0 and <= 1")
    return args


def configure_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    return logging.getLogger("embed_morph")


def ensure_file_exists(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ScriptError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise ScriptError(f"{label} is not a file: {resolved}")
    return resolved


def load_csv(path: Path, label: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Reading %s: %s", label, path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas error surface
        raise ScriptError(f"Failed to read {label} at {path}: {exc}") from exc
    logger.info("Loaded %s rows from %s", len(df), label)
    return df


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    label: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ScriptError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def validate_join_columns(
    features_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    join_columns: list[str],
) -> None:
    if not join_columns:
        raise ScriptError("At least one join column is required.")
    for column in join_columns:
        if column not in features_df.columns:
            raise ScriptError(f"features CSV is missing join column: {column}")
        if column not in embeddings_df.columns:
            raise ScriptError(f"embeddings CSV is missing join column: {column}")


def get_available_class_count_columns(
    features_df: pd.DataFrame,
    logger: logging.Logger,
) -> list[str]:
    available = [column for column in CLASS_COUNT_COLUMNS if column in features_df.columns]
    missing = [column for column in CLASS_COUNT_COLUMNS if column not in features_df.columns]
    if not available:
        raise ScriptError(
            "features CSV does not contain any recognized class-count columns. "
            f"Expected at least one of: {', '.join(CLASS_COUNT_COLUMNS)}"
        )
    if missing:
        logger.warning(
            "features CSV is missing class-count columns: %s. Fraction columns will be "
            "computed only for the available classes, so the fractions may not sum to 1.",
            ", ".join(missing),
        )
    return available


def validate_output_column_collisions(
    features_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    derived_columns: Iterable[str],
) -> None:
    reserved_columns = {EMBEDDING_INPUT_ROW_INDEX_COLUMN, *derived_columns}
    collisions = [
        column
        for column in sorted(reserved_columns)
        if column in features_df.columns or column in embeddings_df.columns
    ]
    if collisions:
        raise ScriptError(
            "Input files already contain reserved output columns: "
            + ", ".join(collisions)
        )


def check_duplicate_keys(
    df: pd.DataFrame,
    join_columns: list[str],
    label: str,
) -> None:
    duplicate_mask = df.duplicated(subset=join_columns, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        sample_keys = (
            df.loc[duplicate_mask, join_columns]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ScriptError(
            f"{label} has {duplicate_count} rows with duplicate join keys for "
            f"{join_columns}. Example keys: {sample_keys}"
        )


def report_unmatched_rows(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_columns: list[str],
    logger: logging.Logger,
) -> tuple[int, int]:
    left_keys = left_df[join_columns].drop_duplicates()
    right_keys = right_df[join_columns].drop_duplicates()
    left_only = left_keys.merge(right_keys, on=join_columns, how="left", indicator=True)
    right_only = right_keys.merge(left_keys, on=join_columns, how="left", indicator=True)
    left_unmatched = int((left_only["_merge"] == "left_only").sum())
    right_unmatched = int((right_only["_merge"] == "left_only").sum())
    logger.info("Unmatched feature rows: %d", left_unmatched)
    logger.info("Unmatched embedding rows: %d", right_unmatched)
    return left_unmatched, right_unmatched


def validate_consistent_shared_columns(
    features_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    join_columns: list[str],
    logger: logging.Logger,
) -> None:
    comparable_columns = [
        column
        for column in (
            "conic_index",
            "image_path",
            "mask_path",
            "class_label_path",
            "image_height",
            "image_width",
            "count_total",
            *CLASS_COUNT_COLUMNS,
        )
        if column in features_df.columns and column in embeddings_df.columns
    ]
    if not comparable_columns:
        return

    joined = features_df[join_columns + comparable_columns].merge(
        embeddings_df[join_columns + comparable_columns],
        on=join_columns,
        how="inner",
        suffixes=("_features", "_embeddings"),
        validate="one_to_one",
    )
    mismatches: list[str] = []
    for column in tqdm(
        comparable_columns,
        desc="Checking shared metadata",
        unit="column",
        disable=not logger.isEnabledFor(logging.DEBUG),
    ):
        left = joined[f"{column}_features"]
        right = joined[f"{column}_embeddings"]
        mismatch_count = int((left.fillna("__NA__") != right.fillna("__NA__")).sum())
        if mismatch_count:
            mismatches.append(f"{column} ({mismatch_count} mismatched rows)")
    if mismatches:
        raise ScriptError(
            "Shared metadata columns disagree between inputs: "
            + ", ".join(mismatches)
        )
    logger.debug("Shared metadata columns agree for %d merged rows", len(joined))


def warn_on_class_total_mismatches(df: pd.DataFrame, logger: logging.Logger) -> None:
    available_class_columns = [column for column in CLASS_COUNT_COLUMNS if column in df.columns]
    class_sum = df.loc[:, available_class_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    total = pd.to_numeric(df["count_total"], errors="coerce")
    mismatch_mask = total.notna() & class_sum.notna() & (total != class_sum)
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count:
        logger.warning(
            "count_total disagrees with the sum of class-count columns in %d rows; "
            "derived fractions will continue to use count_total as the total nuclei column.",
            mismatch_count,
        )


def build_output_columns(
    merged_df: pd.DataFrame,
    join_columns: list[str],
    class_count_columns: list[str],
) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()

    def append(column: str) -> None:
        if column not in seen:
            columns.append(column)
            seen.add(column)

    for column in join_columns:
        append(column)

    feature_priority = [
        "conic_index",
        "image_path",
        "mask_path",
        "class_label_path",
        "image_height",
        "image_width",
        "count_total",
        *class_count_columns,
        "input_row_index",
        "num_objects",
        "total_mask_area",
        "foreground_fraction",
        "mean_area",
        "median_area",
        "std_area",
        "mean_eccentricity",
        "mean_solidity",
        "mean_circularity",
    ]
    for column in feature_priority:
        if column in merged_df.columns:
            append(column)
    embedding_priority = [
        "embedding_input_row_index",
        "embedding_id",
        "resolved_image_path",
        "embedding_path",
        "embedding_format",
        "embedding_row_offset",
        "embedding_dim",
    ]
    for column in embedding_priority:
        if column in merged_df.columns:
            append(column)

    for column in build_derived_columns(class_count_columns):
        if column in merged_df.columns:
            append(column)

    for column in merged_df.columns:
        append(column)
    return columns


def reorder_columns(df: pd.DataFrame, ordered_columns: list[str]) -> pd.DataFrame:
    remaining = [column for column in df.columns if column not in ordered_columns]
    return df[ordered_columns + remaining]


def add_derived_columns(
    df: pd.DataFrame,
    threshold: float,
    class_count_columns: list[str],
) -> pd.DataFrame:
    result = df.copy()
    result["total_nuclei"] = pd.to_numeric(result["count_total"], errors="coerce")

    class_counts = result.loc[:, class_count_columns].apply(pd.to_numeric, errors="coerce")
    total = result["total_nuclei"]

    for column in tqdm(class_count_columns, desc="Computing class fractions", unit="column"):
        fraction_column = f"fraction_{column}"
        # Use NA when there are zero nuclei so downstream code can distinguish
        # "no nuclei present" from a true zero fraction among non-empty patches.
        result[fraction_column] = class_counts[column].where(total > 0).div(total.where(total > 0))

    dominant_count = class_counts.max(axis=1)
    dominant_column = class_counts.idxmax(axis=1)
    tie_count = class_counts.eq(dominant_count, axis=0).sum(axis=1)
    result["dominant_class"] = dominant_column.where(total > 0, "none")
    result.loc[(total > 0) & (tie_count > 1), "dominant_class"] = "tie"
    result["dominant_fraction"] = dominant_count.where(total > 0).div(total.where(total > 0))

    def richness_label(row: pd.Series) -> str:
        total_nuclei = row["total_nuclei"]
        if pd.isna(total_nuclei) or total_nuclei <= 0:
            return "no-nuclei"
        dominant_class = row["dominant_class"]
        dominant_fraction = row["dominant_fraction"]
        # Explicit rule:
        # - no nuclei -> no-nuclei
        # - epithelial / lymphocyte / connective with dominant fraction >= threshold
        #   -> corresponding enriched label
        # - anything else, including immune subclasses dominating, ties, or weak
        #   dominance below threshold -> mixed
        if pd.isna(dominant_fraction):
            return "mixed"
        if dominant_class == "tie":
            return "mixed"
        if dominant_class == "epithelial" and dominant_fraction >= threshold:
            return "epithelial-rich"
        if dominant_class == "lymphocyte" and dominant_fraction >= threshold:
            return "lymphocyte-rich"
        if dominant_class == "connective" and dominant_fraction >= threshold:
            return "connective-tissue-rich"
        return "mixed"

    result["richness_label"] = result.apply(richness_label, axis=1)
    return result


def build_derived_columns(class_count_columns: list[str]) -> list[str]:
    return (
        list(FIXED_DERIVED_COLUMNS[:1])
        + [f"fraction_{column}" for column in class_count_columns]
        + list(FIXED_DERIVED_COLUMNS[1:])
    )


def merge_tables(
    features_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    join_columns: list[str],
    join_how: str,
) -> pd.DataFrame:
    embeddings_for_merge = embeddings_df.copy()
    if (
        "input_row_index" in embeddings_for_merge.columns
        and "input_row_index" not in features_df.columns
    ):
        embeddings_for_merge = embeddings_for_merge.rename(
            columns={"input_row_index": EMBEDDING_INPUT_ROW_INDEX_COLUMN}
        )
    overlap_columns = [
        column
        for column in features_df.columns
        if column in embeddings_for_merge.columns and column not in join_columns
    ]
    merged_df = features_df.merge(
        embeddings_for_merge,
        on=join_columns,
        how=join_how,
        suffixes=("_features", "_embeddings"),
        validate="one_to_one",
    )
    for column in overlap_columns:
        feature_column = f"{column}_features"
        embedding_column = f"{column}_embeddings"
        if column == "input_row_index":
            merged_df[column] = merged_df[feature_column].combine_first(merged_df[embedding_column])
            merged_df[EMBEDDING_INPUT_ROW_INDEX_COLUMN] = merged_df[embedding_column]
        else:
            merged_df[column] = merged_df[feature_column].combine_first(merged_df[embedding_column])
        merged_df = merged_df.drop(columns=[feature_column, embedding_column])
    return merged_df


def write_output(df: pd.DataFrame, out_csv: Path, logger: logging.Logger) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing merged CSV: %s", out_csv)
    try:
        df.to_csv(out_csv, index=False)
    except Exception as exc:  # pragma: no cover - pandas error surface
        raise ScriptError(f"Failed to write output CSV at {out_csv}: {exc}") from exc


def log_summary(
    *,
    logger: logging.Logger,
    features_count: int,
    embeddings_count: int,
    merged_count: int,
    join_columns: list[str],
    join_how: str,
    unmatched_features: int,
    unmatched_embeddings: int,
    derived_columns: list[str],
) -> None:
    logger.info("Summary")
    logger.info("  features rows: %d", features_count)
    logger.info("  embeddings rows: %d", embeddings_count)
    logger.info("  merged rows: %d", merged_count)
    logger.info("  join columns: %s", ", ".join(join_columns))
    logger.info("  join mode: %s", join_how)
    logger.info("  unmatched features rows: %d", unmatched_features)
    logger.info("  unmatched embeddings rows: %d", unmatched_embeddings)
    logger.info("  derived columns: %s", ", ".join(derived_columns))


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.verbose)

    try:
        features_csv = ensure_file_exists(args.features_csv, "Features CSV")
        embeddings_csv = ensure_file_exists(args.embeddings_csv, "Embeddings CSV")

        features_df = load_csv(features_csv, "features CSV", logger)
        embeddings_df = load_csv(embeddings_csv, "embeddings CSV", logger)

        class_count_columns = get_available_class_count_columns(features_df, logger)
        derived_columns = build_derived_columns(class_count_columns)

        validate_required_columns(
            features_df,
            FEATURE_REQUIRED_COLUMNS,
            "features CSV",
        )
        validate_required_columns(
            embeddings_df,
            EMBEDDING_REQUIRED_COLUMNS,
            "embeddings CSV",
        )
        validate_join_columns(features_df, embeddings_df, args.join_columns)
        validate_output_column_collisions(features_df, embeddings_df, derived_columns)
        check_duplicate_keys(features_df, args.join_columns, "features CSV")
        check_duplicate_keys(embeddings_df, args.join_columns, "embeddings CSV")

        unmatched_features, unmatched_embeddings = report_unmatched_rows(
            features_df,
            embeddings_df,
            args.join_columns,
            logger,
        )
        validate_consistent_shared_columns(
            features_df,
            embeddings_df,
            args.join_columns,
            logger,
        )
        warn_on_class_total_mismatches(features_df, logger)

        logger.info("Merging tables with %s join on %s", args.join_how, ", ".join(args.join_columns))
        merged_df = merge_tables(
            features_df,
            embeddings_df,
            args.join_columns,
            args.join_how,
        )
        logger.info("Adding derived columns")
        merged_df = add_derived_columns(merged_df, args.rich_threshold, class_count_columns)
        ordered_columns = build_output_columns(merged_df, args.join_columns, class_count_columns)
        merged_df = reorder_columns(merged_df, ordered_columns)
        write_output(merged_df, args.out_csv.expanduser().resolve(), logger)

        log_summary(
            logger=logger,
            features_count=len(features_df),
            embeddings_count=len(embeddings_df),
            merged_count=len(merged_df),
            join_columns=args.join_columns,
            join_how=args.join_how,
            unmatched_features=unmatched_features,
            unmatched_embeddings=unmatched_embeddings,
            derived_columns=derived_columns,
        )
    except ScriptError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
