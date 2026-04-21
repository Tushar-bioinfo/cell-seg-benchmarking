#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_MANIFEST_PATH = Path("outputs/conic_liz/failure_prediction/patch_manifest.parquet")
DEFAULT_OUTPUT_PATH = Path("outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet")
DEFAULT_EVAL_DIR = Path("outputs/conic_liz")
DEFAULT_EVAL_GLOB = "*_evaluation.csv"

DEFAULT_MANIFEST_PATCH_ID_COL = "patch_id"
DEFAULT_MANIFEST_SLIDE_ID_COL = "slide_id"
DEFAULT_MANIFEST_DATASET_COL = "dataset"
DEFAULT_MANIFEST_SPLIT_COL = "split"
DEFAULT_MANIFEST_EMBEDDING_PATH_COL = "embedding_path"
DEFAULT_MANIFEST_JOIN_COL = "patch_id"

DEFAULT_EVAL_JOIN_COL = "patch_id"
DEFAULT_EVAL_MODEL_COL = "model_name"
DEFAULT_STATUS_COL = "status"
DEFAULT_TARGET_METRICS = ("instance_pq", "pixel_dice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Join a canonical patch manifest with one or more segmentation evaluation files "
            "and create one row per patch_id x model_name."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Input patch manifest (.csv or .parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output joined manifest (.csv or .parquet).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root used to resolve relative manifest/evaluation/output paths.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Directory searched by --eval-glob for evaluation files.",
    )
    parser.add_argument(
        "--eval-glob",
        default=DEFAULT_EVAL_GLOB,
        help="Glob pattern used under --eval-dir to discover evaluation files.",
    )
    parser.add_argument(
        "--eval-files",
        nargs="*",
        type=Path,
        default=[],
        help="Optional explicit evaluation files to include in addition to --eval-dir/--eval-glob matches.",
    )
    parser.add_argument(
        "--manifest-patch-id-col",
        default=DEFAULT_MANIFEST_PATCH_ID_COL,
        help="Manifest column mapped to output column patch_id.",
    )
    parser.add_argument(
        "--manifest-slide-id-col",
        default=DEFAULT_MANIFEST_SLIDE_ID_COL,
        help="Manifest column mapped to output column slide_id.",
    )
    parser.add_argument(
        "--manifest-dataset-col",
        default=DEFAULT_MANIFEST_DATASET_COL,
        help="Manifest column mapped to output column dataset.",
    )
    parser.add_argument(
        "--manifest-split-col",
        default=DEFAULT_MANIFEST_SPLIT_COL,
        help="Optional manifest column mapped to output column split.",
    )
    parser.add_argument(
        "--manifest-embedding-path-col",
        default=DEFAULT_MANIFEST_EMBEDDING_PATH_COL,
        help="Manifest column mapped to output column embedding_path.",
    )
    parser.add_argument(
        "--manifest-join-col",
        default=DEFAULT_MANIFEST_JOIN_COL,
        help="Manifest column used to join to evaluation rows.",
    )
    parser.add_argument(
        "--manifest-extra-cols",
        nargs="*",
        default=[],
        help="Additional manifest columns to preserve in the output.",
    )
    parser.add_argument(
        "--manifest-join-basename",
        action="store_true",
        help="Convert manifest join values to their basename before joining.",
    )
    parser.add_argument(
        "--manifest-join-strip-prefix",
        default="",
        help="Prefix stripped from manifest join values after optional basename normalization.",
    )
    parser.add_argument(
        "--manifest-join-strip-suffix",
        default="",
        help="Suffix stripped from manifest join values after optional basename normalization.",
    )
    parser.add_argument(
        "--eval-join-col",
        default=DEFAULT_EVAL_JOIN_COL,
        help="Evaluation column used to join back to the patch manifest.",
    )
    parser.add_argument(
        "--eval-model-col",
        default=DEFAULT_EVAL_MODEL_COL,
        help="Evaluation column containing model names. If absent, model_name is derived from the filename.",
    )
    parser.add_argument(
        "--status-col",
        default=DEFAULT_STATUS_COL,
        help="Optional evaluation status column used with --status-values.",
    )
    parser.add_argument(
        "--status-values",
        nargs="*",
        default=["ok"],
        help="If the status column exists, keep only these evaluation rows. Pass no values to disable filtering.",
    )
    parser.add_argument(
        "--target-metrics",
        nargs="+",
        default=list(DEFAULT_TARGET_METRICS),
        help="Evaluation metric columns to attach as targets.",
    )
    parser.add_argument(
        "--eval-extra-cols",
        nargs="*",
        default=[],
        help="Additional evaluation columns to preserve in the output.",
    )
    parser.add_argument(
        "--eval-join-basename",
        action="store_true",
        help="Convert evaluation join values to their basename before joining.",
    )
    parser.add_argument(
        "--eval-join-strip-prefix",
        default="",
        help="Prefix stripped from evaluation join values after optional basename normalization.",
    )
    parser.add_argument(
        "--eval-join-strip-suffix",
        default="",
        help="Suffix stripped from evaluation join values after optional basename normalization.",
    )
    parser.add_argument(
        "--eval-dedupe-keep",
        choices=("first", "last"),
        default="first",
        help="Which duplicate evaluation row to keep per model_name x join key.",
    )
    parser.add_argument(
        "--min-manifest-coverage",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of manifest rows that must find a matching evaluation row for each model. "
            "Set above 0 to fail on weak joins."
        ),
    )
    parser.add_argument(
        "--unmatched-example-limit",
        type=int,
        default=5,
        help="Maximum number of unmatched manifest/evaluation join-key examples to log per model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("attach_eval_targets")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def find_repo_root(start: Path | None = None) -> Path | None:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_root(repo_root_arg: Path | None) -> Path | None:
    if repo_root_arg is not None:
        return repo_root_arg.expanduser().resolve()
    return find_repo_root(Path(__file__).resolve())


def resolve_cli_path(path_like: Path, *, repo_root: Path | None) -> Path:
    path = path_like.expanduser()
    if path.is_absolute():
        return path.resolve()
    if repo_root is not None:
        return (repo_root / path).resolve()
    return path.resolve()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format for {path}. Expected .csv or .parquet.")


def write_table(dataframe: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        dataframe.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format for {path}. Expected .csv or .parquet.")


def validate_required_columns(dataframe: pd.DataFrame, required_columns: list[str], *, context: str) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required columns: " + ", ".join(repr(column) for column in missing)
        )


def normalize_join_series(
    series: pd.Series,
    *,
    use_basename: bool,
    strip_prefix: str,
    strip_suffix: str,
) -> pd.Series:
    normalized = series.astype("string").fillna("").str.strip()
    if use_basename:
        normalized = normalized.map(lambda value: Path(value).name if value else value)
    if strip_prefix:
        normalized = normalized.str.removeprefix(strip_prefix)
    if strip_suffix:
        normalized = normalized.str.removesuffix(strip_suffix)
    return normalized


def resolve_eval_paths(
    *,
    eval_dir: Path,
    eval_glob: str,
    explicit_paths: list[Path],
    repo_root: Path | None,
) -> list[Path]:
    discovered: list[Path] = []
    if eval_glob:
        discovered.extend(sorted(eval_dir.glob(eval_glob)))
    for path in explicit_paths:
        discovered.append(resolve_cli_path(path, repo_root=repo_root))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def derive_model_name_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_evaluation"):
        name = name[: -len("_evaluation")]
    return name


def log_unmatched_examples(
    *,
    label: str,
    values: pd.Series,
    limit: int,
    logger: logging.Logger,
) -> None:
    examples = values.dropna().astype(str).loc[lambda s: s.str.strip().ne("")].head(limit).tolist()
    if examples:
        logger.warning("%s examples: %s", label, examples)


def deduplicate_eval_rows(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    join_col: str,
    keep: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    duplicate_mask = dataframe.duplicated(subset=[join_col], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        return dataframe

    duplicate_keys = dataframe.loc[duplicate_mask, join_col].astype(str).nunique()
    logger.warning(
        "Model %s has %d duplicate evaluation row(s) across %d join key(s). Keeping the %s occurrence.",
        model_name,
        duplicate_count,
        duplicate_keys,
        keep,
    )
    return dataframe.drop_duplicates(subset=[join_col], keep=keep).copy()


def build_manifest_base(dataframe: pd.DataFrame, args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    required_manifest_columns = [
        args.manifest_patch_id_col,
        args.manifest_slide_id_col,
        args.manifest_dataset_col,
        args.manifest_embedding_path_col,
        args.manifest_join_col,
    ]
    optional_split_present = args.manifest_split_col in dataframe.columns
    required_manifest_columns.extend(args.manifest_extra_cols)
    validate_required_columns(dataframe, required_manifest_columns, context="Patch manifest")

    manifest = pd.DataFrame(index=dataframe.index)
    manifest["patch_id"] = dataframe[args.manifest_patch_id_col]
    manifest["slide_id"] = dataframe[args.manifest_slide_id_col]
    manifest["dataset"] = dataframe[args.manifest_dataset_col]
    if optional_split_present:
        manifest["split"] = dataframe[args.manifest_split_col]
    else:
        logger.warning(
            "Manifest split column %r is absent. The output will not include split.",
            args.manifest_split_col,
        )
    manifest["embedding_path"] = dataframe[args.manifest_embedding_path_col]

    extra_cols = [
        column
        for column in args.manifest_extra_cols
        if column
        not in {
            args.manifest_patch_id_col,
            args.manifest_slide_id_col,
            args.manifest_dataset_col,
            args.manifest_split_col,
            args.manifest_embedding_path_col,
            args.manifest_join_col,
        }
    ]
    for column in extra_cols:
        manifest[column] = dataframe[column]

    manifest["_join_key"] = normalize_join_series(
        dataframe[args.manifest_join_col],
        use_basename=args.manifest_join_basename,
        strip_prefix=args.manifest_join_strip_prefix,
        strip_suffix=args.manifest_join_strip_suffix,
    )

    missing_join_mask = manifest["_join_key"].eq("")
    if missing_join_mask.any():
        raise ValueError(
            f"Manifest join column {args.manifest_join_col!r} produced {int(missing_join_mask.sum())} blank join key(s)."
        )

    if manifest["patch_id"].duplicated().any():
        duplicate_count = int(manifest["patch_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"Patch manifest contains {duplicate_count} duplicate patch_id row(s). "
            "Run the canonical manifest builder first."
        )

    return manifest


def split_eval_by_model(
    dataframe: pd.DataFrame,
    *,
    eval_path: Path,
    model_col: str,
    logger: logging.Logger,
) -> list[tuple[str, pd.DataFrame]]:
    if model_col in dataframe.columns:
        model_series = dataframe[model_col].astype("string").fillna("").str.strip()
        non_blank = model_series.ne("")
        if non_blank.any():
            grouped: list[tuple[str, pd.DataFrame]] = []
            for model_name, group in dataframe.loc[non_blank].groupby(model_series.loc[non_blank], sort=True):
                grouped.append((str(model_name), group.copy()))
            blank_count = int((~non_blank).sum())
            if blank_count:
                logger.warning(
                    "%s contains %d row(s) with blank %r. They will use a filename-derived model name.",
                    eval_path.name,
                    blank_count,
                    model_col,
                )
                grouped.append((derive_model_name_from_path(eval_path), dataframe.loc[~non_blank].copy()))
            return grouped

    return [(derive_model_name_from_path(eval_path), dataframe.copy())]


def process_eval_group(
    *,
    manifest_base: pd.DataFrame,
    eval_group: pd.DataFrame,
    model_name: str,
    eval_path: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> pd.DataFrame:
    required_eval_columns = [args.eval_join_col, *args.target_metrics]
    validate_required_columns(
        eval_group,
        required_eval_columns + [column for column in args.eval_extra_cols if column],
        context=f"Evaluation file {eval_path.name}",
    )

    filtered = eval_group.copy()
    if args.status_values and args.status_col in filtered.columns:
        before = len(filtered)
        filtered = filtered.loc[filtered[args.status_col].isin(args.status_values)].copy()
        logger.info(
            "Model %s from %s: kept %d/%d row(s) after filtering %r by %s.",
            model_name,
            eval_path.name,
            len(filtered),
            before,
            args.status_col,
            args.status_values,
        )
    elif args.status_values and args.status_col not in filtered.columns:
        logger.warning(
            "Model %s from %s: status column %r is absent. Skipping status filtering.",
            model_name,
            eval_path.name,
            args.status_col,
        )

    filtered["_join_key"] = normalize_join_series(
        filtered[args.eval_join_col],
        use_basename=args.eval_join_basename,
        strip_prefix=args.eval_join_strip_prefix,
        strip_suffix=args.eval_join_strip_suffix,
    )

    blank_join_mask = filtered["_join_key"].eq("")
    if blank_join_mask.any():
        raise ValueError(
            f"Model {model_name} from {eval_path.name} produced {int(blank_join_mask.sum())} blank join key(s)."
        )

    filtered = deduplicate_eval_rows(
        filtered,
        model_name=model_name,
        join_col="_join_key",
        keep=args.eval_dedupe_keep,
        logger=logger,
    )

    eval_keep_cols = ["_join_key", *args.target_metrics]
    extra_cols = [
        column
        for column in args.eval_extra_cols
        if column not in {args.eval_join_col, args.eval_model_col, args.status_col, *args.target_metrics}
    ]
    eval_keep_cols.extend(extra_cols)
    eval_subset = filtered[eval_keep_cols].copy()

    merged = manifest_base.merge(
        eval_subset,
        on="_join_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    merged["model_name"] = model_name

    matched_mask = merged["_merge"].eq("both")
    matched_count = int(matched_mask.sum())
    manifest_count = len(manifest_base)
    unmatched_manifest_count = manifest_count - matched_count
    coverage = matched_count / manifest_count if manifest_count else 0.0

    logger.info(
        "Model %s from %s: matched %d/%d manifest row(s) (coverage=%.4f). Unmatched manifest rows=%d.",
        model_name,
        eval_path.name,
        matched_count,
        manifest_count,
        coverage,
        unmatched_manifest_count,
    )

    if matched_count == 0:
        raise ValueError(
            f"Model {model_name} from {eval_path.name} had zero matched rows. "
            "Check the join column mapping and normalization flags."
        )
    if coverage < args.min_manifest_coverage:
        raise ValueError(
            f"Model {model_name} from {eval_path.name} only matched {coverage:.4f} of manifest rows, "
            f"below --min-manifest-coverage={args.min_manifest_coverage:.4f}."
        )

    manifest_keys = set(manifest_base["_join_key"])
    eval_keys = set(eval_subset["_join_key"])
    unmatched_eval_keys = eval_keys - manifest_keys
    logger.info(
        "Model %s from %s: unmatched evaluation rows=%d.",
        model_name,
        eval_path.name,
        len(unmatched_eval_keys),
    )

    if unmatched_manifest_count:
        log_unmatched_examples(
            label=f"Model {model_name} unmatched manifest join key",
            values=merged.loc[~matched_mask, "_join_key"],
            limit=args.unmatched_example_limit,
            logger=logger,
        )
    if unmatched_eval_keys:
        unmatched_eval_series = pd.Series(sorted(unmatched_eval_keys), dtype="string")
        log_unmatched_examples(
            label=f"Model {model_name} unmatched evaluation join key",
            values=unmatched_eval_series,
            limit=args.unmatched_example_limit,
            logger=logger,
        )

    merged = merged.drop(columns=["_merge"])
    return merged


def order_output_columns(dataframe: pd.DataFrame, *, target_metrics: list[str]) -> pd.DataFrame:
    preferred = ["patch_id", "slide_id", "dataset"]
    if "split" in dataframe.columns:
        preferred.append("split")
    preferred.extend(["model_name", "embedding_path", *target_metrics])
    ordered = [column for column in preferred if column in dataframe.columns]
    ordered.extend(column for column in dataframe.columns if column not in ordered and column != "_join_key")
    return dataframe[ordered]


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.verbose)

    repo_root = resolve_repo_root(args.repo_root)
    manifest_path = resolve_cli_path(args.manifest, repo_root=repo_root)
    output_path = resolve_cli_path(args.output, repo_root=repo_root)
    eval_dir = resolve_cli_path(args.eval_dir, repo_root=repo_root)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Patch manifest does not exist: {manifest_path}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir}")

    eval_paths = resolve_eval_paths(
        eval_dir=eval_dir,
        eval_glob=args.eval_glob,
        explicit_paths=args.eval_files,
        repo_root=repo_root,
    )
    if not eval_paths:
        raise FileNotFoundError(
            f"No evaluation files were found from eval_dir={eval_dir}, eval_glob={args.eval_glob!r}, "
            f"and eval_files={args.eval_files!r}."
        )

    logger.info("Reading patch manifest: %s", manifest_path)
    manifest_df = read_table(manifest_path)
    logger.info("Loaded patch manifest with %d row(s) x %d column(s).", len(manifest_df), len(manifest_df.columns))
    manifest_base = build_manifest_base(manifest_df, args, logger)

    logger.info("Processing %d evaluation file(s).", len(eval_paths))
    joined_frames: list[pd.DataFrame] = []
    for eval_path in tqdm(eval_paths, desc="evaluation files"):
        logger.info("Reading evaluation file: %s", eval_path)
        eval_df = read_table(eval_path)
        model_groups = split_eval_by_model(
            eval_df,
            eval_path=eval_path,
            model_col=args.eval_model_col,
            logger=logger,
        )
        for model_name, eval_group in model_groups:
            joined_frames.append(
                process_eval_group(
                    manifest_base=manifest_base,
                    eval_group=eval_group,
                    model_name=model_name,
                    eval_path=eval_path,
                    args=args,
                    logger=logger,
                )
            )

    if not joined_frames:
        raise RuntimeError("No joined output frames were produced.")

    joined = pd.concat(joined_frames, ignore_index=True)
    joined = order_output_columns(joined, target_metrics=args.target_metrics)
    expected_rows = len(manifest_base) * joined["model_name"].nunique()
    logger.info(
        "Final joined manifest has %d row(s) across %d model(s). Expected rows by patch x model=%d.",
        len(joined),
        joined["model_name"].nunique(),
        expected_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing joined manifest: %s", output_path)
    write_table(joined.drop(columns=["_join_key"], errors="ignore"), output_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
