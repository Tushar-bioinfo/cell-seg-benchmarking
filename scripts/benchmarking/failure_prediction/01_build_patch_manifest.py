#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_PATH = Path("outputs/conic_liz/embeddings/metadata/embeddings_index.parquet")
DEFAULT_OUTPUT_PATH = Path("outputs/conic_liz/failure_prediction/patch_manifest.parquet")
DEFAULT_PATCH_ID_COL = "patch_id"
DEFAULT_SLIDE_ID_COL = "slide_id"
DEFAULT_DATASET_COL = "dataset"
DEFAULT_SPLIT_COL = "split"
DEFAULT_EMBEDDING_PATH_COL = "embedding_path"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Build a clean canonical patch manifest from an embeddings index table. "
            "The output contains exactly one row per patch_id."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input embeddings index table (.csv or .parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output manifest path (.csv or .parquet).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root used to resolve relative input/output and relative paths in the table.",
    )
    parser.add_argument(
        "--path-base-dir",
        type=Path,
        default=None,
        help="Optional base directory to try first when canonicalizing relative paths from the input table.",
    )
    parser.add_argument(
        "--patch-id-col",
        default=DEFAULT_PATCH_ID_COL,
        help="Input column mapped to output column patch_id.",
    )
    parser.add_argument(
        "--slide-id-col",
        default=DEFAULT_SLIDE_ID_COL,
        help="Input column mapped to output column slide_id.",
    )
    parser.add_argument(
        "--slide-id-source-col",
        default=None,
        help=(
            "Optional source column used to derive slide_id. "
            "Defaults to --slide-id-col when not provided."
        ),
    )
    parser.add_argument(
        "--dataset-col",
        default=DEFAULT_DATASET_COL,
        help="Input column mapped to output column dataset. Ignored when --dataset-value is provided.",
    )
    parser.add_argument(
        "--split-col",
        default=DEFAULT_SPLIT_COL,
        help="Input column mapped to output column split. Ignored when --split-value is provided.",
    )
    parser.add_argument(
        "--embedding-path-col",
        default=DEFAULT_EMBEDDING_PATH_COL,
        help="Input column mapped to output column embedding_path.",
    )
    parser.add_argument(
        "--slide-id-value",
        default=None,
        help="Constant value to write to output column slide_id instead of reading an input column.",
    )
    parser.add_argument(
        "--dataset-value",
        default=None,
        help="Constant value to write to output column dataset instead of reading an input column.",
    )
    parser.add_argument(
        "--split-value",
        default=None,
        help="Constant value to write to output column split instead of reading an input column.",
    )
    parser.add_argument(
        "--slide-id-strip-prefix",
        default="",
        help="Optional prefix stripped from slide_id values after reading the source column.",
    )
    parser.add_argument(
        "--slide-id-strip-suffix",
        default="",
        help="Optional suffix stripped from slide_id values after reading the source column.",
    )
    parser.add_argument(
        "--slide-id-strip-regex",
        default="",
        help=(
            "Optional regex pattern removed from slide_id values using re.sub(pattern, '', value). "
            "Useful for converting patch IDs into slide IDs."
        ),
    )
    parser.add_argument(
        "--extra-cols",
        nargs="*",
        default=[],
        help="Additional input columns to preserve in the output manifest.",
    )
    parser.add_argument(
        "--extra-path-cols",
        nargs="*",
        default=[],
        help="Additional path columns to canonicalize if they are included in the output.",
    )
    parser.add_argument(
        "--dedupe-keep",
        choices=("first", "last"),
        default="first",
        help="Which duplicate row to keep when multiple rows share the same patch_id.",
    )
    parser.add_argument(
        "--log-duplicate-examples",
        type=int,
        default=5,
        help="Maximum number of conflicting duplicate patch_id examples to log.",
    )
    parser.add_argument(
        "--require-existing-paths",
        action="store_true",
        help="Fail if a canonicalized path does not exist on disk.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("build_patch_manifest")
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


def validate_required_source_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(repr(column) for column in missing)
        )


def validate_required_output_values(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    for column in required_columns:
        if column not in dataframe.columns:
            raise ValueError(f"Output manifest is missing required column {column!r}.")

        missing_mask = dataframe[column].isna() | dataframe[column].astype(str).str.strip().eq("")
        missing_count = int(missing_mask.sum())
        if missing_count:
            raise ValueError(
                f"Output column {column!r} contains {missing_count} missing/blank value(s). "
                "Fix the source table or supply a CLI mapping/value override."
            )


def derive_slide_id(dataframe: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.slide_id_value is not None:
        return pd.Series(args.slide_id_value, index=dataframe.index, dtype="object")

    source_col = args.slide_id_source_col or args.slide_id_col
    series = dataframe[source_col].astype("object").where(dataframe[source_col].notna(), "")
    derived = series.astype(str).str.strip()

    if args.slide_id_strip_prefix:
        derived = derived.str.removeprefix(args.slide_id_strip_prefix)
    if args.slide_id_strip_suffix:
        derived = derived.str.removesuffix(args.slide_id_strip_suffix)
    if args.slide_id_strip_regex:
        pattern = re.compile(args.slide_id_strip_regex)
        derived = derived.map(lambda value: pattern.sub("", value))

    return derived


def build_candidate_roots(
    *,
    path_base_dir: Path | None,
    repo_root: Path | None,
    input_path: Path,
) -> list[Path]:
    candidates: list[Path] = []
    for candidate in (path_base_dir, repo_root, input_path.parent, Path.cwd()):
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def canonicalize_path_value(
    raw_value: object,
    *,
    candidate_roots: list[Path],
    relative_root: Path | None,
    require_existing_paths: bool,
) -> str:
    if pd.isna(raw_value):
        raise ValueError("Encountered a missing path value during canonicalization.")

    text = str(raw_value).strip()
    if not text:
        raise ValueError("Encountered an empty path value during canonicalization.")

    raw_path = Path(text).expanduser()
    resolved_path: Path | None = None

    if raw_path.is_absolute():
        resolved_path = raw_path.resolve(strict=False)
    else:
        for root in candidate_roots:
            candidate = (root / raw_path).resolve(strict=False)
            if not require_existing_paths or candidate.exists():
                resolved_path = candidate
                break
        if resolved_path is None:
            resolved_path = (candidate_roots[0] / raw_path).resolve(strict=False)

    if require_existing_paths and not resolved_path.exists():
        raise FileNotFoundError(f"Canonicalized path does not exist: {resolved_path}")

    if relative_root is not None:
        try:
            return resolved_path.relative_to(relative_root).as_posix()
        except ValueError:
            pass

    return resolved_path.as_posix()


def canonicalize_path_columns(
    dataframe: pd.DataFrame,
    *,
    path_columns: list[str],
    candidate_roots: list[Path],
    relative_root: Path | None,
    require_existing_paths: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not path_columns:
        return dataframe

    tqdm.pandas()
    result = dataframe.copy()
    for column in path_columns:
        if column not in result.columns:
            logger.warning("Skipping path canonicalization for missing column %r.", column)
            continue

        logger.info("Canonicalizing path column %r.", column)
        result[column] = result[column].progress_apply(
            lambda value: canonicalize_path_value(
                value,
                candidate_roots=candidate_roots,
                relative_root=relative_root,
                require_existing_paths=require_existing_paths,
            )
        )
    return result


def deduplicate_by_patch_id(
    dataframe: pd.DataFrame,
    *,
    keep: str,
    log_duplicate_examples: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    duplicate_mask = dataframe.duplicated(subset=["patch_id"], keep=False)
    duplicate_rows = dataframe.loc[duplicate_mask].copy()

    if duplicate_rows.empty:
        logger.info("No duplicate patch_id values detected.")
        return dataframe

    duplicate_patch_count = int(duplicate_rows["patch_id"].nunique())
    logger.warning(
        "Found %d duplicate row(s) across %d patch_id value(s). Keeping the %s occurrence.",
        int(len(duplicate_rows)),
        duplicate_patch_count,
        keep,
    )

    compare_columns = [column for column in dataframe.columns if column != "patch_id"]
    conflicting_patch_ids: list[str] = []
    for patch_id, group in duplicate_rows.groupby("patch_id", sort=False):
        if group[compare_columns].drop_duplicates().shape[0] > 1:
            conflicting_patch_ids.append(str(patch_id))

    if conflicting_patch_ids:
        logger.warning(
            "Detected %d patch_id value(s) with conflicting duplicate rows.",
            len(conflicting_patch_ids),
        )
        for patch_id in conflicting_patch_ids[:log_duplicate_examples]:
            example_rows = duplicate_rows.loc[duplicate_rows["patch_id"] == patch_id].to_dict(orient="records")
            logger.warning("Conflicting duplicate example for patch_id=%s: %s", patch_id, example_rows)

    deduplicated = dataframe.drop_duplicates(subset=["patch_id"], keep=keep).copy()
    logger.info(
        "Dropped %d duplicate row(s). Final manifest row count: %d.",
        len(dataframe) - len(deduplicated),
        len(deduplicated),
    )
    return deduplicated


def build_manifest(dataframe: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.DataFrame(index=dataframe.index)
    manifest["patch_id"] = dataframe[args.patch_id_col]
    manifest["slide_id"] = derive_slide_id(dataframe, args)
    manifest["dataset"] = args.dataset_value if args.dataset_value is not None else dataframe[args.dataset_col]
    manifest["split"] = args.split_value if args.split_value is not None else dataframe[args.split_col]
    manifest["embedding_path"] = dataframe[args.embedding_path_col]

    extra_cols = [
        column
        for column in args.extra_cols
        if column not in {
            args.patch_id_col,
            args.slide_id_col,
            args.dataset_col,
            args.split_col,
            args.embedding_path_col,
        }
    ]
    for column in extra_cols:
        manifest[column] = dataframe[column]

    return manifest


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.verbose)

    repo_root = resolve_repo_root(args.repo_root)
    input_path = resolve_cli_path(args.input, repo_root=repo_root)
    output_path = resolve_cli_path(args.output, repo_root=repo_root)
    path_base_dir = resolve_cli_path(args.path_base_dir, repo_root=repo_root) if args.path_base_dir else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input table does not exist: {input_path}")

    logger.info("Reading input table: %s", input_path)
    if repo_root is not None:
        logger.info("Using repo root: %s", repo_root)
    if path_base_dir is not None:
        logger.info("Using path base dir: %s", path_base_dir)

    dataframe = read_table(input_path)
    logger.info("Loaded %d row(s) x %d column(s).", len(dataframe), len(dataframe.columns))

    required_source_columns = [args.patch_id_col, args.embedding_path_col]
    if args.slide_id_value is None:
        required_source_columns.append(args.slide_id_source_col or args.slide_id_col)
    if args.dataset_value is None:
        required_source_columns.append(args.dataset_col)
    if args.split_value is None:
        required_source_columns.append(args.split_col)
    required_source_columns.extend(args.extra_cols)
    validate_required_source_columns(dataframe, required_source_columns)

    manifest = build_manifest(dataframe, args)
    validate_required_output_values(
        manifest,
        required_columns=["patch_id", "slide_id", "dataset", "split", "embedding_path"],
    )

    candidate_roots = build_candidate_roots(
        path_base_dir=path_base_dir,
        repo_root=repo_root,
        input_path=input_path,
    )
    relative_root = repo_root if repo_root is not None else input_path.parent.resolve()

    path_columns = list(dict.fromkeys(["embedding_path", *args.extra_path_cols]))
    manifest = canonicalize_path_columns(
        manifest,
        path_columns=path_columns,
        candidate_roots=candidate_roots,
        relative_root=relative_root,
        require_existing_paths=args.require_existing_paths,
        logger=logger,
    )

    before_rows = len(manifest)
    manifest = deduplicate_by_patch_id(
        manifest,
        keep=args.dedupe_keep,
        log_duplicate_examples=args.log_duplicate_examples,
        logger=logger,
    )

    if manifest["patch_id"].duplicated().any():
        raise RuntimeError("patch_id is still duplicated after deduplication.")

    manifest = manifest.reset_index(drop=True)
    logger.info("Manifest reduced from %d to %d row(s).", before_rows, len(manifest))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing canonical patch manifest: %s", output_path)
    write_table(manifest, output_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
