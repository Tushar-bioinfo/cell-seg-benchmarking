#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DEFAULT_INPUT_CSV = Path("outputs/conic_liz/embed_morph.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/conic_liz/embedding_tables")
DEFAULT_FULL_OUTPUT_NAME = "embed_morph_with_vectors.csv"
DEFAULT_EMBEDDINGS_ONLY_NAME = "sample_id_embeddings_only.csv"
DEFAULT_ID_COL = "sample_id"
DEFAULT_EMBEDDING_PATH_COL = "embedding_path"
DEFAULT_EMBEDDING_OFFSET_COL = "embedding_row_offset"
DEFAULT_EMBEDDING_DIM_COL = "embedding_dim"
DEFAULT_EMBEDDING_FORMAT_COL = "embedding_format"
DEFAULT_PREFIX = "emb_"
DEFAULT_FLOAT_FORMAT = "%.10g"
SUPPORTED_EMBEDDING_FORMATS = {"pt"}
COMMON_TENSOR_KEYS = (
    "embeddings",
    "embedding",
    "features",
    "feature",
    "tensor",
    "x",
    "vectors",
    "vector",
    "data",
)
HEURISTIC_KEY_TOKENS = ("emb", "feature", "vector", "tensor")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the embedding-table export workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Expand embedding vectors referenced by a metadata CSV into flat embedding columns and export "
            "full + sample_id-only CSV tables."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input metadata CSV. Relative paths resolve against --repo-root when provided or detected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive the output CSV files.",
    )
    parser.add_argument(
        "--full-output-name",
        default=DEFAULT_FULL_OUTPUT_NAME,
        help="Filename for the metadata + expanded embeddings CSV.",
    )
    parser.add_argument(
        "--embeddings-only-name",
        default=DEFAULT_EMBEDDINGS_ONLY_NAME,
        help="Filename for the sample_id + expanded embeddings CSV.",
    )
    parser.add_argument(
        "--id-col",
        default=DEFAULT_ID_COL,
        help="Identifier column to keep in the embeddings-only output.",
    )
    parser.add_argument(
        "--embedding-path-col",
        default=DEFAULT_EMBEDDING_PATH_COL,
        help="Column containing the embedding file path.",
    )
    parser.add_argument(
        "--embedding-offset-col",
        default=DEFAULT_EMBEDDING_OFFSET_COL,
        help="Column containing the row offset within each embedding file.",
    )
    parser.add_argument(
        "--embedding-dim-col",
        default=DEFAULT_EMBEDDING_DIM_COL,
        help="Column containing the expected embedding dimension.",
    )
    parser.add_argument(
        "--embedding-format-col",
        default=DEFAULT_EMBEDDING_FORMAT_COL,
        help="Column containing the embedding storage format, for example pt.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root used to resolve relative input/output and embedding file paths.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Prefix for expanded embedding columns.",
    )
    parser.add_argument(
        "--float-format",
        default=DEFAULT_FLOAT_FORMAT,
        help="float_format passed to pandas.DataFrame.to_csv, for example '%%.10g'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on input rows for quick debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> logging.Logger:
    """Configure a stream logger for batch and interactive runs."""

    logger = logging.getLogger("export_embedding_tables")
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
    """Walk upward from a starting location until a .git directory is found."""

    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_root(repo_root_arg: Path | None, logger: logging.Logger) -> Path | None:
    """Resolve the repo root from CLI input or by inspecting the script location."""

    if repo_root_arg is not None:
        repo_root = repo_root_arg.expanduser().resolve()
        logger.debug("Using explicit repo root: %s", repo_root)
        return repo_root

    detected = find_repo_root(Path(__file__).resolve())
    if detected is not None:
        logger.debug("Detected repo root from script path: %s", detected)
        return detected

    logger.warning("Could not detect repo root from the script path. Relative paths will resolve from the current working directory.")
    return None


def resolve_cli_path(path_like: Path, *, repo_root: Path | None) -> Path:
    """Resolve a CLI path argument against repo_root when it is relative."""

    path = path_like.expanduser()
    if path.is_absolute():
        return path.resolve()
    if repo_root is not None:
        return (repo_root / path).resolve()
    return path.resolve()


def validate_required_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise if any required columns are absent from the input frame."""

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(repr(column) for column in missing)
        )


def resolve_path(
    raw_path: Any,
    *,
    input_csv: Path,
    repo_root: Path | None,
) -> Path:
    """Resolve an embedding file path using input-CSV and repo-root context."""

    if pd.isna(raw_path):
        raise FileNotFoundError("Encountered a missing embedding file path.")

    text = str(raw_path).strip()
    if not text:
        raise FileNotFoundError("Encountered an empty embedding file path.")

    path = Path(text).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Embedding file does not exist: {path}")

    candidate_roots: list[Path] = [input_csv.parent]
    if repo_root is not None:
        candidate_roots.append(repo_root)
    candidate_roots.append(Path.cwd().resolve())

    seen_roots: set[Path] = set()
    unique_roots: list[Path] = []
    for root in candidate_roots:
        resolved_root = root.resolve()
        if resolved_root not in seen_roots:
            seen_roots.add(resolved_root)
            unique_roots.append(resolved_root)

    attempted_paths: list[str] = []
    for root in unique_roots:
        candidate = (root / path).resolve()
        attempted_paths.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve embedding path {text!r}. Tried: {attempted_paths}"
    )


def normalize_optional_int(value: Any, *, column_name: str, row_context: str) -> int | None:
    """Normalize an optional integer field from pandas data into Python int or None."""

    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{row_context}: column {column_name!r} must be an integer-compatible value, got {value!r}.")
        return int(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{row_context}: column {column_name!r} must be integer-compatible, got {value!r}.") from exc

    if not np.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{row_context}: column {column_name!r} must be integer-compatible, got {value!r}.")
    return int(parsed)


def normalize_required_nonnegative_int(value: Any, *, column_name: str, row_context: str) -> int:
    """Normalize a required non-negative integer column."""

    parsed = normalize_optional_int(value, column_name=column_name, row_context=row_context)
    if parsed is None:
        raise ValueError(f"{row_context}: column {column_name!r} is required and may not be empty.")
    if parsed < 0:
        raise ValueError(f"{row_context}: column {column_name!r} must be non-negative, got {parsed}.")
    return parsed


def normalize_embedding_format(value: Any, *, column_name: str, row_context: str) -> str:
    """Normalize and validate the embedding storage format."""

    if pd.isna(value):
        raise ValueError(f"{row_context}: column {column_name!r} is required and may not be empty.")

    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError(f"{row_context}: column {column_name!r} is required and may not be empty.")
    if normalized not in SUPPORTED_EMBEDDING_FORMATS:
        raise ValueError(
            f"{row_context}: unsupported embedding format {normalized!r}. "
            f"Supported formats: {sorted(SUPPORTED_EMBEDDING_FORMATS)}."
        )
    return normalized


def summarize_object(obj: Any) -> str:
    """Return a compact description of a loaded object for error reporting."""

    if torch.is_tensor(obj):
        return f"torch.Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
    if isinstance(obj, np.ndarray):
        return f"numpy.ndarray(shape={obj.shape}, dtype={obj.dtype})"
    if isinstance(obj, dict):
        keys = list(obj.keys())
        preview = ", ".join(repr(key) for key in keys[:8])
        if len(keys) > 8:
            preview += ", ..."
        return f"dict(keys=[{preview}])"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}(len={len(obj)})"
    return f"{type(obj).__name__}"


def load_pt_object(path: Path) -> Any:
    """Load a .pt payload onto CPU with broad compatibility across torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load .pt embedding file {path}: {type(exc).__name__}: {exc}") from exc


def _coerce_numeric_array(value: Any, *, source_label: str, path: Path) -> np.ndarray:
    """Convert a candidate embedding container into a numeric 1D or 2D numpy array."""

    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise TypeError(f"Loaded empty {type(value).__name__} from {path} at {source_label}; expected embedding data.")
        if all(torch.is_tensor(item) for item in value):
            array = torch.stack([item.detach().cpu() for item in value], dim=0).numpy()
        else:
            try:
                array = np.asarray(value)
            except Exception as exc:
                raise TypeError(
                    f"Could not convert {type(value).__name__} from {path} at {source_label} to a numeric array."
                ) from exc
    else:
        raise TypeError(
            f"Unsupported embedding container type at {source_label} in {path}: {type(value)!r}. "
            "Expected a torch.Tensor, numpy.ndarray, list, or tuple."
        )

    if array.ndim == 0:
        raise TypeError(f"Encountered a scalar at {source_label} in {path}; expected a 1D or 2D embedding container.")
    if array.ndim > 2:
        raise TypeError(
            f"Encountered an array with shape {array.shape} at {source_label} in {path}; "
            "expected a 1D vector or a 2D [n_rows, dim] array."
        )
    if array.size == 0:
        raise TypeError(f"Encountered an empty array with shape {array.shape} at {source_label} in {path}.")
    if not np.issubdtype(array.dtype, np.number):
        try:
            array = array.astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Encountered a non-numeric array with dtype {array.dtype} at {source_label} in {path}."
            ) from exc
    return np.asarray(array, dtype=np.float32)


def _extract_candidate_from_dict(payload: dict[Any, Any], *, path: Path) -> tuple[Any, str]:
    """Choose the most likely embedding container from a dictionary payload."""

    key_lookup = {str(key).lower(): key for key in payload.keys()}
    for preferred in COMMON_TENSOR_KEYS:
        if preferred in key_lookup:
            original_key = key_lookup[preferred]
            return payload[original_key], f"dict[{original_key!r}]"

    heuristic_matches = [
        key for key in payload.keys() if any(token in str(key).lower() for token in HEURISTIC_KEY_TOKENS)
    ]
    if len(heuristic_matches) == 1:
        key = heuristic_matches[0]
        return payload[key], f"dict[{key!r}]"

    if len(payload) == 1:
        only_key = next(iter(payload.keys()))
        return payload[only_key], f"dict[{only_key!r}]"

    item_summaries = ", ".join(f"{key!r}: {summarize_object(value)}" for key, value in list(payload.items())[:8])
    if len(payload) > 8:
        item_summaries += ", ..."
    raise TypeError(
        f"Unsupported dict payload in {path}. Expected an embedding tensor/array directly or under one of the common "
        f"keys {COMMON_TENSOR_KEYS}. Found: {item_summaries}"
    )


def normalize_loaded_embedding_object(loaded_object: Any, *, path: Path) -> np.ndarray:
    """Normalize a loaded .pt object into a numeric 1D or 2D array."""

    candidate = loaded_object
    source_label = "root object"
    if isinstance(loaded_object, dict):
        candidate, source_label = _extract_candidate_from_dict(loaded_object, path=path)

    try:
        return _coerce_numeric_array(candidate, source_label=source_label, path=path)
    except Exception as exc:
        raise TypeError(
            f"Unsupported embedding payload in {path}. Encountered {summarize_object(loaded_object)}. "
            "Expected one of: a 2D torch.Tensor [n_rows, dim], a 1D torch.Tensor [dim], a dict containing an "
            "embedding-like tensor under a common key such as 'embeddings' or 'features', or a list/tuple/numpy-like "
            f"structure convertible to a numeric 1D/2D array. Details: {exc}"
        ) from exc


def extract_embedding_vector(
    normalized_embeddings: np.ndarray,
    *,
    row_offset: int,
    expected_dim: int | None,
    path: Path,
) -> np.ndarray:
    """Extract a single 1D embedding vector from a normalized 1D/2D embedding container."""

    if normalized_embeddings.ndim == 1:
        if row_offset != 0:
            raise IndexError(
                f"Embedding file {path} contains a single 1D vector with length {normalized_embeddings.shape[0]}, "
                f"but row offset {row_offset} was requested."
            )
        vector = normalized_embeddings
    else:
        num_rows, num_dims = normalized_embeddings.shape
        if row_offset >= num_rows:
            raise IndexError(
                f"Embedding row offset {row_offset} is out of range for file {path} with shape "
                f"{normalized_embeddings.shape}. Expected offset in [0, {num_rows - 1}]."
            )
        vector = normalized_embeddings[row_offset]
        if vector.ndim != 1:
            raise TypeError(
                f"Extracted row {row_offset} from {path}, but it has shape {vector.shape}; expected a flat vector."
            )
        if num_dims != vector.shape[0]:
            raise TypeError(
                f"Inconsistent row extraction from {path}: container shape {normalized_embeddings.shape}, "
                f"row shape {vector.shape}."
            )

    flat_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if expected_dim is not None and flat_vector.shape[0] != expected_dim:
        raise ValueError(
            f"Embedding length mismatch for file {path} at row offset {row_offset}: "
            f"expected {expected_dim} from the metadata CSV, got {flat_vector.shape[0]}."
        )
    return flat_vector


def get_embedding_column_names(embedding_dim: int, prefix: str) -> list[str]:
    """Build stable embedding column names such as emb_0000 ... emb_1535."""

    if embedding_dim <= 0:
        raise ValueError(f"Embedding dimension must be positive, got {embedding_dim}.")
    width = max(4, len(str(embedding_dim - 1)))
    return [f"{prefix}{index:0{width}d}" for index in range(embedding_dim)]


def build_embedding_matrix(
    dataframe: pd.DataFrame,
    *,
    input_csv: Path,
    repo_root: Path | None,
    id_col: str,
    embedding_path_col: str,
    embedding_offset_col: str,
    embedding_dim_col: str,
    embedding_format_col: str,
    logger: logging.Logger,
    show_progress: bool,
) -> np.ndarray:
    """Load embedding files once per unique path and return a row-aligned embedding matrix."""

    row_requests: dict[Path, list[tuple[int, int, int | None, str]]] = {}
    for row_index, row in dataframe.iterrows():
        row_context = f"row_index={row_index}"
        if id_col in dataframe.columns:
            row_context += f", {id_col}={row[id_col]!r}"

        embedding_format = normalize_embedding_format(
            row[embedding_format_col],
            column_name=embedding_format_col,
            row_context=row_context,
        )
        if embedding_format != "pt":
            raise ValueError(f"{row_context}: only 'pt' embedding files are supported in this script.")

        resolved_path = resolve_path(
            row[embedding_path_col],
            input_csv=input_csv,
            repo_root=repo_root,
        )
        row_offset = normalize_required_nonnegative_int(
            row[embedding_offset_col],
            column_name=embedding_offset_col,
            row_context=row_context,
        )
        expected_dim = normalize_optional_int(
            row[embedding_dim_col],
            column_name=embedding_dim_col,
            row_context=row_context,
        )
        if expected_dim is not None and expected_dim <= 0:
            raise ValueError(f"{row_context}: column {embedding_dim_col!r} must be positive, got {expected_dim}.")

        row_requests.setdefault(resolved_path, []).append((row_index, row_offset, expected_dim, row_context))

    logger.info("Resolved %d rows across %d unique embedding file(s).", len(dataframe), len(row_requests))

    embedding_matrix: np.ndarray | None = None
    global_dim: int | None = None
    progress = tqdm(
        row_requests.items(),
        total=len(row_requests),
        desc="Embedding files",
        unit="file",
        disable=not show_progress,
    )

    for resolved_path, requests in progress:
        progress.set_postfix_str(resolved_path.name)
        logger.debug("Loading %s for %d row(s).", resolved_path, len(requests))
        loaded_object = load_pt_object(resolved_path)
        normalized_embeddings = normalize_loaded_embedding_object(loaded_object, path=resolved_path)

        for row_index, row_offset, expected_dim, row_context in requests:
            try:
                vector = extract_embedding_vector(
                    normalized_embeddings,
                    row_offset=row_offset,
                    expected_dim=expected_dim,
                    path=resolved_path,
                )
            except Exception as exc:
                raise type(exc)(f"{row_context}: {exc}") from exc

            if global_dim is None:
                global_dim = int(vector.shape[0])
                embedding_matrix = np.empty((len(dataframe), global_dim), dtype=np.float32)
                logger.info("Detected embedding dimension: %d", global_dim)
            elif vector.shape[0] != global_dim:
                raise ValueError(
                    f"{row_context}: inconsistent embedding dimension. Previous rows used dim {global_dim}, "
                    f"but {resolved_path} row offset {row_offset} produced dim {vector.shape[0]}."
                )

            embedding_matrix[row_index, :] = vector

    if embedding_matrix is None or global_dim is None:
        raise ValueError("No embeddings were loaded from the input CSV.")

    return embedding_matrix


def write_outputs(
    dataframe: pd.DataFrame,
    embedding_matrix: np.ndarray,
    *,
    output_dir: Path,
    full_output_name: str,
    embeddings_only_name: str,
    id_col: str,
    prefix: str,
    float_format: str | None,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """Write the full metadata table and the id-only embedding table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_columns = get_embedding_column_names(embedding_matrix.shape[1], prefix)
    embedding_frame = pd.DataFrame(embedding_matrix, columns=embedding_columns, index=dataframe.index)

    full_output_path = output_dir / full_output_name
    embeddings_only_path = output_dir / embeddings_only_name

    full_frame = pd.concat([dataframe.reset_index(drop=True), embedding_frame.reset_index(drop=True)], axis=1)
    embeddings_only_frame = pd.concat(
        [dataframe[[id_col]].reset_index(drop=True), embedding_frame.reset_index(drop=True)],
        axis=1,
    )

    full_frame.to_csv(full_output_path, index=False, float_format=float_format)
    embeddings_only_frame.to_csv(embeddings_only_path, index=False, float_format=float_format)

    logger.info("Wrote full table to %s", full_output_path)
    logger.info("Wrote embeddings-only table to %s", embeddings_only_path)
    return full_output_path, embeddings_only_path


def main() -> int:
    """CLI entrypoint for exporting expanded embedding tables."""

    args = parse_args()
    logger = configure_logging(verbose=args.verbose)

    repo_root = resolve_repo_root(args.repo_root, logger=logger)
    input_csv = resolve_cli_path(args.input_csv, repo_root=repo_root)
    output_dir = resolve_cli_path(args.output_dir, repo_root=repo_root)

    logger.info("Input CSV: %s", input_csv)
    logger.info("Output directory: %s", output_dir)
    if repo_root is not None:
        logger.info("Repo root: %s", repo_root)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    dataframe = pd.read_csv(input_csv)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError(f"--limit must be non-negative, got {args.limit}.")
        dataframe = dataframe.head(args.limit).copy()
        logger.info("Applied row limit: %d", len(dataframe))

    if dataframe.empty:
        raise ValueError("Input CSV contains no rows after applying the requested limit.")

    validate_required_columns(
        dataframe,
        [
            args.id_col,
            args.embedding_path_col,
            args.embedding_format_col,
            args.embedding_offset_col,
            args.embedding_dim_col,
        ],
    )

    duplicate_ids = int(dataframe[args.id_col].duplicated().sum())
    if duplicate_ids:
        logger.warning(
            "Column %r contains %d duplicated value(s). The embeddings-only output will preserve all rows in input order.",
            args.id_col,
            duplicate_ids,
        )

    show_progress = args.verbose or sys.stderr.isatty()
    embedding_matrix = build_embedding_matrix(
        dataframe,
        input_csv=input_csv,
        repo_root=repo_root,
        id_col=args.id_col,
        embedding_path_col=args.embedding_path_col,
        embedding_offset_col=args.embedding_offset_col,
        embedding_dim_col=args.embedding_dim_col,
        embedding_format_col=args.embedding_format_col,
        logger=logger,
        show_progress=show_progress,
    )

    write_outputs(
        dataframe,
        embedding_matrix,
        output_dir=output_dir,
        full_output_name=args.full_output_name,
        embeddings_only_name=args.embeddings_only_name,
        id_col=args.id_col,
        prefix=args.prefix,
        float_format=args.float_format,
        logger=logger,
    )

    logger.info("Completed embedding table export for %d row(s).", len(dataframe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
