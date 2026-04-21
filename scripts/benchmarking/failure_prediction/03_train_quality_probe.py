#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


DEFAULT_INPUT_PATH = Path("outputs/conic_liz/failure_prediction/patch_manifest_with_eval_targets.parquet")
DEFAULT_OUTPUT_DIR = Path("outputs/conic_liz/failure_prediction/quality_probe")

DEFAULT_PATCH_ID_COL = "patch_id"
DEFAULT_SLIDE_ID_COL = "slide_id"
DEFAULT_DATASET_COL = "dataset"
DEFAULT_MODEL_NAME_COL = "model_name"
DEFAULT_EMBEDDING_PATH_COL = "embedding_path"
DEFAULT_EMBEDDING_OFFSET_COL = "embedding_row_offset"
DEFAULT_EMBEDDING_FORMAT_COL = "embedding_format"
DEFAULT_GROUP_COL = "slide_id"
DEFAULT_TARGET_COL = "instance_pq"
DEFAULT_PROBLEM_TYPE = "regression"
DEFAULT_FEATURE_MODE = "embedding_only"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TRAIN_ROWS = None
DEFAULT_PROGRESS = True
DEFAULT_METRICS_OUTPUT_NAME = "metrics.json"
DEFAULT_CONFIG_OUTPUT_NAME = "config.json"
DEFAULT_PREDICTIONS_OUTPUT_NAME = "predictions.csv"
DEFAULT_MODEL_OUTPUT_NAME = "predictor.pkl"
DEFAULT_SCALER_OUTPUT_NAME = "scaler.pkl"

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
FEATURE_MODES = {"embedding_only", "metadata_only", "embedding_plus_metadata"}
PROBLEM_TYPES = {"regression", "classification"}
BASELINE_MODELS = {
    "regression": {"dummy_mean", "ridge"},
    "classification": {"dummy_most_frequent", "logistic"},
}
LEAKAGE_COLUMN_DEFAULTS = {
    "target",
    "target_binary",
    "target_label",
    "target_class",
    "prediction",
    "predictions",
    "pred_score",
    "score",
}


@dataclass
class TrainSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train simple baseline probes to predict segmentation quality from embeddings and/or metadata. "
            "Supports grouped train/test splits and saves model artifacts plus evaluation outputs."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Joined manifest with one row per patch_id x model_name (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive model artifacts and outputs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root used to resolve relative paths.",
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
        "--dataset-col",
        default=DEFAULT_DATASET_COL,
        help="Input column containing dataset labels.",
    )
    parser.add_argument(
        "--model-name-col",
        default=DEFAULT_MODEL_NAME_COL,
        help="Input column containing segmentation model names.",
    )
    parser.add_argument(
        "--embedding-path-col",
        default=DEFAULT_EMBEDDING_PATH_COL,
        help="Input column containing the embedding file path.",
    )
    parser.add_argument(
        "--embedding-offset-col",
        default=DEFAULT_EMBEDDING_OFFSET_COL,
        help=(
            "Optional input column containing the row offset within each embedding file. "
            "If absent or missing, offset 0 is used."
        ),
    )
    parser.add_argument(
        "--embedding-format-col",
        default=DEFAULT_EMBEDDING_FORMAT_COL,
        help=(
            "Optional input column containing the embedding storage format. "
            "If absent or missing, .pt is assumed."
        ),
    )
    parser.add_argument(
        "--group-col",
        default=DEFAULT_GROUP_COL,
        help="Column used for grouped splitting. Often slide_id.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help="Target metric column to predict.",
    )
    parser.add_argument(
        "--problem-type",
        choices=sorted(PROBLEM_TYPES),
        default=DEFAULT_PROBLEM_TYPE,
        help="Train a regression or classification baseline.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        help="Threshold used to derive binary labels when --problem-type=classification.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=sorted(FEATURE_MODES),
        default=DEFAULT_FEATURE_MODE,
        help="Which feature block(s) to use.",
    )
    parser.add_argument(
        "--metadata-cols",
        nargs="*",
        default=[],
        help=(
            "Explicit metadata feature columns to use. "
            "Required for metadata_only or embedding_plus_metadata unless --auto-metadata-cols is enabled."
        ),
    )
    parser.add_argument(
        "--auto-metadata-cols",
        action="store_true",
        help="Infer metadata columns from the manifest by excluding known identifier/target/embedding columns.",
    )
    parser.add_argument(
        "--exclude-metadata-cols",
        nargs="*",
        default=[],
        help="Metadata columns to exclude from training when using --auto-metadata-cols.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of groups assigned to the test split.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for grouped splitting and model fitting.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS,
        help="Optional cap on total rows loaded for debugging.",
    )
    parser.add_argument(
        "--require-existing-embeddings",
        action="store_true",
        help="Fail if any resolved embedding path does not exist on disk.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PROGRESS,
        help="Show tqdm progress while loading embeddings.",
    )
    parser.add_argument(
        "--metrics-output-name",
        default=DEFAULT_METRICS_OUTPUT_NAME,
        help="Filename for saved metrics JSON.",
    )
    parser.add_argument(
        "--config-output-name",
        default=DEFAULT_CONFIG_OUTPUT_NAME,
        help="Filename for saved config JSON.",
    )
    parser.add_argument(
        "--predictions-output-name",
        default=DEFAULT_PREDICTIONS_OUTPUT_NAME,
        help="Filename for saved per-row predictions CSV.",
    )
    parser.add_argument(
        "--model-output-name",
        default=DEFAULT_MODEL_OUTPUT_NAME,
        help="Filename for saved trained predictor pickle.",
    )
    parser.add_argument(
        "--scaler-output-name",
        default=DEFAULT_SCALER_OUTPUT_NAME,
        help="Filename for saved fitted feature preprocessor pickle.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("train_quality_probe")
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


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "This script requires PyTorch to load .pt embedding files. Install torch in the runtime environment."
        ) from exc
    return torch


def validate_required_columns(dataframe: pd.DataFrame, required_columns: list[str], *, context: str) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: " + ", ".join(repr(column) for column in missing))


def summarize_object(obj: Any) -> str:
    torch = _import_torch()
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
    return type(obj).__name__


def load_pt_object(path: Path) -> Any:
    torch = _import_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load .pt embedding file {path}: {type(exc).__name__}: {exc}") from exc


def _coerce_numeric_array(value: Any, *, source_label: str, path: Path) -> np.ndarray:
    torch = _import_torch()
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise TypeError(f"Loaded empty {type(value).__name__} from {path} at {source_label}.")
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
            "Expected torch.Tensor, numpy.ndarray, list, or tuple."
        )

    if array.ndim == 0:
        raise TypeError(f"Encountered a scalar at {source_label} in {path}; expected a 1D or 2D embedding.")
    if array.ndim > 2:
        raise TypeError(
            f"Encountered an array with shape {array.shape} at {source_label} in {path}; expected 1D or 2D."
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


def _find_unique_matrix_candidate_in_dict(payload: dict[Any, Any], *, path: Path) -> tuple[Any, str] | None:
    matrix_candidates: list[tuple[Any, Any]] = []
    for key, value in payload.items():
        try:
            candidate_array = _coerce_numeric_array(value, source_label=f"dict[{key!r}]", path=path)
        except Exception:
            continue
        if candidate_array.ndim == 2:
            matrix_candidates.append((key, value))
            if len(matrix_candidates) > 1:
                return None

    if len(matrix_candidates) == 1:
        key, value = matrix_candidates[0]
        return value, f"dict[{key!r}]"
    return None


def _extract_candidate_from_dict(payload: dict[Any, Any], *, path: Path) -> tuple[Any, str]:
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

    unique_matrix_candidate = _find_unique_matrix_candidate_in_dict(payload, path=path)
    if unique_matrix_candidate is not None:
        return unique_matrix_candidate

    item_summaries = ", ".join(f"{key!r}: {summarize_object(value)}" for key, value in list(payload.items())[:8])
    if len(payload) > 8:
        item_summaries += ", ..."
    raise TypeError(
        f"Unsupported dict payload in {path}. Expected an embedding tensor/array under one of {COMMON_TENSOR_KEYS}. "
        f"Found: {item_summaries}"
    )


def normalize_loaded_embedding_object(loaded_object: Any, *, path: Path) -> np.ndarray:
    candidate = loaded_object
    source_label = "root object"
    if isinstance(loaded_object, dict):
        candidate, source_label = _extract_candidate_from_dict(loaded_object, path=path)

    try:
        return _coerce_numeric_array(candidate, source_label=source_label, path=path)
    except Exception as exc:
        raise TypeError(
            f"Unsupported embedding payload in {path}. Encountered {summarize_object(loaded_object)}. Details: {exc}"
        ) from exc


def resolve_embedding_path(
    raw_path: object,
    *,
    input_path: Path,
    repo_root: Path | None,
    require_existing_embeddings: bool,
) -> Path:
    if pd.isna(raw_path):
        raise FileNotFoundError("Encountered a missing embedding_path value.")

    text = str(raw_path).strip()
    if not text:
        raise FileNotFoundError("Encountered an empty embedding_path value.")

    path = Path(text).expanduser()
    if path.is_absolute():
        resolved = path.resolve(strict=False)
        if require_existing_embeddings and not resolved.exists():
            raise FileNotFoundError(f"Embedding file does not exist: {resolved}")
        return resolved

    candidate_roots: list[Path] = []
    for candidate in (input_path.parent, repo_root, Path.cwd()):
        if candidate is None:
            continue
        resolved_root = candidate.resolve()
        if resolved_root not in candidate_roots:
            candidate_roots.append(resolved_root)

    attempted: list[str] = []
    for root in candidate_roots:
        candidate_path = (root / path).resolve(strict=False)
        attempted.append(str(candidate_path))
        if not require_existing_embeddings or candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Could not resolve embedding path {text!r}. Tried: {attempted}")


def normalize_optional_int(value: object, *, column_name: str, row_context: str) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{row_context}: column {column_name!r} must be integer-compatible, got {value!r}.")
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


def normalize_embedding_format(value: object, *, default_format: str = "pt") -> str:
    if pd.isna(value):
        normalized = default_format
    else:
        normalized = str(value).strip().lower() or default_format
    if normalized not in SUPPORTED_EMBEDDING_FORMATS:
        raise ValueError(
            f"Unsupported embedding format {normalized!r}. Supported formats: {sorted(SUPPORTED_EMBEDDING_FORMATS)}."
        )
    return normalized


def extract_embedding_vector(
    normalized_embeddings: np.ndarray,
    *,
    row_offset: int,
    path: Path,
) -> np.ndarray:
    if normalized_embeddings.ndim == 1:
        if row_offset != 0:
            raise IndexError(
                f"Embedding file {path} contains a single 1D vector, but row offset {row_offset} was requested."
            )
        vector = normalized_embeddings
    else:
        if row_offset >= normalized_embeddings.shape[0]:
            raise IndexError(
                f"Embedding row offset {row_offset} is out of range for file {path} with shape {normalized_embeddings.shape}."
            )
        vector = normalized_embeddings[row_offset]
    return np.asarray(vector, dtype=np.float32).reshape(-1)


def load_single_embedding_vector(
    *,
    embedding_path: Path,
    embedding_format: str,
    row_offset: int,
) -> np.ndarray:
    if embedding_format != "pt":
        raise ValueError(f"Unsupported embedding format {embedding_format!r}. Only .pt baselines are supported.")
    loaded_object = load_pt_object(embedding_path)
    normalized = normalize_loaded_embedding_object(loaded_object, path=embedding_path)
    return extract_embedding_vector(normalized, row_offset=row_offset, path=embedding_path)


def load_embedding_matrix(
    dataframe: pd.DataFrame,
    *,
    input_path: Path,
    repo_root: Path | None,
    embedding_path_col: str,
    embedding_offset_col: str,
    embedding_format_col: str,
    require_existing_embeddings: bool,
    show_progress: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[int]]:
    vectors: list[np.ndarray] = []
    failed_indices: list[int] = []

    iterator = dataframe.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(dataframe), desc="loading embeddings")

    for row_index, row in iterator:
        row_context = f"row_index={row_index}"
        try:
            embedding_path = resolve_embedding_path(
                row[embedding_path_col],
                input_path=input_path,
                repo_root=repo_root,
                require_existing_embeddings=require_existing_embeddings,
            )
            if embedding_format_col in dataframe.columns:
                embedding_format = normalize_embedding_format(row[embedding_format_col])
            else:
                embedding_format = "pt"

            if embedding_offset_col in dataframe.columns:
                row_offset = normalize_optional_int(
                    row[embedding_offset_col],
                    column_name=embedding_offset_col,
                    row_context=row_context,
                )
                if row_offset is None:
                    row_offset = 0
            else:
                row_offset = 0

            vector = load_single_embedding_vector(
                embedding_path=embedding_path,
                embedding_format=embedding_format,
                row_offset=row_offset,
            )
        except Exception as exc:
            logger.error("%s: failed to load embedding: %s", row_context, exc)
            failed_indices.append(row_index)
            continue

        if vectors and vector.shape[0] != vectors[0].shape[0]:
            raise ValueError(
                f"Inconsistent embedding dimension at {row_context}: expected {vectors[0].shape[0]}, got {vector.shape[0]}."
            )
        vectors.append(vector)

    if failed_indices:
        raise RuntimeError(
            f"Failed to load {len(failed_indices)} embedding row(s). "
            "Fix missing/corrupt embedding files before training."
        )
    if not vectors:
        raise RuntimeError("No embeddings were loaded.")

    matrix = np.vstack(vectors).astype(np.float32)
    logger.info("Loaded embedding matrix with shape %s.", matrix.shape)
    return matrix, list(dataframe.index)


def infer_metadata_columns(
    dataframe: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> list[str]:
    excluded = {
        args.patch_id_col,
        args.slide_id_col,
        args.dataset_col,
        args.model_name_col,
        args.embedding_path_col,
        args.embedding_offset_col,
        args.embedding_format_col,
        args.group_col,
        args.target_col,
        *LEAKAGE_COLUMN_DEFAULTS,
    }
    excluded.update(args.exclude_metadata_cols)
    inferred = [column for column in dataframe.columns if column not in excluded]
    return inferred


def check_for_feature_target_leakage(
    *,
    metadata_cols: list[str],
    target_col: str,
    problem_type: str,
    logger: logging.Logger,
) -> None:
    lowered = {column.lower() for column in metadata_cols}
    if target_col in metadata_cols:
        raise ValueError(f"Target column {target_col!r} is included in metadata features. Remove it to prevent leakage.")
    if problem_type == "classification" and "target_binary" in lowered:
        raise ValueError("Derived classification label column target_binary would leak into metadata features.")

    suspicious = [
        column
        for column in metadata_cols
        if column.lower() == target_col.lower()
        or target_col.lower() in column.lower()
        or any(token in column.lower() for token in ("pred", "score", "quality", "pq", "dice"))
    ]
    if suspicious:
        logger.warning(
            "Potential leakage risk: metadata feature columns contain target-like names: %s",
            suspicious,
        )


def validate_non_missing_core_values(dataframe: pd.DataFrame, *, required_columns: list[str]) -> None:
    for column in required_columns:
        missing_mask = dataframe[column].isna() | dataframe[column].astype(str).str.strip().eq("")
        missing_count = int(missing_mask.sum())
        if missing_count:
            raise ValueError(
                f"Column {column!r} contains {missing_count} missing/blank value(s). "
                "Fix the joined manifest before training."
            )


def build_grouped_split(
    dataframe: pd.DataFrame,
    *,
    group_col: str,
    test_size: float,
    random_seed: int,
) -> TrainSplit:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"--test-size must be in (0, 1), got {test_size}.")

    groups = dataframe[group_col].astype("string").fillna("").to_numpy()
    if np.any(groups == ""):
        raise ValueError(f"Group column {group_col!r} contains missing/blank values.")

    unique_groups = pd.Index(pd.unique(groups))
    if len(unique_groups) < 2:
        raise ValueError(
            f"Need at least two distinct groups in column {group_col!r} for grouped splitting, got {len(unique_groups)}."
        )

    n_test_groups = max(1, int(round(len(unique_groups) * test_size)))
    if n_test_groups >= len(unique_groups):
        n_test_groups = len(unique_groups) - 1
    if n_test_groups <= 0:
        raise ValueError("Computed zero test groups; increase dataset size or test-size.")

    rng = np.random.default_rng(random_seed)
    shuffled_groups = unique_groups.to_numpy(copy=True)
    rng.shuffle(shuffled_groups)
    test_groups = set(shuffled_groups[:n_test_groups])
    train_mask = ~pd.Series(groups).isin(test_groups).to_numpy()
    test_mask = ~train_mask

    train_groups = pd.Series(groups[train_mask]).unique()
    test_groups_array = pd.Series(groups[test_mask]).unique()
    overlap = set(train_groups).intersection(set(test_groups_array))
    if overlap:
        raise RuntimeError(
            f"Grouped split leakage detected: {len(overlap)} group(s) appear in both train and test: {sorted(overlap)[:10]}"
        )

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("Grouped split produced an empty train or test set.")

    return TrainSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_groups=np.asarray(train_groups),
        test_groups=np.asarray(test_groups_array),
    )


def build_feature_dataframe(
    dataframe: pd.DataFrame,
    *,
    metadata_cols: list[str],
) -> pd.DataFrame:
    return dataframe[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=dataframe.index)


def build_model_pipeline(
    *,
    problem_type: str,
    baseline_name: str,
    embedding_dim: int,
    metadata_frame: pd.DataFrame,
    feature_mode: str,
    random_seed: int,
) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        embedding_cols = [f"emb_{index:04d}" for index in range(embedding_dim)]
        transformers.append(
            (
                "embedding",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                embedding_cols,
            )
        )

    if feature_mode in {"metadata_only", "embedding_plus_metadata"} and not metadata_frame.empty:
        numeric_cols = metadata_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = [column for column in metadata_frame.columns if column not in numeric_cols]
        if numeric_cols:
            transformers.append(
                (
                    "metadata_numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            )
        if categorical_cols:
            transformers.append(
                (
                    "metadata_categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                )
            )

    if not transformers:
        raise ValueError("No usable feature transformers were constructed. Check feature mode and metadata columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if problem_type == "regression":
        if baseline_name == "dummy_mean":
            estimator = DummyRegressor(strategy="mean")
        elif baseline_name == "ridge":
            estimator = Ridge(alpha=1.0, random_state=random_seed)
        else:
            raise ValueError(f"Unsupported regression baseline {baseline_name!r}.")
    else:
        if baseline_name == "dummy_most_frequent":
            estimator = DummyClassifier(strategy="most_frequent")
        elif baseline_name == "logistic":
            estimator = LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=random_seed,
            )
        else:
            raise ValueError(f"Unsupported classification baseline {baseline_name!r}.")

    return Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])


def materialize_training_frame(
    *,
    embedding_matrix: np.ndarray | None,
    metadata_frame: pd.DataFrame,
    feature_mode: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        if embedding_matrix is None:
            raise ValueError("Embedding matrix is required for this feature mode.")
        embedding_cols = [f"emb_{index:04d}" for index in range(embedding_matrix.shape[1])]
        frames.append(pd.DataFrame(embedding_matrix, columns=embedding_cols, index=metadata_frame.index))
    if feature_mode in {"metadata_only", "embedding_plus_metadata"}:
        frames.append(metadata_frame)
    if not frames:
        raise ValueError(f"Unsupported feature mode {feature_mode!r}.")
    return pd.concat(frames, axis=1)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serializable[key] = serialize_metrics(value)
        elif isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.verbose)

    repo_root = resolve_repo_root(args.repo_root)
    input_path = resolve_cli_path(args.input, repo_root=repo_root)
    output_dir = resolve_cli_path(args.output_dir, repo_root=repo_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Joined manifest does not exist: {input_path}")

    logger.info("Reading joined manifest: %s", input_path)
    dataframe = read_table(input_path)
    if args.max_train_rows is not None:
        dataframe = dataframe.head(args.max_train_rows).copy()
        logger.info("Applied row limit: %d row(s).", len(dataframe))

    required_columns = [
        args.patch_id_col,
        args.slide_id_col,
        args.dataset_col,
        args.model_name_col,
        args.embedding_path_col,
        args.group_col,
        args.target_col,
    ]
    validate_required_columns(dataframe, required_columns, context="Joined manifest")
    validate_non_missing_core_values(
        dataframe,
        required_columns=[
            args.patch_id_col,
            args.slide_id_col,
            args.dataset_col,
            args.model_name_col,
            args.embedding_path_col,
            args.group_col,
            args.target_col,
        ],
    )

    if args.feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode {args.feature_mode!r}.")
    if args.problem_type not in PROBLEM_TYPES:
        raise ValueError(f"Unsupported problem type {args.problem_type!r}.")

    explicit_metadata_cols = [column for column in args.metadata_cols if column]
    if args.auto_metadata_cols:
        inferred_metadata_cols = infer_metadata_columns(dataframe, args=args)
        metadata_cols = list(dict.fromkeys(explicit_metadata_cols + inferred_metadata_cols))
    else:
        metadata_cols = explicit_metadata_cols

    if args.feature_mode in {"metadata_only", "embedding_plus_metadata"} and not metadata_cols:
        raise ValueError(
            "Metadata features are required for this feature mode. Supply --metadata-cols or enable --auto-metadata-cols."
        )
    validate_required_columns(dataframe, metadata_cols, context="Joined manifest metadata features")
    check_for_feature_target_leakage(
        metadata_cols=metadata_cols,
        target_col=args.target_col,
        problem_type=args.problem_type,
        logger=logger,
    )

    logger.info("Using feature mode %s.", args.feature_mode)
    logger.info("Using metadata columns: %s", metadata_cols)

    y_regression = pd.to_numeric(dataframe[args.target_col], errors="coerce")
    if y_regression.isna().any():
        missing_count = int(y_regression.isna().sum())
        raise ValueError(
            f"Target column {args.target_col!r} contains {missing_count} non-numeric or missing value(s)."
        )

    metadata_frame = build_feature_dataframe(dataframe, metadata_cols=metadata_cols)

    embedding_matrix: np.ndarray | None = None
    if args.feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        if args.embedding_path_col not in dataframe.columns:
            raise ValueError(
                f"Feature mode {args.feature_mode!r} requires column {args.embedding_path_col!r} to load embeddings."
            )
        embedding_matrix, loaded_indices = load_embedding_matrix(
            dataframe,
            input_path=input_path,
            repo_root=repo_root,
            embedding_path_col=args.embedding_path_col,
            embedding_offset_col=args.embedding_offset_col,
            embedding_format_col=args.embedding_format_col,
            require_existing_embeddings=args.require_existing_embeddings,
            show_progress=args.progress,
            logger=logger,
        )
        if loaded_indices != list(dataframe.index):
            raise RuntimeError("Embedding load order drift detected; loaded row indices do not match the manifest order.")

    X = materialize_training_frame(
        embedding_matrix=embedding_matrix,
        metadata_frame=metadata_frame,
        feature_mode=args.feature_mode,
    )

    split = build_grouped_split(
        dataframe,
        group_col=args.group_col,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    logger.info(
        "Grouped split complete: train_rows=%d test_rows=%d train_groups=%d test_groups=%d.",
        len(split.train_indices),
        len(split.test_indices),
        len(split.train_groups),
        len(split.test_groups),
    )

    baseline_name = (
        "ridge" if args.problem_type == "regression" else "logistic"
    )
    logger.info("Training baseline model: %s", baseline_name)

    if args.problem_type == "regression":
        y = y_regression.to_numpy(dtype=np.float32)
    else:
        threshold = args.classification_threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"--classification-threshold must be in [0, 1], got {threshold}.")
        y = (y_regression.to_numpy(dtype=np.float32) >= threshold).astype(np.int64)
        class_values = np.unique(y)
        if class_values.size < 2:
            raise ValueError(
                f"Classification target derived from {args.target_col!r} at threshold {threshold} "
                f"has only one class: {class_values.tolist()}."
            )

    pipeline = build_model_pipeline(
        problem_type=args.problem_type,
        baseline_name=baseline_name,
        embedding_dim=0 if embedding_matrix is None else embedding_matrix.shape[1],
        metadata_frame=metadata_frame,
        feature_mode=args.feature_mode,
        random_seed=args.random_seed,
    )

    X_train = X.iloc[split.train_indices]
    X_test = X.iloc[split.test_indices]
    y_train = y[split.train_indices]
    y_test = y[split.test_indices]

    pipeline.fit(X_train, y_train)
    estimator = pipeline.named_steps["estimator"]
    preprocessor = pipeline.named_steps["preprocessor"]

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    metrics: dict[str, Any] = {
        "problem_type": args.problem_type,
        "baseline_name": baseline_name,
        "feature_mode": args.feature_mode,
        "target_col": args.target_col,
        "group_col": args.group_col,
        "classification_threshold": args.classification_threshold if args.problem_type == "classification" else None,
        "n_rows": int(len(dataframe)),
        "n_train_rows": int(len(split.train_indices)),
        "n_test_rows": int(len(split.test_indices)),
        "n_train_groups": int(len(split.train_groups)),
        "n_test_groups": int(len(split.test_groups)),
        "train_groups": split.train_groups.tolist(),
        "test_groups": split.test_groups.tolist(),
        "metadata_cols": metadata_cols,
        "embedding_dim": None if embedding_matrix is None else int(embedding_matrix.shape[1]),
    }

    predictions_df = dataframe.copy()
    predictions_df["split"] = "train"
    predictions_df.loc[predictions_df.index[split.test_indices], "split"] = "test"
    predictions_df["target"] = y if args.problem_type == "classification" else y_regression.to_numpy(dtype=np.float32)
    predictions_df["prediction"] = np.nan
    predictions_df.loc[predictions_df.index[split.train_indices], "prediction"] = train_pred
    predictions_df.loc[predictions_df.index[split.test_indices], "prediction"] = test_pred

    if args.problem_type == "regression":
        metrics["train_metrics"] = compute_regression_metrics(y_train, train_pred)
        metrics["test_metrics"] = compute_regression_metrics(y_test, test_pred)
    else:
        train_score = None
        test_score = None
        if hasattr(pipeline, "predict_proba"):
            train_score = pipeline.predict_proba(X_train)[:, 1]
            test_score = pipeline.predict_proba(X_test)[:, 1]
            predictions_df["prediction_score"] = np.nan
            predictions_df.loc[predictions_df.index[split.train_indices], "prediction_score"] = train_score
            predictions_df.loc[predictions_df.index[split.test_indices], "prediction_score"] = test_score
        metrics["train_metrics"] = compute_classification_metrics(y_train, train_pred, y_score=train_score)
        metrics["test_metrics"] = compute_classification_metrics(y_test, test_pred, y_score=test_score)

    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        key: value
        for key, value in vars(args).items()
        if key not in {"input", "output_dir", "repo_root"}
    }
    config_payload.update(
        {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "repo_root": None if repo_root is None else str(repo_root),
            "baseline_name": baseline_name,
            "train_groups": split.train_groups.tolist(),
            "test_groups": split.test_groups.tolist(),
        }
    )

    model_output_path = output_dir / args.model_output_name
    scaler_output_path = output_dir / args.scaler_output_name
    config_output_path = output_dir / args.config_output_name
    metrics_output_path = output_dir / args.metrics_output_name
    predictions_output_path = output_dir / args.predictions_output_name

    logger.info("Writing predictor: %s", model_output_path)
    with model_output_path.open("wb") as handle:
        pickle.dump(estimator, handle)

    logger.info("Writing preprocessor/scaler: %s", scaler_output_path)
    with scaler_output_path.open("wb") as handle:
        pickle.dump(preprocessor, handle)

    logger.info("Writing config: %s", config_output_path)
    config_output_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")

    logger.info("Writing metrics: %s", metrics_output_path)
    metrics_output_path.write_text(json.dumps(serialize_metrics(metrics), indent=2, sort_keys=True) + "\n")

    logger.info("Writing predictions: %s", predictions_output_path)
    predictions_df.to_csv(predictions_output_path, index=False)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
