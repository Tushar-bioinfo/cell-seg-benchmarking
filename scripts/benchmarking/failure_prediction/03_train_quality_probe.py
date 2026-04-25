#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
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
DEFAULT_PROBLEM_TYPE = "classification"
DEFAULT_FEATURE_MODE = "embedding_only"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TRAIN_ROWS = None
DEFAULT_PROGRESS = True
DEFAULT_OPTUNA_TRIALS = 40
DEFAULT_CV_FOLDS = 5
DEFAULT_CLASSIFIER_FAMILIES = ("logistic_regression", "random_forest", "svm")
DEFAULT_TARGET_AGGREGATION = "median"
DEFAULT_THRESHOLD_STRATEGY = "train_median"
DEFAULT_CLASSIFICATION_THRESHOLD = None
DEFAULT_REQUIRE_COMPLETE_MODEL_COVERAGE = True
DEFAULT_POSITIVE_CLASS_NAME = "failure"
DEFAULT_PLOTS_DIR_NAME = "plots"
DEFAULT_OPTUNA_TRIALS_OUTPUT_NAME = "study_trials.csv"
DEFAULT_BEST_PARAMS_OUTPUT_NAME = "best_params.json"
DEFAULT_AGGREGATED_OUTPUT_NAME = "aggregated_patch_targets.parquet"
DEFAULT_FAMILY_METRICS_OUTPUT_NAME = "candidate_family_metrics.json"
DEFAULT_METRICS_OUTPUT_NAME = "metrics.json"
DEFAULT_CONFIG_OUTPUT_NAME = "config.json"
DEFAULT_PREDICTIONS_OUTPUT_NAME = "predictions.csv"
DEFAULT_MODEL_OUTPUT_NAME = "predictor.pkl"
DEFAULT_SCALER_OUTPUT_NAME = "scaler.pkl"
DEFAULT_N_JOBS = 20

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
PROBLEM_TYPES = {"classification"}
CLASSIFIER_FAMILIES = {"logistic_regression", "random_forest", "svm"}
TARGET_AGGREGATIONS = {"median", "mean"}
THRESHOLD_STRATEGIES = {"train_median", "fixed"}
LEAKAGE_COLUMN_DEFAULTS = {
    "target",
    "target_binary",
    "target_label",
    "target_class",
    "prediction",
    "predictions",
    "pred_score",
    "prediction_score",
    "score",
    "quality",
    "pq",
    "dice",
}
AUTO_METADATA_EXCLUDE_TOKENS = (
    "path",
    "offset",
    "format",
    "embedding",
    "index",
    "row",
    "mask",
    "class_label",
)


@dataclass
class OuterSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train a patch-level failure predictor from embeddings by first aggregating per-model segmentation "
            "quality into a single patch target, then tuning multiple classifier families with grouped CV."
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
        help="Column used for grouped splitting. Usually slide_id.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help="Per-model metric column aggregated into a patch-level target, for example instance_pq.",
    )
    parser.add_argument(
        "--target-aggregation",
        choices=sorted(TARGET_AGGREGATIONS),
        default=DEFAULT_TARGET_AGGREGATION,
        help="Aggregation applied across model rows for each patch.",
    )
    parser.add_argument(
        "--problem-type",
        choices=sorted(PROBLEM_TYPES),
        default=DEFAULT_PROBLEM_TYPE,
        help="Only grouped classification is supported in this Optuna-tuned training flow.",
    )
    parser.add_argument(
        "--classification-threshold-strategy",
        choices=sorted(THRESHOLD_STRATEGIES),
        default=DEFAULT_THRESHOLD_STRATEGY,
        help="How to convert the aggregated continuous target into a binary label.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        help=(
            "Optional fixed threshold for binary labels. "
            "If omitted with --classification-threshold-strategy=train_median, the threshold is learned from the outer training split."
        ),
    )
    parser.add_argument(
        "--positive-class-name",
        choices=("failure", "quality"),
        default=DEFAULT_POSITIVE_CLASS_NAME,
        help="Which semantic class is encoded as label 1.",
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
            "Explicit metadata feature columns to use after patch-level aggregation. "
            "Required for metadata_only or embedding_plus_metadata unless --auto-metadata-cols is enabled."
        ),
    )
    parser.add_argument(
        "--auto-metadata-cols",
        action="store_true",
        help="Infer metadata columns from the aggregated patch table by excluding known identifier, path, and target columns.",
    )
    parser.add_argument(
        "--exclude-metadata-cols",
        nargs="*",
        default=[],
        help="Metadata columns to exclude from training when using --auto-metadata-cols.",
    )
    parser.add_argument(
        "--require-complete-model-coverage",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_COMPLETE_MODEL_COVERAGE,
        help="Drop patches that do not have valid target values from every expected model.",
    )
    parser.add_argument(
        "--classifier-families",
        nargs="+",
        choices=sorted(CLASSIFIER_FAMILIES),
        default=list(DEFAULT_CLASSIFIER_FAMILIES),
        help="Classifier families tuned independently with Optuna.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=DEFAULT_OPTUNA_TRIALS,
        help="Number of Optuna trials run per classifier family.",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds per classifier family.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Number of grouped CV folds inside the Optuna objective.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Thread count used by selected estimators and some plotting helpers.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of groups assigned to the outer held-out test split.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for grouped splitting, Optuna samplers, and model fitting.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS,
        help="Optional cap on aggregated patch rows loaded for debugging.",
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
        "--plots-dir-name",
        default=DEFAULT_PLOTS_DIR_NAME,
        help="Subdirectory under --output-dir used for generated plots.",
    )
    parser.add_argument(
        "--optuna-trials-output-name",
        default=DEFAULT_OPTUNA_TRIALS_OUTPUT_NAME,
        help="Filename for the combined Optuna trial table.",
    )
    parser.add_argument(
        "--best-params-output-name",
        default=DEFAULT_BEST_PARAMS_OUTPUT_NAME,
        help="Filename for saved best parameter JSON.",
    )
    parser.add_argument(
        "--aggregated-output-name",
        default=DEFAULT_AGGREGATED_OUTPUT_NAME,
        help="Filename for the saved aggregated patch-level target table.",
    )
    parser.add_argument(
        "--family-metrics-output-name",
        default=DEFAULT_FAMILY_METRICS_OUTPUT_NAME,
        help="Filename for saved family comparison metrics JSON.",
    )
    parser.add_argument(
        "--metrics-output-name",
        default=DEFAULT_METRICS_OUTPUT_NAME,
        help="Filename for saved summary metrics JSON.",
    )
    parser.add_argument(
        "--config-output-name",
        default=DEFAULT_CONFIG_OUTPUT_NAME,
        help="Filename for saved config JSON.",
    )
    parser.add_argument(
        "--predictions-output-name",
        default=DEFAULT_PREDICTIONS_OUTPUT_NAME,
        help="Filename for saved patch-level predictions CSV.",
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


def _import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "This script requires Optuna for hyperparameter tuning. Add optuna to the Pixi environment first."
        ) from exc
    return optuna


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
    for candidate in (repo_root, input_path.parent, Path.cwd()):
        if candidate is None:
            continue
        resolved_root = candidate.resolve()
        if resolved_root not in candidate_roots:
            candidate_roots.append(resolved_root)

    attempted_paths: list[Path] = []
    for root in candidate_roots:
        candidate_path = (root / path).resolve(strict=False)
        attempted_paths.append(candidate_path)
        if candidate_path.exists():
            return candidate_path

    if require_existing_embeddings:
        raise FileNotFoundError(
            f"Could not resolve embedding path {text!r}. Tried: {[str(candidate) for candidate in attempted_paths]}"
        )

    if attempted_paths:
        return attempted_paths[0]
    raise FileNotFoundError(f"Could not resolve embedding path {text!r}; no candidate roots were available.")


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
    continuous_target_col: str,
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
        continuous_target_col,
        "target_binary",
        "target_label",
        "source_row_count",
        "n_unique_models",
        "n_models_with_target",
        "model_names",
        *LEAKAGE_COLUMN_DEFAULTS,
    }
    excluded.update(args.exclude_metadata_cols)

    inferred: list[str] = []
    for column in dataframe.columns:
        lowered = column.lower()
        if column in excluded:
            continue
        if any(token in lowered for token in AUTO_METADATA_EXCLUDE_TOKENS):
            continue
        if lowered.endswith("_id"):
            continue
        inferred.append(column)
    return inferred


def check_for_feature_target_leakage(
    *,
    metadata_cols: list[str],
    target_col: str,
    continuous_target_col: str,
    logger: logging.Logger,
) -> None:
    lowered = {column.lower() for column in metadata_cols}
    if target_col in metadata_cols or continuous_target_col in metadata_cols:
        raise ValueError("Target-derived columns are included in metadata features. Remove them to prevent leakage.")
    if "target_binary" in lowered or "target_label" in lowered:
        raise ValueError("Derived target label columns would leak into metadata features.")

    suspicious = [
        column
        for column in metadata_cols
        if target_col.lower() in column.lower()
        or continuous_target_col.lower() in column.lower()
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


def detect_patch_constant_columns(
    dataframe: pd.DataFrame,
    *,
    patch_id_col: str,
    excluded_columns: set[str],
) -> list[str]:
    constant_columns: list[str] = []
    grouped = dataframe.groupby(patch_id_col, sort=False)
    for column in dataframe.columns:
        if column in excluded_columns:
            continue
        if grouped[column].nunique(dropna=False).le(1).all():
            constant_columns.append(column)
    return constant_columns


def aggregate_target_values(series: pd.Series, *, aggregation: str) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    if aggregation == "median":
        return float(numeric.median())
    if aggregation == "mean":
        return float(numeric.mean())
    raise ValueError(f"Unsupported aggregation {aggregation!r}.")


def aggregate_patch_level_dataset(
    dataframe: pd.DataFrame,
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    duplicate_patch_model_mask = dataframe.duplicated(
        subset=[args.patch_id_col, args.model_name_col],
        keep=False,
    )
    if duplicate_patch_model_mask.any():
        duplicate_count = int(duplicate_patch_model_mask.sum())
        raise ValueError(
            f"Joined manifest contains {duplicate_count} duplicate patch_id x model_name row(s). "
            "Fix the upstream join before training."
        )

    excluded_columns = {args.model_name_col, args.target_col}
    constant_columns = detect_patch_constant_columns(
        dataframe,
        patch_id_col=args.patch_id_col,
        excluded_columns=excluded_columns,
    )
    required_constant_columns = [
        args.patch_id_col,
        args.slide_id_col,
        args.dataset_col,
        args.group_col,
    ]
    if args.feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        required_constant_columns.append(args.embedding_path_col)
    if args.embedding_offset_col in dataframe.columns:
        required_constant_columns.append(args.embedding_offset_col)
    if args.embedding_format_col in dataframe.columns:
        required_constant_columns.append(args.embedding_format_col)
    missing_constant_columns = [
        column for column in required_constant_columns if column not in constant_columns
    ]
    if missing_constant_columns:
        raise ValueError(
            "Some required patch-level columns vary across model rows and cannot be aggregated safely: "
            + ", ".join(repr(column) for column in missing_constant_columns)
        )

    aggregation_dict = {column: "first" for column in constant_columns if column != args.patch_id_col}
    aggregated = (
        dataframe.groupby(args.patch_id_col, sort=False, as_index=False)
        .agg(aggregation_dict)
        .copy()
    )

    target_numeric = pd.to_numeric(dataframe[args.target_col], errors="coerce")
    target_stats = dataframe[[args.patch_id_col, args.model_name_col]].copy()
    target_stats["_target_numeric"] = target_numeric
    grouped_target = target_stats.groupby(args.patch_id_col, sort=False)
    target_summary = grouped_target.agg(
        source_row_count=(args.model_name_col, "size"),
        n_unique_models=(args.model_name_col, "nunique"),
        n_models_with_target=("_target_numeric", lambda s: int(s.notna().sum())),
        model_names=(args.model_name_col, lambda s: json.dumps(sorted({str(value) for value in s if pd.notna(value)}))),
        target_continuous=("_target_numeric", lambda s: aggregate_target_values(s, aggregation=args.target_aggregation)),
    )
    aggregated = aggregated.merge(
        target_summary.reset_index(),
        on=args.patch_id_col,
        how="left",
        validate="one_to_one",
    )

    expected_model_count = int(
        dataframe[args.model_name_col].astype("string").fillna("").str.strip().loc[lambda s: s.ne("")].nunique()
    )
    dropped_incomplete_patches = 0
    if args.require_complete_model_coverage:
        complete_mask = (
            aggregated["n_unique_models"].eq(expected_model_count)
            & aggregated["n_models_with_target"].eq(expected_model_count)
        )
        dropped_incomplete_patches = int((~complete_mask).sum())
        if dropped_incomplete_patches:
            logger.warning(
                "Dropping %d patch row(s) without complete model coverage (%d expected models).",
                dropped_incomplete_patches,
                expected_model_count,
            )
        aggregated = aggregated.loc[complete_mask].copy()

    missing_target_mask = aggregated["target_continuous"].isna()
    dropped_missing_target_patches = int(missing_target_mask.sum())
    if dropped_missing_target_patches:
        logger.warning(
            "Dropping %d aggregated patch row(s) with missing %s target.",
            dropped_missing_target_patches,
            args.target_aggregation,
        )
        aggregated = aggregated.loc[~missing_target_mask].copy()

    aggregated.reset_index(drop=True, inplace=True)
    aggregation_stats = {
        "expected_model_count": expected_model_count,
        "dropped_incomplete_patch_rows": dropped_incomplete_patches,
        "dropped_missing_target_patch_rows": dropped_missing_target_patches,
        "constant_columns": constant_columns,
    }
    logger.info(
        "Aggregated patch-level table has %d row(s) from %d patch x model row(s).",
        len(aggregated),
        len(dataframe),
    )
    return aggregated, aggregation_stats


def build_grouped_outer_split(
    dataframe: pd.DataFrame,
    *,
    group_col: str,
    test_size: float,
    random_seed: int,
) -> OuterSplit:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"--test-size must be in (0, 1), got {test_size}.")

    groups = dataframe[group_col].astype("string").fillna("").to_numpy()
    if np.any(groups == ""):
        raise ValueError(f"Group column {group_col!r} contains missing/blank values.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_indices, test_indices = next(splitter.split(dataframe, groups=groups))

    train_groups = pd.Series(groups[train_indices]).unique().astype(str)
    test_groups = pd.Series(groups[test_indices]).unique().astype(str)
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        raise RuntimeError(
            f"Grouped split leakage detected: {len(overlap)} group(s) appear in both train and test: {sorted(overlap)[:10]}"
        )

    return OuterSplit(
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        train_groups=np.asarray(train_groups, dtype=object),
        test_groups=np.asarray(test_groups, dtype=object),
    )


def determine_classification_threshold(train_targets: pd.Series, *, args: argparse.Namespace) -> float:
    if args.classification_threshold is not None:
        return float(args.classification_threshold)
    if args.classification_threshold_strategy == "train_median":
        return float(pd.to_numeric(train_targets, errors="coerce").median())
    raise ValueError(
        "A fixed threshold is required when --classification-threshold-strategy is not train_median."
    )


def build_binary_labels(
    continuous_target: pd.Series,
    *,
    threshold: float,
    positive_class_name: str,
) -> pd.Series:
    if positive_class_name == "failure":
        labels = continuous_target < threshold
    else:
        labels = continuous_target >= threshold
    return labels.astype(np.int64)


def build_feature_dataframe(
    dataframe: pd.DataFrame,
    *,
    metadata_cols: list[str],
) -> pd.DataFrame:
    return dataframe[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=dataframe.index)


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


def get_embedding_columns(feature_frame: pd.DataFrame) -> list[str]:
    return [column for column in feature_frame.columns if column.startswith("emb_")]


def get_metadata_columns(feature_frame: pd.DataFrame) -> list[str]:
    return [column for column in feature_frame.columns if not column.startswith("emb_")]


def sample_classifier_params(
    trial: Any,
    *,
    family: str,
    feature_mode: str,
    embedding_dim: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {"family": family}

    pca_candidates = [value for value in (32, 64, 128, 256, 512) if value < embedding_dim]
    can_use_pca = feature_mode in {"embedding_only", "embedding_plus_metadata"} and bool(pca_candidates)

    if family == "logistic_regression":
        params["C"] = trial.suggest_float("C", 1e-3, 50.0, log=True)
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        params["use_pca"] = trial.suggest_categorical("use_pca", [False, True]) if can_use_pca else False
        if params["use_pca"]:
            params["pca_n_components"] = trial.suggest_categorical("pca_n_components", pca_candidates)
    elif family == "random_forest":
        params["n_estimators"] = trial.suggest_int("n_estimators", 200, 800, step=100)
        params["max_depth"] = trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40])
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
        params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.5])
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
    elif family == "svm":
        params["kernel"] = trial.suggest_categorical("kernel", ["linear", "rbf"])
        params["C"] = trial.suggest_float("C", 1e-2, 50.0, log=True)
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        params["use_pca"] = trial.suggest_categorical("use_pca", [False, True]) if can_use_pca else False
        if params["use_pca"]:
            params["pca_n_components"] = trial.suggest_categorical("pca_n_components", pca_candidates)
        if params["kernel"] == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-5, 1e-1, log=True)
    else:
        raise ValueError(f"Unsupported classifier family {family!r}.")
    return params


def build_preprocessor(
    *,
    feature_frame: pd.DataFrame,
    feature_mode: str,
    family: str,
    params: dict[str, Any],
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []

    embedding_cols = get_embedding_columns(feature_frame)
    metadata_cols = get_metadata_columns(feature_frame)
    metadata_frame = feature_frame[metadata_cols] if metadata_cols else pd.DataFrame(index=feature_frame.index)

    if feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        if not embedding_cols:
            raise ValueError("No embedding columns were found for embedding-based training.")
        embedding_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
        if family in {"logistic_regression", "svm"}:
            embedding_steps.append(("scaler", StandardScaler()))
            if params.get("use_pca"):
                embedding_steps.append(("pca", PCA(n_components=int(params["pca_n_components"]), random_state=0)))
        transformers.append(("embedding", Pipeline(steps=embedding_steps), embedding_cols))

    if feature_mode in {"metadata_only", "embedding_plus_metadata"} and not metadata_frame.empty:
        numeric_cols = metadata_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = [column for column in metadata_frame.columns if column not in numeric_cols]
        if numeric_cols:
            numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
            if family in {"logistic_regression", "svm"}:
                numeric_steps.append(("scaler", StandardScaler()))
            transformers.append(("metadata_numeric", Pipeline(steps=numeric_steps), numeric_cols))
        if categorical_cols:
            categorical_steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            transformers.append(("metadata_categorical", Pipeline(steps=categorical_steps), categorical_cols))

    if not transformers:
        raise ValueError("No usable feature transformers were constructed. Check feature mode and metadata columns.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_estimator(
    *,
    family: str,
    params: dict[str, Any],
    random_seed: int,
    n_jobs: int,
) -> Any:
    if family == "logistic_regression":
        return LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=2000,
            solver="lbfgs",
            random_state=random_seed,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=random_seed,
            n_jobs=n_jobs,
        )
    if family == "svm":
        estimator_kwargs: dict[str, Any] = {
            "kernel": params["kernel"],
            "C": float(params["C"]),
            "class_weight": params["class_weight"],
            "probability": True,
            "random_state": random_seed,
        }
        if params["kernel"] == "rbf":
            estimator_kwargs["gamma"] = float(params["gamma"])
        return SVC(**estimator_kwargs)
    raise ValueError(f"Unsupported classifier family {family!r}.")


def get_positive_class_score(model: Any, X: pd.DataFrame, *, positive_class_value: int = 1) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise RuntimeError("predict_proba is available but classes_ is missing from the estimator.")
        class_list = list(classes)
        if positive_class_value not in class_list:
            raise RuntimeError(f"Positive class value {positive_class_value} is not present in estimator.classes_.")
        positive_index = class_list.index(positive_class_value)
        return np.asarray(probabilities[:, positive_index], dtype=np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    raise RuntimeError("Estimator does not support predict_proba or decision_function.")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1.0 - 1e-6))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def run_optuna_for_family(
    *,
    family: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    feature_mode: str,
    random_seed: int,
    cv_folds: int,
    optuna_trials: int,
    optuna_timeout: int | None,
    n_jobs: int,
    plots_dir: Path,
    logger: logging.Logger,
) -> tuple[Any, pd.DataFrame]:
    optuna = _import_optuna()

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels contain only one class; Optuna tuning cannot proceed.")
    unique_groups = pd.Index(pd.unique(groups_train))
    if len(unique_groups) < cv_folds:
        raise ValueError(
            f"Need at least {cv_folds} distinct groups for grouped CV, got {len(unique_groups)}."
        )

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(5, max(1, optuna_trials // 5)))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"quality_probe_{family}",
    )

    embedding_dim = len(get_embedding_columns(X_train))
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    def objective(trial: Any) -> float:
        params = sample_classifier_params(
            trial,
            family=family,
            feature_mode=feature_mode,
            embedding_dim=embedding_dim if embedding_dim > 0 else 1,
        )
        train_auc_scores: list[float] = []
        valid_auc_scores: list[float] = []
        train_loss_scores: list[float] = []
        valid_loss_scores: list[float] = []

        for fold_index, (inner_train_idx, inner_valid_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train)):
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_valid = X_train.iloc[inner_valid_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_valid = y_train[inner_valid_idx]

            if len(np.unique(y_inner_train)) < 2 or len(np.unique(y_inner_valid)) < 2:
                raise ValueError("A grouped CV fold produced only one class. Reduce cv-folds or adjust the split.")

            preprocessor = build_preprocessor(
                feature_frame=X_inner_train,
                feature_mode=feature_mode,
                family=family,
                params=params,
            )
            estimator = build_estimator(
                family=family,
                params=params,
                random_seed=random_seed,
                n_jobs=n_jobs,
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
            pipeline.fit(X_inner_train, y_inner_train)

            train_score = get_positive_class_score(pipeline, X_inner_train, positive_class_value=1)
            valid_score = get_positive_class_score(pipeline, X_inner_valid, positive_class_value=1)
            train_pred = pipeline.predict(X_inner_train)
            valid_pred = pipeline.predict(X_inner_valid)

            train_auc_scores.append(float(roc_auc_score(y_inner_train, train_score)))
            valid_auc_scores.append(float(roc_auc_score(y_inner_valid, valid_score)))
            train_loss_scores.append(float(log_loss(y_inner_train, np.clip(train_score, 1e-6, 1.0 - 1e-6))))
            valid_loss_scores.append(float(log_loss(y_inner_valid, np.clip(valid_score, 1e-6, 1.0 - 1e-6))))

            trial.report(float(np.mean(valid_auc_scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("family", family)
        trial.set_user_attr("mean_train_roc_auc", float(np.mean(train_auc_scores)))
        trial.set_user_attr("mean_valid_roc_auc", float(np.mean(valid_auc_scores)))
        trial.set_user_attr("std_valid_roc_auc", float(np.std(valid_auc_scores, ddof=0)))
        trial.set_user_attr("mean_train_log_loss", float(np.mean(train_loss_scores)))
        trial.set_user_attr("mean_valid_log_loss", float(np.mean(valid_loss_scores)))
        return float(np.mean(valid_auc_scores))

    logger.info("Starting Optuna study for %s with %d trial(s).", family, optuna_trials)
    study.optimize(objective, n_trials=optuna_trials, timeout=optuna_timeout, show_progress_bar=False)
    trials_df = study.trials_dataframe()
    if "params_family" not in trials_df.columns:
        trials_df["params_family"] = family
    trials_df["family"] = family

    save_optuna_family_plots(study=study, trials_df=trials_df, family=family, plots_dir=plots_dir, logger=logger)
    return study, trials_df


def fit_family_pipeline(
    *,
    family: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_seed: int,
    n_jobs: int,
) -> Pipeline:
    preprocessor = build_preprocessor(
        feature_frame=X_train,
        feature_mode="embedding_plus_metadata" if get_metadata_columns(X_train) and get_embedding_columns(X_train) else (
            "metadata_only" if get_metadata_columns(X_train) else "embedding_only"
        ),
        family=family,
        params=params,
    )
    estimator = build_estimator(
        family=family,
        params=params,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    pipeline.fit(X_train, y_train)
    return pipeline


def plot_and_save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_optuna_family_plots(
    *,
    study: Any,
    trials_df: pd.DataFrame,
    family: str,
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    optuna = _import_optuna()
    family_plot_dir = plots_dir / family
    family_plot_dir.mkdir(parents=True, exist_ok=True)

    complete_trials = trials_df.loc[trials_df["state"] == "COMPLETE"].copy()
    if complete_trials.empty:
        logger.warning("No completed Optuna trials were available for %s; skipping Optuna plots.", family)
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    trial_numbers = complete_trials["number"].to_numpy()
    trial_values = complete_trials["value"].to_numpy(dtype=float)
    best_so_far = np.maximum.accumulate(trial_values)
    ax.plot(trial_numbers, trial_values, marker="o", linestyle="-", alpha=0.75, label="validation ROC-AUC")
    ax.plot(trial_numbers, best_so_far, linestyle="--", linewidth=2.0, label="best so far")
    ax.set_title(f"{family} optimization history")
    ax.set_xlabel("trial")
    ax.set_ylabel("ROC-AUC")
    ax.legend()
    plot_and_save_figure(fig, family_plot_dir / "optimization_history.png")

    history_columns = {"user_attrs_mean_train_roc_auc", "user_attrs_mean_valid_roc_auc"}
    if history_columns.issubset(complete_trials.columns):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            trial_numbers,
            complete_trials["user_attrs_mean_train_roc_auc"].to_numpy(dtype=float),
            marker="o",
            label="train ROC-AUC",
        )
        ax.plot(
            trial_numbers,
            complete_trials["user_attrs_mean_valid_roc_auc"].to_numpy(dtype=float),
            marker="o",
            label="validation ROC-AUC",
        )
        ax.set_title(f"{family} train vs validation ROC-AUC")
        ax.set_xlabel("trial")
        ax.set_ylabel("ROC-AUC")
        ax.legend()
        plot_and_save_figure(fig, family_plot_dir / "train_validation_auc_history.png")

    loss_columns = {"user_attrs_mean_train_log_loss", "user_attrs_mean_valid_log_loss"}
    if loss_columns.issubset(complete_trials.columns):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            trial_numbers,
            complete_trials["user_attrs_mean_train_log_loss"].to_numpy(dtype=float),
            marker="o",
            label="train log loss",
        )
        ax.plot(
            trial_numbers,
            complete_trials["user_attrs_mean_valid_log_loss"].to_numpy(dtype=float),
            marker="o",
            label="validation log loss",
        )
        ax.set_title(f"{family} train vs validation log loss")
        ax.set_xlabel("trial")
        ax.set_ylabel("log loss")
        ax.legend()
        plot_and_save_figure(fig, family_plot_dir / "train_validation_loss_history.png")

    importances = optuna.importance.get_param_importances(study)
    if importances:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(importances.keys())[::-1]
        values = [importances[label] for label in labels]
        ax.barh(labels, values)
        ax.set_title(f"{family} parameter importances")
        ax.set_xlabel("importance")
        plot_and_save_figure(fig, family_plot_dir / "param_importances.png")

    param_columns = [column for column in complete_trials.columns if column.startswith("params_")]
    if param_columns:
        best_param = param_columns[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        x_values = complete_trials[best_param].astype(str)
        ax.scatter(x_values, complete_trials["value"].to_numpy(dtype=float), alpha=0.75)
        ax.set_title(f"{family} score vs {best_param.removeprefix('params_')}")
        ax.set_xlabel(best_param.removeprefix("params_"))
        ax.set_ylabel("ROC-AUC")
        ax.tick_params(axis="x", rotation=45)
        plot_and_save_figure(fig, family_plot_dir / "score_vs_primary_param.png")


def save_confusion_matrix_plot(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    title: str,
    negative_class_name: str,
    positive_class_name: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=[negative_class_name, positive_class_name])
    ax.set_yticks([0, 1], labels=[negative_class_name, positive_class_name])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plot_and_save_figure(fig, path)


def save_roc_curve_plot(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plot_and_save_figure(fig, path)


def save_precision_recall_curve_plot(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    title: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {average_precision:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    plot_and_save_figure(fig, path)


def save_learning_curve_plot(
    *,
    family: str,
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    feature_mode: str,
    random_seed: int,
    cv_folds: int,
    n_jobs: int,
    path: Path,
    logger: logging.Logger,
) -> None:
    if len(np.unique(y_train)) < 2:
        logger.warning("Skipping learning-curve plot because the training labels contain only one class.")
        return

    try:
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        preprocessor = build_preprocessor(
            feature_frame=X_train,
            feature_mode=feature_mode,
            family=family,
            params=best_params,
        )
        estimator = build_estimator(
            family=family,
            params=best_params,
            random_seed=random_seed,
            n_jobs=n_jobs,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

        train_sizes, train_scores, valid_scores = learning_curve(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="roc_auc",
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=1,
        )
    except Exception as exc:
        logger.warning("Skipping learning-curve plot for %s because it failed: %s", family, exc)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="train ROC-AUC")
    ax.plot(train_sizes, valid_scores.mean(axis=1), marker="o", label="validation ROC-AUC")
    ax.set_xlabel("training rows")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{family} learning curve")
    ax.legend()
    plot_and_save_figure(fig, path)


def serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serializable[key] = serialize_metrics(value)
        elif isinstance(value, list):
            serializable[key] = [
                serialize_metrics(item) if isinstance(item, dict) else (
                    item.item() if isinstance(item, (np.floating, np.integer)) else item
                )
                for item in value
            ]
        elif isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.verbose)
    _import_optuna()

    repo_root = resolve_repo_root(args.repo_root)
    input_path = resolve_cli_path(args.input, repo_root=repo_root)
    output_dir = resolve_cli_path(args.output_dir, repo_root=repo_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Joined manifest does not exist: {input_path}")

    logger.info("Reading joined manifest: %s", input_path)
    dataframe = read_table(input_path)
    required_columns = [
        args.patch_id_col,
        args.slide_id_col,
        args.dataset_col,
        args.model_name_col,
        args.group_col,
        args.target_col,
    ]
    if args.feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        required_columns.append(args.embedding_path_col)
    validate_required_columns(dataframe, required_columns, context="Joined manifest")
    validate_non_missing_core_values(
        dataframe,
        required_columns=[
            args.patch_id_col,
            args.slide_id_col,
            args.dataset_col,
            args.group_col,
        ],
    )

    aggregated, aggregation_stats = aggregate_patch_level_dataset(dataframe, args=args, logger=logger)
    if args.max_train_rows is not None:
        aggregated = aggregated.head(args.max_train_rows).copy()
        logger.info("Applied aggregated patch row limit: %d row(s).", len(aggregated))

    if aggregated.empty:
        raise RuntimeError("No aggregated patch rows were available after preprocessing.")

    if args.feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode {args.feature_mode!r}.")

    continuous_target_col = "target_continuous"
    explicit_metadata_cols = [column for column in args.metadata_cols if column]
    if args.auto_metadata_cols:
        inferred_metadata_cols = infer_metadata_columns(
            aggregated,
            args=args,
            continuous_target_col=continuous_target_col,
        )
        metadata_cols = list(dict.fromkeys(explicit_metadata_cols + inferred_metadata_cols))
    else:
        metadata_cols = explicit_metadata_cols

    if args.feature_mode in {"metadata_only", "embedding_plus_metadata"} and not metadata_cols:
        raise ValueError(
            "Metadata features are required for this feature mode. Supply --metadata-cols or enable --auto-metadata-cols."
        )
    validate_required_columns(aggregated, metadata_cols, context="Aggregated patch metadata features")
    check_for_feature_target_leakage(
        metadata_cols=metadata_cols,
        target_col=args.target_col,
        continuous_target_col=continuous_target_col,
        logger=logger,
    )

    logger.info("Using feature mode %s.", args.feature_mode)
    logger.info("Using metadata columns: %s", metadata_cols)

    metadata_frame = build_feature_dataframe(aggregated, metadata_cols=metadata_cols)
    embedding_matrix: np.ndarray | None = None
    if args.feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        embedding_matrix, loaded_indices = load_embedding_matrix(
            aggregated,
            input_path=input_path,
            repo_root=repo_root,
            embedding_path_col=args.embedding_path_col,
            embedding_offset_col=args.embedding_offset_col,
            embedding_format_col=args.embedding_format_col,
            require_existing_embeddings=args.require_existing_embeddings,
            show_progress=args.progress,
            logger=logger,
        )
        if loaded_indices != list(aggregated.index):
            raise RuntimeError("Embedding load order drift detected; loaded row indices do not match the aggregated table.")

    feature_frame = materialize_training_frame(
        embedding_matrix=embedding_matrix,
        metadata_frame=metadata_frame,
        feature_mode=args.feature_mode,
    )
    split = build_grouped_outer_split(
        aggregated,
        group_col=args.group_col,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    train_continuous_target = aggregated.loc[split.train_indices, continuous_target_col]
    threshold = determine_classification_threshold(train_continuous_target, args=args)
    negative_class_name = "quality" if args.positive_class_name == "failure" else "failure"
    aggregated["target_binary"] = build_binary_labels(
        aggregated[continuous_target_col],
        threshold=threshold,
        positive_class_name=args.positive_class_name,
    )
    aggregated["target_label"] = aggregated["target_binary"].map({1: args.positive_class_name, 0: negative_class_name})
    aggregated["split"] = "train"
    aggregated.loc[split.test_indices, "split"] = "test"

    y_binary = aggregated["target_binary"].to_numpy(dtype=np.int64)
    y_train = y_binary[split.train_indices]
    y_test = y_binary[split.test_indices]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Outer grouped split produced only one class in train or test. Adjust the split settings.")

    X_train = feature_frame.iloc[split.train_indices].reset_index(drop=True)
    X_test = feature_frame.iloc[split.test_indices].reset_index(drop=True)
    groups_train = aggregated.iloc[split.train_indices][args.group_col].astype(str).to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / args.plots_dir_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_trials: list[pd.DataFrame] = []
    family_metrics_summary: dict[str, Any] = {}
    best_params_by_family: dict[str, dict[str, Any]] = {}
    failed_families: dict[str, str] = {}

    for family in args.classifier_families:
        try:
            study, trials_df = run_optuna_for_family(
                family=family,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                feature_mode=args.feature_mode,
                random_seed=args.random_seed,
                cv_folds=args.cv_folds,
                optuna_trials=args.optuna_trials,
                optuna_timeout=args.optuna_timeout,
                n_jobs=args.n_jobs,
                plots_dir=plots_dir,
                logger=logger,
            )
            trials_df["selected_threshold"] = threshold
            all_trials.append(trials_df)

            best_trial = study.best_trial
            raw_params = dict(best_trial.params)
            best_params_by_family[family] = raw_params.copy()

            family_pipeline = fit_family_pipeline(
                family=family,
                params={"family": family, **raw_params},
                X_train=X_train,
                y_train=y_train,
                random_seed=args.random_seed,
                n_jobs=args.n_jobs,
            )

            train_score = get_positive_class_score(family_pipeline, X_train, positive_class_value=1)
            test_score = get_positive_class_score(family_pipeline, X_test, positive_class_value=1)
            train_pred = family_pipeline.predict(X_train)
            test_pred = family_pipeline.predict(X_test)

            family_metrics_summary[family] = {
                "status": "ok",
                "best_cv_roc_auc": float(study.best_value),
                "best_params": raw_params,
                "train_metrics": compute_classification_metrics(y_train, train_pred, y_score=train_score),
                "test_metrics": compute_classification_metrics(y_test, test_pred, y_score=test_score),
            }
        except Exception as exc:
            failed_families[family] = str(exc)
            logger.exception("Classifier family %s failed during tuning or evaluation.", family)
            family_metrics_summary[family] = {
                "status": "failed",
                "failure_reason": str(exc),
            }

    if not best_params_by_family:
        raise RuntimeError("All classifier families failed during tuning. Check the logged errors and input data.")

    combined_trials = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    trials_output_path = output_dir / args.optuna_trials_output_name
    if not combined_trials.empty:
        logger.info("Writing Optuna trials: %s", trials_output_path)
        combined_trials.to_csv(trials_output_path, index=False)

    best_params_output_path = output_dir / args.best_params_output_name
    family_metrics_output_path = output_dir / args.family_metrics_output_name
    best_params_output_path.write_text(json.dumps(best_params_by_family, indent=2, sort_keys=True) + "\n")
    family_metrics_output_path.write_text(json.dumps(serialize_metrics(family_metrics_summary), indent=2, sort_keys=True) + "\n")

    selected_family = max(
        (
            (family, metrics)
            for family, metrics in family_metrics_summary.items()
            if metrics.get("status") == "ok"
        ),
        key=lambda item: item[1]["best_cv_roc_auc"],
    )[0]
    selected_params = {"family": selected_family, **best_params_by_family[selected_family]}
    selected_pipeline_eval = fit_family_pipeline(
        family=selected_family,
        params=selected_params,
        X_train=X_train,
        y_train=y_train,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )

    train_score = get_positive_class_score(selected_pipeline_eval, X_train, positive_class_value=1)
    test_score = get_positive_class_score(selected_pipeline_eval, X_test, positive_class_value=1)
    train_pred = selected_pipeline_eval.predict(X_train)
    test_pred = selected_pipeline_eval.predict(X_test)

    test_plot_dir = plots_dir / selected_family
    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=test_pred,
        path=test_plot_dir / "confusion_matrix_test.png",
        title=f"{selected_family} test confusion matrix",
        negative_class_name=negative_class_name,
        positive_class_name=args.positive_class_name,
    )
    save_roc_curve_plot(
        y_true=y_test,
        y_score=test_score,
        path=test_plot_dir / "roc_curve_test.png",
        title=f"{selected_family} test ROC curve",
    )
    save_precision_recall_curve_plot(
        y_true=y_test,
        y_score=test_score,
        path=test_plot_dir / "precision_recall_curve_test.png",
        title=f"{selected_family} test precision-recall curve",
    )
    save_learning_curve_plot(
        family=selected_family,
        best_params=selected_params,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        feature_mode=args.feature_mode,
        random_seed=args.random_seed,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        path=test_plot_dir / "learning_curve_roc_auc.png",
        logger=logger,
    )

    predictions_df = aggregated.copy()
    predictions_df["target"] = predictions_df["target_binary"].astype(np.int64)
    predictions_df["prediction"] = pd.Series(pd.NA, index=predictions_df.index, dtype="Int64")
    predictions_df["prediction_score"] = np.nan
    predictions_df["predicted_failure_label"] = pd.Series(pd.NA, index=predictions_df.index, dtype="Int64")
    predictions_df["predicted_failure_probability"] = np.nan
    predictions_df["predicted_quality_label"] = pd.Series(pd.NA, index=predictions_df.index, dtype="Int64")
    predictions_df["predicted_quality_probability"] = np.nan
    predictions_df["selected_classifier_family"] = selected_family
    predictions_df["classification_threshold"] = threshold

    predictions_df.loc[predictions_df.index[split.train_indices], "prediction"] = train_pred
    predictions_df.loc[predictions_df.index[split.test_indices], "prediction"] = test_pred
    predictions_df.loc[predictions_df.index[split.train_indices], "prediction_score"] = train_score
    predictions_df.loc[predictions_df.index[split.test_indices], "prediction_score"] = test_score
    if args.positive_class_name == "failure":
        predictions_df.loc[predictions_df.index[split.train_indices], "predicted_failure_label"] = train_pred
        predictions_df.loc[predictions_df.index[split.test_indices], "predicted_failure_label"] = test_pred
        predictions_df.loc[predictions_df.index[split.train_indices], "predicted_failure_probability"] = train_score
        predictions_df.loc[predictions_df.index[split.test_indices], "predicted_failure_probability"] = test_score
        predictions_df["predicted_quality_probability"] = 1.0 - predictions_df["predicted_failure_probability"]
        predictions_df["predicted_quality_label"] = (1 - predictions_df["predicted_failure_label"]).astype("Int64")
    else:
        predictions_df.loc[predictions_df.index[split.train_indices], "predicted_quality_label"] = train_pred
        predictions_df.loc[predictions_df.index[split.test_indices], "predicted_quality_label"] = test_pred
        predictions_df.loc[predictions_df.index[split.train_indices], "predicted_quality_probability"] = train_score
        predictions_df.loc[predictions_df.index[split.test_indices], "predicted_quality_probability"] = test_score
        predictions_df["predicted_failure_probability"] = 1.0 - predictions_df["predicted_quality_probability"]
        predictions_df["predicted_failure_label"] = (1 - predictions_df["predicted_quality_label"]).astype("Int64")

    aggregated_output_path = output_dir / args.aggregated_output_name
    logger.info("Writing aggregated patch targets: %s", aggregated_output_path)
    write_table(predictions_df, aggregated_output_path)

    predictions_output_path = output_dir / args.predictions_output_name
    logger.info("Writing predictions: %s", predictions_output_path)
    predictions_df.to_csv(predictions_output_path, index=False)

    final_training_pipeline = fit_family_pipeline(
        family=selected_family,
        params=selected_params,
        X_train=feature_frame.reset_index(drop=True),
        y_train=y_binary,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    estimator = final_training_pipeline.named_steps["estimator"]
    preprocessor = final_training_pipeline.named_steps["preprocessor"]

    model_output_path = output_dir / args.model_output_name
    scaler_output_path = output_dir / args.scaler_output_name
    logger.info("Writing predictor: %s", model_output_path)
    with model_output_path.open("wb") as handle:
        pickle.dump(estimator, handle)
    logger.info("Writing preprocessor/scaler: %s", scaler_output_path)
    with scaler_output_path.open("wb") as handle:
        pickle.dump(preprocessor, handle)

    selected_train_metrics = compute_classification_metrics(y_train, train_pred, y_score=train_score)
    selected_test_metrics = compute_classification_metrics(y_test, test_pred, y_score=test_score)

    metrics_payload: dict[str, Any] = {
        "problem_type": args.problem_type,
        "feature_mode": args.feature_mode,
        "target_col": args.target_col,
        "target_aggregation": args.target_aggregation,
        "aggregated_target_col": continuous_target_col,
        "group_col": args.group_col,
        "positive_class_name": args.positive_class_name,
        "positive_class_value": 1,
        "negative_class_name": negative_class_name,
        "classification_threshold_strategy": args.classification_threshold_strategy,
        "classification_threshold": threshold,
        "expected_model_count": aggregation_stats["expected_model_count"],
        "dropped_incomplete_patch_rows": aggregation_stats["dropped_incomplete_patch_rows"],
        "dropped_missing_target_patch_rows": aggregation_stats["dropped_missing_target_patch_rows"],
        "n_raw_rows": int(len(dataframe)),
        "n_rows": int(len(aggregated)),
        "n_train_rows": int(len(split.train_indices)),
        "n_test_rows": int(len(split.test_indices)),
        "n_train_groups": int(len(split.train_groups)),
        "n_test_groups": int(len(split.test_groups)),
        "train_groups": split.train_groups.tolist(),
        "test_groups": split.test_groups.tolist(),
        "metadata_cols": metadata_cols,
        "embedding_dim": None if embedding_matrix is None else int(embedding_matrix.shape[1]),
        "classifier_families": args.classifier_families,
        "selected_classifier_family": selected_family,
        "selected_best_cv_roc_auc": family_metrics_summary[selected_family]["best_cv_roc_auc"],
        "candidate_family_summary": {
            family: {
                "status": family_metrics_summary[family]["status"],
                "best_cv_roc_auc": family_metrics_summary[family].get("best_cv_roc_auc"),
                "test_roc_auc": family_metrics_summary[family].get("test_metrics", {}).get("roc_auc"),
                "test_balanced_accuracy": family_metrics_summary[family].get("test_metrics", {}).get("balanced_accuracy"),
                "failure_reason": family_metrics_summary[family].get("failure_reason"),
            }
            for family in family_metrics_summary
        },
        "train_metrics": selected_train_metrics,
        "test_metrics": selected_test_metrics,
    }

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
            "problem_type": args.problem_type,
            "aggregated_target_col": continuous_target_col,
            "classification_threshold": threshold,
            "positive_class_name": args.positive_class_name,
            "positive_class_value": 1,
            "negative_class_name": negative_class_name,
            "selected_classifier_family": selected_family,
            "selected_best_params": best_params_by_family[selected_family],
            "failed_classifier_families": failed_families,
            "expected_model_count": aggregation_stats["expected_model_count"],
            "metadata_cols": metadata_cols,
            "embedding_dim": None if embedding_matrix is None else int(embedding_matrix.shape[1]),
            "train_groups": split.train_groups.tolist(),
            "test_groups": split.test_groups.tolist(),
            "plots_dir": str(plots_dir),
            "aggregated_output_path": str(aggregated_output_path),
            "trials_output_path": str(trials_output_path),
            "best_params_output_path": str(best_params_output_path),
            "family_metrics_output_path": str(family_metrics_output_path),
        }
    )

    config_output_path = output_dir / args.config_output_name
    metrics_output_path = output_dir / args.metrics_output_name
    logger.info("Writing config: %s", config_output_path)
    config_output_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
    logger.info("Writing metrics: %s", metrics_output_path)
    metrics_output_path.write_text(json.dumps(serialize_metrics(metrics_payload), indent=2, sort_keys=True) + "\n")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
