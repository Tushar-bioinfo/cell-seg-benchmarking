#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_PATH = Path("outputs/conic_liz/failure_prediction/patch_manifest.parquet")
DEFAULT_ARTIFACT_DIR = Path("outputs/conic_liz/failure_prediction/quality_probe")
DEFAULT_OUTPUT_PATH = Path("outputs/conic_liz/failure_prediction/quality_probe/inference_predictions.parquet")
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_MODEL_NAME = "predictor.pkl"
DEFAULT_SCALER_NAME = "scaler.pkl"
DEFAULT_PROGRESS = True

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run a trained quality probe on new patches without ground truth. "
            "Loads the saved predictor, preprocessor, and config, then writes a clean prediction table."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Patch manifest with embedding_path and identifiers (.csv or .parquet).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing the saved model artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output prediction table (.csv or .parquet).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root used to resolve relative paths.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Filename of the saved training config JSON inside --artifact-dir.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Filename of the saved predictor pickle inside --artifact-dir.",
    )
    parser.add_argument(
        "--scaler-name",
        default=DEFAULT_SCALER_NAME,
        help="Filename of the saved preprocessor/scaler pickle inside --artifact-dir.",
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
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("infer_quality_probe")
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
        raise ValueError(f"{context} is missing required columns: " + ", ".join(repr(column) for column in missing))


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "This script requires PyTorch to load .pt embedding files. Install torch in the runtime environment."
        ) from exc
    return torch


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


def load_single_embedding_vector(
    *,
    embedding_path: Path,
    embedding_format: str,
    row_offset: int,
) -> np.ndarray:
    if embedding_format != "pt":
        raise ValueError(f"Unsupported embedding format {embedding_format!r}. Only .pt inference is supported.")
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
) -> np.ndarray:
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
            f"Failed to load {len(failed_indices)} embedding row(s). Fix missing/corrupt embedding files before inference."
        )
    if not vectors:
        raise RuntimeError("No embeddings were loaded.")

    matrix = np.vstack(vectors).astype(np.float32)
    logger.info("Loaded embedding matrix with shape %s.", matrix.shape)
    return matrix


def build_metadata_frame(dataframe: pd.DataFrame, *, metadata_cols: list[str]) -> pd.DataFrame:
    if not metadata_cols:
        return pd.DataFrame(index=dataframe.index)
    validate_required_columns(dataframe, metadata_cols, context="Inference manifest metadata features")
    return dataframe[metadata_cols].copy()


def materialize_feature_frame(
    *,
    embedding_matrix: np.ndarray | None,
    metadata_frame: pd.DataFrame,
    feature_mode: str,
    embedding_dim: int | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        if embedding_matrix is None:
            raise ValueError("Embedding matrix is required for this feature mode.")
        if embedding_dim is not None and embedding_matrix.shape[1] != int(embedding_dim):
            raise ValueError(
                f"Loaded embedding dimension {embedding_matrix.shape[1]} does not match trained embedding_dim={embedding_dim}."
            )
        embedding_cols = [f"emb_{index:04d}" for index in range(embedding_matrix.shape[1])]
        frames.append(pd.DataFrame(embedding_matrix, columns=embedding_cols, index=metadata_frame.index))

    if feature_mode in {"metadata_only", "embedding_plus_metadata"}:
        frames.append(metadata_frame)

    if not frames:
        raise ValueError(f"Unsupported feature mode {feature_mode!r}.")
    return pd.concat(frames, axis=1)


def load_pickle(path: Path, *, label: str) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {label} from {path}: {type(exc).__name__}: {exc}") from exc


def load_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to load config JSON from {path}: {type(exc).__name__}: {exc}") from exc


def determine_output_columns(
    dataframe: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> list[str]:
    preferred = [
        config.get("patch_id_col", "patch_id"),
        config.get("slide_id_col", "slide_id"),
        config.get("dataset_col", "dataset"),
    ]
    model_name_col = config.get("model_name_col")
    if model_name_col:
        preferred.append(model_name_col)
    group_col = config.get("group_col")
    if group_col and group_col not in preferred:
        preferred.append(group_col)
    preferred.append(config.get("embedding_path_col", "embedding_path"))
    return [column for column in preferred if column in dataframe.columns]


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.verbose)

    repo_root = resolve_repo_root(args.repo_root)
    input_path = resolve_cli_path(args.input, repo_root=repo_root)
    artifact_dir = resolve_cli_path(args.artifact_dir, repo_root=repo_root)
    output_path = resolve_cli_path(args.output, repo_root=repo_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Inference manifest does not exist: {input_path}")
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")

    config_path = artifact_dir / args.config_name
    model_path = artifact_dir / args.model_name
    scaler_path = artifact_dir / args.scaler_name
    for path, label in (
        (config_path, "config"),
        (model_path, "predictor"),
        (scaler_path, "preprocessor/scaler"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")

    logger.info("Loading artifacts from %s", artifact_dir)
    config = load_config(config_path)
    estimator = load_pickle(model_path, label="predictor")
    preprocessor = load_pickle(scaler_path, label="preprocessor/scaler")

    problem_type = config.get("problem_type")
    feature_mode = config.get("feature_mode")
    if problem_type not in {"regression", "classification"}:
        raise ValueError(f"Saved config has unsupported problem_type={problem_type!r}.")
    if feature_mode not in {"embedding_only", "metadata_only", "embedding_plus_metadata"}:
        raise ValueError(f"Saved config has unsupported feature_mode={feature_mode!r}.")

    patch_id_col = config.get("patch_id_col", "patch_id")
    slide_id_col = config.get("slide_id_col", "slide_id")
    dataset_col = config.get("dataset_col", "dataset")
    model_name_col = config.get("model_name_col", "model_name")
    embedding_path_col = config.get("embedding_path_col", "embedding_path")
    embedding_offset_col = config.get("embedding_offset_col", "embedding_row_offset")
    embedding_format_col = config.get("embedding_format_col", "embedding_format")
    metadata_cols = list(config.get("metadata_cols", []))
    embedding_dim = config.get("embedding_dim")

    manifest = read_table(input_path)
    required_cols = [patch_id_col, slide_id_col, dataset_col]
    if feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        required_cols.append(embedding_path_col)
    validate_required_columns(manifest, required_cols, context="Inference manifest")

    logger.info("Loaded inference manifest with %d row(s) x %d column(s).", len(manifest), len(manifest.columns))

    metadata_frame = build_metadata_frame(manifest, metadata_cols=metadata_cols)
    embedding_matrix: np.ndarray | None = None
    if feature_mode in {"embedding_only", "embedding_plus_metadata"}:
        embedding_matrix = load_embedding_matrix(
            manifest,
            input_path=input_path,
            repo_root=repo_root,
            embedding_path_col=embedding_path_col,
            embedding_offset_col=embedding_offset_col,
            embedding_format_col=embedding_format_col,
            require_existing_embeddings=args.require_existing_embeddings,
            show_progress=args.progress,
            logger=logger,
        )

    X = materialize_feature_frame(
        embedding_matrix=embedding_matrix,
        metadata_frame=metadata_frame,
        feature_mode=feature_mode,
        embedding_dim=embedding_dim,
    )

    try:
        X_transformed = preprocessor.transform(X)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to transform inference features with the saved preprocessor. "
            f"This usually means the manifest schema does not match training. Details: {exc}"
        ) from exc

    prediction_table = manifest.loc[:, determine_output_columns(manifest, config=config)].copy()
    prediction_table["artifact_dir"] = str(artifact_dir)
    prediction_table["feature_mode"] = feature_mode
    prediction_table["problem_type"] = problem_type
    if "selected_classifier_family" in config:
        prediction_table["selected_classifier_family"] = config["selected_classifier_family"]
    if "classification_threshold" in config:
        prediction_table["classification_threshold"] = config["classification_threshold"]

    if problem_type == "regression":
        predicted_quality = estimator.predict(X_transformed)
        prediction_table["predicted_quality_score"] = np.asarray(predicted_quality, dtype=np.float64)
    else:
        predicted_label = np.asarray(estimator.predict(X_transformed), dtype=np.int64)
        positive_class_name = str(config.get("positive_class_name", "quality"))
        negative_class_name = str(
            config.get("negative_class_name", "failure" if positive_class_name == "quality" else "quality")
        )
        positive_class_value = int(config.get("positive_class_value", 1))

        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X_transformed)
            classes = getattr(estimator, "classes_", None)
            if classes is None:
                raise RuntimeError("Saved classification estimator exposes predict_proba but not classes_.")
            classes = list(classes)
            if positive_class_value not in classes:
                raise RuntimeError(
                    f"Saved classification estimator does not contain positive class value {positive_class_value!r}."
                )
            positive_index = classes.index(positive_class_value)
            if probabilities.ndim != 2 or probabilities.shape[1] <= positive_index:
                raise RuntimeError(
                    f"Expected binary classification probabilities with a positive-class column, got {probabilities.shape}."
                )
            positive_probability = np.asarray(probabilities[:, positive_index], dtype=np.float64)
        else:
            raise RuntimeError(
                "Saved classification estimator does not support predict_proba, so failure probability cannot be computed."
            )

        prediction_table["positive_class_name"] = positive_class_name
        prediction_table["negative_class_name"] = negative_class_name

        if positive_class_name == "failure":
            failure_probability = positive_probability
            failure_label = (predicted_label == positive_class_value).astype(np.int64)
            quality_probability = 1.0 - failure_probability
            quality_label = 1 - failure_label
        else:
            quality_probability = positive_probability
            quality_label = (predicted_label == positive_class_value).astype(np.int64)
            failure_probability = 1.0 - quality_probability
            failure_label = 1 - quality_label

        prediction_table["predicted_failure_label"] = np.asarray(failure_label, dtype=np.int64)
        prediction_table["predicted_failure_probability"] = np.asarray(failure_probability, dtype=np.float64)
        prediction_table["predicted_quality_label"] = np.asarray(quality_label, dtype=np.int64)
        prediction_table["predicted_quality_probability"] = np.asarray(quality_probability, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing inference predictions: %s", output_path)
    write_table(prediction_table, output_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
