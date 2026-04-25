"""Shared utilities for the sklearn-first modeling workflow.

Why this file exists:
- Centralize the repeated workflow logic used by the model-table build step,
  the three modeling stages, and the final summary step.
- Keep the stage scripts small and consistent on validation, logging, feature
  loading, grouped splitting, Optuna tuning, metric calculation, and artifact
  writing.

What it reads:
- Stage scripts call the helpers here to read CSV, CSV.GZ, JSON, and the
  upstream parquet target table.
- Embedding feature modes use the embedding metadata in the canonical modeling
  table and read the referenced embedding files on demand.

What it writes:
- The helper functions write JSON, Markdown, plots, `run.log`, and
  `timing.json` into stage output directories.

What validation it performs:
- Shared checks for required columns, duplicate IDs, missingness summaries,
  grouped train/test disjointness, artifact existence, and output-table
  readability.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort:skip

try:
    import optuna

    OPTUNA_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on environment
    optuna = None
    OPTUNA_IMPORT_ERROR = str(exc)

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only when xgboost is absent
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = str(exc)

FEATURE_SET_CHOICES = (
    "embedding_only",
    "metadata_only",
    "embedding_plus_metadata",
)

REGRESSION_FAMILIES = ("ridge", "svr", "random_forest", "xgboost")
CLASSIFICATION_FAMILIES = (
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
)

DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_OPTUNA_TRIALS = 20
DEFAULT_OPTUNA_TIMEOUT = 600

REPORTED_TARGET_COLUMNS = (
    "pq_median",
    "rq_median",
    "sq_median",
    "pixel_precision_median",
    "pixel_recall_median",
)

FAILURE_MODE_METRIC_COLUMNS = (
    "rq_median",
    "sq_median",
    "pixel_precision_median",
    "pixel_recall_median",
)

EMBEDDING_METADATA_COLUMNS = (
    "embedding_path",
    "embedding_format",
    "embedding_row_offset",
    "embedding_dim",
)

NON_FEATURE_COLUMNS = {
    "patch_id",
    "slide_id",
    "dataset",
    "split",
    "model_name",
    "train_test_split",
    "report_label",
    "failure_mode_label",
    "is_hard_patch",
    "hard_threshold",
}
NON_FEATURE_COLUMNS.update(REPORTED_TARGET_COLUMNS)

METRIC_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pq": ("pq", "instance_pq", "panoptic_quality"),
    "rq": ("rq", "recognition_quality"),
    "sq": ("sq", "segmentation_quality"),
    "pixel_precision": ("pixel_precision", "precision"),
    "pixel_recall": ("pixel_recall", "recall"),
}


def has_xgboost() -> bool:
    """Return whether xgboost is importable in the current environment."""

    return XGBClassifier is not None and XGBRegressor is not None


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the current working directory."""

    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return (Path.cwd() / path_obj).resolve()


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    """Convert numpy and pandas objects into JSON-safe Python values."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (pd.Series, np.ndarray, list, tuple, set)):
        return [_json_ready(item) for item in list(value)]
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON file with stable formatting."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path_obj


def write_text(path: str | Path, text: str) -> Path:
    """Write plain text to disk."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    path_obj.write_text(text, encoding="utf-8")
    return path_obj


def setup_stage_logging(output_dir: str | Path, stage_name: str) -> logging.Logger:
    """Create a stage logger that writes to both stderr and `run.log`."""

    output_path = ensure_directory(output_dir)
    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_path / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("stage=%s output_dir=%s", stage_name, output_path)
    return logger


def write_timing_json(
    output_dir: str | Path,
    started_time: float,
    stage_name: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write stage timing information."""

    ended_time = time.time()
    payload: dict[str, Any] = {
        "stage_name": stage_name,
        "started_at_utc": datetime.fromtimestamp(started_time, tz=timezone.utc).isoformat(),
        "ended_at_utc": datetime.fromtimestamp(ended_time, tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(ended_time - started_time, 6),
    }
    if extra:
        payload.update(extra)
    return write_json(resolve_path(output_dir) / "timing.json", payload)


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV, CSV.GZ, or parquet table."""

    path_obj = resolve_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input table does not exist: {path_obj}")
    suffixes = path_obj.suffixes
    if suffixes[-2:] == [".csv", ".gz"] or path_obj.suffix == ".csv":
        return pd.read_csv(path_obj, low_memory=False)
    if path_obj.suffix == ".parquet":
        return pd.read_parquet(path_obj)
    raise ValueError(
        f"Unsupported table format for {path_obj}. Expected .csv, .csv.gz, or .parquet."
    )


def save_csv_table(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a DataFrame as CSV or CSV.GZ."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    suffixes = path_obj.suffixes
    if suffixes[-2:] == [".csv", ".gz"]:
        frame.to_csv(path_obj, index=index, compression="gzip")
    elif path_obj.suffix == ".csv":
        frame.to_csv(path_obj, index=index)
    else:
        raise ValueError(f"Output must be .csv or .csv.gz, received {path_obj}")
    return path_obj


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        available = ", ".join(sorted(frame.columns))
        raise ValueError(f"{label} is missing required columns {missing}. Available columns: {available}")


def normalize_model_name(model_name: str) -> str:
    """Normalize model names for stable output column names."""

    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(model_name).strip().lower()).strip("_")
    return normalized or "unknown_model"


def infer_metric_source_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map canonical metric names to the first matching source column."""

    column_set = set(columns)
    resolved: dict[str, str] = {}
    for canonical_name, candidates in METRIC_COLUMN_ALIASES.items():
        for candidate in candidates:
            if candidate in column_set:
                resolved[canonical_name] = candidate
                break
    return resolved


def missingness_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a table-friendly missingness summary."""

    if frame.empty:
        return []
    rows = len(frame)
    summary: list[dict[str, Any]] = []
    for column in frame.columns:
        missing_count = int(frame[column].isna().sum())
        if missing_count == 0:
            continue
        summary.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_fraction": round(missing_count / rows, 6),
            }
        )
    summary.sort(key=lambda item: (-item["missing_count"], item["column"]))
    return summary


def file_existence_rows(paths: Mapping[str, str | Path]) -> list[dict[str, Any]]:
    """Describe whether the expected artifact paths exist on disk."""

    rows: list[dict[str, Any]] = []
    for label, raw_path in paths.items():
        path_obj = resolve_path(raw_path)
        rows.append(
            {
                "label": label,
                "path": str(path_obj),
                "exists": path_obj.exists(),
            }
        )
    return rows


def dataframe_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    """Render a short Markdown table from list-like records."""

    if not records:
        return "_None_\n"
    frame = pd.DataFrame(records)
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        rendered_cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                rendered_cells.append("")
            else:
                rendered_cells.append(str(value).replace("\n", "<br>"))
        rows.append("| " + " | ".join(rendered_cells) + " |")
    return "\n".join(rows) + "\n"


def render_validation_markdown(report: Mapping[str, Any]) -> str:
    """Render a validation report in Markdown."""

    lines = [
        "# Validation Report",
        "",
        f"- Stage: `{report.get('stage_name', 'unknown')}`",
        f"- Success: `{bool(report.get('success', False))}`",
        f"- Generated at (UTC): `{report.get('generated_at_utc', 'unknown')}`",
    ]
    if "input_path" in report:
        lines.append(f"- Input path: `{report['input_path']}`")
    if "output_dir" in report:
        lines.append(f"- Output dir: `{report['output_dir']}`")
    if "summary" in report:
        lines.extend(["", "## Summary", "", dataframe_markdown(report["summary"]).rstrip()])
    if "artifact_checks" in report:
        lines.extend(["", "## Artifact Checks", "", dataframe_markdown(report["artifact_checks"]).rstrip()])
    if "missingness_summary" in report:
        lines.extend(["", "## Missingness Summary", "", dataframe_markdown(report["missingness_summary"]).rstrip()])
    if "notes" in report:
        lines.extend(["", "## Notes", ""])
        for note in report["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_validation_reports(output_dir: str | Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write validation reports in JSON and Markdown."""

    payload = dict(report)
    payload.setdefault("generated_at_utc", now_utc_iso())
    json_path = write_json(resolve_path(output_dir) / "validation.json", payload)
    markdown_path = write_text(
        resolve_path(output_dir) / "validation.md",
        render_validation_markdown(payload),
    )
    return json_path, markdown_path


def verify_no_duplicate_patch_ids(frame: pd.DataFrame, patch_id_col: str = "patch_id") -> int:
    """Return the number of duplicated patch IDs."""

    return int(frame.duplicated(subset=[patch_id_col]).sum())


def split_train_test_by_group(
    frame: pd.DataFrame,
    group_col: str,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one grouped train/test split and verify group disjointness."""

    require_columns(frame, [group_col], "modeling table")
    if not 0 < test_size < 1:
        raise ValueError(f"--test-size must be between 0 and 1, received {test_size}")
    unique_group_count = frame[group_col].nunique(dropna=True)
    if unique_group_count < 2:
        raise ValueError(
            f"Grouped splitting requires at least 2 unique `{group_col}` values, found {unique_group_count}"
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_index, test_index = next(splitter.split(frame, groups=frame[group_col]))
    train_frame = frame.iloc[train_index].copy()
    test_frame = frame.iloc[test_index].copy()
    train_groups = set(train_frame[group_col].dropna().astype(str))
    test_groups = set(test_frame[group_col].dropna().astype(str))
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise AssertionError(f"Train/test group overlap detected for `{group_col}`: {overlap[:10]}")
    return train_frame, test_frame


def build_group_kfold(groups: pd.Series, cv_folds: int) -> GroupKFold:
    """Build a `GroupKFold` with a valid number of splits."""

    unique_group_count = groups.nunique(dropna=True)
    if unique_group_count < 2:
        raise ValueError(
            f"Grouped CV requires at least 2 unique groups, found {unique_group_count}"
        )
    n_splits = min(cv_folds, int(unique_group_count))
    if n_splits < 2:
        raise ValueError(f"Grouped CV needs at least 2 folds, computed {n_splits}")
    return GroupKFold(n_splits=n_splits)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | None]:
    """Compute the required regression metrics."""

    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(observed, predicted))
    metrics: dict[str, float | None] = {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(observed, predicted)),
    }
    if len(np.unique(observed)) > 1 and len(observed) > 1:
        metrics["pearson"] = float(pearsonr(observed, predicted).statistic)
        metrics["spearman"] = float(spearmanr(observed, predicted).statistic)
    else:
        metrics["pearson"] = None
        metrics["spearman"] = None
    return metrics


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    ordered_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute the required classification metrics."""

    truth = np.asarray(list(y_true))
    predictions = np.asarray(list(y_pred))
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(truth, predictions, labels=list(labels), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "per_class_metrics": classification_report(
            truth,
            predictions,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(labels)).tolist(),
        "labels": list(labels),
    }
    if ordered_labels is not None:
        metrics["quadratic_weighted_kappa"] = float(
            cohen_kappa_score(
                truth,
                predictions,
                labels=list(ordered_labels),
                weights="quadratic",
            )
        )
    return metrics


def plot_predicted_vs_observed(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    path: str | Path,
    title: str,
) -> Path:
    """Write a predicted-vs-observed scatter plot."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    lower = float(np.nanmin([observed.min(), predicted.min()]))
    upper = float(np.nanmax([observed.max(), predicted.max()]))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(observed, predicted, alpha=0.7, edgecolors="none")
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.0)
    axis.set_xlabel("Observed")
    axis.set_ylabel("Predicted")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    path: str | Path,
    title: str,
) -> Path:
    """Write a residual plot for regression outputs."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = observed - predicted
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(predicted, residuals, alpha=0.7, edgecolors="none")
    axis.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Residual (observed - predicted)")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_residual_histogram(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    path: str | Path,
    title: str,
) -> Path:
    """Write a residual histogram for regression outputs."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = observed - predicted
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.9)
    axis.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
    axis.set_xlabel("Residual (observed - predicted)")
    axis.set_ylabel("Count")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_confusion_matrix(
    labels: Sequence[str],
    matrix: np.ndarray,
    path: str | Path,
    title: str,
) -> Path:
    """Write a labeled confusion matrix plot."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                int(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def per_class_metrics_frame(
    per_class_metrics: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
) -> pd.DataFrame:
    """Return a tidy per-class metric table."""

    rows: list[dict[str, Any]] = []
    for label in labels:
        metric_row = per_class_metrics.get(label, {})
        rows.append(
            {
                "label": label,
                "precision": metric_row.get("precision"),
                "recall": metric_row.get("recall"),
                "f1_score": metric_row.get("f1-score"),
                "support": metric_row.get("support"),
            }
        )
    return pd.DataFrame(rows)


def plot_per_class_metrics(
    per_class_metrics: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    path: str | Path,
    title: str,
) -> Path:
    """Write a grouped per-class precision/recall/F1 plot."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    metric_frame = per_class_metrics_frame(per_class_metrics, labels)
    x_positions = np.arange(len(metric_frame))
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4.5))
    axis.bar(x_positions - width, metric_frame["precision"], width=width, label="Precision")
    axis.bar(x_positions, metric_frame["recall"], width=width, label="Recall")
    axis.bar(x_positions + width, metric_frame["f1_score"], width=width, label="F1")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(metric_frame["label"], rotation=30, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_label_distribution(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    path: str | Path,
    title: str,
) -> Path:
    """Write side-by-side true/predicted label count bars."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    truth_counts = pd.Series(list(y_true)).value_counts().reindex(labels).fillna(0)
    prediction_counts = pd.Series(list(y_pred)).value_counts().reindex(labels).fillna(0)
    x_positions = np.arange(len(labels))
    width = 0.35
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4.5))
    axis.bar(x_positions - width / 2, truth_counts.values, width=width, label="True")
    axis.bar(x_positions + width / 2, prediction_counts.values, width=width, label="Predicted")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel("Count")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_prediction_confidence(
    predicted_probability: Sequence[float],
    correct_mask: Sequence[bool],
    path: str | Path,
    title: str,
) -> Path:
    """Write a confidence histogram split by correct versus incorrect predictions."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    probabilities = np.asarray(predicted_probability, dtype=float)
    correctness = np.asarray(correct_mask, dtype=bool)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(
        probabilities[correctness],
        bins=20,
        alpha=0.7,
        label="Correct",
        color="seagreen",
        edgecolor="white",
    )
    axis.hist(
        probabilities[~correctness],
        bins=20,
        alpha=0.7,
        label="Incorrect",
        color="indianred",
        edgecolor="white",
    )
    axis.set_xlabel("Predicted probability of assigned class")
    axis.set_ylabel("Count")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_family_cv_scores(
    family_results: Sequence[Mapping[str, Any]],
    path: str | Path,
    title: str,
    score_label: str,
    direction: str,
) -> Path:
    """Write a bar plot comparing the best CV score from each family."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    result_frame = pd.DataFrame(family_results)
    ascending = direction == "minimize"
    result_frame = result_frame.sort_values("best_cv_score", ascending=ascending).reset_index(drop=True)
    x_positions = np.arange(len(result_frame))
    figure, axis = plt.subplots(figsize=(max(6, len(result_frame) * 1.5), 4.5))
    axis.bar(x_positions, result_frame["best_cv_score"], color="slateblue", alpha=0.9)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(result_frame["family"], rotation=30, ha="right")
    axis.set_ylabel(score_label)
    axis.set_title(title)
    subtitle = "Lower is better" if direction == "minimize" else "Higher is better"
    axis.text(0.99, 0.98, subtitle, transform=axis.transAxes, ha="right", va="top", fontsize=9)
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_optuna_history(
    trials_frame: pd.DataFrame,
    path: str | Path,
    title: str,
    direction: str,
    score_label: str,
) -> Path | None:
    """Write an optimization-history plot from an Optuna trial table."""

    if trials_frame.empty or "value" not in trials_frame.columns:
        return None
    filtered = trials_frame.copy()
    if "state" in filtered.columns:
        filtered = filtered.loc[filtered["state"] == "COMPLETE"].copy()
    filtered = filtered.loc[filtered["value"].notna()].sort_values("number")
    if filtered.empty:
        return None
    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    trial_numbers = filtered["number"].astype(int).to_numpy()
    trial_values = filtered["value"].astype(float).to_numpy()
    if direction == "minimize":
        best_so_far = np.minimum.accumulate(trial_values)
    else:
        best_so_far = np.maximum.accumulate(trial_values)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(trial_numbers, trial_values, alpha=0.8, label="Trial value")
    axis.plot(trial_numbers, best_so_far, color="black", linewidth=1.5, label="Best so far")
    axis.set_xlabel("Trial number")
    axis.set_ylabel(score_label)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def plot_optuna_param_importance(
    param_importances: Mapping[str, float],
    path: str | Path,
    title: str,
) -> Path | None:
    """Write a horizontal bar plot of Optuna parameter importances."""

    if not param_importances:
        return None
    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    items = sorted(param_importances.items(), key=lambda item: item[1])
    labels = [item[0] for item in items]
    values = [float(item[1]) for item in items]
    figure, axis = plt.subplots(figsize=(6, max(3, len(labels) * 0.5)))
    axis.barh(labels, values, color="darkorange", alpha=0.9)
    axis.set_xlabel("Importance")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path_obj, dpi=200)
    plt.close(figure)
    return path_obj


def write_optuna_diagnostics(
    *,
    output_dir: str | Path,
    family_results: Sequence[Mapping[str, Any]],
    trials_frame: pd.DataFrame,
    stage_name: str,
    score_label: str,
) -> dict[str, str]:
    """Write reusable Optuna diagnostics for a stage."""

    output_path = resolve_path(output_dir)
    plots_dir = ensure_directory(output_path / "plots")
    family_scores_frame = pd.DataFrame(family_results)
    family_scores_path = save_csv_table(
        family_scores_frame,
        output_path / "optuna_family_scores.csv",
        index=False,
    )
    diagnostics: dict[str, str] = {
        "optuna_family_scores_csv": str(family_scores_path),
    }
    if not family_scores_frame.empty:
        direction = str(family_scores_frame.iloc[0]["direction"])
        family_comparison_path = plot_family_cv_scores(
            family_results=family_results,
            path=plots_dir / "optuna_family_cv_scores.png",
            title=f"{stage_name}: Optuna family comparison",
            score_label=score_label,
            direction=direction,
        )
        diagnostics["optuna_family_cv_scores_png"] = str(family_comparison_path)

    importances_payload: dict[str, dict[str, float]] = {}
    for family_result in family_results:
        family = str(family_result["family"])
        safe_family = normalize_model_name(family)
        family_trials = trials_frame.loc[trials_frame["family"] == family].copy()
        history_path = plot_optuna_history(
            trials_frame=family_trials,
            path=plots_dir / f"optuna_history_{safe_family}.png",
            title=f"{stage_name}: Optuna history ({family})",
            direction=str(family_result["direction"]),
            score_label=score_label,
        )
        if history_path is not None:
            diagnostics[f"optuna_history_{safe_family}_png"] = str(history_path)

        param_importances = {
            str(key): float(value)
            for key, value in dict(family_result.get("param_importances", {})).items()
        }
        if param_importances:
            importances_payload[family] = param_importances
            importance_path = plot_optuna_param_importance(
                param_importances=param_importances,
                path=plots_dir / f"optuna_param_importance_{safe_family}.png",
                title=f"{stage_name}: Optuna param importance ({family})",
            )
            if importance_path is not None:
                diagnostics[f"optuna_param_importance_{safe_family}_png"] = str(importance_path)

    importances_json_path = write_json(
        output_path / "optuna_param_importances.json",
        {
            "stage_name": stage_name,
            "score_label": score_label,
            "by_family": importances_payload,
        },
    )
    diagnostics["optuna_param_importances_json"] = str(importances_json_path)
    return diagnostics


def maybe_parse_probability_frame(
    estimator: Pipeline,
    features: pd.DataFrame,
    class_labels: Sequence[str],
) -> pd.DataFrame:
    """Return class probabilities when the estimator exposes `predict_proba`."""

    if not hasattr(estimator, "predict_proba"):
        return pd.DataFrame(index=features.index)
    probabilities = estimator.predict_proba(features)
    if probabilities.ndim != 2:
        return pd.DataFrame(index=features.index)
    probability_columns = {
        f"prob_{label}": probabilities[:, index] for index, label in enumerate(class_labels)
    }
    probability_frame = pd.DataFrame(probability_columns, index=features.index)
    probability_frame["predicted_probability"] = probability_frame.max(axis=1)
    return probability_frame


def infer_metadata_columns(
    frame: pd.DataFrame,
    explicit_metadata_cols: Sequence[str] | None = None,
) -> list[str]:
    """Choose metadata columns while excluding leakage-prone modeling targets."""

    if explicit_metadata_cols:
        return list(explicit_metadata_cols)

    excluded_columns = set(NON_FEATURE_COLUMNS)
    excluded_columns.update(EMBEDDING_METADATA_COLUMNS)
    inferred: list[str] = []
    for column in frame.columns:
        if column in excluded_columns:
            continue
        if column.endswith("_median") or column.endswith("_model_count"):
            continue
        if "__" in column:
            continue
        inferred.append(column)
    return inferred


def infer_join_key(frame: pd.DataFrame, preferred_key: str) -> str:
    """Infer the patch-level join key for optional metadata inputs."""

    candidates = [preferred_key, "patch_id", "sample_id"]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Unable to infer a patch-level join key. Tried columns: {candidates}. "
        f"Available columns: {list(frame.columns)}"
    )


class EmbeddingStoreCache:
    """Cache loaded embedding stores so repeated feature builds do not reload files."""

    def __init__(self) -> None:
        self._arrays: dict[Path, np.ndarray] = {}

    def _load_pt_file(self, path: Path) -> np.ndarray:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                f"Embedding file {path} requires torch for `.pt` loading, but torch is unavailable: {exc}"
            ) from exc
        tensor_like = torch.load(path, map_location="cpu")
        if hasattr(tensor_like, "detach"):
            return tensor_like.detach().cpu().numpy()
        if isinstance(tensor_like, np.ndarray):
            return tensor_like
        if isinstance(tensor_like, Mapping):
            for key in ("embeddings", "tensor", "data", "features"):
                if key in tensor_like:
                    value = tensor_like[key]
                    if hasattr(value, "detach"):
                        return value.detach().cpu().numpy()
                    return np.asarray(value)
            first_value = next(iter(tensor_like.values()))
            if hasattr(first_value, "detach"):
                return first_value.detach().cpu().numpy()
            return np.asarray(first_value)
        return np.asarray(tensor_like)

    def _load_array(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Embedding file does not exist: {path}")
        if path in self._arrays:
            return self._arrays[path]
        suffixes = path.suffixes
        if path.suffix == ".pt":
            array = self._load_pt_file(path)
        elif path.suffix == ".npy":
            array = np.load(path)
        elif path.suffix == ".npz":
            archive = np.load(path)
            if "embeddings" in archive:
                array = archive["embeddings"]
            else:
                first_key = archive.files[0]
                array = archive[first_key]
        elif suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".csv":
            array = pd.read_csv(path, low_memory=False).to_numpy()
        else:
            raise ValueError(
                f"Unsupported embedding file format for {path}. Expected .pt, .npy, .npz, .csv, or .csv.gz."
            )
        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError(f"Embedding file {path} must resolve to a 2D array, found shape {array.shape}")
        self._arrays[path] = array
        return array

    def load_rows(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Load embedding rows for a frame using `embedding_path` and `embedding_row_offset`."""

        require_columns(frame, ["embedding_path", "embedding_row_offset"], "embedding feature inputs")
        if frame["embedding_path"].isna().any():
            raise ValueError("Embedding feature inputs contain missing `embedding_path` values.")
        if frame["embedding_row_offset"].isna().any():
            raise ValueError("Embedding feature inputs contain missing `embedding_row_offset` values.")
        loaded_chunks: list[np.ndarray] = []
        ordered_indices: list[pd.Index] = []
        for raw_path, subset in frame.groupby("embedding_path", sort=False):
            resolved_path = resolve_path(raw_path)
            store = self._load_array(resolved_path)
            offsets = subset["embedding_row_offset"].astype(int).to_numpy()
            max_offset = int(offsets.max()) if offsets.size else -1
            if max_offset >= len(store):
                raise IndexError(
                    f"Embedding row offset out of range for {resolved_path}: "
                    f"max offset {max_offset} with store length {len(store)}"
                )
            loaded_chunks.append(store[offsets, :])
            ordered_indices.append(subset.index)
        if not loaded_chunks:
            return np.empty((0, 0), dtype=float), []
        embedding_frame = pd.DataFrame(
            np.vstack(loaded_chunks),
            index=pd.Index(np.concatenate([index.to_numpy() for index in ordered_indices])),
        ).sort_index()
        embedding_frame.columns = [f"embedding_{index:04d}" for index in range(embedding_frame.shape[1])]
        return embedding_frame.to_numpy(dtype=float), list(embedding_frame.columns)


def build_feature_frame(
    frame: pd.DataFrame,
    feature_set: str,
    metadata_cols: Sequence[str] | None,
    embedding_cache: EmbeddingStoreCache | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a feature frame for the requested feature set."""

    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unsupported feature set `{feature_set}`. Choose from {FEATURE_SET_CHOICES}.")
    feature_parts: list[pd.DataFrame] = []
    feature_summary: dict[str, Any] = {
        "feature_set": feature_set,
        "metadata_columns": list(metadata_cols or []),
        "embedding_feature_count": 0,
        "metadata_feature_count": 0,
    }
    if feature_set in {"embedding_only", "embedding_plus_metadata"}:
        cache = embedding_cache or EmbeddingStoreCache()
        embedding_array, embedding_columns = cache.load_rows(frame)
        embedding_frame = pd.DataFrame(embedding_array, index=frame.index, columns=embedding_columns)
        feature_parts.append(embedding_frame)
        feature_summary["embedding_feature_count"] = len(embedding_columns)
    if feature_set in {"metadata_only", "embedding_plus_metadata"}:
        metadata_columns = list(metadata_cols or [])
        if not metadata_columns:
            raise ValueError(
                f"`{feature_set}` requires metadata columns, but none were provided or inferred."
            )
        require_columns(frame, metadata_columns, "metadata feature inputs")
        metadata_frame = frame.loc[:, metadata_columns].copy()
        feature_parts.append(metadata_frame)
        feature_summary["metadata_feature_count"] = len(metadata_columns)
    if not feature_parts:
        raise ValueError(f"No feature parts were produced for feature set `{feature_set}`.")
    combined = pd.concat(feature_parts, axis=1)
    if combined.columns.duplicated().any():
        duplicates = combined.columns[combined.columns.duplicated()].tolist()
        raise ValueError(f"Feature construction produced duplicate columns: {duplicates}")
    return combined, feature_summary


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create a simple preprocessor for numeric and categorical features."""

    numeric_columns = feature_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No numeric or categorical features were available after feature construction.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return preprocessor, numeric_columns, categorical_columns


def make_regression_estimator(
    family: str,
    params: Mapping[str, Any],
    preprocessor: ColumnTransformer,
    random_seed: int,
    n_jobs: int,
) -> Pipeline:
    """Create a regression pipeline from a family name and sampled params."""

    if family == "ridge":
        estimator = Ridge(alpha=float(params["alpha"]), random_state=random_seed)
    elif family == "svr":
        estimator = SVR(
            C=float(params["C"]),
            epsilon=float(params["epsilon"]),
            gamma=params["gamma"],
        )
    elif family == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_seed,
            n_jobs=n_jobs,
        )
    elif family == "xgboost":
        if not has_xgboost():
            raise ValueError(f"xgboost requested but unavailable: {XGBOOST_IMPORT_ERROR}")
        estimator = XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unsupported regression family `{family}`")
    return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", estimator)])


def make_classification_estimator(
    family: str,
    params: Mapping[str, Any],
    preprocessor: ColumnTransformer,
    random_seed: int,
    n_jobs: int,
    num_classes: int,
) -> Pipeline:
    """Create a classification pipeline from a family name and sampled params."""

    if family == "logistic_regression":
        estimator = LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=2000,
            multi_class="auto",
            n_jobs=n_jobs,
            random_state=random_seed,
        )
    elif family == "svm":
        estimator = SVC(
            C=float(params["C"]),
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            probability=True,
            random_state=random_seed,
        )
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=random_seed,
            n_jobs=n_jobs,
        )
    elif family == "xgboost":
        if not has_xgboost():
            raise ValueError(f"xgboost requested but unavailable: {XGBOOST_IMPORT_ERROR}")
        estimator = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            num_class=num_classes if num_classes > 2 else None,
            random_state=random_seed,
            n_jobs=n_jobs,
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
        )
    else:
        raise ValueError(f"Unsupported classification family `{family}`")
    return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", estimator)])


def sample_regression_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    """Sample a small focused regression search space."""

    if family == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
    if family == "svr":
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 0.5, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
    raise ValueError(f"Unsupported regression family `{family}`")


def sample_classification_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    """Sample a small focused classification search space."""

    if family == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if family == "svm":
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
    raise ValueError(f"Unsupported classification family `{family}`")


def sanitize_model_families(
    requested_families: Sequence[str],
    allowed_families: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return valid family names and notes about skipped families."""

    allowed = set(allowed_families)
    selected: list[str] = []
    notes: list[str] = []
    for family in requested_families:
        if family not in allowed:
            notes.append(f"Skipped unsupported model family `{family}`.")
            continue
        if family == "xgboost" and not has_xgboost():
            notes.append(f"Skipped `xgboost` because it is unavailable: {XGBOOST_IMPORT_ERROR}")
            continue
        selected.append(family)
    if not selected:
        raise ValueError(f"No usable model families remained after filtering: {list(requested_families)}")
    return selected, notes


def run_optuna_search(
    *,
    families: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv_folds: int,
    optuna_trials: int,
    optuna_timeout: int | None,
    random_seed: int,
    n_jobs: int,
    problem_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Run per-family Optuna tuning and return the winning configuration."""

    if optuna is None:
        raise ImportError(
            "Optuna is required for model-family search but is unavailable in the current environment: "
            f"{OPTUNA_IMPORT_ERROR}"
        )
    if optuna_trials < 1:
        raise ValueError(f"--optuna-trials must be at least 1, received {optuna_trials}")

    splitter = build_group_kfold(groups_train, cv_folds)
    family_results: list[dict[str, Any]] = []
    trials_frames: list[pd.DataFrame] = []

    for family_index, family in enumerate(families):
        sampler = optuna.samplers.TPESampler(seed=random_seed + family_index)
        direction = "minimize" if problem_type == "regression" else "maximize"
        study = optuna.create_study(direction=direction, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            if problem_type == "regression":
                params = sample_regression_params(trial, family)
                estimator = make_regression_estimator(
                    family=family,
                    params=params,
                    preprocessor=preprocessor,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                )
                scores = cross_val_score(
                    estimator,
                    X_train,
                    y_train,
                    groups=groups_train,
                    cv=splitter.split(X_train, y_train, groups_train),
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,
                )
                return float(-scores.mean())

            params = sample_classification_params(trial, family)
            estimator = make_classification_estimator(
                family=family,
                params=params,
                preprocessor=preprocessor,
                random_seed=random_seed,
                n_jobs=n_jobs,
                num_classes=int(pd.Series(y_train).nunique()),
            )
            scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                groups=groups_train,
                cv=splitter.split(X_train, y_train, groups_train),
                scoring="f1_macro",
                n_jobs=1,
            )
            return float(scores.mean())

        study.optimize(objective, n_trials=optuna_trials, timeout=optuna_timeout)
        best_value = float(study.best_value)
        best_params = {key: _json_ready(value) for key, value in study.best_params.items()}
        param_importances: dict[str, float] = {}
        try:
            if len(study.best_params) > 0 and len(study.trials) > 1:
                param_importances = {
                    str(key): float(value)
                    for key, value in optuna.importance.get_param_importances(study).items()
                }
        except Exception:
            param_importances = {}

        result = {
            "family": family,
            "best_cv_score": best_value,
            "best_params": best_params,
            "direction": direction,
            "problem_type": problem_type,
            "best_trial_number": int(study.best_trial.number),
            "trial_count": int(len(study.trials)),
            "complete_trial_count": int(
                sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
            ),
            "param_importances": param_importances,
        }
        family_results.append(result)
        trials_frame = study.trials_dataframe()
        trials_frame["family"] = family
        trials_frames.append(trials_frame)

    if problem_type == "regression":
        best_result = min(family_results, key=lambda item: item["best_cv_score"])
    else:
        best_result = max(family_results, key=lambda item: item["best_cv_score"])
    combined_trials = pd.concat(trials_frames, ignore_index=True) if trials_frames else pd.DataFrame()
    return best_result, family_results, combined_trials


def dump_joblib(path: str | Path, payload: Any) -> Path:
    """Persist a joblib artifact."""

    path_obj = resolve_path(path)
    ensure_directory(path_obj.parent)
    joblib.dump(payload, path_obj)
    return path_obj
