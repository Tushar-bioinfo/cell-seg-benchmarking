#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIR = Path("outputs/conic_liz/failure_prediction/quality_probe")
DEFAULT_OUTPUT_PATH = Path("outputs/conic_liz/failure_prediction/quality_probe_summary.csv")
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_METRICS_NAME = "metrics.json"
DEFAULT_PREDICTIONS_NAME = "predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Summarize one or more quality-probe artifact directories and optional inference prediction tables "
            "into a single flat report."
        ),
    )
    parser.add_argument(
        "--artifact-dirs",
        nargs="*",
        type=Path,
        default=[DEFAULT_ARTIFACT_DIR],
        help="Artifact directories produced by 03_train_quality_probe.py.",
    )
    parser.add_argument(
        "--artifact-dir-glob",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Optional glob expressions whose matches are added to --artifact-dirs, "
            "for example outputs/conic_liz/failure_prediction/quality_probe*."
        ),
    )
    parser.add_argument(
        "--inference-tables",
        nargs="*",
        type=Path,
        default=[],
        help="Optional inference prediction tables produced by 04_infer_quality_probe.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Summary output (.csv or .parquet).",
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
        help="Filename of the saved config JSON inside each artifact directory.",
    )
    parser.add_argument(
        "--metrics-name",
        default=DEFAULT_METRICS_NAME,
        help="Filename of the saved metrics JSON inside each artifact directory.",
    )
    parser.add_argument(
        "--predictions-name",
        default=DEFAULT_PREDICTIONS_NAME,
        help="Filename of the saved training predictions CSV inside each artifact directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("summarize_quality_probe")
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


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to load {label} JSON from {path}: {type(exc).__name__}: {exc}") from exc


def flatten_mapping(
    mapping: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        flat_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, prefix=f"{flat_key}."))
        elif isinstance(value, list):
            flattened[flat_key] = json.dumps(value)
        else:
            flattened[flat_key] = value
    return flattened


def resolve_artifact_dirs(
    *,
    artifact_dirs: list[Path],
    artifact_dir_globs: list[Path],
    repo_root: Path | None,
) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for artifact_dir in artifact_dirs:
        path = resolve_cli_path(artifact_dir, repo_root=repo_root)
        if path not in seen:
            seen.add(path)
            resolved.append(path)

    for artifact_glob in artifact_dir_globs:
        pattern = artifact_glob.expanduser()
        if pattern.is_absolute():
            matches = sorted(Path("/").glob(str(pattern)[1:]))
        else:
            root = repo_root if repo_root is not None else Path.cwd().resolve()
            matches = sorted(root.glob(str(pattern)))
        for match in matches:
            if match not in seen:
                seen.add(match)
                resolved.append(match)

    return resolved


def summarize_predictions_table(predictions: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "predictions.n_rows": int(len(predictions)),
    }

    if "split" in predictions.columns:
        split_counts = predictions["split"].value_counts(dropna=False).to_dict()
        for split_name, count in split_counts.items():
            summary[f"predictions.split_count.{split_name}"] = int(count)

    if "target" in predictions.columns and "prediction" in predictions.columns:
        valid_mask = predictions["target"].notna() & predictions["prediction"].notna()
        if valid_mask.any():
            summary["predictions.target_prediction_corr"] = float(
                predictions.loc[valid_mask, ["target", "prediction"]].corr().iloc[0, 1]
            )

    if "prediction_score" in predictions.columns:
        score_series = pd.to_numeric(predictions["prediction_score"], errors="coerce")
        valid_scores = score_series.dropna()
        if not valid_scores.empty:
            summary["predictions.prediction_score_mean"] = float(valid_scores.mean())
            summary["predictions.prediction_score_std"] = float(valid_scores.std(ddof=0))

    return summary


def summarize_artifact_dir(
    artifact_dir: Path,
    *,
    config_name: str,
    metrics_name: str,
    predictions_name: str,
) -> dict[str, Any]:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
    if not artifact_dir.is_dir():
        raise NotADirectoryError(f"Artifact path is not a directory: {artifact_dir}")

    config_path = artifact_dir / config_name
    metrics_path = artifact_dir / metrics_name
    predictions_path = artifact_dir / predictions_name
    for path, label in (
        (config_path, "config"),
        (metrics_path, "metrics"),
        (predictions_path, "predictions"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")

    config = load_json(config_path, label="config")
    metrics = load_json(metrics_path, label="metrics")
    predictions = pd.read_csv(predictions_path)

    summary: dict[str, Any] = {
        "summary_type": "artifact",
        "artifact_dir": str(artifact_dir),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
    summary.update({f"config.{key}": value for key, value in flatten_mapping(config).items()})
    summary.update({f"metrics.{key}": value for key, value in flatten_mapping(metrics).items()})
    summary.update(summarize_predictions_table(predictions))
    return summary


def summarize_inference_table(path: Path) -> dict[str, Any]:
    table = read_table(path)
    summary: dict[str, Any] = {
        "summary_type": "inference",
        "inference_table": str(path),
        "inference.n_rows": int(len(table)),
    }

    numeric_columns = [
        column
        for column in (
            "predicted_quality_score",
            "predicted_quality_probability",
            "predicted_failure_probability",
        )
        if column in table.columns
    ]
    for column in numeric_columns:
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[f"inference.{column}.mean"] = float(values.mean())
        summary[f"inference.{column}.std"] = float(values.std(ddof=0))
        summary[f"inference.{column}.min"] = float(values.min())
        summary[f"inference.{column}.max"] = float(values.max())

    if "predicted_quality_label" in table.columns:
        label_counts = table["predicted_quality_label"].value_counts(dropna=False).to_dict()
        for label, count in label_counts.items():
            summary[f"inference.predicted_quality_label_count.{label}"] = int(count)

    return summary


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.verbose)

    repo_root = resolve_repo_root(args.repo_root)
    output_path = resolve_cli_path(args.output, repo_root=repo_root)
    artifact_dirs = resolve_artifact_dirs(
        artifact_dirs=args.artifact_dirs,
        artifact_dir_globs=args.artifact_dir_glob,
        repo_root=repo_root,
    )

    if not artifact_dirs and not args.inference_tables:
        raise ValueError("Provide at least one artifact directory or inference table to summarize.")

    summary_rows: list[dict[str, Any]] = []

    for artifact_dir in artifact_dirs:
        logger.info("Summarizing artifact directory: %s", artifact_dir)
        summary_rows.append(
            summarize_artifact_dir(
                artifact_dir,
                config_name=args.config_name,
                metrics_name=args.metrics_name,
                predictions_name=args.predictions_name,
            )
        )

    for inference_table in args.inference_tables:
        resolved_table = resolve_cli_path(inference_table, repo_root=repo_root)
        logger.info("Summarizing inference table: %s", resolved_table)
        if not resolved_table.exists():
            raise FileNotFoundError(f"Inference table does not exist: {resolved_table}")
        summary_rows.append(summarize_inference_table(resolved_table))

    if not summary_rows:
        raise RuntimeError("No summary rows were produced.")

    summary_df = pd.DataFrame(summary_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing summary report: %s", output_path)
    write_table(summary_df, output_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
