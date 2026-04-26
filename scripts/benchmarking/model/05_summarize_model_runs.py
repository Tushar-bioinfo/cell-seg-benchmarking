"""Summarize the three modeling stages into compact report tables.

Why this script exists:
- Collect the main, report, and failure-mode stage artifacts into one compact
  summary that is easy to inspect and compare.
- Provide a final integrity gate that checks the expected stage artifacts exist
  before downstream reporting uses them.

What it reads:
- Stage directories produced by `02_train_main_model.py`,
  `03_train_report_model.py`, and `04_train_failure_mode_model.py`
- Each stage's `metrics.json`, `config.json`, `validation.json`, and core
  inspectable artifacts

What it writes:
- `stage_metrics.csv`, `stage_metrics.md`, `report_asset_index.csv`,
  `report_asset_index.md`, `summary.json`, `validation.json`,
  `validation.md`, `run.log`, and `timing.json`

What validation it performs:
- Verifies required upstream stage artifacts exist before writing outputs
- Verifies summary outputs exist after writing
- Records a compact artifact status table and summary counts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from _common import (
    dataframe_markdown,
    file_existence_rows,
    now_utc_iso,
    resolve_path,
    save_csv_table,
    setup_stage_logging,
    write_json,
    write_text,
    write_timing_json,
    write_validation_reports,
)

REQUIRED_STAGE_ARTIFACTS = {
    "metrics_json": "metrics.json",
    "config_json": "config.json",
    "validation_json": "validation.json",
    "predictions_csv": "predictions.csv",
    "model_joblib": "model.joblib",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the summary stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-dir",
        default="outputs/conic_liz/model/main_model",
        help="Stage directory for the continuous regression model.",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/conic_liz/model/report_model",
        help="Stage directory for the easy/medium/hard classifier.",
    )
    parser.add_argument(
        "--failure-dir",
        default="outputs/conic_liz/model/failure_mode",
        help="Stage directory for the hard-patch failure-mode classifier.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/conic_liz/model/summary",
        help="Output directory for the compact summary artifacts.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    """Read a JSON file as a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stage_artifact_table(stage_name: str, stage_dir: Path) -> list[dict]:
    """Return a per-stage artifact existence table."""

    rows = []
    for artifact_label, relative_path in REQUIRED_STAGE_ARTIFACTS.items():
        path = stage_dir / relative_path
        rows.append(
            {
                "stage_name": stage_name,
                "artifact_label": artifact_label,
                "path": str(path),
                "exists": path.exists(),
            }
        )
    return rows


def asset_kind(path: Path) -> str:
    """Infer a compact asset type label from a file path."""

    suffixes = path.suffixes
    if path.suffix == ".png":
        return "plot"
    if suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".csv":
        return "table"
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".md":
        return "markdown"
    if path.suffix == ".joblib":
        return "model"
    if path.suffix == ".log":
        return "log"
    return "other"


def collect_stage_assets(stage_name: str, stage_dir: Path) -> list[dict]:
    """Recursively collect stage assets for the summary index."""

    required_paths = set(REQUIRED_STAGE_ARTIFACTS.values())
    rows: list[dict] = []
    for path in sorted(stage_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = str(path.relative_to(stage_dir))
        rows.append(
            {
                "stage_name": stage_name,
                "relative_path": relative_path,
                "artifact_kind": asset_kind(path),
                "required": relative_path in required_paths,
                "path": str(path),
                "exists": True,
            }
        )
    return rows


def build_stage_metric_row(stage_name: str, stage_dir: Path) -> dict:
    """Flatten one stage's metrics/config into a compact summary row."""

    metrics = read_json(stage_dir / "metrics.json")
    config = read_json(stage_dir / "config.json")
    common = {
        "stage_name": stage_name,
        "stage_dir": str(stage_dir),
        "selected_family": metrics.get("selected_family"),
        "feature_set": metrics.get("feature_set", config.get("feature_set")),
        "target_col": metrics.get("target_col", config.get("target_col")),
        "group_col": config.get("group_col"),
        "test_size": config.get("test_size"),
        "cv_folds": config.get("cv_folds"),
    }
    if stage_name == "main_model":
        test_metrics = metrics["test_metrics"]
        common.update(
            {
                "primary_metric_name": "mae",
                "primary_metric_value": test_metrics["mae"],
                "secondary_metric_1_name": "rmse",
                "secondary_metric_1_value": test_metrics["rmse"],
                "secondary_metric_2_name": "r2",
                "secondary_metric_2_value": test_metrics["r2"],
                "secondary_metric_3_name": "pearson",
                "secondary_metric_3_value": test_metrics["pearson"],
                "secondary_metric_4_name": "spearman",
                "secondary_metric_4_value": test_metrics["spearman"],
            }
        )
    elif stage_name == "report_model":
        test_metrics = metrics["test_metrics"]
        common.update(
            {
                "primary_metric_name": "macro_f1",
                "primary_metric_value": test_metrics["macro_f1"],
                "secondary_metric_1_name": "balanced_accuracy",
                "secondary_metric_1_value": test_metrics["balanced_accuracy"],
                "secondary_metric_2_name": "quadratic_weighted_kappa",
                "secondary_metric_2_value": test_metrics["quadratic_weighted_kappa"],
                "secondary_metric_3_name": "class_thresholds",
                "secondary_metric_3_value": ",".join(str(value) for value in metrics["class_thresholds"]),
                "secondary_metric_4_name": "class_labels_low_to_high",
                "secondary_metric_4_value": ",".join(metrics["class_labels_low_to_high"]),
            }
        )
    elif stage_name == "failure_mode":
        test_metrics = metrics["test_metrics"]
        common.update(
            {
                "primary_metric_name": "macro_f1",
                "primary_metric_value": test_metrics["macro_f1"],
                "secondary_metric_1_name": "balanced_accuracy",
                "secondary_metric_1_value": test_metrics["balanced_accuracy"],
                "secondary_metric_2_name": "hard_threshold",
                "secondary_metric_2_value": metrics["hard_threshold"],
                "secondary_metric_3_name": "retained_classes",
                "secondary_metric_3_value": ",".join(metrics["retained_classes"]),
                "secondary_metric_4_name": "dominance_margin",
                "secondary_metric_4_value": metrics["dominance_margin"],
            }
        )
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported stage name `{stage_name}`")
    return common


def main() -> None:
    """Run the summary stage."""

    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    logger = setup_stage_logging(output_dir, stage_name="summarize_model_runs")
    started_time = time.time()

    stage_dirs = {
        "main_model": resolve_path(args.main_dir),
        "report_model": resolve_path(args.report_dir),
        "failure_mode": resolve_path(args.failure_dir),
    }

    required_artifact_rows: list[dict] = []
    discovered_asset_rows: list[dict] = []
    available_stage_dirs: dict[str, Path] = {}
    skipped_stage_rows: list[dict] = []
    for stage_name, stage_dir in stage_dirs.items():
        if not stage_dir.exists():
            skipped_stage_rows.append(
                {
                    "stage_name": stage_name,
                    "reason": "stage_dir_missing",
                    "path": str(stage_dir),
                }
            )
            continue
        stage_rows = stage_artifact_table(stage_name, stage_dir)
        missing_rows = [row for row in stage_rows if not row["exists"]]
        if missing_rows:
            raise FileNotFoundError(
                f"Stage `{stage_name}` is missing required artifacts: {missing_rows}"
            )
        available_stage_dirs[stage_name] = stage_dir
        required_artifact_rows.extend(stage_rows)
        discovered_asset_rows.extend(collect_stage_assets(stage_name, stage_dir))
    if not available_stage_dirs:
        raise FileNotFoundError(
            "No stage directories were available for summary. "
            f"Requested stage dirs: { {name: str(path) for name, path in stage_dirs.items()} }"
        )

    stage_metric_rows = [
        build_stage_metric_row(stage_name=stage_name, stage_dir=stage_dir)
        for stage_name, stage_dir in available_stage_dirs.items()
    ]
    stage_metrics_frame = pd.DataFrame(stage_metric_rows)
    asset_index_frame = pd.DataFrame(discovered_asset_rows)

    stage_metrics_csv = save_csv_table(stage_metrics_frame, output_dir / "stage_metrics.csv", index=False)
    asset_index_csv = save_csv_table(asset_index_frame, output_dir / "report_asset_index.csv", index=False)
    stage_metrics_md = write_text(
        output_dir / "stage_metrics.md",
        "# Stage Metrics\n\n" + dataframe_markdown(stage_metric_rows),
    )
    asset_index_md = write_text(
        output_dir / "report_asset_index.md",
        "# Report Asset Index\n\n" + dataframe_markdown(discovered_asset_rows),
    )
    summary_json = write_json(
        output_dir / "summary.json",
        {
            "generated_at_utc": now_utc_iso(),
            "requested_stage_dirs": {name: str(path) for name, path in stage_dirs.items()},
            "available_stage_dirs": {name: str(path) for name, path in available_stage_dirs.items()},
            "n_stages": len(available_stage_dirs),
            "n_required_stage_artifacts": len(required_artifact_rows),
            "n_discovered_assets": len(discovered_asset_rows),
            "stage_metric_rows": stage_metric_rows,
            "required_artifact_rows": required_artifact_rows,
            "artifact_rows": discovered_asset_rows,
            "skipped_stage_rows": skipped_stage_rows,
        },
    )

    output_artifact_checks = file_existence_rows(
        {
            "stage_metrics_csv": stage_metrics_csv,
            "stage_metrics_md": stage_metrics_md,
            "report_asset_index_csv": asset_index_csv,
            "report_asset_index_md": asset_index_md,
            "summary_json": summary_json,
            "config_json": output_dir / "config.json",
        }
    )
    write_json(
        output_dir / "config.json",
        {
            "stage_name": "summarize_model_runs",
            "output_dir": str(output_dir),
            "requested_stage_dirs": {name: str(path) for name, path in stage_dirs.items()},
            "available_stage_dirs": {name: str(path) for name, path in available_stage_dirs.items()},
            "generated_at_utc": now_utc_iso(),
        },
    )
    output_artifact_checks = file_existence_rows(
        {
            "stage_metrics_csv": stage_metrics_csv,
            "stage_metrics_md": stage_metrics_md,
            "report_asset_index_csv": asset_index_csv,
            "report_asset_index_md": asset_index_md,
            "summary_json": summary_json,
            "config_json": output_dir / "config.json",
        }
    )

    report = {
        "stage_name": "summarize_model_runs",
        "success": True,
        "output_dir": str(output_dir),
        "summary": [
            {"name": "stage_count", "value": int(len(available_stage_dirs))},
            {"name": "stage_metric_rows", "value": int(len(stage_metrics_frame))},
            {"name": "report_asset_rows", "value": int(len(asset_index_frame))},
            {"name": "required_artifact_rows", "value": int(len(required_artifact_rows))},
            {"name": "skipped_stage_rows", "value": int(len(skipped_stage_rows))},
        ],
        "artifact_checks": output_artifact_checks,
        "notes": [
            "Verified required artifacts for each available stage before writing summary outputs.",
            "Wrote compact CSV/Markdown comparison tables and a recursive report asset index, including plot files.",
            *[
                f"Skipped stage `{row['stage_name']}` because `{row['path']}` was missing."
                for row in skipped_stage_rows
            ],
        ],
    }
    write_validation_reports(output_dir, report)
    write_timing_json(
        output_dir,
        started_time,
        stage_name="summarize_model_runs",
        extra={
            "stage_count": int(len(available_stage_dirs)),
            "report_asset_rows": int(len(asset_index_frame)),
            "required_artifact_rows": int(len(required_artifact_rows)),
            "skipped_stage_rows": int(len(skipped_stage_rows)),
        },
    )
    logger.info(
        "finished summary stage_count=%s asset_rows=%s output_dir=%s",
        len(available_stage_dirs),
        len(asset_index_frame),
        output_dir,
    )


if __name__ == "__main__":
    main()
