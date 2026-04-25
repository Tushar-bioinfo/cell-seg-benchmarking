"""Train the continuous patch-quality regression model.

Why this script exists:
- Fit the main regression baseline for consensus patch quality using the shared
  canonical modeling table.
- Provide a simple sklearn-first reference model for predicting continuous
  patch difficulty before any label discretization.

What it reads:
- `modeling_table.csv.gz` from `01_build_model_table.py`
- Embedding files referenced by the modeling table when the selected feature set
  includes embeddings

What it writes:
- `model.joblib`, `metrics.json`, `best_params.json`,
  `family_search_results.json`, `predictions.csv`, `study_trials.csv`,
  plots, `validation.json`, `validation.md`, `run.log`, and `timing.json`

What validation it performs:
- Required input-column checks and missing-target filtering
- Grouped train/test disjointness checks
- Output artifact existence checks for metrics, predictions, model, and plots
- Output row-count checks plus missingness summaries for the prediction table
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from _common import (
    DEFAULT_CV_FOLDS,
    DEFAULT_OPTUNA_TIMEOUT,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    FEATURE_SET_CHOICES,
    REGRESSION_FAMILIES,
    EmbeddingStoreCache,
    build_feature_frame,
    build_preprocessor,
    dump_joblib,
    file_existence_rows,
    infer_metadata_columns,
    load_table,
    make_regression_estimator,
    missingness_summary,
    now_utc_iso,
    plot_predicted_vs_observed,
    plot_residual_histogram,
    plot_residuals,
    regression_metrics,
    require_columns,
    resolve_path,
    run_optuna_search,
    sanitize_model_families,
    save_csv_table,
    setup_stage_logging,
    split_train_test_by_group,
    write_json,
    write_optuna_diagnostics,
    write_timing_json,
    write_validation_reports,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the main-model stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-table",
        default="outputs/conic_liz/model/model_table/modeling_table.csv.gz",
        help="Canonical modeling table from the prep step.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/conic_liz/model/main_model",
        help="Output directory for the regression stage artifacts.",
    )
    parser.add_argument(
        "--target-col",
        default="pq_median",
        help="Continuous consensus target to regress.",
    )
    parser.add_argument(
        "--feature-set",
        default="embedding_only",
        choices=FEATURE_SET_CHOICES,
        help="Feature family to use for training.",
    )
    parser.add_argument(
        "--metadata-cols",
        nargs="*",
        default=None,
        help="Optional explicit metadata columns. If omitted, columns are inferred.",
    )
    parser.add_argument(
        "--model-families",
        nargs="*",
        default=["ridge", "svr", "random_forest"],
        help="Regression families to search.",
    )
    parser.add_argument(
        "--group-col",
        default="slide_id",
        help="Grouping column used for train/test splitting and CV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of groups assigned to the held-out test split.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Maximum grouped CV folds inside train.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=DEFAULT_OPTUNA_TRIALS,
        help="Trials per model family.",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=DEFAULT_OPTUNA_TIMEOUT,
        help="Timeout in seconds per model family. Use 0 to disable the timeout.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers forwarded to sklearn/xgboost estimators.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Fixed random seed for split and model reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the main regression stage."""

    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    logger = setup_stage_logging(output_dir, stage_name="main_model")
    started_time = time.time()

    config = {
        "stage_name": "main_model",
        "input_table": str(resolve_path(args.input_table)),
        "output_dir": str(output_dir),
        "target_col": args.target_col,
        "feature_set": args.feature_set,
        "metadata_cols": list(args.metadata_cols or []),
        "requested_model_families": list(args.model_families),
        "group_col": args.group_col,
        "test_size": args.test_size,
        "cv_folds": args.cv_folds,
        "optuna_trials": args.optuna_trials,
        "optuna_timeout": args.optuna_timeout,
        "n_jobs": args.n_jobs,
        "random_seed": args.random_seed,
        "generated_at_utc": now_utc_iso(),
    }
    write_json(output_dir / "config.json", config)

    modeling_frame = load_table(args.input_table).copy()
    require_columns(modeling_frame, ["patch_id", args.group_col, args.target_col], "modeling table")

    initial_row_count = len(modeling_frame)
    modeling_frame = modeling_frame.loc[modeling_frame[args.target_col].notna()].copy()
    dropped_missing_target = initial_row_count - len(modeling_frame)
    if modeling_frame.empty:
        raise ValueError(
            f"No rows remained after dropping missing `{args.target_col}` values from {args.input_table}."
        )

    metadata_cols = infer_metadata_columns(modeling_frame, args.metadata_cols)
    selected_families, family_notes = sanitize_model_families(args.model_families, REGRESSION_FAMILIES)
    logger.info(
        "rows=%s dropped_missing_target=%s feature_set=%s metadata_cols=%s model_families=%s",
        len(modeling_frame),
        dropped_missing_target,
        args.feature_set,
        len(metadata_cols),
        selected_families,
    )

    train_frame, test_frame = split_train_test_by_group(
        modeling_frame,
        group_col=args.group_col,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    embedding_cache = EmbeddingStoreCache() if args.feature_set != "metadata_only" else None
    X_train, train_feature_summary = build_feature_frame(
        train_frame,
        feature_set=args.feature_set,
        metadata_cols=metadata_cols,
        embedding_cache=embedding_cache,
    )
    X_test, _ = build_feature_frame(
        test_frame,
        feature_set=args.feature_set,
        metadata_cols=metadata_cols,
        embedding_cache=embedding_cache,
    )
    y_train = train_frame[args.target_col].astype(float)
    y_test = test_frame[args.target_col].astype(float)

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    timeout = None if args.optuna_timeout <= 0 else args.optuna_timeout
    best_result, family_results, trials_frame = run_optuna_search(
        families=selected_families,
        X_train=X_train,
        y_train=y_train,
        groups_train=train_frame[args.group_col],
        preprocessor=preprocessor,
        cv_folds=args.cv_folds,
        optuna_trials=args.optuna_trials,
        optuna_timeout=timeout,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
        problem_type="regression",
    )
    best_family = str(best_result["family"])
    best_params = best_result["best_params"]

    estimator = make_regression_estimator(
        family=best_family,
        params=best_params,
        preprocessor=preprocessor,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    metrics = regression_metrics(y_test, predictions)

    plots_dir = output_dir / "plots"
    predicted_vs_observed_path = plot_predicted_vs_observed(
        y_true=y_test,
        y_pred=predictions,
        path=plots_dir / "predicted_vs_observed.png",
        title=f"Main Model: predicted vs observed ({best_family})",
    )
    residual_plot_path = plot_residuals(
        y_true=y_test,
        y_pred=predictions,
        path=plots_dir / "residuals.png",
        title=f"Main Model: residuals ({best_family})",
    )
    residual_histogram_path = plot_residual_histogram(
        y_true=y_test,
        y_pred=predictions,
        path=plots_dir / "residual_histogram.png",
        title=f"Main Model: residual histogram ({best_family})",
    )

    predictions_frame = pd.DataFrame(
        {
            "patch_id": test_frame["patch_id"].values,
            args.group_col: test_frame[args.group_col].values,
            args.target_col: y_test.values,
            "predicted_value": predictions,
            "residual": y_test.values - predictions,
            "selected_family": best_family,
        }
    )
    predictions_path = save_csv_table(predictions_frame, output_dir / "predictions.csv", index=False)
    trials_path = save_csv_table(trials_frame, output_dir / "study_trials.csv", index=False)
    optuna_artifacts = write_optuna_diagnostics(
        output_dir=output_dir,
        family_results=family_results,
        trials_frame=trials_frame,
        stage_name="main_model",
        score_label="CV MAE",
    )

    metrics_payload = {
        "stage_name": "main_model",
        "selected_family": best_family,
        "target_col": args.target_col,
        "feature_set": args.feature_set,
        "metadata_cols": metadata_cols,
        "test_metrics": metrics,
        "best_cv_score_mae": best_result["best_cv_score"],
        "n_rows_total": int(len(modeling_frame)),
        "n_rows_train": int(len(train_frame)),
        "n_rows_test": int(len(test_frame)),
        "n_groups_train": int(train_frame[args.group_col].nunique()),
        "n_groups_test": int(test_frame[args.group_col].nunique()),
        "dropped_missing_target_rows": int(dropped_missing_target),
        "feature_summary": train_feature_summary,
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "family_notes": family_notes,
        "plot_artifacts": {
            "predicted_vs_observed_png": str(predicted_vs_observed_path),
            "residuals_png": str(residual_plot_path),
            "residual_histogram_png": str(residual_histogram_path),
            **optuna_artifacts,
        },
    }
    metrics_path = write_json(output_dir / "metrics.json", metrics_payload)
    best_params_path = write_json(
        output_dir / "best_params.json",
        {
            "selected_family": best_family,
            "best_cv_score_mae": best_result["best_cv_score"],
            "best_params": best_params,
        },
    )
    family_results_path = write_json(
        output_dir / "family_search_results.json",
        {
            "problem_type": "regression",
            "results": family_results,
        },
    )
    model_path = dump_joblib(
        output_dir / "model.joblib",
        {
            "estimator": estimator,
            "stage_name": "main_model",
            "target_col": args.target_col,
            "feature_set": args.feature_set,
            "metadata_cols": metadata_cols,
            "selected_family": best_family,
            "best_params": best_params,
            "group_col": args.group_col,
            "random_seed": args.random_seed,
        },
    )

    reloaded_predictions = load_table(predictions_path)
    if len(reloaded_predictions) != len(test_frame):
        raise AssertionError(
            f"Predictions row count {len(reloaded_predictions)} did not match held-out row count {len(test_frame)}."
        )
    artifact_checks = file_existence_rows(
        {
            "model_joblib": model_path,
            "metrics_json": metrics_path,
            "best_params_json": best_params_path,
            "family_search_results_json": family_results_path,
            "predictions_csv": predictions_path,
            "study_trials_csv": trials_path,
            "predicted_vs_observed_plot": predicted_vs_observed_path,
            "residual_plot": residual_plot_path,
            "residual_histogram_plot": residual_histogram_path,
            "config_json": output_dir / "config.json",
            **optuna_artifacts,
        }
    )
    report = {
        "stage_name": "main_model",
        "success": True,
        "input_path": str(resolve_path(args.input_table)),
        "output_dir": str(output_dir),
        "summary": [
            {"name": "total_rows_after_target_filter", "value": int(len(modeling_frame))},
            {"name": "train_rows", "value": int(len(train_frame))},
            {"name": "test_rows", "value": int(len(test_frame))},
            {"name": "train_groups", "value": int(train_frame[args.group_col].nunique())},
            {"name": "test_groups", "value": int(test_frame[args.group_col].nunique())},
            {"name": "selected_family", "value": best_family},
            {"name": "best_cv_score_mae", "value": float(best_result["best_cv_score"])},
            {"name": "test_mae", "value": float(metrics["mae"])},
            {"name": "test_rmse", "value": float(metrics["rmse"])},
            {"name": "test_r2", "value": float(metrics["r2"])},
        ],
        "artifact_checks": artifact_checks,
        "missingness_summary": missingness_summary(reloaded_predictions),
        "notes": [
            f"Dropped {dropped_missing_target} rows with missing `{args.target_col}` before splitting.",
            f"Train/test groups are disjoint under `{args.group_col}`.",
            f"Feature set `{args.feature_set}` used {train_feature_summary['embedding_feature_count']} embedding features and {train_feature_summary['metadata_feature_count']} metadata columns.",
            *family_notes,
        ],
    }
    write_validation_reports(output_dir, report)
    write_timing_json(
        output_dir,
        started_time,
        stage_name="main_model",
        extra={
            "selected_family": best_family,
            "test_mae": float(metrics["mae"]),
            "test_rmse": float(metrics["rmse"]),
            "test_r2": float(metrics["r2"]),
        },
    )
    logger.info(
        "finished stage=main_model family=%s mae=%.6f rmse=%.6f r2=%.6f",
        best_family,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
    )


if __name__ == "__main__":
    main()
