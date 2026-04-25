"""Train the 3-class easy/medium/hard report model.

Why this script exists:
- Convert continuous consensus patch quality into report-friendly ordinal
  classes using train-only quantiles, then fit a simple sklearn classifier.
- Provide the main classification view used for presentation-ready difficulty
  reporting without changing the canonical modeling table.

What it reads:
- `modeling_table.csv.gz` from `01_build_model_table.py`
- Embedding files referenced by the modeling table when the selected feature set
  includes embeddings

What it writes:
- `model.joblib`, `metrics.json`, `best_params.json`,
  `family_search_results.json`, `predictions.csv`, `study_trials.csv`,
  confusion-matrix assets, `validation.json`, `validation.md`, `run.log`, and
  `timing.json`

What validation it performs:
- Required input-column checks and train-only label-threshold derivation checks
- Grouped train/test disjointness checks
- Metrics/model/predictions/confusion-matrix artifact existence checks
- Prediction-table row counts and missingness summaries
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import (
    CLASSIFICATION_FAMILIES,
    DEFAULT_CV_FOLDS,
    DEFAULT_OPTUNA_TIMEOUT,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    FEATURE_SET_CHOICES,
    EmbeddingStoreCache,
    build_feature_frame,
    build_preprocessor,
    classification_metrics,
    dump_joblib,
    file_existence_rows,
    infer_metadata_columns,
    load_table,
    make_classification_estimator,
    maybe_parse_probability_frame,
    missingness_summary,
    now_utc_iso,
    per_class_metrics_frame,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_per_class_metrics,
    plot_prediction_confidence,
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
    """Parse CLI arguments for the report-model stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-table",
        default="outputs/conic_liz/model/model_table/modeling_table.csv.gz",
        help="Canonical modeling table from the prep step.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/conic_liz/model/report_model",
        help="Output directory for the report-model artifacts.",
    )
    parser.add_argument(
        "--target-col",
        default="pq_median",
        help="Continuous target column used to derive easy/medium/hard classes.",
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
        default=["logistic_regression", "svm", "random_forest"],
        help="Classification families to search.",
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
    parser.add_argument(
        "--class-quantiles",
        nargs="*",
        type=float,
        default=[1.0 / 3.0, 2.0 / 3.0],
        help="Quantile cutpoints used on train-only target values.",
    )
    parser.add_argument(
        "--class-labels",
        nargs="*",
        default=["hard", "medium", "easy"],
        help="Labels ordered from lowest target bin to highest target bin.",
    )
    return parser.parse_args()


def validate_quantile_config(class_quantiles: list[float], class_labels: list[str]) -> None:
    """Validate the label and quantile configuration."""

    if sorted(class_quantiles) != class_quantiles:
        raise ValueError(f"--class-quantiles must be sorted ascending, received {class_quantiles}")
    if len(class_labels) != len(class_quantiles) + 1:
        raise ValueError(
            "--class-labels must contain exactly one more label than --class-quantiles. "
            f"Received labels={class_labels} quantiles={class_quantiles}"
        )
    for quantile in class_quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Each class quantile must be between 0 and 1, received {quantile}")


def assign_quantile_labels(
    values: pd.Series,
    thresholds: list[float],
    labels: list[str],
) -> pd.Series:
    """Assign ordinal labels from low target values to high target values."""

    bin_indices = np.digitize(values.to_numpy(dtype=float), bins=np.asarray(thresholds, dtype=float), right=True)
    return pd.Series([labels[index] for index in bin_indices], index=values.index, name="report_label")


def main() -> None:
    """Run the report-model stage."""

    args = parse_args()
    validate_quantile_config(args.class_quantiles, args.class_labels)
    output_dir = resolve_path(args.output_dir)
    logger = setup_stage_logging(output_dir, stage_name="report_model")
    started_time = time.time()

    config = {
        "stage_name": "report_model",
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
        "class_quantiles": list(args.class_quantiles),
        "class_labels_low_to_high": list(args.class_labels),
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
    selected_families, family_notes = sanitize_model_families(args.model_families, CLASSIFICATION_FAMILIES)
    train_frame, test_frame = split_train_test_by_group(
        modeling_frame,
        group_col=args.group_col,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    quantile_thresholds = [
        float(train_frame[args.target_col].quantile(quantile)) for quantile in args.class_quantiles
    ]
    if sorted(quantile_thresholds) != quantile_thresholds or len(set(quantile_thresholds)) != len(quantile_thresholds):
        raise ValueError(
            f"Train-only class thresholds must be strictly increasing, received {quantile_thresholds}."
        )

    train_labels = assign_quantile_labels(train_frame[args.target_col], quantile_thresholds, args.class_labels)
    test_labels = assign_quantile_labels(test_frame[args.target_col], quantile_thresholds, args.class_labels)
    train_frame = train_frame.assign(report_label=train_labels)
    test_frame = test_frame.assign(report_label=test_labels)

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
    y_train = train_frame["report_label"].astype(str)
    y_test = test_frame["report_label"].astype(str)

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
        problem_type="classification",
    )
    best_family = str(best_result["family"])
    best_params = best_result["best_params"]

    estimator = make_classification_estimator(
        family=best_family,
        params=best_params,
        preprocessor=preprocessor,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
        num_classes=len(args.class_labels),
    )
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    metrics = classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        labels=args.class_labels,
        ordered_labels=args.class_labels,
    )

    confusion_array = np.asarray(metrics["confusion_matrix"], dtype=int)
    plots_dir = output_dir / "plots"
    confusion_plot_path = plot_confusion_matrix(
        labels=args.class_labels,
        matrix=confusion_array,
        path=plots_dir / "confusion_matrix.png",
        title=f"Report Model: confusion matrix ({best_family})",
    )
    confusion_csv_path = save_csv_table(
        pd.DataFrame(confusion_array, index=args.class_labels, columns=args.class_labels),
        output_dir / "confusion_matrix.csv",
        index=True,
    )
    per_class_metrics_df = per_class_metrics_frame(metrics["per_class_metrics"], args.class_labels)
    per_class_metrics_csv_path = save_csv_table(
        per_class_metrics_df,
        output_dir / "per_class_metrics.csv",
        index=False,
    )
    per_class_metrics_plot_path = plot_per_class_metrics(
        per_class_metrics=metrics["per_class_metrics"],
        labels=args.class_labels,
        path=plots_dir / "per_class_metrics.png",
        title=f"Report Model: per-class metrics ({best_family})",
    )
    label_distribution_plot_path = plot_label_distribution(
        y_true=y_test,
        y_pred=predictions,
        labels=args.class_labels,
        path=plots_dir / "label_distribution.png",
        title=f"Report Model: label distribution ({best_family})",
    )

    predictions_frame = pd.DataFrame(
        {
            "patch_id": test_frame["patch_id"].values,
            args.group_col: test_frame[args.group_col].values,
            args.target_col: test_frame[args.target_col].values,
            "true_label": y_test.values,
            "predicted_label": predictions,
            "selected_family": best_family,
        }
    )
    probability_frame = maybe_parse_probability_frame(
        estimator=estimator,
        features=X_test,
        class_labels=args.class_labels,
    )
    if not probability_frame.empty:
        predictions_frame = pd.concat(
            [predictions_frame.reset_index(drop=True), probability_frame.reset_index(drop=True)],
            axis=1,
        )
    confidence_plot_path = None
    if not probability_frame.empty and "predicted_probability" in probability_frame.columns:
        confidence_plot_path = plot_prediction_confidence(
            predicted_probability=probability_frame["predicted_probability"],
            correct_mask=(pd.Series(predictions, index=y_test.index) == y_test),
            path=plots_dir / "prediction_confidence.png",
            title=f"Report Model: prediction confidence ({best_family})",
        )

    predictions_path = save_csv_table(predictions_frame, output_dir / "predictions.csv", index=False)
    trials_path = save_csv_table(trials_frame, output_dir / "study_trials.csv", index=False)
    optuna_artifacts = write_optuna_diagnostics(
        output_dir=output_dir,
        family_results=family_results,
        trials_frame=trials_frame,
        stage_name="report_model",
        score_label="CV macro-F1",
    )

    plot_artifacts = {
        "confusion_matrix_png": str(confusion_plot_path),
        "per_class_metrics_png": str(per_class_metrics_plot_path),
        "label_distribution_png": str(label_distribution_plot_path),
        **optuna_artifacts,
    }
    if confidence_plot_path is not None:
        plot_artifacts["prediction_confidence_png"] = str(confidence_plot_path)

    metrics_payload = {
        "stage_name": "report_model",
        "selected_family": best_family,
        "target_col": args.target_col,
        "feature_set": args.feature_set,
        "metadata_cols": metadata_cols,
        "class_quantiles": args.class_quantiles,
        "class_labels_low_to_high": args.class_labels,
        "class_thresholds": quantile_thresholds,
        "test_metrics": metrics,
        "best_cv_score_macro_f1": best_result["best_cv_score"],
        "n_rows_total": int(len(modeling_frame)),
        "n_rows_train": int(len(train_frame)),
        "n_rows_test": int(len(test_frame)),
        "n_groups_train": int(train_frame[args.group_col].nunique()),
        "n_groups_test": int(test_frame[args.group_col].nunique()),
        "dropped_missing_target_rows": int(dropped_missing_target),
        "feature_summary": train_feature_summary,
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "train_class_counts": train_frame["report_label"].value_counts().reindex(args.class_labels).fillna(0).astype(int).to_dict(),
        "test_class_counts": test_frame["report_label"].value_counts().reindex(args.class_labels).fillna(0).astype(int).to_dict(),
        "family_notes": family_notes,
        "plot_artifacts": plot_artifacts,
    }
    metrics_path = write_json(output_dir / "metrics.json", metrics_payload)
    best_params_path = write_json(
        output_dir / "best_params.json",
        {
            "selected_family": best_family,
            "best_cv_score_macro_f1": best_result["best_cv_score"],
            "best_params": best_params,
        },
    )
    family_results_path = write_json(
        output_dir / "family_search_results.json",
        {
            "problem_type": "classification",
            "results": family_results,
        },
    )
    model_path = dump_joblib(
        output_dir / "model.joblib",
        {
            "estimator": estimator,
            "stage_name": "report_model",
            "target_col": args.target_col,
            "feature_set": args.feature_set,
            "metadata_cols": metadata_cols,
            "class_quantiles": args.class_quantiles,
            "class_labels_low_to_high": args.class_labels,
            "class_thresholds": quantile_thresholds,
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
    artifact_map = {
        "model_joblib": model_path,
        "metrics_json": metrics_path,
        "best_params_json": best_params_path,
        "family_search_results_json": family_results_path,
        "predictions_csv": predictions_path,
        "study_trials_csv": trials_path,
        "confusion_matrix_png": confusion_plot_path,
        "confusion_matrix_csv": confusion_csv_path,
        "per_class_metrics_csv": per_class_metrics_csv_path,
        "per_class_metrics_png": per_class_metrics_plot_path,
        "label_distribution_png": label_distribution_plot_path,
        "config_json": output_dir / "config.json",
        **optuna_artifacts,
    }
    if confidence_plot_path is not None:
        artifact_map["prediction_confidence_png"] = confidence_plot_path
    artifact_checks = file_existence_rows(artifact_map)
    report = {
        "stage_name": "report_model",
        "success": True,
        "input_path": str(resolve_path(args.input_table)),
        "output_dir": str(output_dir),
        "summary": [
            {"name": "total_rows_after_target_filter", "value": int(len(modeling_frame))},
            {"name": "train_rows", "value": int(len(train_frame))},
            {"name": "test_rows", "value": int(len(test_frame))},
            {"name": "selected_family", "value": best_family},
            {"name": "best_cv_score_macro_f1", "value": float(best_result["best_cv_score"])},
            {"name": "test_macro_f1", "value": float(metrics["macro_f1"])},
            {"name": "test_balanced_accuracy", "value": float(metrics["balanced_accuracy"])},
            {"name": "test_quadratic_weighted_kappa", "value": float(metrics["quadratic_weighted_kappa"])},
        ],
        "artifact_checks": artifact_checks,
        "missingness_summary": missingness_summary(reloaded_predictions),
        "notes": [
            f"Dropped {dropped_missing_target} rows with missing `{args.target_col}` before splitting.",
            f"Train-only thresholds for `{args.target_col}` were {quantile_thresholds}.",
            f"Labels are ordered low-to-high quality as {args.class_labels}.",
            f"Train/test groups are disjoint under `{args.group_col}`.",
            f"Feature set `{args.feature_set}` used {train_feature_summary['embedding_feature_count']} embedding features and {train_feature_summary['metadata_feature_count']} metadata columns.",
            *family_notes,
        ],
    }
    write_validation_reports(output_dir, report)
    write_timing_json(
        output_dir,
        started_time,
        stage_name="report_model",
        extra={
            "selected_family": best_family,
            "test_macro_f1": float(metrics["macro_f1"]),
            "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        },
    )
    logger.info(
        "finished stage=report_model family=%s macro_f1=%.6f balanced_accuracy=%.6f",
        best_family,
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
    )


if __name__ == "__main__":
    main()
