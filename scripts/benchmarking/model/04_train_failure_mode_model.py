"""Train the hard-patch failure-mode classifier.

Why this script exists:
- Focus on the hardest patches only, then learn which dominant failure pattern
  is most likely for those difficult regions.
- Separate "how hard is this patch?" from "what kind of failure does it show?"
  so the workflow can support both ranking and failure-type analysis.

What it reads:
- `modeling_table.csv.gz` from `01_build_model_table.py`
- Embedding files referenced by the modeling table when the selected feature set
  includes embeddings

What it writes:
- `model.joblib`, `metrics.json`, `best_params.json`,
  `family_search_results.json`, `predictions.csv`, `hard_patch_labels.csv`,
  `study_trials.csv`, confusion-matrix assets, `validation.json`,
  `validation.md`, `run.log`, and `timing.json`

What validation it performs:
- Required input-column checks for hard-patch gating and failure-mode metrics
- Train-only hard-threshold derivation checks and group disjointness checks
- Metrics/model/predictions/confusion-matrix artifact existence checks
- Prediction-table and hard-label-table row-count plus missingness checks
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
    FAILURE_MODE_METRIC_COLUMNS,
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

FAILURE_LABEL_MAP = {
    "rq_median": "rq_deficit",
    "sq_median": "sq_deficit",
    "pixel_precision_median": "pixel_precision_deficit",
    "pixel_recall_median": "pixel_recall_deficit",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the failure-mode stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-table",
        default="outputs/conic_liz/model/model_table/modeling_table.csv.gz",
        help="Canonical modeling table from the prep step.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/conic_liz/model/failure_mode",
        help="Output directory for the failure-mode artifacts.",
    )
    parser.add_argument(
        "--target-col",
        default="pq_median",
        help="Continuous target column used to define hard patches.",
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
        "--hard-quantile",
        type=float,
        default=1.0 / 3.0,
        help="Train-only quantile threshold below which patches are considered hard.",
    )
    parser.add_argument(
        "--dominance-margin",
        type=float,
        default=0.05,
        help="Minimum gap between the top two deficits required for a dominant failure label.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=25,
        help="Minimum train rows required to retain a failure-mode class.",
    )
    return parser.parse_args()


def derive_failure_mode_labels(frame: pd.DataFrame, dominance_margin: float) -> pd.DataFrame:
    """Derive dominant failure-mode labels from consensus deficits."""

    require_columns(frame, list(FAILURE_MODE_METRIC_COLUMNS), "failure-mode inputs")
    metric_frame = frame.loc[:, FAILURE_MODE_METRIC_COLUMNS].astype(float)
    deficit_frame = 1.0 - metric_frame
    deficit_frame = deficit_frame.rename(columns=FAILURE_LABEL_MAP)

    missing_metric_mask = metric_frame.isna().any(axis=1)
    deficit_values = deficit_frame.to_numpy(dtype=float)
    top_indices = deficit_values.argmax(axis=1)
    sorted_deficits = np.sort(deficit_values, axis=1)
    top_values = sorted_deficits[:, -1]
    second_values = sorted_deficits[:, -2] if deficit_values.shape[1] > 1 else np.zeros(len(frame))
    dominance_gap = top_values - second_values

    label_columns = list(deficit_frame.columns)
    dominant_labels = [label_columns[index] for index in top_indices]
    label_series = pd.Series(dominant_labels, index=frame.index, name="failure_mode_label")
    label_series = label_series.where(~missing_metric_mask & (dominance_gap >= dominance_margin))

    result = frame.copy()
    result["failure_mode_label"] = label_series
    result["dominance_gap"] = dominance_gap
    result["max_deficit"] = top_values
    result["second_max_deficit"] = second_values
    result["missing_failure_metrics"] = missing_metric_mask
    for column in deficit_frame.columns:
        result[column] = deficit_frame[column]
    return result


def main() -> None:
    """Run the failure-mode stage."""

    args = parse_args()
    if not 0.0 < args.hard_quantile < 1.0:
        raise ValueError(f"--hard-quantile must be between 0 and 1, received {args.hard_quantile}")
    if args.dominance_margin < 0.0:
        raise ValueError(
            f"--dominance-margin must be non-negative, received {args.dominance_margin}"
        )
    if args.min_class_count < 1:
        raise ValueError(f"--min-class-count must be at least 1, received {args.min_class_count}")

    output_dir = resolve_path(args.output_dir)
    logger = setup_stage_logging(output_dir, stage_name="failure_mode")
    started_time = time.time()

    config = {
        "stage_name": "failure_mode",
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
        "hard_quantile": args.hard_quantile,
        "dominance_margin": args.dominance_margin,
        "min_class_count": args.min_class_count,
        "generated_at_utc": now_utc_iso(),
    }
    write_json(output_dir / "config.json", config)

    modeling_frame = load_table(args.input_table).copy()
    require_columns(
        modeling_frame,
        ["patch_id", args.group_col, args.target_col, *FAILURE_MODE_METRIC_COLUMNS],
        "modeling table",
    )
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

    hard_threshold = float(train_frame[args.target_col].quantile(args.hard_quantile))
    train_hard = train_frame.loc[train_frame[args.target_col] <= hard_threshold].copy()
    test_hard = test_frame.loc[test_frame[args.target_col] <= hard_threshold].copy()
    if train_hard.empty:
        raise ValueError("No hard patches were found in the training split.")
    if test_hard.empty:
        raise ValueError("No hard patches were found in the held-out test split.")

    train_hard = derive_failure_mode_labels(train_hard, dominance_margin=args.dominance_margin)
    test_hard = derive_failure_mode_labels(test_hard, dominance_margin=args.dominance_margin)
    train_hard["split_name"] = "train"
    test_hard["split_name"] = "test"
    hard_label_table = pd.concat([train_hard, test_hard], ignore_index=True)

    train_labelled = train_hard.loc[train_hard["failure_mode_label"].notna()].copy()
    test_labelled = test_hard.loc[test_hard["failure_mode_label"].notna()].copy()
    if train_labelled.empty:
        raise ValueError(
            "No train hard patches received a dominant failure label. Relax --dominance-margin or inspect the inputs."
        )

    retained_class_counts = train_labelled["failure_mode_label"].value_counts()
    retained_classes = retained_class_counts.loc[retained_class_counts >= args.min_class_count].index.tolist()
    if len(retained_classes) < 2:
        raise ValueError(
            "Fewer than 2 failure-mode classes met --min-class-count in the training split. "
            f"Counts: {retained_class_counts.to_dict()}"
        )

    train_labelled = train_labelled.loc[train_labelled["failure_mode_label"].isin(retained_classes)].copy()
    test_labelled = test_labelled.loc[test_labelled["failure_mode_label"].isin(retained_classes)].copy()
    if train_labelled.empty:
        raise ValueError("No train rows remained after applying the retained-class filter.")
    if test_labelled.empty:
        raise ValueError("No test rows remained after applying the retained-class filter.")

    hard_label_table["retained_for_training"] = hard_label_table["failure_mode_label"].isin(retained_classes)
    hard_label_table["is_hard_patch"] = True
    hard_label_table["hard_threshold"] = hard_threshold
    hard_label_table_path = save_csv_table(
        hard_label_table,
        output_dir / "hard_patch_labels.csv",
        index=False,
    )

    embedding_cache = EmbeddingStoreCache() if args.feature_set != "metadata_only" else None
    X_train, train_feature_summary = build_feature_frame(
        train_labelled,
        feature_set=args.feature_set,
        metadata_cols=metadata_cols,
        embedding_cache=embedding_cache,
    )
    X_test, _ = build_feature_frame(
        test_labelled,
        feature_set=args.feature_set,
        metadata_cols=metadata_cols,
        embedding_cache=embedding_cache,
    )
    y_train = train_labelled["failure_mode_label"].astype(str)
    y_test = test_labelled["failure_mode_label"].astype(str)

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    timeout = None if args.optuna_timeout <= 0 else args.optuna_timeout
    best_result, family_results, trials_frame = run_optuna_search(
        families=selected_families,
        X_train=X_train,
        y_train=y_train,
        groups_train=train_labelled[args.group_col],
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
        num_classes=len(retained_classes),
    )
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    metrics = classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        labels=retained_classes,
        ordered_labels=None,
    )

    confusion_array = np.asarray(metrics["confusion_matrix"], dtype=int)
    plots_dir = output_dir / "plots"
    confusion_plot_path = plot_confusion_matrix(
        labels=retained_classes,
        matrix=confusion_array,
        path=plots_dir / "confusion_matrix.png",
        title=f"Failure Mode: confusion matrix ({best_family})",
    )
    confusion_csv_path = save_csv_table(
        pd.DataFrame(confusion_array, index=retained_classes, columns=retained_classes),
        output_dir / "confusion_matrix.csv",
        index=True,
    )
    per_class_metrics_df = per_class_metrics_frame(metrics["per_class_metrics"], retained_classes)
    per_class_metrics_csv_path = save_csv_table(
        per_class_metrics_df,
        output_dir / "per_class_metrics.csv",
        index=False,
    )
    per_class_metrics_plot_path = plot_per_class_metrics(
        per_class_metrics=metrics["per_class_metrics"],
        labels=retained_classes,
        path=plots_dir / "per_class_metrics.png",
        title=f"Failure Mode: per-class metrics ({best_family})",
    )
    label_distribution_plot_path = plot_label_distribution(
        y_true=y_test,
        y_pred=predictions,
        labels=retained_classes,
        path=plots_dir / "label_distribution.png",
        title=f"Failure Mode: label distribution ({best_family})",
    )

    predictions_frame = pd.DataFrame(
        {
            "patch_id": test_labelled["patch_id"].values,
            args.group_col: test_labelled[args.group_col].values,
            args.target_col: test_labelled[args.target_col].values,
            "true_label": y_test.values,
            "predicted_label": predictions,
            "selected_family": best_family,
            "hard_threshold": hard_threshold,
        }
    )
    probability_frame = maybe_parse_probability_frame(
        estimator=estimator,
        features=X_test,
        class_labels=retained_classes,
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
            title=f"Failure Mode: prediction confidence ({best_family})",
        )
    predictions_path = save_csv_table(predictions_frame, output_dir / "predictions.csv", index=False)
    trials_path = save_csv_table(trials_frame, output_dir / "study_trials.csv", index=False)
    optuna_artifacts = write_optuna_diagnostics(
        output_dir=output_dir,
        family_results=family_results,
        trials_frame=trials_frame,
        stage_name="failure_mode",
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
        "stage_name": "failure_mode",
        "selected_family": best_family,
        "target_col": args.target_col,
        "feature_set": args.feature_set,
        "metadata_cols": metadata_cols,
        "hard_quantile": args.hard_quantile,
        "hard_threshold": hard_threshold,
        "dominance_margin": args.dominance_margin,
        "min_class_count": args.min_class_count,
        "retained_classes": retained_classes,
        "test_metrics": metrics,
        "best_cv_score_macro_f1": best_result["best_cv_score"],
        "n_rows_total": int(len(modeling_frame)),
        "n_rows_train": int(len(train_frame)),
        "n_rows_test": int(len(test_frame)),
        "n_rows_train_hard": int(len(train_hard)),
        "n_rows_test_hard": int(len(test_hard)),
        "n_rows_train_labelled": int(len(train_labelled)),
        "n_rows_test_labelled": int(len(test_labelled)),
        "n_groups_train": int(train_labelled[args.group_col].nunique()),
        "n_groups_test": int(test_labelled[args.group_col].nunique()),
        "dropped_missing_target_rows": int(dropped_missing_target),
        "feature_summary": train_feature_summary,
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "train_class_counts": train_labelled["failure_mode_label"].value_counts().reindex(retained_classes).fillna(0).astype(int).to_dict(),
        "test_class_counts": test_labelled["failure_mode_label"].value_counts().reindex(retained_classes).fillna(0).astype(int).to_dict(),
        "all_train_label_counts_before_min_count_filter": retained_class_counts.astype(int).to_dict(),
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
            "stage_name": "failure_mode",
            "target_col": args.target_col,
            "feature_set": args.feature_set,
            "metadata_cols": metadata_cols,
            "hard_quantile": args.hard_quantile,
            "hard_threshold": hard_threshold,
            "dominance_margin": args.dominance_margin,
            "min_class_count": args.min_class_count,
            "retained_classes": retained_classes,
            "selected_family": best_family,
            "best_params": best_params,
            "group_col": args.group_col,
            "random_seed": args.random_seed,
        },
    )

    reloaded_predictions = load_table(predictions_path)
    reloaded_hard_labels = load_table(hard_label_table_path)
    if len(reloaded_predictions) != len(test_labelled):
        raise AssertionError(
            f"Predictions row count {len(reloaded_predictions)} did not match held-out labelled row count {len(test_labelled)}."
        )
    if len(reloaded_hard_labels) != len(hard_label_table):
        raise AssertionError(
            f"Hard-label table row count {len(reloaded_hard_labels)} did not match in-memory row count {len(hard_label_table)}."
        )

    artifact_map = {
        "model_joblib": model_path,
        "metrics_json": metrics_path,
        "best_params_json": best_params_path,
        "family_search_results_json": family_results_path,
        "predictions_csv": predictions_path,
        "hard_patch_labels_csv": hard_label_table_path,
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
        "stage_name": "failure_mode",
        "success": True,
        "input_path": str(resolve_path(args.input_table)),
        "output_dir": str(output_dir),
        "summary": [
            {"name": "total_rows_after_target_filter", "value": int(len(modeling_frame))},
            {"name": "train_hard_rows", "value": int(len(train_hard))},
            {"name": "test_hard_rows", "value": int(len(test_hard))},
            {"name": "train_labelled_rows", "value": int(len(train_labelled))},
            {"name": "test_labelled_rows", "value": int(len(test_labelled))},
            {"name": "selected_family", "value": best_family},
            {"name": "best_cv_score_macro_f1", "value": float(best_result["best_cv_score"])},
            {"name": "test_macro_f1", "value": float(metrics["macro_f1"])},
            {"name": "test_balanced_accuracy", "value": float(metrics["balanced_accuracy"])},
        ],
        "artifact_checks": artifact_checks,
        "missingness_summary": [
            *missingness_summary(reloaded_predictions),
            *missingness_summary(reloaded_hard_labels),
        ],
        "notes": [
            f"Dropped {dropped_missing_target} rows with missing `{args.target_col}` before splitting.",
            f"Train-only hard threshold for `{args.target_col}` was {hard_threshold}.",
            f"Retained failure-mode classes were {retained_classes}.",
            f"Dominance margin was {args.dominance_margin} and min class count was {args.min_class_count}.",
            f"Train/test groups are disjoint under `{args.group_col}`.",
            f"Feature set `{args.feature_set}` used {train_feature_summary['embedding_feature_count']} embedding features and {train_feature_summary['metadata_feature_count']} metadata columns.",
            *family_notes,
        ],
    }
    write_validation_reports(output_dir, report)
    write_timing_json(
        output_dir,
        started_time,
        stage_name="failure_mode",
        extra={
            "selected_family": best_family,
            "hard_threshold": hard_threshold,
            "test_macro_f1": float(metrics["macro_f1"]),
            "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        },
    )
    logger.info(
        "finished stage=failure_mode family=%s macro_f1=%.6f balanced_accuracy=%.6f retained_classes=%s",
        best_family,
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        retained_classes,
    )


if __name__ == "__main__":
    main()
