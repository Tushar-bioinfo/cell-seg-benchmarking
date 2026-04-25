from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cell-seg-mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SUBSET_LABELS = {
    "consep": "CoNSeP",
    "crag": "CRAG",
    "dpath": "DPath",
    "glas": "GlaS",
    "pannuke": "PanNuke",
}

FAMILY_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "svm": "SVM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready failure-probe tables and figures.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("tmp/extra/outputs_22_04_26/conic_liz/failure_prediction/quality_probe"),
        help="Directory with saved quality_probe artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/conic_liz/analysis/failure_prediction"),
        help="Directory for generated analysis outputs.",
    )
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "font.size": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=450)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def friendly_subset(slide_id: str) -> str:
    prefix = str(slide_id).split("_", 1)[0].lower()
    return SUBSET_LABELS.get(prefix, prefix)


def compute_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    x_rank = compute_rank(x)
    y_rank = compute_rank(y)
    return float(x_rank.corr(y_rank))


def format_float(value: float | int | str | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def write_markdown_table(df: pd.DataFrame, path: Path, digits: int = 4) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_float(row[col], digits=digits) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def build_classifier_comparison(candidate_metrics: dict) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for family, metrics in candidate_metrics.items():
        rows.append(
            {
                "Classifier": FAMILY_LABELS.get(family, family),
                "Best CV ROC-AUC": float(metrics["best_cv_roc_auc"]),
                "Test ROC-AUC": float(metrics["test_metrics"]["roc_auc"]),
                "Test Balanced Accuracy": float(metrics["test_metrics"]["balanced_accuracy"]),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["Test ROC-AUC", "Test Balanced Accuracy"], ascending=False).reset_index(drop=True)


def build_selected_vs_baselines(predictions: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    test_df = predictions.loc[predictions["split"].eq("test")].copy()
    y_true = test_df["target"].astype(int)
    majority_label = int(y_true.mean() >= 0.5)
    majority_accuracy = float((y_true.eq(majority_label)).mean())
    positive_rate = float(y_true.mean())
    selected = metrics["test_metrics"]
    rows = [
        {
            "Method": f"Selected probe ({FAMILY_LABELS.get(metrics['selected_classifier_family'], metrics['selected_classifier_family'])})",
            "Accuracy": float(selected["accuracy"]),
            "Balanced Accuracy": float(selected["balanced_accuracy"]),
            "ROC-AUC": float(selected["roc_auc"]),
            "Average Precision": float(selected["average_precision"]),
            "Notes": "Held-out test performance",
        },
        {
            "Method": f"Majority-class baseline (always {'failure' if majority_label == 1 else 'quality'})",
            "Accuracy": majority_accuracy,
            "Balanced Accuracy": 0.5,
            "ROC-AUC": np.nan,
            "Average Precision": np.nan,
            "Notes": "Label-only baseline on the held-out test set",
        },
        {
            "Method": "Random/prevalence ranking baseline",
            "Accuracy": np.nan,
            "Balanced Accuracy": np.nan,
            "ROC-AUC": 0.5,
            "Average Precision": positive_rate,
            "Notes": "AP equals failure prevalence in the test set",
        },
    ]
    return pd.DataFrame(rows)


def build_per_class_table(predictions: pd.DataFrame) -> pd.DataFrame:
    test_df = predictions.loc[predictions["split"].eq("test")].copy()
    y_true = test_df["target"].astype(int)
    y_pred = test_df["predicted_failure_label"].astype(int)
    rows: list[dict[str, float | str | int]] = []
    for class_value, class_label in [(0, "quality"), (1, "failure")]:
        class_true = y_true.eq(class_value)
        class_pred = y_pred.eq(class_value)
        rows.append(
            {
                "Class": class_label,
                "Support": int(class_true.sum()),
                "Predicted Count": int(class_pred.sum()),
                "Precision": float(precision_score(class_true, class_pred, zero_division=0)),
                "Recall": float(recall_score(class_true, class_pred, zero_division=0)),
                "F1": float(f1_score(class_true, class_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def build_subset_table(predictions: pd.DataFrame) -> pd.DataFrame:
    test_df = predictions.loc[predictions["split"].eq("test")].copy()
    test_df["Source Subset"] = test_df["slide_id"].map(friendly_subset)

    rows: list[dict[str, float | str | int]] = []
    for subset, subset_df in test_df.groupby("Source Subset", sort=False):
        y_true = subset_df["target"].astype(int)
        y_pred = subset_df["predicted_failure_label"].astype(int)
        y_score = subset_df["predicted_failure_probability"].astype(float)
        supports_both = y_true.nunique() == 2
        rows.append(
            {
                "Source Subset": subset,
                "Test Patches": int(len(subset_df)),
                "Held-out Slides": int(subset_df["slide_id"].nunique()),
                "Failure Rate": float(y_true.mean()),
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "ROC-AUC": float(roc_auc_score(y_true, y_score)) if supports_both else np.nan,
                "Average Precision": float(average_precision_score(y_true, y_score)) if supports_both else np.nan,
                "Failure Precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "Failure Recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "Caution": "small_n" if len(subset_df) < 30 else "",
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["Test Patches", "Source Subset"], ascending=[False, True]).reset_index(drop=True)


def plot_failure_probability_vs_quality(predictions: pd.DataFrame, output_dir: Path, threshold: float) -> dict[str, float]:
    test_df = predictions.loc[predictions["split"].eq("test")].copy()
    x = test_df["target_continuous"].astype(float)
    y = test_df["predicted_failure_probability"].astype(float)
    pearson_r = float(x.corr(y))
    spearman_r = spearman_corr(x, y)

    plot_df = test_df.copy()
    plot_df["target_continuous"] = x
    plot_df["predicted_failure_probability"] = y
    plot_df["pq_bin"] = pd.qcut(plot_df["target_continuous"], q=20, duplicates="drop")
    trend = (
        plot_df.groupby("pq_bin", observed=False)
        .agg(
            pq_center=("target_continuous", "median"),
            mean_failure_probability=("predicted_failure_probability", "mean"),
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    hb = ax.hexbin(
        x,
        y,
        gridsize=32,
        cmap="YlOrRd",
        mincnt=1,
        linewidths=0.0,
    )
    ax.plot(
        trend["pq_center"],
        trend["mean_failure_probability"],
        color="#1f4e79",
        linewidth=2.4,
        marker="o",
        markersize=3.8,
        label="Binned mean failure probability",
    )
    ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.2, label=f"Failure threshold = {threshold:.3f}")
    ax.set_xlabel("Median cross-model instance PQ")
    ax.set_ylabel("Predicted failure probability")
    ax.set_title("Held-out failure scores versus continuous patch quality")
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.985,
        0.03,
        f"Pearson r = {pearson_r:.3f}\nSpearman rho = {spearman_r:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )
    ax.legend(frameon=False, loc="upper right")
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Test patch count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_dir, "failure_probability_vs_instance_pq_test")
    return {"pearson_r": pearson_r, "spearman_rho": spearman_r}


def plot_calibration_curve(predictions: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    test_df = predictions.loc[predictions["split"].eq("test")].copy()
    y_true = test_df["target"].astype(int)
    y_prob = test_df["predicted_failure_probability"].astype(float)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    brier = float(brier_score_loss(y_true, y_prob))

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.15], hspace=0.15)

    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2, label="Perfect calibration")
    ax.plot(
        prob_pred,
        prob_true,
        color="#1f4e79",
        marker="o",
        linewidth=2.2,
        markersize=4.5,
        label="Held-out test",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Observed failure fraction")
    ax.set_title("Held-out calibration for failure probabilities")
    ax.text(
        0.98,
        0.03,
        f"Brier score = {brier:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_hist = fig.add_subplot(gs[1], sharex=ax)
    ax_hist.hist(y_prob, bins=np.linspace(0, 1, 11), color="#d97706", edgecolor="white")
    ax_hist.set_xlabel("Predicted failure probability")
    ax_hist.set_ylabel("Test patches")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    save_figure(fig, output_dir, "failure_probability_calibration_test")
    return {"brier_score": brier}


def write_summary_markdown(
    output_dir: Path,
    classifier_comparison: pd.DataFrame,
    selected_vs_baselines: pd.DataFrame,
    per_class: pd.DataFrame,
    subset_table: pd.DataFrame,
    threshold: float,
    corr_stats: dict[str, float],
    calibration_stats: dict[str, float],
) -> None:
    major_subset = subset_table.loc[subset_table["Test Patches"].ge(100)].copy()
    major_subset = major_subset.sort_values("Test Patches", ascending=False)
    subset_lines = []
    for _, row in major_subset.iterrows():
        subset_lines.append(
            "- "
            f"{row['Source Subset']}: n = {int(row['Test Patches'])}, "
            f"balanced accuracy = {row['Balanced Accuracy']:.3f}, "
            f"ROC-AUC = {row['ROC-AUC']:.3f}, "
            f"failure recall = {row['Failure Recall']:.3f}"
        )
    small_subset_rows = subset_table.loc[subset_table["Caution"].eq("small_n")]
    small_subset_note = ""
    if not small_subset_rows.empty:
        names = ", ".join(small_subset_rows["Source Subset"].tolist())
        small_subset_note = (
            f"Small held-out subsets (`{names}`) are included in the table but should be interpreted cautiously "
            "because their test sample sizes are too small for strong comparative claims."
        )

    lines = [
        "# Failure Probe Follow-up Assets",
        "",
        "This summary was generated from the saved `quality_probe` artifacts without rerunning training.",
        "",
        "## Key Additions",
        "",
        "- Report table: `classifier_comparison.md`",
        "- Report table: `selected_model_vs_baselines.md`",
        "- Report table: `per_class_performance.md`",
        "- Report table: `test_source_subset_performance.md`",
        "- Figure: `failure_probability_vs_instance_pq_test.png`",
        "- Figure: `failure_probability_calibration_test.png`",
        "",
        "## Main Follow-up Findings",
        "",
        f"- The held-out binary failure threshold remains `{threshold:.4f}` on median cross-model `instance_pq`.",
        f"- Predicted failure probability tracks continuous patch quality strongly on the held-out test set: Pearson `r = {corr_stats['pearson_r']:.3f}`, Spearman `rho = {corr_stats['spearman_rho']:.3f}`.",
        f"- The held-out calibration curve has Brier score `{calibration_stats['brier_score']:.3f}`. The probabilities are directionally useful, but calibration is not perfect across all bins.",
        "- Subgroup performance is heterogeneous across source collections:",
        *subset_lines,
    ]
    if small_subset_note:
        lines.extend(["", small_subset_note])

    lines.extend(
        [
            "",
            "## Tables",
            "",
            "### Classifier Comparison",
            "",
            (output_dir / "classifier_comparison.md").read_text().rstrip(),
            "",
            "### Selected Model Versus Baselines",
            "",
            (output_dir / "selected_model_vs_baselines.md").read_text().rstrip(),
            "",
            "### Per-class Performance",
            "",
            (output_dir / "per_class_performance.md").read_text().rstrip(),
            "",
            "### Held-out Source-subset Performance",
            "",
            (output_dir / "test_source_subset_performance.md").read_text().rstrip(),
            "",
        ]
    )
    (output_dir / "follow_up_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    metrics = load_json(artifact_dir / "metrics.json")
    candidate_metrics = load_json(artifact_dir / "candidate_family_metrics.json")
    predictions = pd.read_csv(artifact_dir / "predictions.csv")

    classifier_comparison = build_classifier_comparison(candidate_metrics)
    selected_vs_baselines = build_selected_vs_baselines(predictions, metrics)
    per_class = build_per_class_table(predictions)
    subset_table = build_subset_table(predictions)

    classifier_comparison.to_csv(output_dir / "classifier_comparison.csv", index=False)
    write_markdown_table(classifier_comparison, output_dir / "classifier_comparison.md")

    selected_vs_baselines.to_csv(output_dir / "selected_model_vs_baselines.csv", index=False)
    write_markdown_table(selected_vs_baselines, output_dir / "selected_model_vs_baselines.md")

    per_class.to_csv(output_dir / "per_class_performance.csv", index=False)
    write_markdown_table(per_class, output_dir / "per_class_performance.md")

    subset_table.to_csv(output_dir / "test_source_subset_performance.csv", index=False)
    write_markdown_table(subset_table, output_dir / "test_source_subset_performance.md")

    threshold = float(metrics["classification_threshold"])
    corr_stats = plot_failure_probability_vs_quality(predictions, output_dir, threshold)
    calibration_stats = plot_calibration_curve(predictions, output_dir)

    summary_payload = {
        "classification_threshold": threshold,
        "selected_classifier_family": metrics["selected_classifier_family"],
        "correlation_with_continuous_quality": corr_stats,
        "calibration": calibration_stats,
    }
    (output_dir / "follow_up_metrics_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

    write_summary_markdown(
        output_dir=output_dir,
        classifier_comparison=classifier_comparison,
        selected_vs_baselines=selected_vs_baselines,
        per_class=per_class,
        subset_table=subset_table,
        threshold=threshold,
        corr_stats=corr_stats,
        calibration_stats=calibration_stats,
    )


if __name__ == "__main__":
    main()
