from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVAL_DIR = Path("tmp/extra/outputs_22_04_26/conic_liz")
OUTPUT_DIR = Path("outputs/analysis/conic_liz")
EMBED_MORPH_PATH = EVAL_DIR / "embed_morph.csv"

MODEL_FILES = {
    "cellsam": "cellsam_evaluation.csv",
    "cellpose_sam": "cellpose_sam_evaluation.csv",
    "cellvit_sam": "cellvit_sam_evaluation.csv",
    "stardist": "stardist_evaluation.csv",
}

MODEL_LABELS = {
    "cellsam": "CellSAM",
    "cellpose_sam": "Cellpose-SAM",
    "cellvit_sam": "CellViT-SAM-H",
    "stardist": "StarDist",
    "consensus_pq": "Consensus PQ",
    "tie": "Tie",
}

MODEL_ORDER = ["cellpose_sam", "cellsam", "cellvit_sam", "stardist"]

MODEL_COLORS = {
    "cellpose_sam": "#1f4e79",
    "cellsam": "#d97706",
    "cellvit_sam": "#2e8b57",
    "stardist": "#b22222",
    "consensus_pq": "#111111",
}

CONTINUOUS_FEATURES = [
    ("foreground_fraction", "Foreground fraction"),
    ("total_nuclei", "Total nuclei"),
    ("mean_area", "Mean nuclei area"),
    ("mean_circularity", "Mean circularity"),
]

COMPOSITION_ORDER = [
    "connective-tissue-rich",
    "epithelial-rich",
    "mixed",
    "lymphocyte-rich",
    "no-nuclei",
]

COMPOSITION_LABELS = {
    "connective-tissue-rich": "Connective-rich",
    "epithelial-rich": "Epithelial-rich",
    "mixed": "Mixed",
    "lymphocyte-rich": "Lymphocyte-rich",
    "no-nuclei": "No nuclei",
}


def patch_id_from_image_id(image_id: str) -> str:
    name = Path(str(image_id)).name
    if name.endswith("_image.png"):
        return name[: -len("_image.png")]
    return Path(name).stem


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


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=450)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def load_eval_wide() -> pd.DataFrame:
    frames = []
    for model_key, filename in MODEL_FILES.items():
        path = EVAL_DIR / filename
        df = pd.read_csv(path, usecols=["status", "image_id", "instance_pq"])
        df = df.loc[df["status"].eq("ok")].copy()
        df["patch_id"] = df["image_id"].map(patch_id_from_image_id)
        frames.append(df[["patch_id", "instance_pq"]].rename(columns={"instance_pq": model_key}))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="patch_id", how="inner")

    merged["consensus_pq"] = merged[MODEL_ORDER].median(axis=1)
    merged["mean_pq_across_models"] = merged[MODEL_ORDER].mean(axis=1)
    return merged


def load_eval_long() -> pd.DataFrame:
    frames = []
    metric_cols = [
        "status",
        "image_id",
        "instance_pq",
        "instance_rq",
        "instance_sq",
        "pixel_precision",
        "pixel_recall",
    ]
    for model_key, filename in MODEL_FILES.items():
        path = EVAL_DIR / filename
        df = pd.read_csv(path, usecols=metric_cols)
        df = df.loc[df["status"].eq("ok")].copy()
        df["patch_id"] = df["image_id"].map(patch_id_from_image_id)
        df["model_key"] = model_key
        frames.append(
            df[
                [
                    "patch_id",
                    "model_key",
                    "instance_pq",
                    "instance_rq",
                    "instance_sq",
                    "pixel_precision",
                    "pixel_recall",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def load_embed_morph() -> pd.DataFrame:
    cols = [
        "sample_id",
        "foreground_fraction",
        "total_nuclei",
        "mean_area",
        "mean_circularity",
        "mean_eccentricity",
        "mean_solidity",
        "dominant_class",
        "dominant_fraction",
        "richness_label",
    ]
    return pd.read_csv(EMBED_MORPH_PATH, usecols=cols)


def make_qc_summary(merged_wide: pd.DataFrame, embed_morph: pd.DataFrame) -> None:
    joined = merged_wide.merge(embed_morph, left_on="patch_id", right_on="sample_id", how="left")
    lines = [
        "Eval + embed_morph merged analysis QC",
        f"Matched patch rows: {len(merged_wide)}",
        f"Rows with embed_morph match: {joined['sample_id'].notna().sum()}",
        "",
        "Missing values in selected embed_morph columns on matched patches:",
    ]
    qc_cols = [
        "foreground_fraction",
        "total_nuclei",
        "mean_area",
        "mean_circularity",
        "dominant_class",
        "dominant_fraction",
        "richness_label",
    ]
    for col in qc_cols:
        lines.append(f"- {col}: {int(joined[col].isna().sum())}")
    (OUTPUT_DIR / "eval_embed_morph_qc_summary.txt").write_text("\n".join(lines) + "\n")


def morphology_pq_analysis(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    quartile_rows = []
    metrics = ["consensus_pq", *MODEL_ORDER]
    for feature, feature_label in CONTINUOUS_FEATURES:
        tmp = merged[["patch_id", feature, *metrics]].dropna().copy()
        quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
        tmp["feature_quartile"] = pd.qcut(tmp[feature], 4, labels=quartile_labels)
        for metric in metrics:
            rho = tmp[[metric, feature]].corr(method="spearman").iloc[0, 1]
            rows.append(
                {
                    "feature": feature,
                    "feature_label": feature_label,
                    "metric": metric,
                    "metric_label": MODEL_LABELS[metric],
                    "spearman_rho": float(rho),
                    "n": int(len(tmp)),
                }
            )
            quartile_summary = (
                tmp.groupby("feature_quartile", observed=False)[metric]
                .agg(["median", "mean", "count"])
                .reset_index()
            )
            for _, qrow in quartile_summary.iterrows():
                quartile_rows.append(
                    {
                        "feature": feature,
                        "feature_label": feature_label,
                        "metric": metric,
                        "metric_label": MODEL_LABELS[metric],
                        "feature_quartile": str(qrow["feature_quartile"]),
                        "median_instance_pq": float(qrow["median"]),
                        "mean_instance_pq": float(qrow["mean"]),
                        "n": int(qrow["count"]),
                    }
                )

    corr_df = pd.DataFrame(rows)
    quartile_df = pd.DataFrame(quartile_rows)
    corr_df.to_csv(OUTPUT_DIR / "morphology_pq_spearman.csv", index=False)
    quartile_df.to_csv(OUTPUT_DIR / "morphology_pq_quartile_summary.csv", index=False)

    heatmap_matrix = corr_df.pivot(index="feature_label", columns="metric_label", values="spearman_rho")
    feature_order = [label for _, label in CONTINUOUS_FEATURES]
    heatmap_matrix = heatmap_matrix[
        ["Consensus PQ", "Cellpose-SAM", "CellSAM", "CellViT-SAM-H", "StarDist"]
    ]
    heatmap_matrix = heatmap_matrix.loc[feature_order]
    heatmap_matrix.to_csv(OUTPUT_DIR / "morphology_pq_spearman_matrix.csv", index=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    values = heatmap_matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(np.arange(len(heatmap_matrix.columns)), heatmap_matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heatmap_matrix.index)), heatmap_matrix.index)
    ax.set_title("Morphology vs patch-quality correlation (Spearman rho)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if abs(val) >= 0.23 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    save_figure(fig, "morphology_pq_spearman_heatmap")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), sharey=True)
    axes = axes.ravel()
    for ax, (feature, feature_label) in zip(axes, CONTINUOUS_FEATURES):
        panel = quartile_df.loc[quartile_df["feature"].eq(feature)].copy()
        for metric in ["consensus_pq", *MODEL_ORDER]:
            metric_panel = panel.loc[panel["metric"].eq(metric)].copy()
            metric_panel["quartile_order"] = metric_panel["feature_quartile"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
            metric_panel = metric_panel.sort_values("quartile_order")
            style = "--" if metric == "consensus_pq" else "-"
            marker = "D" if metric == "consensus_pq" else "o"
            ax.plot(
                metric_panel["quartile_order"],
                metric_panel["median_instance_pq"],
                label=MODEL_LABELS[metric],
                color=MODEL_COLORS[metric],
                linestyle=style,
                marker=marker,
                linewidth=2.0 if metric == "consensus_pq" else 1.8,
                markersize=5,
            )
        ax.set_title(feature_label)
        ax.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("Feature quartile")
        ax.set_ylim(0.0, 0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Median instance PQ")
    axes[2].set_ylabel("Median instance PQ")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Patch quality across morphology feature quartiles", y=0.98, fontsize=13)
    fig.subplots_adjust(top=0.88, bottom=0.14, wspace=0.18, hspace=0.25)
    save_figure(fig, "morphology_pq_feature_quartiles")

    return corr_df, quartile_df


def composition_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_counts = merged["richness_label"].value_counts()
    for richness in COMPOSITION_ORDER:
        subset = merged.loc[merged["richness_label"].eq(richness)].copy()
        for metric in ["consensus_pq", *MODEL_ORDER]:
            rows.append(
                {
                    "richness_label": richness,
                    "richness_label_display": COMPOSITION_LABELS[richness],
                    "patch_count": int(len(subset)),
                    "metric": metric,
                    "metric_label": MODEL_LABELS[metric],
                    "median_instance_pq": float(subset[metric].median()),
                    "mean_instance_pq": float(subset[metric].mean()),
                    "n": int(group_counts.get(richness, 0)),
                }
            )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "pq_by_richness_label.csv", index=False)

    heatmap = summary_df.pivot(index="richness_label_display", columns="metric_label", values="median_instance_pq")
    heatmap = heatmap.loc[
        [COMPOSITION_LABELS[r] for r in COMPOSITION_ORDER],
        ["Consensus PQ", "Cellpose-SAM", "CellSAM", "CellViT-SAM-H", "StarDist"],
    ]
    heatmap.to_csv(OUTPUT_DIR / "pq_by_richness_label_matrix.csv", index=True)

    row_labels = [
        f"{COMPOSITION_LABELS[r]} (n={int(group_counts.get(r, 0))})"
        for r in COMPOSITION_ORDER
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    values = heatmap.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=0.7)
    ax.set_xticks(np.arange(len(heatmap.columns)), heatmap.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Patch quality by composition label")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val >= 0.42 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median instance PQ")
    save_figure(fig, "pq_by_richness_label_heatmap")

    return summary_df


def winner_by_phenotype_analysis(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = merged[MODEL_ORDER]
    max_values = values.max(axis=1)
    num_max = values.eq(max_values, axis=0).sum(axis=1)
    winner = values.idxmax(axis=1)
    winner = winner.where(num_max.eq(1), other="tie")

    tmp = merged[["patch_id", "richness_label"]].copy()
    tmp["winner"] = winner
    counts = (
        tmp.groupby(["richness_label", "winner"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=COMPOSITION_ORDER, columns=[*MODEL_ORDER, "tie"], fill_value=0)
    )
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0

    counts.to_csv(OUTPUT_DIR / "winner_by_richness_counts.csv", index=True)
    percentages.to_csv(OUTPUT_DIR / "winner_by_richness_percentages.csv", index=True)

    row_labels = [f"{COMPOSITION_LABELS[r]} (n={int(counts.loc[r].sum())})" for r in COMPOSITION_ORDER]
    col_labels = [MODEL_LABELS[c] for c in [*MODEL_ORDER, "tie"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    values = percentages.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Unique winner by patch phenotype")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val >= 55 else "#222222"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Within-phenotype win rate (%)")
    save_figure(fig, "winner_by_richness_heatmap")

    count_rows = []
    pct_rows = []
    for richness in COMPOSITION_ORDER:
        for winner_key in [*MODEL_ORDER, "tie"]:
            count_rows.append(
                {
                    "richness_label": richness,
                    "richness_label_display": COMPOSITION_LABELS[richness],
                    "winner": winner_key,
                    "winner_label": MODEL_LABELS[winner_key],
                    "count": int(counts.loc[richness, winner_key]),
                }
            )
            pct_rows.append(
                {
                    "richness_label": richness,
                    "richness_label_display": COMPOSITION_LABELS[richness],
                    "winner": winner_key,
                    "winner_label": MODEL_LABELS[winner_key],
                    "percentage": float(percentages.loc[richness, winner_key]),
                }
            )

    counts_long = pd.DataFrame(count_rows)
    pct_long = pd.DataFrame(pct_rows)
    counts_long.to_csv(OUTPUT_DIR / "winner_by_richness_counts_long.csv", index=False)
    pct_long.to_csv(OUTPUT_DIR / "winner_by_richness_percentages_long.csv", index=False)
    return counts_long, pct_long


def error_profile_density_analysis(long_df: pd.DataFrame, embed_morph: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(embed_morph[["sample_id", "foreground_fraction"]], left_on="patch_id", right_on="sample_id", how="left")
    merged["density_quartile"] = pd.qcut(
        merged["foreground_fraction"],
        4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
    )

    summary = (
        merged.groupby(["model_key", "density_quartile"], observed=False)[
            ["instance_pq", "instance_rq", "instance_sq", "pixel_precision", "pixel_recall"]
        ]
        .median()
        .reset_index()
    )
    summary["model_label"] = summary["model_key"].map(MODEL_LABELS)
    summary.to_csv(OUTPUT_DIR / "error_profile_by_density_quartile.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0), sharex=True)
    axes = axes.ravel()
    metric_specs = [
        ("instance_rq", "Instance RQ"),
        ("instance_sq", "Instance SQ"),
        ("pixel_precision", "Pixel precision"),
        ("pixel_recall", "Pixel recall"),
    ]
    quartile_order = ["Q1 low", "Q2", "Q3", "Q4 high"]
    x = np.arange(len(quartile_order))
    for ax, (metric, title) in zip(axes, metric_specs):
        for model_key in MODEL_ORDER:
            panel = summary.loc[summary["model_key"].eq(model_key), ["density_quartile", metric]].copy()
            panel["density_quartile"] = pd.Categorical(panel["density_quartile"], quartile_order, ordered=True)
            panel = panel.sort_values("density_quartile")
            ax.plot(
                x,
                panel[metric],
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
            )
        ax.set_title(title)
        ax.set_xticks(x, quartile_order)
        ax.set_ylim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].set_xlabel("Foreground-fraction quartile")
    axes[3].set_xlabel("Foreground-fraction quartile")
    axes[0].set_ylabel("Median metric value")
    axes[2].set_ylabel("Median metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Error profile across patch-density strata", y=0.98, fontsize=13)
    fig.subplots_adjust(top=0.88, bottom=0.14, wspace=0.20, hspace=0.22)
    save_figure(fig, "error_profile_by_density_quartile")

    cutoffs = pd.qcut(merged["foreground_fraction"], 4, retbins=True, duplicates="drop")[1]
    cutoff_df = pd.DataFrame(
        {
            "density_quartile": ["Q1 low", "Q2", "Q3", "Q4 high"],
            "lower_bound": cutoffs[:-1],
            "upper_bound": cutoffs[1:],
        }
    )
    cutoff_df.to_csv(OUTPUT_DIR / "density_quartile_cutoffs.csv", index=False)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    eval_wide = load_eval_wide()
    eval_long = load_eval_long()
    embed_morph = load_embed_morph()
    merged = eval_wide.merge(embed_morph, left_on="patch_id", right_on="sample_id", how="left")

    make_qc_summary(eval_wide, embed_morph)
    morphology_pq_analysis(merged)
    composition_analysis(merged)
    winner_by_phenotype_analysis(merged)
    error_profile_density_analysis(eval_long, embed_morph)


if __name__ == "__main__":
    main()
