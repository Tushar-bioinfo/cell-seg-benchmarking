from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = Path("tmp/extra/outputs_22_04_26/conic_liz")
OUTPUT_DIR = Path("outputs/analysis/conic_liz")

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
}

MODEL_COLORS = {
    "cellpose_sam": "#1f4e79",
    "cellsam": "#d97706",
    "cellvit_sam": "#2e8b57",
    "stardist": "#b22222",
}

KEY_METRICS = [
    "instance_pq",
    "pixel_dice",
    "instance_rq",
    "instance_sq",
    "pixel_precision",
    "pixel_recall",
]

USECOLS = ["status", "image_id", *KEY_METRICS]


def patch_id_from_image_id(image_id: str) -> str:
    name = Path(str(image_id)).name
    if name.endswith("_image.png"):
        return name[: -len("_image.png")]
    return Path(name).stem


def load_model_table(model_key: str) -> pd.DataFrame:
    path = INPUT_DIR / MODEL_FILES[model_key]
    df = pd.read_csv(path, usecols=USECOLS)
    df = df.loc[df["status"].eq("ok")].copy()
    df["patch_id"] = df["image_id"].map(patch_id_from_image_id)

    if df["patch_id"].duplicated().any():
        duplicated = df.loc[df["patch_id"].duplicated(), "patch_id"].head().tolist()
        raise ValueError(f"{model_key} has duplicate patch IDs: {duplicated}")

    missing_mask = df[["patch_id", *KEY_METRICS]].isna().any(axis=1)
    if missing_mask.any():
        missing_rows = int(missing_mask.sum())
        raise ValueError(f"{model_key} has {missing_rows} rows with missing matched-analysis fields")

    return df.set_index("patch_id").sort_index()


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def metric_summary(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)),
        "iqr": iqr(series),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
    }


def set_common_style() -> None:
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
            "legend.fontsize": 10,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=450)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def make_overall_summary(matched_tables: dict[str, pd.DataFrame], raw_counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model_key, df in matched_tables.items():
        row: dict[str, float | int | str] = {
            "model_key": model_key,
            "model": MODEL_LABELS[model_key],
            "n_evaluated_raw": raw_counts[model_key],
            "n_compared_matched": int(len(df)),
        }
        for metric in KEY_METRICS:
            stats = metric_summary(df[metric])
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["sort_metric"] = summary["instance_pq_median"]
    summary = summary.sort_values(["sort_metric", "instance_pq_mean"], ascending=False).drop(columns=["sort_metric"])
    return summary.reset_index(drop=True)


def write_pretty_table(summary: pd.DataFrame) -> None:
    metrics = [
        ("instance_pq", "PQ"),
        ("pixel_dice", "Dice"),
        ("instance_rq", "RQ"),
        ("instance_sq", "SQ"),
        ("pixel_precision", "PixPrec"),
        ("pixel_recall", "PixRec"),
    ]
    headers = [
        "Model",
        "RawN",
        "MatchN",
        *[f"{label}_{stat}" for _, label in metrics for stat in ("mean", "median", "std", "iqr")],
    ]
    lines = ["\t".join(headers)]
    for _, row in summary.iterrows():
        values = [
            row["model"],
            str(int(row["n_evaluated_raw"])),
            str(int(row["n_compared_matched"])),
        ]
        for metric, _ in metrics:
            values.extend(f"{row[f'{metric}_{stat}']:.4f}" for stat in ("mean", "median", "std", "iqr"))
        lines.append("\t".join(values))
    (OUTPUT_DIR / "overall_benchmark_summary_pretty.txt").write_text("\n".join(lines) + "\n")


def plot_pq_distribution(summary: pd.DataFrame, matched_tables: dict[str, pd.DataFrame]) -> None:
    order = summary["model_key"].tolist()
    labels = [MODEL_LABELS[m] for m in order]
    colors = [MODEL_COLORS[m] for m in order]
    data = [matched_tables[m]["instance_pq"].to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.5},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        boxprops={"linewidth": 1.0, "edgecolor": "#333333"},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#666666",
            "markeredgecolor": "#666666",
            "alpha": 0.15,
        },
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Instance PQ")
    ax.set_xlabel("Model")
    ax.set_title("Matched-patch instance PQ distribution")
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "matched_patch_instance_pq_boxplot")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model_key in order:
        values = np.sort(matched_tables[model_key]["instance_pq"].to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(
            values,
            y,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLORS[model_key],
            linewidth=2.0,
        )
    ax.set_xlabel("Instance PQ")
    ax.set_ylabel("ECDF")
    ax.set_title("Matched-patch instance PQ ECDF")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "matched_patch_instance_pq_ecdf")


def make_pq_distribution_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_key",
        "model",
        "instance_pq_mean",
        "instance_pq_median",
        "instance_pq_std",
        "instance_pq_iqr",
        "instance_pq_q1",
        "instance_pq_q3",
    ]
    return summary.loc[:, columns].copy()


def dominance_analysis(summary: pd.DataFrame, matched_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = summary["model_key"].tolist()
    pq_table = pd.concat(
        [matched_tables[model_key][["instance_pq"]].rename(columns={"instance_pq": model_key}) for model_key in order],
        axis=1,
    )

    winner_idx = pq_table.to_numpy().argmax(axis=1)
    winners = pd.Index([order[idx] for idx in winner_idx], name="winner_model")
    win_counts = winners.value_counts().reindex(order, fill_value=0)

    wins_df = pd.DataFrame(
        {
            "model_key": order,
            "model": [MODEL_LABELS[m] for m in order],
            "win_count": [int(win_counts[m]) for m in order],
            "win_percentage": [100.0 * win_counts[m] / len(pq_table) for m in order],
        }
    )

    pairwise_rows: list[dict[str, float | str | int]] = []
    pairwise_matrix = pd.DataFrame(index=[MODEL_LABELS[m] for m in order], columns=[MODEL_LABELS[m] for m in order], dtype=float)
    for a in order:
        for b in order:
            if a == b:
                pairwise_matrix.loc[MODEL_LABELS[a], MODEL_LABELS[b]] = np.nan
                continue
            gt = (pq_table[a] > pq_table[b]).sum()
            lt = (pq_table[a] < pq_table[b]).sum()
            ties = len(pq_table) - gt - lt
            win_rate = 100.0 * gt / len(pq_table)
            pairwise_rows.append(
                {
                    "model_a_key": a,
                    "model_a": MODEL_LABELS[a],
                    "model_b_key": b,
                    "model_b": MODEL_LABELS[b],
                    "a_beats_b_count": int(gt),
                    "b_beats_a_count": int(lt),
                    "tie_count": int(ties),
                    "a_beats_b_percentage": win_rate,
                }
            )
            pairwise_matrix.loc[MODEL_LABELS[a], MODEL_LABELS[b]] = win_rate

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df = pairwise_df.sort_values(["model_a", "model_b"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(len(order))
    heights = wins_df["win_count"].to_numpy()
    bars = ax.bar(x, heights, color=[MODEL_COLORS[m] for m in order], width=0.62)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in order])
    ax.set_ylabel("Matched-patch wins")
    ax.set_xlabel("Model")
    ax.set_title("Patch-level dominance by highest instance PQ")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, pct in zip(bars, wins_df["win_percentage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(heights) * 0.015,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, "patch_dominance_wins_bar")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    matrix_values = pairwise_matrix.to_numpy(dtype=float)
    cmap = plt.get_cmap("Blues")
    im = ax.imshow(matrix_values, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(pairwise_matrix.columns)), pairwise_matrix.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(pairwise_matrix.index)), pairwise_matrix.index)
    ax.set_title("Pairwise win rate by instance PQ")
    for i in range(matrix_values.shape[0]):
        for j in range(matrix_values.shape[1]):
            if math.isnan(matrix_values[i, j]):
                text = "—"
                color = "#444444"
            else:
                text = f"{matrix_values[i, j]:.1f}"
                color = "white" if matrix_values[i, j] >= 55 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("A beats B (%)")
    save_figure(fig, "patch_dominance_pairwise_heatmap")

    pairwise_matrix.to_csv(OUTPUT_DIR / "patch_dominance_pairwise_win_rates_matrix.csv", index=True)
    return wins_df, pairwise_df


def error_profile_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_key",
        "model",
        "instance_rq_mean",
        "instance_rq_median",
        "instance_rq_std",
        "instance_rq_iqr",
        "instance_sq_mean",
        "instance_sq_median",
        "instance_sq_std",
        "instance_sq_iqr",
        "pixel_precision_mean",
        "pixel_precision_median",
        "pixel_precision_std",
        "pixel_precision_iqr",
        "pixel_recall_mean",
        "pixel_recall_median",
        "pixel_recall_std",
        "pixel_recall_iqr",
    ]
    return summary.loc[:, columns].copy()


def plot_error_profiles(summary: pd.DataFrame, matched_tables: dict[str, pd.DataFrame]) -> None:
    order = summary["model_key"].tolist()
    metric_specs = [
        ("instance_rq", "Instance RQ"),
        ("instance_sq", "Instance SQ"),
        ("pixel_precision", "Pixel precision"),
        ("pixel_recall", "Pixel recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), sharex=False)
    axes = axes.ravel()
    for idx, (ax, (metric, title)) in enumerate(zip(axes, metric_specs)):
        data = [matched_tables[m][metric].to_numpy() for m in order]
        box = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": "#444444", "linewidth": 1.0},
            capprops={"color": "#444444", "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "edgecolor": "#333333"},
            flierprops={
                "marker": "o",
                "markersize": 2.0,
                "markerfacecolor": "#666666",
                "markeredgecolor": "#666666",
                "alpha": 0.12,
            },
        )
        for patch, model_key in zip(box["boxes"], order):
            patch.set_facecolor(MODEL_COLORS[model_key])
            patch.set_alpha(0.85)
        ax.set_title(title, pad=10)
        ax.set_xticks(range(1, len(order) + 1))
        if idx < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=18, ha="right")
            ax.set_xlabel("Model")
        ax.set_ylim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Error profile decomposition on matched patches", y=0.98, fontsize=13)
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.20, hspace=0.22)
    save_figure(fig, "error_profile_decomposition")


def make_qc_report(raw_counts: dict[str, int], matched_tables: dict[str, pd.DataFrame]) -> None:
    matched_patch_ids = sorted(set.intersection(*[set(df.index) for df in matched_tables.values()]))
    lines = [
        "Matched-patch QC summary",
        f"Matched patch count: {len(matched_patch_ids)}",
        "",
        "Raw evaluated counts after status == ok:",
    ]
    for model_key, count in raw_counts.items():
        lines.append(f"- {MODEL_LABELS[model_key]}: {count}")
    lines.append("")
    lines.append("Missing-value checks for matched key metrics:")
    for model_key, df in matched_tables.items():
        missing = int(df[KEY_METRICS].isna().sum().sum())
        lines.append(f"- {MODEL_LABELS[model_key]}: {missing} missing values across {len(KEY_METRICS)} metrics")
    (OUTPUT_DIR / "matched_patch_qc_summary.txt").write_text("\n".join(lines) + "\n")
    (OUTPUT_DIR / "matched_patch_ids.txt").write_text("\n".join(matched_patch_ids) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_common_style()

    raw_tables = {model_key: load_model_table(model_key) for model_key in MODEL_FILES}
    raw_counts = {model_key: int(len(df)) for model_key, df in raw_tables.items()}

    common_patch_ids = sorted(set.intersection(*[set(df.index) for df in raw_tables.values()]))
    matched_tables = {model_key: df.loc[common_patch_ids].copy() for model_key, df in raw_tables.items()}

    for model_key, df in matched_tables.items():
        if not df.index.equals(pd.Index(common_patch_ids)):
            raise ValueError(f"{model_key} index alignment mismatch")
        if df[KEY_METRICS].isna().any().any():
            raise ValueError(f"{model_key} has missing matched analysis metrics")

    make_qc_report(raw_counts, matched_tables)

    overall = make_overall_summary(matched_tables, raw_counts)
    overall.to_csv(OUTPUT_DIR / "overall_benchmark_summary.csv", index=False)
    write_pretty_table(overall)

    pq_summary = make_pq_distribution_summary(overall)
    pq_summary.to_csv(OUTPUT_DIR / "matched_patch_pq_distribution_summary.csv", index=False)
    plot_pq_distribution(overall, matched_tables)

    wins_df, pairwise_df = dominance_analysis(overall, matched_tables)
    wins_df.to_csv(OUTPUT_DIR / "patch_dominance_wins.csv", index=False)
    pairwise_df.to_csv(OUTPUT_DIR / "patch_dominance_pairwise_win_rates.csv", index=False)

    err_summary = error_profile_summary(overall)
    err_summary.to_csv(OUTPUT_DIR / "error_profile_summary.csv", index=False)
    plot_error_profiles(overall, matched_tables)


if __name__ == "__main__":
    main()
