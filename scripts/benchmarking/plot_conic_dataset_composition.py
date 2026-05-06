from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cell-seg-mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter


SOURCE_LABELS = {
    "consep": "CoNSeP",
    "crag": "CRAG",
    "dpath": "DPath",
    "glas": "GlaS",
    "pannuke": "PanNuke",
}

SOURCE_COLORS = {
    "consep": "#3D5A80",
    "crag": "#7FB069",
    "dpath": "#E09F3E",
    "glas": "#9C89B8",
    "pannuke": "#D1495B",
}

CELL_TYPE_LABELS = {
    "epithelial": "Epithelial",
    "connective": "Connective",
    "lymphocyte": "Lymphocyte",
    "plasma": "Plasma",
    "neutrophil": "Neutrophil",
    "eosinophil": "Eosinophil",
}

CELL_TYPE_COLORS = {
    "epithelial": "#E76F51",
    "connective": "#59A14F",
    "lymphocyte": "#4C78A8",
    "plasma": "#B279A2",
    "neutrophil": "#2A9D8F",
    "eosinophil": "#F4A261",
}

DEFAULT_MANIFEST = Path("data/conic_lizard/dataset_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/conic_liz/analysis/dataset_composition")
PNG_DPI = 450


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-style CoNIC/Lizard dataset composition figures."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the CoNIC/Lizard dataset manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2C2C2C",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=PNG_DPI)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def friendly_source(source: str) -> str:
    return SOURCE_LABELS.get(source, source)


def friendly_cell_type(cell_type: str) -> str:
    return CELL_TYPE_LABELS.get(cell_type, cell_type.replace("_", " ").title())


def format_count(value: float | int) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{int(round(value)):,}"


def setup_clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)


def annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#1F1F1F",
    )


def pie_autopct(min_pct: float = 4.0) -> callable:
    def _formatter(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= min_pct else ""

    return _formatter


def draw_donut_chart(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    colors: list[str],
    center_value: str,
    center_label: str,
) -> None:
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=pie_autopct(),
        pctdistance=0.76,
        wedgeprops={
            "width": 0.42,
            "edgecolor": "white",
            "linewidth": 1.6,
        },
        textprops={
            "fontsize": 9.3,
            "fontweight": "semibold",
        },
    )

    total = float(sum(values))
    for wedge, autotext, value, label in zip(wedges, autotexts, values, labels):
        if not autotext.get_text():
            continue
        red, green, blue, _ = wedge.get_facecolor()
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        autotext.set_color("#1F1F1F" if luminance > 0.62 else "white")
        autotext.set_fontsize(9.1)

    legend_labels = [
        f"{label}  {format_count(value)} ({100.0 * value / total:.1f}%)"
        for label, value in zip(labels, values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handlelength=1.1,
        handletextpad=0.8,
        labelspacing=1.0,
        borderaxespad=0.0,
    )
    ax.text(
        0.0,
        0.08,
        center_value,
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="semibold",
        color="#1F1F1F",
    )
    ax.text(
        0.0,
        -0.11,
        center_label,
        ha="center",
        va="center",
        fontsize=9.2,
        color="#5B5B5B",
    )
    ax.set_aspect("equal")


def load_manifest(manifest_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(manifest_path)
    if df.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    cell_type_columns = [
        column
        for column in df.columns
        if column not in {"sample_id", "conic_index", "image_path", "mask_path", "class_label_path", "image_height", "image_width", "count_total"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if "count_total" not in df.columns:
        raise ValueError("Manifest must include 'count_total'.")
    if not cell_type_columns:
        raise ValueError("Manifest does not include numeric cell-type count columns.")

    df = df.copy()
    df["source"] = df["sample_id"].astype(str).str.split("_").str[0].str.lower()
    df["source_label"] = df["source"].map(friendly_source)
    return df, cell_type_columns


def build_summaries(df: pd.DataFrame, cell_type_columns: list[str]) -> dict[str, object]:
    source_summary = (
        df.groupby("source", dropna=False)
        .agg(
            patches=("sample_id", "size"),
            total_cells=("count_total", "sum"),
        )
        .sort_values(["patches", "total_cells"], ascending=False)
    )
    source_summary["mean_cells_per_patch"] = source_summary["total_cells"] / source_summary["patches"]
    source_summary["source_label"] = [friendly_source(source) for source in source_summary.index]

    source_order = source_summary.index.tolist()

    cell_type_totals = df[cell_type_columns].sum().sort_values(ascending=False)
    cell_type_order = cell_type_totals.index.tolist()

    source_cell_totals = df.groupby("source")[cell_type_order].sum().reindex(source_order)
    source_cell_fractions = source_cell_totals.div(source_cell_totals.sum(axis=1), axis=0)
    patch_presence = (df[cell_type_order] > 0).assign(source=df["source"]).groupby("source")[cell_type_order].mean()
    patch_presence = patch_presence.reindex(source_order)

    return {
        "source_summary": source_summary,
        "source_order": source_order,
        "cell_type_totals": cell_type_totals,
        "cell_type_order": cell_type_order,
        "source_cell_totals": source_cell_totals,
        "source_cell_fractions": source_cell_fractions,
        "patch_presence": patch_presence,
    }


def plot_source_footprint(source_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    y_positions = np.arange(len(source_summary))
    labels = source_summary["source_label"].tolist()
    colors = [SOURCE_COLORS.get(source, "#808080") for source in source_summary.index]

    axes[0].barh(y_positions, source_summary["patches"], color=colors, height=0.66)
    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Patch count")
    axes[0].set_title("Source patch footprint")
    setup_clean_axis(axes[0])
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    for ypos, value in zip(y_positions, source_summary["patches"]):
        axes[0].text(value + source_summary["patches"].max() * 0.015, ypos, f"{int(value):,}", va="center", ha="left")

    axes[1].barh(y_positions, source_summary["total_cells"], color=colors, height=0.66)
    axes[1].set_yticks(y_positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Total annotated nuclei")
    axes[1].set_title("Source nuclei footprint")
    setup_clean_axis(axes[1])
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_count(value)))
    for ypos, row in enumerate(source_summary.itertuples()):
        axes[1].text(
            row.total_cells + source_summary["total_cells"].max() * 0.015,
            ypos,
            f"{format_count(row.total_cells)}  ({row.mean_cells_per_patch:.1f}/patch)",
            va="center",
            ha="left",
        )

    save_figure(fig, output_dir, "conic_lizard_source_footprint")


def plot_cell_type_totals(cell_type_totals: pd.Series, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    y_positions = np.arange(len(cell_type_totals))
    labels = [friendly_cell_type(cell_type) for cell_type in cell_type_totals.index]
    colors = [CELL_TYPE_COLORS.get(cell_type, "#808080") for cell_type in cell_type_totals.index]
    grand_total = float(cell_type_totals.sum())

    ax.barh(y_positions, cell_type_totals.values, color=colors, height=0.68)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Total annotated nuclei")
    ax.set_title("Overall cell-type distribution")
    setup_clean_axis(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_count(value)))

    for ypos, (cell_type, value) in enumerate(cell_type_totals.items()):
        ax.text(
            value + cell_type_totals.max() * 0.018,
            ypos,
            f"{format_count(value)}  ({100.0 * value / grand_total:.1f}%)",
            va="center",
            ha="left",
        )

    save_figure(fig, output_dir, "conic_lizard_cell_type_totals")


def plot_source_patch_count_pie(source_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    labels = source_summary["source_label"].tolist()
    values = source_summary["patches"].astype(float).tolist()
    colors = [SOURCE_COLORS.get(source, "#808080") for source in source_summary.index]
    draw_donut_chart(
        ax=ax,
        labels=labels,
        values=values,
        colors=colors,
        center_value=f"{int(sum(values)):,}",
        center_label="patches",
    )
    save_figure(fig, output_dir, "conic_lizard_source_patch_counts_pie")


def plot_cell_type_total_pie(cell_type_totals: pd.Series, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    labels = [friendly_cell_type(cell_type) for cell_type in cell_type_totals.index]
    values = cell_type_totals.astype(float).tolist()
    colors = [CELL_TYPE_COLORS.get(cell_type, "#808080") for cell_type in cell_type_totals.index]
    draw_donut_chart(
        ax=ax,
        labels=labels,
        values=values,
        colors=colors,
        center_value=format_count(sum(values)),
        center_label="annotated nuclei",
    )
    save_figure(fig, output_dir, "conic_lizard_cell_type_totals_pie")


def plot_source_cell_type_proportions(
    source_cell_fractions: pd.DataFrame,
    source_summary: pd.DataFrame,
    cell_type_order: list[str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.4), constrained_layout=True)
    y_positions = np.arange(len(source_cell_fractions))
    labels = source_summary["source_label"].tolist()
    left = np.zeros(len(source_cell_fractions), dtype=float)

    for cell_type in cell_type_order:
        values = source_cell_fractions[cell_type].values
        ax.barh(
            y_positions,
            values,
            left=left,
            color=CELL_TYPE_COLORS.get(cell_type, "#808080"),
            edgecolor="white",
            linewidth=1.0,
            height=0.68,
            label=friendly_cell_type(cell_type),
        )
        left += values

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Within-source share of annotated nuclei")
    ax.set_title("Cell-type composition within each source")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    setup_clean_axis(ax)
    ax.legend(
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.2,
        columnspacing=1.2,
    )

    save_figure(fig, output_dir, "conic_lizard_source_cell_type_proportions")


def plot_patch_presence_heatmap(
    patch_presence: pd.DataFrame,
    source_summary: pd.DataFrame,
    cell_type_order: list[str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    data = patch_presence[cell_type_order].values
    image = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(cell_type_order)), [friendly_cell_type(cell_type) for cell_type in cell_type_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(source_summary)), source_summary["source_label"].tolist())
    ax.set_title("Patch-level class prevalence")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Source dataset")

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            text_color = "white" if value >= 0.55 else "#1F1F1F"
            ax.text(col, row, f"{value * 100:.0f}%", ha="center", va="center", color=text_color, fontsize=8.6)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Share of patches containing at least one cell")
    colorbar.formatter = PercentFormatter(xmax=1.0, decimals=0)
    colorbar.update_ticks()

    save_figure(fig, output_dir, "conic_lizard_source_cell_type_prevalence_heatmap")


def plot_bubble_matrix(
    source_cell_totals: pd.DataFrame,
    source_cell_fractions: pd.DataFrame,
    source_summary: pd.DataFrame,
    cell_type_order: list[str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

    counts = source_cell_totals[cell_type_order].values.astype(float)
    fractions = source_cell_fractions[cell_type_order].values.astype(float)
    positive_counts = counts[counts > 0]
    min_count = float(positive_counts.min())
    max_count = float(positive_counts.max())
    sqrt_counts = np.sqrt(counts)
    sizes = np.interp(sqrt_counts, [np.sqrt(min_count), np.sqrt(max_count)], [120.0, 2100.0])
    norm = Normalize(vmin=0.0, vmax=float(fractions.max()))

    y_positions = np.arange(len(source_summary))
    x_positions = np.arange(len(cell_type_order))

    for row, y in enumerate(y_positions):
        for col, x in enumerate(x_positions):
            ax.scatter(
                x,
                y,
                s=sizes[row, col],
                c=[fractions[row, col]],
                cmap="cividis",
                norm=norm,
                edgecolors="white",
                linewidths=1.0,
            )

    ax.set_xticks(x_positions, [friendly_cell_type(cell_type) for cell_type in cell_type_order], rotation=25, ha="right")
    ax.set_yticks(y_positions, source_summary["source_label"].tolist())
    ax.set_xlim(-0.6, len(cell_type_order) - 0.4)
    ax.set_ylim(len(source_summary) - 0.4, -0.6)
    ax.set_title("Source-by-cell-type bubble matrix")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Source dataset")
    ax.grid(color="#E1E1E1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap="cividis")
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Within-source cell-type share")
    colorbar.formatter = PercentFormatter(xmax=1.0, decimals=0)
    colorbar.update_ticks()

    legend_counts = [1_000, 10_000, 50_000]
    legend_sizes = np.interp(
        np.sqrt(legend_counts),
        [np.sqrt(min_count), np.sqrt(max_count)],
        [120.0, 2100.0],
    )
    handles = [
        ax.scatter([], [], s=size, color="#9E9E9E", edgecolors="white", linewidths=1.0)
        for size in legend_sizes
    ]
    labels = [format_count(value) for value in legend_counts]
    ax.legend(
        handles,
        labels,
        title="Absolute nuclei",
        frameon=False,
        scatterpoints=1,
        loc="upper left",
        bbox_to_anchor=(1.14, 1.0),
        labelspacing=1.3,
        borderpad=0.3,
    )

    save_figure(fig, output_dir, "conic_lizard_source_cell_type_bubble_matrix")


def plot_mosaic(
    source_cell_totals: pd.DataFrame,
    source_summary: pd.DataFrame,
    cell_type_order: list[str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    grand_total = float(source_cell_totals.values.sum())
    current_x = 0.0
    centers: list[float] = []
    widths: list[float] = []

    for source in source_summary.index:
        source_total = float(source_cell_totals.loc[source].sum())
        width = source_total / grand_total
        current_y = 0.0
        centers.append(current_x + width / 2.0)
        widths.append(width)

        for cell_type in cell_type_order:
            cell_value = float(source_cell_totals.loc[source, cell_type])
            height = 0.0 if source_total == 0 else cell_value / source_total
            ax.add_patch(
                Rectangle(
                    (current_x, current_y),
                    width,
                    height,
                    facecolor=CELL_TYPE_COLORS.get(cell_type, "#808080"),
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            current_y += height

        ax.add_patch(
            Rectangle(
                (current_x, 0.0),
                width,
                1.0,
                fill=False,
                edgecolor="#222222",
                linewidth=0.9,
            )
        )
        current_x += width

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks([])
    ax.set_xlabel("Source dataset (bar width encodes share of all nuclei)")
    ax.set_ylabel("Within-source cell-type share")
    ax.set_title("Single-view composition mosaic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)

    for center, width, row in zip(centers, widths, source_summary.itertuples()):
        if width >= 0.06:
            ax.text(
                center,
                -0.055,
                row.source_label,
                ha="center",
                va="top",
                fontsize=9.2,
                color="#2F2F2F",
            )
            continue
        ax.text(
            center,
            1.01,
            row.source_label,
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#2F2F2F",
            rotation=90,
        )

    ax.legend(
        [Rectangle((0, 0), 1, 1, facecolor=CELL_TYPE_COLORS[cell_type]) for cell_type in cell_type_order],
        [friendly_cell_type(cell_type) for cell_type in cell_type_order],
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.14),
        columnspacing=1.3,
        handlelength=1.1,
    )

    save_figure(fig, output_dir, "conic_lizard_source_cell_type_mosaic")


def plot_multipanel(
    source_summary: pd.DataFrame,
    cell_type_totals: pd.Series,
    source_cell_fractions: pd.DataFrame,
    patch_presence: pd.DataFrame,
    cell_type_order: list[str],
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(12.4, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1.0, 1.06])

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    y_positions = np.arange(len(source_summary))
    source_labels = source_summary["source_label"].tolist()
    source_colors = [SOURCE_COLORS.get(source, "#808080") for source in source_summary.index]

    ax_a.barh(y_positions, source_summary["patches"], color=source_colors, height=0.66)
    ax_a.set_yticks(y_positions, source_labels)
    ax_a.invert_yaxis()
    ax_a.set_title("Source patch counts")
    ax_a.set_xlabel("Patches")
    setup_clean_axis(ax_a)
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    annotate_panel(ax_a, "A")

    y_positions_cell = np.arange(len(cell_type_totals))
    ax_b.barh(
        y_positions_cell,
        cell_type_totals.values,
        color=[CELL_TYPE_COLORS.get(cell_type, "#808080") for cell_type in cell_type_totals.index],
        height=0.68,
    )
    ax_b.set_yticks(y_positions_cell, [friendly_cell_type(cell_type) for cell_type in cell_type_totals.index])
    ax_b.invert_yaxis()
    ax_b.set_title("Overall cell-type totals")
    ax_b.set_xlabel("Annotated nuclei")
    setup_clean_axis(ax_b)
    ax_b.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_count(value)))
    annotate_panel(ax_b, "B")

    left = np.zeros(len(source_cell_fractions), dtype=float)
    for cell_type in cell_type_order:
        ax_c.barh(
            y_positions,
            source_cell_fractions[cell_type].values,
            left=left,
            color=CELL_TYPE_COLORS.get(cell_type, "#808080"),
            edgecolor="white",
            linewidth=0.9,
            height=0.68,
            label=friendly_cell_type(cell_type),
        )
        left += source_cell_fractions[cell_type].values
    ax_c.set_yticks(y_positions, source_labels)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 1)
    ax_c.set_title("Within-source cell-type composition")
    ax_c.set_xlabel("Share of nuclei")
    ax_c.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    setup_clean_axis(ax_c)
    annotate_panel(ax_c, "C")

    heatmap_values = patch_presence[cell_type_order].values
    image = ax_d.imshow(heatmap_values, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax_d.set_xticks(
        np.arange(len(cell_type_order)),
        [friendly_cell_type(cell_type) for cell_type in cell_type_order],
        rotation=25,
        ha="right",
    )
    ax_d.set_yticks(np.arange(len(source_labels)), source_labels)
    ax_d.set_title("Patch-level class prevalence")
    ax_d.set_xlabel("Cell type")
    for row in range(heatmap_values.shape[0]):
        for col in range(heatmap_values.shape[1]):
            value = heatmap_values[row, col]
            ax_d.text(
                col,
                row,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "#1F1F1F",
                fontsize=8.2,
            )
    annotate_panel(ax_d, "D")
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.046, pad=0.03)
    colorbar.formatter = PercentFormatter(xmax=1.0, decimals=0)
    colorbar.update_ticks()

    fig.suptitle("CoNIC/Lizard dataset composition overview", fontsize=14, fontweight="semibold")
    fig.text(
        0.5,
        0.015,
        "Tissue labels are not present in the current export manifest, so source dataset is used as the available provenance field.",
        ha="center",
        va="center",
        fontsize=9,
        color="#575757",
    )

    save_figure(fig, output_dir, "conic_lizard_dataset_composition_multipanel")


def main() -> None:
    args = parse_args()
    set_plot_style()

    df, cell_type_columns = load_manifest(args.manifest)
    summaries = build_summaries(df, cell_type_columns)

    output_dir = args.output_dir
    plot_source_footprint(summaries["source_summary"], output_dir)
    plot_cell_type_totals(summaries["cell_type_totals"], output_dir)
    plot_source_patch_count_pie(summaries["source_summary"], output_dir)
    plot_cell_type_total_pie(summaries["cell_type_totals"], output_dir)
    plot_source_cell_type_proportions(
        summaries["source_cell_fractions"],
        summaries["source_summary"],
        summaries["cell_type_order"],
        output_dir,
    )
    plot_patch_presence_heatmap(
        summaries["patch_presence"],
        summaries["source_summary"],
        summaries["cell_type_order"],
        output_dir,
    )
    plot_bubble_matrix(
        summaries["source_cell_totals"],
        summaries["source_cell_fractions"],
        summaries["source_summary"],
        summaries["cell_type_order"],
        output_dir,
    )
    plot_mosaic(
        summaries["source_cell_totals"],
        summaries["source_summary"],
        summaries["cell_type_order"],
        output_dir,
    )
    plot_multipanel(
        summaries["source_summary"],
        summaries["cell_type_totals"],
        summaries["source_cell_fractions"],
        summaries["patch_presence"],
        summaries["cell_type_order"],
        output_dir,
    )


if __name__ == "__main__":
    main()
