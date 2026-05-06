from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd

from conic_visualization_utils import (
    DEFAULT_MODEL_ORDER,
    PredictionLocator,
    StoryCase,
    add_default_umap_context_columns,
    build_matched_patch_table,
    compute_embedding_projection,
    default_dataset_root,
    default_embedding_csv,
    default_metadata_csv,
    default_output_dir,
    load_embedding_table,
    plot_archetype_gallery,
    plot_dominant_class_atlas,
    plot_metric_caseboards,
    plot_patch_view_grid,
    plot_umap_patch_neighborhoods,
    plot_umap_scatter_grid,
    save_figure,
    select_dominant_class_exemplars,
    select_umap_fraction_connective_cases,
    select_umap_story_cases,
    ViewSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build story-ready CoNIC patch visualizations for the final report using "
            "local image/mask assets and matched evaluation tables."
        )
    )
    parser.add_argument("--eval-dir", type=Path, default=None, help="Directory containing *_evaluation.csv files.")
    parser.add_argument("--metadata-csv", type=Path, default=default_metadata_csv(), help="Path to embed_morph.csv.")
    parser.add_argument(
        "--embedding-csv",
        type=Path,
        default=default_embedding_csv(),
        help="Path to embed_morph_with_vectors.csv for the UMAP-backed figures.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help="Root directory for data/conic_lizard image and mask exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for saved PNG story figures.",
    )
    parser.add_argument(
        "--prediction-root",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help=(
            "Optional prediction output root for a model, for example "
            "--prediction-root cellpose_sam=/path/to/inference/benchmarking/conic_liz/cellpose_sam . "
            "When provided, the script can also render model-specific predicted overlays."
        ),
    )
    return parser.parse_args()


def parse_prediction_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction root arguments must use MODEL=PATH format, got: {value!r}")
        model_name, path_text = value.split("=", 1)
        roots[model_name.strip()] = Path(path_text).expanduser().resolve()
    return roots


def select_case_by_query(
    dataframe: pd.DataFrame,
    query: str,
    *,
    sort_columns: list[str],
    ascending: list[bool],
) -> str:
    filtered = dataframe.query(query).sort_values(sort_columns, ascending=ascending)
    if filtered.empty:
        raise ValueError(f"No rows matched query: {query}")
    return str(filtered.iloc[0]["sample_id"])


def select_story_cases(matched: pd.DataFrame) -> dict[str, list[StoryCase]]:
    hard_connective = select_case_by_query(
        matched,
        "story_label == 'connective-rich' and total_nuclei >= 10 and cellpose_sam__instance_pq >= 0.2",
        sort_columns=["pq_median", "cellpose_sam__instance_pq", "foreground_fraction"],
        ascending=[True, False, True],
    )
    easy_epithelial = select_case_by_query(
        matched,
        "story_label == 'epithelial-rich'",
        sort_columns=["pq_median", "foreground_fraction"],
        ascending=[False, False],
    )
    easy_mixed = select_case_by_query(
        matched,
        "story_label == 'mixed'",
        sort_columns=["pq_median", "foreground_fraction", "total_nuclei"],
        ascending=[False, False, False],
    )
    favorable_lymphocyte = select_case_by_query(
        matched,
        "story_label == 'lymphocyte-rich'",
        sort_columns=["pq_median", "foreground_fraction"],
        ascending=[False, False],
    )
    disagreement_lymphocyte = select_case_by_query(
        matched,
        "story_label == 'lymphocyte-rich' and total_nuclei >= 150",
        sort_columns=["pq_std", "pq_median"],
        ascending=[False, False],
    )

    return {
        "archetypes": [
            StoryCase(
                sample_id=hard_connective,
                title="Sparse connective",
                takeaway="Connective-heavy structure with very little recoverable foreground is the clearest hard regime.",
            ),
            StoryCase(
                sample_id=easy_epithelial,
                title="Epithelial gland",
                takeaway="Larger, visually obvious nuclei make the easier epithelial regime concrete.",
            ),
            StoryCase(
                sample_id=easy_mixed,
                title="Dense mixed field",
                takeaway="Dense nuclear burden can still be easy when the nuclei are clear and plentiful.",
            ),
            StoryCase(
                sample_id=favorable_lymphocyte,
                title="Lymphocyte-dense field",
                takeaway="A small lymphocyte-rich subgroup often remains segmentable despite high object counts.",
            ),
        ],
        "morphology": [
            StoryCase(
                sample_id=hard_connective,
                title="Sparse connective",
                takeaway="Low foreground fraction and low nuclei burden align with near-zero consensus PQ.",
            ),
            StoryCase(
                sample_id=easy_mixed,
                title="Dense mixed",
                takeaway="High nuclear burden and higher occupied area align with markedly better median PQ.",
            ),
        ],
        "caseboards": [
            StoryCase(
                sample_id=hard_connective,
                title="Case 1: hard connective patch",
                takeaway="Cellpose-SAM partially recovers this patch while the rest of the field largely collapses.",
            ),
            StoryCase(
                sample_id=disagreement_lymphocyte,
                title="Case 2: crowded lymphocyte patch",
                takeaway="Dense lymphocyte-rich patches can still separate the middle tier, especially CellSAM versus CellViT-SAM-H.",
            ),
        ],
    }


def story_case_table(case_groups: dict[str, list[StoryCase]]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for group_name, cases in case_groups.items():
        for case in cases:
            records.append(
                {
                    "group": group_name,
                    "sample_id": case.sample_id,
                    "title": case.title,
                    "takeaway": case.takeaway,
                }
            )
    return pd.DataFrame.from_records(records)


def select_major_dominant_classes(embedding_table: pd.DataFrame) -> list[str]:
    counts = (
        embedding_table["dominant_class"]
        .astype(str)
        .value_counts()
        .rename_axis("dominant_class")
        .reset_index(name="count")
    )
    counts = counts.loc[~counts["dominant_class"].isin({"none", "tie"})]
    counts = counts.loc[counts["count"].ge(30)]
    return counts["dominant_class"].astype(str).head(4).tolist()


def build_dominant_class_atlas_map(
    projection_frame: pd.DataFrame,
    *,
    dominant_classes: list[str],
) -> dict[str, list[str]]:
    atlas_map: dict[str, list[str]] = {}
    for dominant_class in dominant_classes:
        atlas_map[dominant_class] = select_dominant_class_exemplars(
            projection_frame,
            class_name=dominant_class,
            n_examples=3,
        )
    return atlas_map


def attach_umap_report_columns(projection_frame: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "sample_id",
        "pq_median",
        "pq_mean",
        "cellvit_sam__instance_pq",
    ]
    available_columns = [column for column in metric_columns if column in matched.columns]
    enriched = projection_frame.merge(
        matched[available_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    return add_default_umap_context_columns(enriched)


def build_figures(args: argparse.Namespace) -> list[Path]:
    matched = build_matched_patch_table(args.eval_dir, metadata_csv=args.metadata_csv)
    case_groups = select_story_cases(matched)
    saved_paths: list[Path] = []
    prediction_roots = parse_prediction_roots(args.prediction_root)
    prediction_locator = PredictionLocator(prediction_roots=prediction_roots) if prediction_roots else None
    embedding_table = load_embedding_table(args.embedding_csv)
    projection_frame = compute_embedding_projection(embedding_table)
    projection_frame = attach_umap_report_columns(projection_frame, matched)
    dominant_classes = select_major_dominant_classes(projection_frame)
    atlas_map = build_dominant_class_atlas_map(projection_frame, dominant_classes=dominant_classes)
    umap_cases = select_umap_story_cases(projection_frame)
    connective_umap_cases = select_umap_fraction_connective_cases(projection_frame)

    archetype_fig, _ = plot_archetype_gallery(
        case_groups["archetypes"],
        matched,
        dataset_root=args.dataset_root,
        suptitle="Patch archetypes behind the summary statistics",
        show_takeaways=False,
    )
    saved_paths.append(save_figure(archetype_fig, args.output_dir, "patch_archetype_gallery.png"))
    plt.close(archetype_fig)

    atlas_fig, _ = plot_dominant_class_atlas(
        atlas_map,
        dataset_root=args.dataset_root,
        suptitle="Major dominant classes as actual patches",
    )
    saved_paths.append(save_figure(atlas_fig, args.output_dir, "dominant_class_patch_atlas.png"))
    plt.close(atlas_fig)

    morphology_fig, _ = plot_patch_view_grid(
        [case.sample_id for case in case_groups["morphology"]],
        [
            ViewSpec(kind="image", title="Original patch"),
            ViewSpec(kind="gt_mask", title="Ground-truth mask"),
            ViewSpec(kind="gt_overlay", title="Ground-truth outline"),
        ],
        dataset_root=args.dataset_root,
        row_titles=[case.title for case in case_groups["morphology"]],
        suptitle="Morphology contrast: sparse connective versus dense mixed tissue",
    )
    saved_paths.append(save_figure(morphology_fig, args.output_dir, "morphology_contrast_grid.png"))
    plt.close(morphology_fig)

    umap_fig, _ = plot_umap_patch_neighborhoods(
        projection_frame,
        umap_cases,
        dataset_root=args.dataset_root,
        suptitle=None,
        show_scatter_title=False,
    )
    saved_paths.append(save_figure(umap_fig, args.output_dir, "umap_patch_neighborhoods.png"))
    plt.close(umap_fig)

    connective_umap_fig, _ = plot_umap_patch_neighborhoods(
        projection_frame,
        connective_umap_cases,
        dataset_root=args.dataset_root,
        color_column="fraction_connective_level",
        suptitle=None,
        show_scatter_title=False,
    )
    saved_paths.append(save_figure(connective_umap_fig, args.output_dir, "umap_fraction_connective_neighborhoods.png"))
    plt.close(connective_umap_fig)

    umap_grid_fig, _ = plot_umap_scatter_grid(
        projection_frame,
        [
            ("fraction_connective_level", "Connective fraction"),
            ("foreground_fraction_level", "Foreground fraction"),
            ("total_nuclei_level", "Total nuclei"),
            ("consensus_pq_tier", "Consensus PQ"),
        ],
        ncols=2,
    )
    saved_paths.append(save_figure(umap_grid_fig, args.output_dir, "umap_embedding_signal_grid.png"))
    plt.close(umap_grid_fig)

    caseboard_prediction_models = []
    if prediction_locator is not None:
        caseboard_prediction_models = [
            model_name for model_name in DEFAULT_MODEL_ORDER if model_name in prediction_locator.available_models()
        ]

    caseboard_fig, _ = plot_metric_caseboards(
        case_groups["caseboards"],
        matched,
        dataset_root=args.dataset_root,
        prediction_locator=prediction_locator,
        prediction_models=caseboard_prediction_models,
        suptitle="Representative patch caseboards with per-model instance PQ",
    )
    saved_paths.append(save_figure(caseboard_fig, args.output_dir, "model_caseboards.png"))
    plt.close(caseboard_fig)

    case_table = story_case_table(case_groups)
    case_table_path = args.output_dir / "story_case_selection.csv"
    case_table.to_csv(case_table_path, index=False)
    saved_paths.append(case_table_path.resolve())

    atlas_table = pd.DataFrame(
        [
            {"dominant_class": dominant_class, "sample_id": sample_id, "example_index": example_index + 1}
            for dominant_class, sample_ids in atlas_map.items()
            for example_index, sample_id in enumerate(sample_ids)
        ]
    )
    atlas_table_path = args.output_dir / "dominant_class_atlas_selection.csv"
    atlas_table.to_csv(atlas_table_path, index=False)
    saved_paths.append(atlas_table_path.resolve())

    umap_case_table = pd.DataFrame(
        [
            {"sample_id": case.sample_id, "title": case.title, "takeaway": case.takeaway}
            for case in umap_cases
        ]
    )
    umap_case_table_path = args.output_dir / "umap_story_selection.csv"
    umap_case_table.to_csv(umap_case_table_path, index=False)
    saved_paths.append(umap_case_table_path.resolve())

    connective_umap_case_table = pd.DataFrame(
        [
            {"sample_id": case.sample_id, "title": case.title, "takeaway": case.takeaway}
            for case in connective_umap_cases
        ]
    )
    connective_umap_case_table_path = args.output_dir / "umap_fraction_connective_selection.csv"
    connective_umap_case_table.to_csv(connective_umap_case_table_path, index=False)
    saved_paths.append(connective_umap_case_table_path.resolve())

    projection_save_columns = [
        "sample_id",
        "dominant_class",
        "dominant_fraction",
        "richness_label",
        "story_label",
        "fraction_connective",
        "fraction_connective_level",
        "foreground_fraction_level",
        "foreground_fraction",
        "total_nuclei_level",
        "total_nuclei",
        "mean_area_level",
        "mean_area",
        "pq_median",
        "pq_mean",
        "consensus_pq_tier",
        "umap_1",
        "umap_2",
        "pca_1",
        "pca_2",
    ]
    projection_output = projection_frame.loc[:, [column for column in projection_save_columns if column in projection_frame.columns]]
    projection_output_path = args.output_dir / "embedding_umap_projection.csv"
    projection_output.to_csv(projection_output_path, index=False)
    saved_paths.append(projection_output_path.resolve())
    return saved_paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = build_figures(args)
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
