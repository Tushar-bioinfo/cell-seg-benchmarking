from __future__ import annotations

import json
import textwrap
from pathlib import Path


def markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


NOTEBOOK_PATH = Path(
    "/Users/tusharsingh/Library/CloudStorage/Box-Box/internship-2026/cell-seg/bencharmking-v1/scripts/benchmarking/monusac_evaluation_visualizer.ipynb"
)


cells: list[dict[str, object]] = [
    markdown_cell(
        """
        # MoNuSAC Segmentation Evaluation Visualizer

        Use this notebook to inspect segmentation evaluation outputs and compare model predictions for the same sample.

        Workflow:
        1. Keep or edit the evaluation root in the configuration cell.
        2. Load the evaluation table and inspect the `table_row` values.
        3. Pick one or more rows using `56`, `[3, 56, 6]`, or `"3:23"`.
        4. Plot a clean side-by-side comparison across the discovered models.

        By default the notebook looks for `benchmarking/monusac/_evaluation`. If that folder is missing, it automatically falls back to the repo-standard `inference/benchmarking/monusac/_evaluation`.
        """
    ),
    markdown_cell(
        """
        ## Configuration

        Keep the defaults for the standard four-model benchmark, or edit the paths and figure settings here before running the rest of the notebook.
        """
    ),
    code_cell(
        """
        import ast
        import re
        from pathlib import Path
        from typing import Any

        import imageio.v3 as iio
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from PIL import Image

        if "display" not in globals():
            def display(value):
                print(value)

        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.titlesize": 11,
                "axes.titleweight": "semibold",
            }
        )

        DEFAULT_EVALUATION_ROOT = Path("benchmarking") / "monusac" / "_evaluation"
        FALLBACK_EVALUATION_ROOTS = (
            Path("inference") / "benchmarking" / "monusac" / "_evaluation",
        )
        DEFAULT_COMBINED_CSV_NAMES = ("all_models_evaluation.csv", "evaluation.csv")
        DEFAULT_MODEL_ORDER = ["cellpose_sam", "cellsam", "cellvit_sam", "stardist"]
        MAX_MODELS_PER_FIGURE = 4
        DEFAULT_ROW_SELECTION = 0
        DEFAULT_STATUS_FOR_PLOTTING = "ok"
        MASK_COLOR_SEED = 7
        FIGURE_WIDTH = 14
        ROW_HEIGHT = 4.0

        IMAGE_SUFFIX = "_image.png"
        MASK_SUFFIX = "_mask.png"
        ROW_ID_COLUMNS = (
            "patch_id",
            "relative_image_path",
            "image_path",
            "predicted_mask_relative_path",
            "predicted_mask_name",
            "predicted_mask_path",
        )
        PREVIEW_COLUMNS = [
            "table_row",
            "model_name",
            "status",
            "patient_id",
            "image_id",
            "match_key",
            "pixel_dice",
            "instance_pq",
            "gt_relative_path",
            "pred_relative_path",
            "pred_file_name",
            "source_csv",
        ]

        SAVE_FIGURES = False
        SAVE_DIRECTORY = None
        SAVE_DPI = 200

        PREDICTION_MANIFEST_CACHE: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
        """
    ),
    markdown_cell(
        """
        ## File Discovery And Evaluation Loading

        This cell resolves the evaluation root, discovers the available evaluation CSVs, and builds one combined table with a stable `table_row` column for row selection.
        """
    ),
    code_cell(
        """
        def find_repo_root(start: Path | None = None) -> Path:
            start_path = (start or Path.cwd()).resolve()
            for candidate in (start_path, *start_path.parents):
                if (candidate / ".git").exists():
                    return candidate
            raise FileNotFoundError("Could not find the repository root from the current working directory.")


        def resolve_evaluation_root(
            requested_root: str | Path,
            *,
            repo_root: Path,
            fallback_paths: tuple[Path, ...] = (),
        ) -> Path:
            requested = Path(requested_root).expanduser()
            candidate_paths: list[Path] = []

            if requested.is_absolute():
                candidate_paths.append(requested)
            else:
                candidate_paths.extend(
                    [
                        repo_root / requested,
                        Path.cwd().resolve() / requested,
                    ]
                )

            for fallback in fallback_paths:
                fallback_path = Path(fallback).expanduser()
                if fallback_path.is_absolute():
                    candidate_paths.append(fallback_path)
                else:
                    candidate_paths.extend(
                        [
                            repo_root / fallback_path,
                            Path.cwd().resolve() / fallback_path,
                        ]
                    )

            checked: set[Path] = set()
            for candidate in candidate_paths:
                resolved = candidate.resolve()
                if resolved in checked:
                    continue
                checked.add(resolved)
                if resolved.exists():
                    return resolved

            rendered_candidates = "\\n".join(f"  - {path.resolve()}" for path in candidate_paths)
            raise FileNotFoundError(
                "Could not resolve the evaluation root. Checked:\\n"
                f"{rendered_candidates}"
            )


        def read_evaluation_frame(csv_path: Path) -> pd.DataFrame:
            frame = pd.read_csv(csv_path)
            frame = frame.copy()
            frame["source_csv"] = csv_path.name

            if "model_name" not in frame.columns:
                inferred_model_name = csv_path.name.removesuffix("_evaluation.csv").removesuffix(".csv")
                frame["model_name"] = inferred_model_name
            else:
                inferred_model_name = csv_path.name.removesuffix("_evaluation.csv").removesuffix(".csv")
                missing_model_name = frame["model_name"].isna() | frame["model_name"].astype(str).str.strip().eq("")
                frame.loc[missing_model_name, "model_name"] = inferred_model_name

            return frame


        def load_evaluation_table(evaluation_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
            combined_csv_path = next(
                (
                    evaluation_root / csv_name
                    for csv_name in DEFAULT_COMBINED_CSV_NAMES
                    if (evaluation_root / csv_name).is_file()
                ),
                None,
            )

            if combined_csv_path is not None:
                evaluation_table = read_evaluation_frame(combined_csv_path)
                evaluation_sources = pd.DataFrame(
                    [
                        {
                            "source_csv": combined_csv_path.name,
                            "csv_path": str(combined_csv_path),
                            "row_count": len(evaluation_table),
                            "is_combined": True,
                        }
                    ]
                )
                return evaluation_table, evaluation_sources, combined_csv_path

            model_csv_paths = sorted(
                path
                for path in evaluation_root.rglob("*_evaluation.csv")
                if path.is_file()
            )
            if not model_csv_paths:
                raise FileNotFoundError(
                    f"No evaluation CSV files were found under {evaluation_root}."
                )

            evaluation_frames = [read_evaluation_frame(csv_path) for csv_path in model_csv_paths]
            evaluation_table = pd.concat(evaluation_frames, ignore_index=True, sort=False)
            evaluation_sources = pd.DataFrame(
                [
                    {
                        "source_csv": csv_path.name,
                        "csv_path": str(csv_path),
                        "row_count": len(frame),
                        "is_combined": False,
                    }
                    for csv_path, frame in zip(model_csv_paths, evaluation_frames)
                ]
            )
            return evaluation_table, evaluation_sources, None


        REPO_ROOT = find_repo_root()
        EVALUATION_ROOT = resolve_evaluation_root(
            DEFAULT_EVALUATION_ROOT,
            repo_root=REPO_ROOT,
            fallback_paths=FALLBACK_EVALUATION_ROOTS,
        )
        PREDICTION_ROOT = EVALUATION_ROOT.parent

        evaluation_table, evaluation_sources, combined_csv_path = load_evaluation_table(EVALUATION_ROOT)
        evaluation_table = evaluation_table.copy()
        evaluation_table.insert(0, "table_row", np.arange(len(evaluation_table), dtype=int))
        available_preview_columns = [column for column in PREVIEW_COLUMNS if column in evaluation_table.columns]

        print(f"Repository root: {REPO_ROOT}")
        print(f"Evaluation root: {EVALUATION_ROOT}")
        print(f"Prediction root: {PREDICTION_ROOT}")
        if combined_csv_path is not None:
            print(f"Using combined evaluation CSV: {combined_csv_path.name}")
        else:
            print("Using concatenated per-model evaluation CSV files.")

        if "status" in evaluation_table.columns:
            print("\\nRow counts by status:")
            display(evaluation_table["status"].value_counts(dropna=False).rename("rows").to_frame())

        print("\\nDiscovered evaluation files:")
        display(evaluation_sources)

        print("\\nTable preview:")
        display(evaluation_table[available_preview_columns].head(20))
        """
    ),
    markdown_cell(
        """
        ## Helper Functions

        These helpers handle path resolution, prediction-manifest lookup, image and mask loading, deterministic mask coloring, and red outline overlays. The row-ID logic matches `scripts/benchmarking/monusac_segmentation_evaluation.py`, and the image/mask rendering logic follows the repo's inference and visualization utilities.
        """
    ),
    code_cell(
        """
        def first_non_empty(row: pd.Series | dict[str, Any], columns: list[str] | tuple[str, ...]) -> str:
            for column in columns:
                value = row.get(column, "")
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if text and text.lower() != "nan":
                    return text
            return ""


        def manifest_row_image_id(manifest_row: pd.Series) -> str:
            # Keep the same image-id priority used by monusac_segmentation_evaluation.py.
            return first_non_empty(manifest_row, list(ROW_ID_COLUMNS))


        def resolve_recorded_path(
            path_value: Any,
            *,
            base_dir: Path | None = None,
            repo_root: Path | None = None,
        ) -> Path | None:
            if path_value is None or pd.isna(path_value):
                return None

            raw_text = str(path_value).strip()
            if not raw_text:
                return None

            raw_path = Path(raw_text).expanduser()
            candidate_paths: list[Path] = []

            if raw_path.is_absolute():
                candidate_paths.append(raw_path)
            else:
                if base_dir is not None:
                    candidate_paths.append(base_dir / raw_path)
                if repo_root is not None:
                    candidate_paths.append(repo_root / raw_path)
                candidate_paths.append(Path.cwd().resolve() / raw_path)
                candidate_paths.append(raw_path)

            checked: set[Path] = set()
            for candidate in candidate_paths:
                resolved = candidate.resolve()
                if resolved in checked:
                    continue
                checked.add(resolved)
                if resolved.exists():
                    return resolved

            return None


        def derive_image_path_from_mask(mask_path: Path) -> Path | None:
            candidate_names: list[str] = []

            if mask_path.name.endswith(MASK_SUFFIX):
                candidate_names.append(mask_path.name[: -len(MASK_SUFFIX)] + IMAGE_SUFFIX)

            if mask_path.stem.endswith("_mask"):
                candidate_names.append(mask_path.stem[: -len("_mask")] + "_image" + mask_path.suffix)

            candidate_names.append(mask_path.stem + "_image" + mask_path.suffix)
            candidate_names.append(mask_path.stem.replace("_mask", "_image") + mask_path.suffix)

            checked: set[Path] = set()
            for candidate_name in candidate_names:
                candidate_path = (mask_path.parent / candidate_name).resolve()
                if candidate_path in checked:
                    continue
                checked.add(candidate_path)
                if candidate_path.exists():
                    return candidate_path

            return None


        def load_prediction_manifest(model_name: str, prediction_root: Path) -> tuple[Path, pd.DataFrame]:
            cache_key = (str(prediction_root.resolve()), model_name)
            if cache_key in PREDICTION_MANIFEST_CACHE:
                return PREDICTION_MANIFEST_CACHE[cache_key]

            manifest_path = (prediction_root / model_name / "predictions.csv").resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing predictions.csv for model '{model_name}': {manifest_path}")

            manifest_table = pd.read_csv(manifest_path).copy()
            manifest_table["_manifest_image_id"] = manifest_table.apply(manifest_row_image_id, axis=1)
            manifest_table["_manifest_row"] = np.arange(len(manifest_table), dtype=int)
            PREDICTION_MANIFEST_CACHE[cache_key] = (manifest_path, manifest_table)
            return manifest_path, manifest_table


        def find_prediction_manifest_row(
            evaluation_row: pd.Series,
            *,
            prediction_root: Path,
        ) -> tuple[Path | None, pd.Series | None]:
            model_name = first_non_empty(evaluation_row, ["model_name"])
            image_id = first_non_empty(evaluation_row, ["image_id", "match_key"])

            if not model_name or not image_id:
                return None, None

            try:
                manifest_path, manifest_table = load_prediction_manifest(model_name, prediction_root)
            except FileNotFoundError:
                return None, None

            matching_rows = manifest_table.loc[manifest_table["_manifest_image_id"].astype(str) == image_id].copy()

            pred_relative_path = first_non_empty(evaluation_row, ["pred_relative_path"])
            if len(matching_rows) > 1 and pred_relative_path and "predicted_mask_relative_path" in matching_rows.columns:
                narrowed_rows = matching_rows.loc[
                    matching_rows["predicted_mask_relative_path"].astype(str) == pred_relative_path
                ]
                if not narrowed_rows.empty:
                    matching_rows = narrowed_rows

            pred_path = first_non_empty(evaluation_row, ["pred_path"])
            if len(matching_rows) > 1 and pred_path and "predicted_mask_path" in matching_rows.columns:
                narrowed_rows = matching_rows.loc[
                    matching_rows["predicted_mask_path"].astype(str) == pred_path
                ]
                if not narrowed_rows.empty:
                    matching_rows = narrowed_rows

            if matching_rows.empty:
                return manifest_path, None

            return manifest_path, matching_rows.iloc[0]


        def resolve_source_image_path(
            evaluation_row: pd.Series,
            *,
            prediction_root: Path,
            repo_root: Path,
        ) -> Path:
            manifest_path, manifest_row = find_prediction_manifest_row(
                evaluation_row,
                prediction_root=prediction_root,
            )

            if manifest_row is not None and manifest_path is not None:
                for column in ("source_image_path", "image_path", "patch_image_path", "relative_image_path"):
                    resolved = resolve_recorded_path(
                        manifest_row.get(column),
                        base_dir=manifest_path.parent,
                        repo_root=repo_root,
                    )
                    if resolved is not None and resolved.exists():
                        return resolved

            gt_path = resolve_recorded_path(evaluation_row.get("gt_path"), repo_root=repo_root)
            if gt_path is not None and gt_path.exists():
                derived_image_path = derive_image_path_from_mask(gt_path)
                if derived_image_path is not None and derived_image_path.exists():
                    return derived_image_path

            raise FileNotFoundError(
                "Could not resolve the original image for table row "
                f"{evaluation_row.get('table_row')}."
            )


        def load_rgb_image(path: str | Path) -> np.ndarray:
            # Mirrors benchmark_inference_utils.read_rgb_image: always load as RGB uint8.
            resolved_path = Path(path).expanduser().resolve()
            with Image.open(resolved_path) as handle:
                return np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()


        def safely_reduce_loaded_mask(array: np.ndarray, path: Path) -> np.ndarray:
            if array.ndim == 2:
                return array

            if array.ndim != 3:
                raise ValueError(f"Mask file must be 2D, got shape {array.shape} for {path}")

            squeezed = np.squeeze(array)
            if squeezed.ndim == 2:
                return np.asarray(squeezed)

            if array.shape[-1] in (3, 4):
                rgb = array[..., :3]
                if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
                    return np.asarray(rgb[..., 0])

            if array.shape[0] in (3, 4):
                rgb = array[:3, ...]
                if np.array_equal(rgb[0, ...], rgb[1, ...]) and np.array_equal(rgb[0, ...], rgb[2, ...]):
                    return np.asarray(rgb[0, ...])

            raise ValueError(f"Mask file must decode to a 2D label image, got shape {array.shape} for {path}")


        def load_label_mask(path: str | Path) -> np.ndarray:
            resolved_path = Path(path).expanduser().resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Mask file not found: {resolved_path}")

            loaded = iio.imread(resolved_path)
            mask = safely_reduce_loaded_mask(np.asarray(loaded), resolved_path)
            if mask.size == 0:
                raise ValueError(f"Mask file is empty: {resolved_path}")
            return np.asarray(mask)


        def colorize_instance_mask(instance_mask: np.ndarray, seed: int = MASK_COLOR_SEED) -> np.ndarray:
            labels = np.unique(instance_mask)
            labels = labels[labels > 0]
            colored_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)

            for label in labels:
                rng = np.random.default_rng(seed + int(label))
                color = rng.integers(32, 224, size=3, dtype=np.uint8)
                colored_mask[instance_mask == label] = color

            return colored_mask


        def find_instance_boundaries(instance_mask: np.ndarray) -> np.ndarray:
            # Matches the boundary logic used in scripts/benchmarking/monusac_visualization_utils.py.
            boundaries = np.zeros(instance_mask.shape, dtype=bool)
            boundaries[1:, :] |= instance_mask[1:, :] != instance_mask[:-1, :]
            boundaries[:-1, :] |= instance_mask[:-1, :] != instance_mask[1:, :]
            boundaries[:, 1:] |= instance_mask[:, 1:] != instance_mask[:, :-1]
            boundaries[:, :-1] |= instance_mask[:, :-1] != instance_mask[:, 1:]
            boundaries &= instance_mask > 0
            return boundaries


        def draw_red_outlines(image: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
            overlay = np.asarray(image, dtype=np.uint8).copy()
            if overlay.ndim == 2:
                overlay = np.repeat(overlay[..., None], 3, axis=-1)

            boundaries = find_instance_boundaries(instance_mask)
            overlay[boundaries] = np.array([255, 0, 0], dtype=np.uint8)
            return overlay


        def build_mask_visualization(instance_mask: np.ndarray) -> np.ndarray:
            mask_visual = colorize_instance_mask(instance_mask)
            boundaries = find_instance_boundaries(instance_mask)
            mask_visual[boundaries] = np.array([255, 255, 255], dtype=np.uint8)
            return mask_visual


        def metric_summary(row: pd.Series) -> str:
            metric_parts: list[str] = []
            metric_mapping = {
                "pixel_dice": "Dice",
                "instance_pq": "PQ",
                "instance_rq": "RQ",
                "pixel_precision": "Precision",
            }

            for column_name, label in metric_mapping.items():
                if column_name in row.index and not pd.isna(row[column_name]):
                    try:
                        metric_parts.append(f"{label}={float(row[column_name]):.3f}")
                    except (TypeError, ValueError):
                        continue

            return " | ".join(metric_parts)


        def sanitize_for_filename(text: str) -> str:
            sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
            return sanitized[:120] or "comparison"


        def render_message_panel(ax: plt.Axes, title: str, message: str) -> None:
            ax.set_facecolor("#f5f5f5")
            ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                fontsize=10,
                wrap=True,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        """
    ),
    markdown_cell(
        """
        ## Row Selection Logic

        Select rows using the stable `table_row` values shown in the preview table. The range syntax is intentionally beginner-friendly: `"3:23"` means rows 3 through 23 inclusive.
        """
    ),
    code_cell(
        """
        def parse_row_selection(selection: Any, available_rows: pd.Series | list[int]) -> list[int]:
            available_list = [int(value) for value in list(available_rows)]
            available_set = set(available_list)
            minimum_row = min(available_list)
            maximum_row = max(available_list)

            if isinstance(selection, slice):
                start = minimum_row if selection.start is None else int(selection.start)
                stop = maximum_row if selection.stop is None else int(selection.stop)
                step = 1 if selection.step is None else int(selection.step)
                stop_inclusive = stop + (1 if step > 0 else -1)
                raw_items = list(range(start, stop_inclusive, step))
            elif isinstance(selection, (int, np.integer)):
                raw_items = [int(selection)]
            elif isinstance(selection, str):
                text = selection.strip()
                if not text:
                    raise ValueError("Row selection cannot be empty.")

                if text.startswith("[") and text.endswith("]"):
                    raw_items = [int(item) for item in ast.literal_eval(text)]
                elif ":" in text:
                    parts = [part.strip() for part in text.split(":")]
                    if len(parts) not in {2, 3}:
                        raise ValueError(
                            "Range selection must use 'start:stop' or 'start:stop:step'."
                        )

                    start = minimum_row if parts[0] == "" else int(parts[0])
                    stop = maximum_row if parts[1] == "" else int(parts[1])
                    step = 1 if len(parts) == 2 or parts[2] == "" else int(parts[2])
                    if step == 0:
                        raise ValueError("Range step cannot be zero.")
                    if (stop - start) * step < 0:
                        raise ValueError("The provided range direction does not match the step.")

                    stop_inclusive = stop + (1 if step > 0 else -1)
                    raw_items = list(range(start, stop_inclusive, step))
                elif "," in text:
                    raw_items = [int(part.strip()) for part in text.split(",") if part.strip()]
                else:
                    raw_items = [int(text)]
            elif isinstance(selection, (list, tuple, set, range, np.ndarray, pd.Series)):
                raw_items = [int(item) for item in list(selection)]
            else:
                raise TypeError(
                    "Unsupported row selection type. Use an int, a list, a range string, or a Python slice."
                )

            normalized_rows: list[int] = []
            seen_rows: set[int] = set()

            for item in raw_items:
                row_number = int(item)
                if row_number not in available_set:
                    raise IndexError(
                        f"Row {row_number} is not present in the evaluation table."
                    )
                if row_number in seen_rows:
                    continue
                seen_rows.add(row_number)
                normalized_rows.append(row_number)

            return normalized_rows


        def select_evaluation_rows(
            evaluation_table: pd.DataFrame,
            selection: Any,
        ) -> pd.DataFrame:
            selected_table_rows = parse_row_selection(selection, evaluation_table["table_row"])
            table_lookup = evaluation_table.set_index("table_row", drop=False)
            return table_lookup.loc[selected_table_rows].reset_index(drop=True)
        """
    ),
    markdown_cell(
        """
        ## Plotting Function

        Each selected evaluation row becomes one comparison figure. The notebook uses `match_key` or `image_id` to gather the same sample across models, then renders:

        - left column: prediction outline overlay on the original image
        - right column: predicted mask visualization

        When `DEFAULT_MODEL_ORDER` contains four model names, the figure stays in the requested `2 columns x 4 rows` layout even if one model is missing.
        """
    ),
    code_cell(
        """
        def find_comparison_rows(
            anchor_row: pd.Series,
            evaluation_table: pd.DataFrame,
            *,
            status: str | None = DEFAULT_STATUS_FOR_PLOTTING,
        ) -> tuple[pd.DataFrame, str, str]:
            working_table = evaluation_table.copy()
            if status is not None and "status" in working_table.columns:
                working_table = working_table.loc[
                    working_table["status"].astype(str).str.lower().eq(status.lower())
                ].copy()

            for column_name in ("match_key", "image_id", "gt_relative_path", "relative_mask_path", "gt_path"):
                if column_name not in working_table.columns:
                    continue

                anchor_value = first_non_empty(anchor_row, [column_name])
                if not anchor_value:
                    continue

                matching_rows = working_table.loc[
                    working_table[column_name].astype(str) == anchor_value
                ].copy()
                if not matching_rows.empty:
                    return matching_rows, column_name, anchor_value

            raise ValueError(
                f"Could not identify the comparison group for table row {anchor_row.get('table_row')}."
            )


        def prepare_plot_entries(
            anchor_row: pd.Series,
            evaluation_table: pd.DataFrame,
            *,
            model_names: list[str] | None = None,
            max_models: int = MAX_MODELS_PER_FIGURE,
            status: str | None = DEFAULT_STATUS_FOR_PLOTTING,
        ) -> tuple[list[tuple[str, pd.Series | None]], str, str]:
            comparison_rows, match_column, match_value = find_comparison_rows(
                anchor_row,
                evaluation_table,
                status=status,
            )

            comparison_rows = comparison_rows.sort_values(["model_name", "table_row"]).copy()
            comparison_rows = comparison_rows.drop_duplicates(subset=["model_name"], keep="first")

            if model_names is not None:
                ordered_model_names = [str(model_name) for model_name in model_names][:max_models]
            else:
                ordered_model_names = comparison_rows["model_name"].astype(str).tolist()[:max_models]

            plot_entries: list[tuple[str, pd.Series | None]] = []
            for model_name in ordered_model_names:
                model_rows = comparison_rows.loc[
                    comparison_rows["model_name"].astype(str) == model_name
                ]
                plot_entries.append((model_name, None if model_rows.empty else model_rows.iloc[0]))

            return plot_entries, match_column, match_value


        def plot_comparison_for_row(
            anchor_row: pd.Series,
            *,
            evaluation_table: pd.DataFrame,
            prediction_root: Path,
            model_names: list[str] | None = None,
            max_models: int = MAX_MODELS_PER_FIGURE,
            status: str | None = DEFAULT_STATUS_FOR_PLOTTING,
            figure_width: float = FIGURE_WIDTH,
            row_height: float = ROW_HEIGHT,
            save_directory: str | Path | None = None,
            save_dpi: int = SAVE_DPI,
        ) -> tuple[plt.Figure, dict[str, Any]]:
            plot_entries, match_column, match_value = prepare_plot_entries(
                anchor_row,
                evaluation_table,
                model_names=model_names,
                max_models=max_models,
                status=status,
            )

            figure, axes = plt.subplots(
                nrows=len(plot_entries),
                ncols=2,
                figsize=(figure_width, row_height * len(plot_entries)),
                squeeze=False,
            )

            for row_index, (model_name, model_row) in enumerate(plot_entries):
                outline_ax, mask_ax = axes[row_index]

                if model_row is None:
                    missing_message = "No matching evaluation row was found for this model."
                    render_message_panel(outline_ax, f"{model_name} | outline overlay", missing_message)
                    render_message_panel(mask_ax, f"{model_name} | predicted mask", missing_message)
                    continue

                try:
                    image_path = resolve_source_image_path(
                        model_row,
                        prediction_root=prediction_root,
                        repo_root=REPO_ROOT,
                    )
                    pred_path = resolve_recorded_path(model_row.get("pred_path"), repo_root=REPO_ROOT)
                    if pred_path is None or not pred_path.exists():
                        raise FileNotFoundError(
                            f"Missing predicted mask file for row {model_row.get('table_row')}."
                        )

                    image = load_rgb_image(image_path)
                    pred_mask = load_label_mask(pred_path)

                    if image.shape[:2] != pred_mask.shape:
                        raise ValueError(
                            "Image and prediction mask shapes do not match: "
                            f"{image.shape[:2]} vs {pred_mask.shape}"
                        )

                    outline_overlay = draw_red_outlines(image, pred_mask)
                    predicted_mask_visual = build_mask_visualization(pred_mask)

                    outline_ax.imshow(outline_overlay)
                    mask_ax.imshow(predicted_mask_visual)

                    title_suffix = metric_summary(model_row)
                    outline_title = f"{model_name} | outline overlay"
                    if title_suffix:
                        outline_title += f"\\n{title_suffix}"

                    outline_ax.set_title(outline_title)
                    mask_ax.set_title(f"{model_name} | predicted mask")

                    for ax in (outline_ax, mask_ax):
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_frame_on(False)

                except Exception as error:
                    error_message = str(error)
                    render_message_panel(outline_ax, f"{model_name} | outline overlay", error_message)
                    render_message_panel(mask_ax, f"{model_name} | predicted mask", error_message)

            anchor_table_row = int(anchor_row.get("table_row", -1))
            figure.suptitle(
                f"Table row {anchor_table_row} | {match_column} = {match_value}",
                fontsize=14,
                y=0.995,
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))

            saved_path = None
            if save_directory is not None:
                resolved_save_directory = Path(save_directory).expanduser()
                if not resolved_save_directory.is_absolute():
                    resolved_save_directory = (REPO_ROOT / resolved_save_directory).resolve()
                resolved_save_directory.mkdir(parents=True, exist_ok=True)

                file_stem = sanitize_for_filename(f"row_{anchor_table_row}_{match_value}")
                saved_path = resolved_save_directory / f"{file_stem}.png"
                figure.savefig(saved_path, dpi=save_dpi, bbox_inches="tight")
                print(f"Saved figure to {saved_path}")

            metadata = {
                "table_row": anchor_table_row,
                "match_column": match_column,
                "match_value": match_value,
                "saved_path": None if saved_path is None else str(saved_path),
            }
            return figure, metadata


        def plot_selected_rows(
            selected_rows: pd.DataFrame,
            *,
            evaluation_table: pd.DataFrame,
            prediction_root: Path,
            model_names: list[str] | None = None,
            max_models: int = MAX_MODELS_PER_FIGURE,
            status: str | None = DEFAULT_STATUS_FOR_PLOTTING,
            save_directory: str | Path | None = None,
            save_dpi: int = SAVE_DPI,
        ) -> list[tuple[plt.Figure, dict[str, Any]]]:
            plot_results: list[tuple[plt.Figure, dict[str, Any]]] = []
            backend_name = plt.get_backend().lower()
            should_show = "agg" not in backend_name or "inline" in backend_name

            for _, anchor_row in selected_rows.iterrows():
                figure, metadata = plot_comparison_for_row(
                    anchor_row,
                    evaluation_table=evaluation_table,
                    prediction_root=prediction_root,
                    model_names=model_names,
                    max_models=max_models,
                    status=status,
                    save_directory=save_directory,
                    save_dpi=save_dpi,
                )
                if should_show:
                    plt.show()
                plot_results.append((figure, metadata))

            return plot_results
        """
    ),
    markdown_cell(
        """
        ## Optional Save Section

        Keep `SAVE_FIGURES = False` while exploring. When you want PNG exports, switch it to `True` and set `SAVE_DIRECTORY` to any relative or absolute folder.
        """
    ),
    code_cell(
        """
        # Optional figure saving.
        # Leave this off during normal exploration, then switch it on when you want PNG files.
        SAVE_FIGURES = False

        # Example save targets:
        # SAVE_DIRECTORY = EVALUATION_ROOT / "_plots"
        # SAVE_DIRECTORY = REPO_ROOT / "tmp" / "evaluation_plots"
        SAVE_DIRECTORY = EVALUATION_ROOT / "_plots"

        # Increase or decrease this if you need larger or smaller exports.
        SAVE_DPI = 200
        """
    ),
    markdown_cell(
        """
        ## Example Usage

        Start by inspecting the table preview, then set `ROW_SELECTION` to one row, a list of rows, or an inclusive range string.
        """
    ),
    code_cell(
        """
        display(evaluation_table[available_preview_columns].head(25))
        """
    ),
    code_cell(
        """
        ROW_SELECTION = DEFAULT_ROW_SELECTION
        # Examples:
        # ROW_SELECTION = 56
        # ROW_SELECTION = [3, 56, 6]
        # ROW_SELECTION = "3:23"

        selected_rows = select_evaluation_rows(evaluation_table, ROW_SELECTION)
        display(selected_rows[available_preview_columns])
        """
    ),
    code_cell(
        """
        plot_results = plot_selected_rows(
            selected_rows,
            evaluation_table=evaluation_table,
            prediction_root=PREDICTION_ROOT,
            model_names=DEFAULT_MODEL_ORDER,
            max_models=MAX_MODELS_PER_FIGURE,
            status=DEFAULT_STATUS_FOR_PLOTTING,
            save_directory=SAVE_DIRECTORY if SAVE_FIGURES else None,
            save_dpi=SAVE_DPI,
        )

        print(f"Rendered {len(plot_results)} comparison figure(s).")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n")
print(f"Wrote {NOTEBOOK_PATH}")
