from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_MODEL_ORDER = ("cellpose_sam", "cellsam", "cellvit_sam", "stardist")
EMBEDDING_PREFIX = "emb_"
DEFAULT_RANDOM_SEED = 7
DEFAULT_PCA_COMPONENTS = 400
DEFAULT_UMAP_INPUT_COMPONENTS = 400
DEFAULT_UMAP_N_NEIGHBORS = 75
DEFAULT_UMAP_MIN_DIST = 0.3
CONTINUOUS_CATEGORY_SUFFIX = "_level"
LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75
CATEGORY_ORDER = ("low", "mid", "high", "Missing")
DEFAULT_UMAP_SELECTION_CLUSTERS = 8
MODEL_DISPLAY_NAMES = {
    "cellpose_sam": "Cellpose-SAM",
    "cellsam": "CellSAM",
    "cellvit_sam": "CellViT-SAM-H",
    "stardist": "StarDist",
}
MODEL_COLORS = {
    "cellpose_sam": "#1b9e77",
    "cellsam": "#d95f02",
    "cellvit_sam": "#7570b3",
    "stardist": "#e7298a",
}
DOMINANT_CLASS_COLORS = {
    "epithelial": "#b8376c",
    "connective": "#6f78c9",
    "lymphocyte": "#2f8b57",
    "plasma": "#c97b2f",
    "neutrophil": "#8a4fb8",
    "eosinophil": "#8d6e63",
    "tie": "#6f6f6f",
    "none": "#bdbdbd",
}
STORY_LABEL_COLORS = {
    "connective-rich": "#4c78a8",
    "epithelial-rich": "#d64e7b",
    "mixed": "#c99000",
    "lymphocyte-rich": "#3ba272",
    "no-nuclei": "#bdbdbd",
}
LEVEL_COLORS = {
    "low": "#3ba272",
    "mid": "#c99000",
    "high": "#d64e7b",
    "Missing": "#bdbdbd",
}
DIFFICULTY_TIER_COLORS = {
    "hard": "#d64e7b",
    "medium": "#c99000",
    "easy": "#3ba272",
    "Missing": "#bdbdbd",
}
COLUMN_CATEGORY_COLORS = {
    "fraction_connective_level": LEVEL_COLORS,
    "foreground_fraction_level": LEVEL_COLORS,
    "total_nuclei_level": LEVEL_COLORS,
    "mean_area_level": LEVEL_COLORS,
    "consensus_pq_tier": DIFFICULTY_TIER_COLORS,
}
VIEW_TITLE_DEFAULTS = {
    "image": "Original patch",
    "gt_mask": "Ground-truth mask",
    "gt_overlay": "Ground-truth outline",
    "pred_mask": "Predicted mask",
    "pred_overlay": "Predicted outline",
}
RICHNESS_LABEL_MAP = {
    "connective-tissue-rich": "connective-rich",
    "epithelial-rich": "epithelial-rich",
    "lymphocyte-rich": "lymphocyte-rich",
    "mixed": "mixed",
    "no-nuclei": "no-nuclei",
}
UMAP_CASE_CONFIG = {
    "connective-rich": {
        "query": "foreground_fraction >= 0.03 and total_nuclei >= 15 and dominant_class == 'connective'",
        "title": "Sparse connective",
        "takeaway": "Sparse stromal neighborhoods sit toward the low-signal side of the manifold.",
    },
    "epithelial-rich": {
        "query": "foreground_fraction >= 0.12 and total_nuclei >= 40 and dominant_class == 'epithelial'",
        "title": "Glandular epithelium",
        "takeaway": "Epithelial-rich neighborhoods concentrate around obvious glandular structure and larger nuclei.",
    },
    "mixed": {
        "query": "foreground_fraction >= 0.12 and total_nuclei >= 80 and dominant_fraction <= 0.6",
        "title": "Mixed transition",
        "takeaway": "Mixed patches usually bridge neighboring tissue regimes instead of forming a perfectly pure island.",
    },
    "lymphocyte-rich": {
        "query": "foreground_fraction >= 0.18 and total_nuclei >= 120 and dominant_class == 'lymphocyte'",
        "title": "Lymphocyte-dense field",
        "takeaway": "Immune-dense neighborhoods shift toward the high-burden side of the manifold.",
    },
}
UMAP_CONNECTIVE_LEVEL_CASE_CONFIG = {
    "low": {
        "query": "foreground_fraction >= 0.12 and total_nuclei >= 40 and dominant_class != 'none'",
        "title": "Low connective",
        "takeaway": "Low-connective neighborhoods usually correspond to cell-rich patches with limited stromal background.",
    },
    "mid": {
        "query": "foreground_fraction >= 0.10 and total_nuclei >= 50 and dominant_class != 'none'",
        "title": "Mid connective",
        "takeaway": "Mid-connective neighborhoods tend to mix epithelial structure with visible stromal context.",
    },
    "high": {
        "query": "foreground_fraction >= 0.03 and total_nuclei >= 20 and dominant_class == 'connective'",
        "title": "High connective",
        "takeaway": "High-connective neighborhoods shift toward stromal, lower-signal tissue with more visible connective background.",
    },
    "Missing": {
        "query": "total_nuclei == 0 or dominant_class == 'none'",
        "title": "Missing / no nuclei",
        "takeaway": "Missing connective-fraction labels correspond to no-nuclei patches where class fractions are undefined.",
    },
}
DEFAULT_OVERLAY_ALPHA = 0.35


@dataclass(frozen=True)
class ViewSpec:
    kind: str
    title: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class StoryCase:
    sample_id: str
    title: str
    takeaway: str


class PredictionLocator:
    def __init__(
        self,
        *,
        prediction_roots: dict[str, Path] | None = None,
        prediction_manifests: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.prediction_roots = {
            str(model_name): Path(path).expanduser().resolve()
            for model_name, path in (prediction_roots or {}).items()
        }
        self.prediction_manifests = {
            str(model_name): dataframe.copy()
            for model_name, dataframe in (prediction_manifests or {}).items()
        }

    def available_models(self) -> list[str]:
        return sorted(set(self.prediction_roots) | set(self.prediction_manifests))

    def resolve(self, sample_id: str, model_name: str) -> Path:
        manifest = self.prediction_manifests.get(model_name)
        if manifest is not None and not manifest.empty:
            matched = manifest.loc[manifest["sample_id"].astype(str).eq(sample_id)]
            if not matched.empty:
                row = matched.iloc[0]
                for column in (
                    "predicted_mask_path",
                    "predicted_mask_relative_path",
                    "pred_path",
                    "pred_relative_path",
                    "relative_image_path",
                    "image_id",
                ):
                    value = row.get(column)
                    if isinstance(value, str) and value.strip():
                        path = self._resolve_from_candidate(value, model_name=model_name)
                        if path is not None:
                            return path

        root = self.prediction_roots.get(model_name)
        if root is not None:
            for relative_candidate in (
                Path("images") / f"{sample_id}_image.png",
                Path(f"{sample_id}_image.png"),
                Path("masks") / f"{sample_id}_mask.png",
                Path(f"{sample_id}_mask.png"),
            ):
                candidate = (root / relative_candidate).resolve()
                if candidate.exists():
                    return candidate

        raise FileNotFoundError(
            f"Could not resolve a predicted mask for sample_id={sample_id!r} model={model_name!r}."
        )

    def _resolve_from_candidate(self, path_like: str, *, model_name: str) -> Path | None:
        raw_path = Path(path_like).expanduser()
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path.resolve()

        root = self.prediction_roots.get(model_name)
        relative_text = path_like.replace("\\", "/")
        for prefix in ("data/conic_liz/", "data/conic_lizard/"):
            if relative_text.startswith(prefix):
                relative_text = relative_text[len(prefix) :]
                break

        if root is not None:
            for candidate in (
                root / path_like,
                root / relative_text,
                root / Path(relative_text).name,
            ):
                resolved = candidate.expanduser().resolve()
                if resolved.exists():
                    return resolved

        return None


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not find the repository root from the provided start path.")


def default_dataset_root(start: Path | None = None) -> Path:
    return find_repo_root(start or Path(__file__).resolve()) / "data" / "conic_lizard"


def default_eval_dir(start: Path | None = None) -> Path:
    return find_repo_root(start or Path(__file__).resolve()) / "outputs" / "conic_liz"


def default_metadata_csv(start: Path | None = None) -> Path:
    return default_eval_dir(start) / "embed_morph.csv"


def default_embedding_csv(start: Path | None = None) -> Path:
    return default_eval_dir(start) / "embedding_tables" / "embed_morph_with_vectors.csv"


def default_output_dir(start: Path | None = None) -> Path:
    return default_eval_dir(start) / "analysis" / "visuals"


def ensure_directory(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_id_from_image_ref(image_ref: str | Path) -> str:
    text = str(image_ref).replace("\\", "/")
    if text.startswith("samples/"):
        parts = text.split("/")
        if len(parts) >= 2:
            return parts[1]

    name = Path(text).name
    for suffix in ("_image.png", "_mask.png", "_class_labels.png"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def load_conic_metadata(metadata_csv: str | Path | None = None) -> pd.DataFrame:
    path = Path(metadata_csv or default_metadata_csv()).expanduser().resolve()
    dataframe = pd.read_csv(path)
    if "sample_id" not in dataframe.columns:
        raise ValueError(f"{path} does not contain the required 'sample_id' column.")
    if dataframe["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_id values.")
    dataframe["story_label"] = dataframe.get("richness_label", pd.Series(index=dataframe.index)).map(
        RICHNESS_LABEL_MAP
    )
    return dataframe


def load_embedding_table(embedding_csv: str | Path | None = None) -> pd.DataFrame:
    path = Path(embedding_csv or default_embedding_csv()).expanduser().resolve()
    dataframe = pd.read_csv(path)
    if "sample_id" not in dataframe.columns:
        raise ValueError(f"{path} does not contain the required 'sample_id' column.")
    dataframe["story_label"] = dataframe.get("richness_label", pd.Series(index=dataframe.index)).map(
        RICHNESS_LABEL_MAP
    )
    if "fraction_connective" in dataframe.columns and "fraction_connective_level" not in dataframe.columns:
        dataframe = add_quantile_category_column(dataframe, "fraction_connective")
    return dataframe


def add_quantile_category_column(
    frame: pd.DataFrame,
    source_column: str,
    *,
    suffix: str = CONTINUOUS_CATEGORY_SUFFIX,
    low_quantile: float = LOW_QUANTILE,
    high_quantile: float = HIGH_QUANTILE,
) -> pd.DataFrame:
    level_column = f"{source_column}{suffix}"
    if level_column in frame.columns:
        return frame
    if source_column not in frame.columns:
        raise KeyError(f"add_quantile_category_column missing source column: {source_column}")

    output = frame.copy()
    numeric_values = pd.to_numeric(output[source_column], errors="coerce")
    valid_values = numeric_values.dropna()
    categorized = pd.Series("Missing", index=output.index, dtype="object")

    if not valid_values.empty:
        low_threshold = valid_values.quantile(low_quantile)
        high_threshold = valid_values.quantile(high_quantile)
        categorized.loc[numeric_values.notna()] = "mid"
        if np.isclose(low_threshold, high_threshold):
            categorized.loc[numeric_values.notna()] = "mid"
        else:
            categorized.loc[numeric_values <= low_threshold] = "low"
            categorized.loc[numeric_values >= high_threshold] = "high"

    output[level_column] = pd.Categorical(categorized, categories=list(CATEGORY_ORDER), ordered=True)
    return output


def add_difficulty_tier_column(
    frame: pd.DataFrame,
    source_column: str = "pq_median",
    *,
    output_column: str = "consensus_pq_tier",
    low_quantile: float = LOW_QUANTILE,
    high_quantile: float = HIGH_QUANTILE,
) -> pd.DataFrame:
    if output_column in frame.columns:
        return frame
    if source_column not in frame.columns:
        raise KeyError(f"add_difficulty_tier_column missing source column: {source_column}")

    output = frame.copy()
    numeric_values = pd.to_numeric(output[source_column], errors="coerce")
    valid_values = numeric_values.dropna()
    categorized = pd.Series("Missing", index=output.index, dtype="object")

    if not valid_values.empty:
        low_threshold = valid_values.quantile(low_quantile)
        high_threshold = valid_values.quantile(high_quantile)
        categorized.loc[numeric_values.notna()] = "medium"
        if np.isclose(low_threshold, high_threshold):
            categorized.loc[numeric_values.notna()] = "medium"
        else:
            categorized.loc[numeric_values <= low_threshold] = "hard"
            categorized.loc[numeric_values >= high_threshold] = "easy"

    output[output_column] = pd.Categorical(
        categorized,
        categories=["hard", "medium", "easy", "Missing"],
        ordered=True,
    )
    return output


def add_default_umap_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for source_column in ("fraction_connective", "foreground_fraction", "total_nuclei", "mean_area"):
        if source_column in output.columns:
            output = add_quantile_category_column(output, source_column)
    if "pq_median" in output.columns:
        output = add_difficulty_tier_column(output, "pq_median")
    return output


def embedding_column_names(frame: pd.DataFrame, *, prefix: str = EMBEDDING_PREFIX) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def compute_embedding_projection(
    frame: pd.DataFrame,
    *,
    embedding_prefix: str = EMBEDDING_PREFIX,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    umap_input_components: int = DEFAULT_UMAP_INPUT_COMPONENTS,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    metric: str = "euclidean",
    init: str = "spectral",
    random_seed: int = DEFAULT_RANDOM_SEED,
    standardize_embeddings: bool = False,
    shuffle_rows: bool = True,
) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "compute_embedding_projection requires umap-learn. "
            "Install it in the local environment before building the UMAP-based visuals."
        ) from error

    embedding_columns = embedding_column_names(frame, prefix=embedding_prefix)
    if not embedding_columns:
        raise ValueError(f"No embedding columns starting with prefix {embedding_prefix!r} were found.")

    working_df = frame.copy()
    if shuffle_rows and len(working_df) > 1:
        working_df = working_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    else:
        working_df = working_df.reset_index(drop=True)

    X_model = working_df[embedding_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid_rows = np.isfinite(X_model).all(axis=1)
    if not np.all(valid_rows):
        working_df = working_df.loc[valid_rows].reset_index(drop=True)
        X_model = X_model[valid_rows]
    if X_model.shape[0] < 2:
        raise ValueError("Need at least two valid rows to compute PCA and UMAP projections.")

    if standardize_embeddings:
        means = X_model.mean(axis=0, keepdims=True)
        stds = X_model.std(axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        X_model = (X_model - means) / stds

    max_pca_components = min(pca_components, X_model.shape[0], X_model.shape[1])
    if max_pca_components < 2:
        raise ValueError(f"Need at least 2 PCA components, got {max_pca_components}.")
    pca = PCA(n_components=max_pca_components, random_state=random_seed)
    X_pca = pca.fit_transform(X_model)

    effective_umap_components = min(umap_input_components, X_pca.shape[1])
    if effective_umap_components < 2:
        raise ValueError(f"Need at least 2 PCA components for UMAP, got {effective_umap_components}.")
    X_umap_input = X_pca[:, :effective_umap_components]

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        init=init,
        random_state=random_seed,
    )
    X_umap = umap_model.fit_transform(X_umap_input)

    projection_df = pd.DataFrame(
        {
            "sample_id": working_df["sample_id"].astype(str).to_numpy(),
            "pca_1": X_pca[:, 0],
            "pca_2": X_pca[:, 1],
            "umap_1": X_umap[:, 0],
            "umap_2": X_umap[:, 1],
        }
    )
    return working_df.merge(projection_df, on="sample_id", how="left", validate="one_to_one")


def load_model_evaluations(
    eval_dir: str | Path | None = None,
    *,
    model_order: Iterable[str] = DEFAULT_MODEL_ORDER,
) -> dict[str, pd.DataFrame]:
    resolved_eval_dir = Path(eval_dir or default_eval_dir()).expanduser().resolve()
    evaluations: dict[str, pd.DataFrame] = {}

    for model_name in model_order:
        path = resolved_eval_dir / f"{model_name}_evaluation.csv"
        dataframe = pd.read_csv(path)
        if "image_id" not in dataframe.columns:
            raise ValueError(f"{path} does not contain the required 'image_id' column.")
        dataframe["sample_id"] = dataframe["image_id"].map(sample_id_from_image_ref)
        dataframe["model_name"] = model_name
        evaluations[model_name] = dataframe

    return evaluations


def build_matched_patch_table(
    eval_dir: str | Path | None = None,
    *,
    metadata_csv: str | Path | None = None,
    model_order: Iterable[str] = DEFAULT_MODEL_ORDER,
) -> pd.DataFrame:
    evaluations = load_model_evaluations(eval_dir, model_order=model_order)
    metric_columns = [
        "instance_pq",
        "instance_rq",
        "instance_sq",
        "pixel_precision",
        "pixel_recall",
        "pixel_dice",
    ]

    merged: pd.DataFrame | None = None
    for model_name, dataframe in evaluations.items():
        available_columns = ["sample_id", *[column for column in metric_columns if column in dataframe.columns]]
        renamed = dataframe[available_columns].rename(
            columns={column: f"{model_name}__{column}" for column in available_columns if column != "sample_id"}
        )
        merged = renamed if merged is None else merged.merge(renamed, on="sample_id", how="inner")

    if merged is None:
        raise ValueError("No evaluation tables were loaded.")

    metadata = load_conic_metadata(metadata_csv)
    merged = merged.merge(metadata, on="sample_id", how="left", validate="one_to_one")

    pq_columns = [f"{model_name}__instance_pq" for model_name in model_order]
    merged["pq_median"] = merged[pq_columns].median(axis=1)
    merged["pq_mean"] = merged[pq_columns].mean(axis=1)
    merged["pq_std"] = merged[pq_columns].std(axis=1)
    merged["cellsam_recall_minus_precision"] = (
        merged.get("cellsam__pixel_recall", 0.0) - merged.get("cellsam__pixel_precision", 0.0)
    )
    merged["cellvit_precision_minus_recall"] = (
        merged.get("cellvit_sam__pixel_precision", 0.0) - merged.get("cellvit_sam__pixel_recall", 0.0)
    )
    return merged


def load_patch_arrays(
    sample_id: str,
    *,
    dataset_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_dataset_root = Path(dataset_root or default_dataset_root()).expanduser().resolve()
    image_path = resolved_dataset_root / "images" / f"{sample_id}_image.png"
    mask_path = resolved_dataset_root / "masks" / f"{sample_id}_mask.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image patch for sample_id={sample_id!r}: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Could not find mask patch for sample_id={sample_id!r}: {mask_path}")

    with Image.open(image_path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(mask_path) as mask_handle:
        mask = np.asarray(mask_handle, dtype=np.uint16).copy()

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for sample_id={sample_id!r}: {image.shape[:2]} vs {mask.shape}."
        )
    return image, mask


def colorize_instance_mask(instance_mask: np.ndarray, seed: int = 7) -> np.ndarray:
    labels = np.unique(instance_mask)
    labels = labels[labels > 0]
    color_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)

    for label in labels:
        rng = np.random.default_rng(seed + int(label))
        color = rng.integers(40, 224, size=3, dtype=np.uint8)
        color_mask[instance_mask == label] = color

    return color_mask


def find_instance_boundaries(instance_mask: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(instance_mask.shape, dtype=bool)
    boundaries[1:, :] |= instance_mask[1:, :] != instance_mask[:-1, :]
    boundaries[:-1, :] |= instance_mask[:-1, :] != instance_mask[1:, :]
    boundaries[:, 1:] |= instance_mask[:, 1:] != instance_mask[:, :-1]
    boundaries[:, :-1] |= instance_mask[:, :-1] != instance_mask[:, 1:]
    boundaries &= instance_mask > 0
    return boundaries


def overlay_instance_mask(
    image: np.ndarray,
    instance_mask: np.ndarray,
    *,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    seed: int = 7,
) -> np.ndarray:
    if image.shape[:2] != instance_mask.shape:
        raise ValueError(
            f"Overlay expects matching image/mask shapes, got {image.shape[:2]} and {instance_mask.shape}."
        )

    overlay = image.astype(np.float32).copy()
    color_mask = colorize_instance_mask(instance_mask, seed=seed).astype(np.float32)
    foreground = instance_mask > 0
    overlay[foreground] = (1.0 - alpha) * overlay[foreground] + alpha * color_mask[foreground]
    overlay[find_instance_boundaries(instance_mask)] = 255.0
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_view(
    sample_id: str,
    view_spec: ViewSpec,
    *,
    dataset_root: str | Path | None = None,
    prediction_locator: PredictionLocator | None = None,
) -> np.ndarray:
    image, gt_mask = load_patch_arrays(sample_id, dataset_root=dataset_root)

    if view_spec.kind == "image":
        return image
    if view_spec.kind == "gt_mask":
        return colorize_instance_mask(gt_mask)
    if view_spec.kind == "gt_overlay":
        return overlay_instance_mask(image, gt_mask)

    if view_spec.kind in {"pred_mask", "pred_overlay"}:
        if prediction_locator is None or not view_spec.model_name:
            raise ValueError(
                f"View kind={view_spec.kind!r} for sample_id={sample_id!r} requires both "
                "prediction_locator and view_spec.model_name."
            )
        prediction_path = prediction_locator.resolve(sample_id, view_spec.model_name)
        with Image.open(prediction_path) as prediction_handle:
            pred_mask = np.asarray(prediction_handle, dtype=np.uint16).copy()
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"Prediction mask shape mismatch for sample_id={sample_id!r} model={view_spec.model_name!r}: "
                f"{pred_mask.shape} vs {gt_mask.shape}."
            )
        if view_spec.kind == "pred_mask":
            return colorize_instance_mask(pred_mask)
        return overlay_instance_mask(image, pred_mask)

    raise ValueError(f"Unsupported view kind: {view_spec.kind!r}")


def plot_patch_view_grid(
    sample_ids: list[str],
    view_specs: list[ViewSpec],
    *,
    dataset_root: str | Path | None = None,
    prediction_locator: PredictionLocator | None = None,
    row_titles: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    nrows = len(sample_ids)
    ncols = len(view_specs)
    if nrows == 0 or ncols == 0:
        raise ValueError("plot_patch_view_grid requires at least one sample and one view spec.")

    if figsize is None:
        figsize = (4.6 * ncols, 4.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)

    for row_index, sample_id in enumerate(sample_ids):
        for col_index, view_spec in enumerate(view_specs):
            axis = axes[row_index, col_index]
            rendered = render_view(
                sample_id,
                view_spec,
                dataset_root=dataset_root,
                prediction_locator=prediction_locator,
            )
            axis.imshow(rendered)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_frame_on(False)

            title = view_spec.title or VIEW_TITLE_DEFAULTS.get(view_spec.kind, view_spec.kind)
            if nrows == 1 or row_index == 0:
                axis.set_title(title, fontsize=11)

            if row_titles and col_index == 0:
                axis.set_ylabel(row_titles[row_index], fontsize=11, rotation=90, labelpad=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def plot_archetype_gallery(
    cases: list[StoryCase],
    matched_table: pd.DataFrame,
    *,
    dataset_root: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    show_takeaways: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    if not cases:
        raise ValueError("plot_archetype_gallery requires at least one story case.")

    ncols = 2
    nrows = int(np.ceil(len(cases) / ncols))
    if figsize is None:
        figsize = (11.5, 5.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_array = np.atleast_2d(axes)
    flat_axes = list(axes_array.flat)
    metrics_index = matched_table.set_index("sample_id")

    for axis, case in zip(flat_axes, cases):
        rendered = render_view(case.sample_id, ViewSpec(kind="gt_overlay"), dataset_root=dataset_root)
        axis.imshow(rendered)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)

        row = metrics_index.loc[case.sample_id]
        title = (
            f"{case.title}\n"
            f"PQ {row['pq_median']:.3f} | fg {row['foreground_fraction']:.3f} | n {int(row['total_nuclei'])}"
        )
        axis.set_title(title, fontsize=11)
        if show_takeaways:
            axis.text(
                0.02,
                0.02,
                textwrap.fill(case.takeaway, width=46),
                transform=axis.transAxes,
                fontsize=8.5,
                color="white",
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 4},
            )

    for axis in flat_axes[len(cases) :]:
        axis.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    return fig, axes_array


def plot_metric_caseboards(
    cases: list[StoryCase],
    matched_table: pd.DataFrame,
    *,
    dataset_root: str | Path | None = None,
    prediction_locator: PredictionLocator | None = None,
    prediction_models: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if not cases:
        raise ValueError("plot_metric_caseboards requires at least one story case.")

    include_predictions = bool(prediction_locator and prediction_models)
    ordered_prediction_models = [model for model in (prediction_models or []) if model in prediction_locator.available_models()] if include_predictions else []
    n_prediction_cols = len(ordered_prediction_models)
    ncols = 4 + n_prediction_cols
    if figsize is None:
        figsize = (4.0 * ncols, 4.0 * len(cases))

    width_ratios = [1.0, 1.0, 1.0, 1.3, *([1.0] * n_prediction_cols)]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gridspec = fig.add_gridspec(len(cases), ncols, width_ratios=width_ratios)
    axes = np.empty((len(cases), ncols), dtype=object)

    metrics_index = matched_table.set_index("sample_id")

    for row_index, case in enumerate(cases):
        row = metrics_index.loc[case.sample_id]
        base_views = [
            ViewSpec(kind="image", title="Original patch"),
            ViewSpec(kind="gt_mask", title="Ground-truth mask"),
            ViewSpec(kind="gt_overlay", title="Ground-truth outline"),
        ]
        prediction_views = [
            ViewSpec(kind="pred_overlay", title=MODEL_DISPLAY_NAMES.get(model_name, model_name), model_name=model_name)
            for model_name in ordered_prediction_models
        ]
        views = [*base_views, *prediction_views]

        for col_index, view_spec in enumerate(views):
            axis = fig.add_subplot(gridspec[row_index, col_index])
            rendered = render_view(
                case.sample_id,
                view_spec,
                dataset_root=dataset_root,
                prediction_locator=prediction_locator,
            )
            axis.imshow(rendered)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_frame_on(False)
            if row_index == 0:
                axis.set_title(view_spec.title or VIEW_TITLE_DEFAULTS.get(view_spec.kind, view_spec.kind), fontsize=11)
            axes[row_index, col_index] = axis

        metric_axis = fig.add_subplot(gridspec[row_index, 3 + n_prediction_cols])
        axes[row_index, 3 + n_prediction_cols] = metric_axis
        model_names = list(DEFAULT_MODEL_ORDER)
        bar_values = [float(row[f"{model_name}__instance_pq"]) for model_name in model_names]
        y_positions = np.arange(len(model_names))
        metric_axis.barh(
            y_positions,
            bar_values,
            color=[MODEL_COLORS[model_name] for model_name in model_names],
            edgecolor="none",
        )
        metric_axis.set_xlim(0, 1)
        metric_axis.set_yticks(y_positions)
        metric_axis.set_yticklabels([MODEL_DISPLAY_NAMES[model_name] for model_name in model_names], fontsize=9)
        metric_axis.invert_yaxis()
        metric_axis.grid(axis="x", alpha=0.2)
        metric_axis.set_title(f"{case.title}\nPatch-level instance PQ", fontsize=11)
        metric_axis.set_xlabel("PQ", fontsize=9)
        for spine in ("top", "right"):
            metric_axis.spines[spine].set_visible(False)

        summary_lines = [
            (
                f"{row.get('story_label', row.get('richness_label', 'unknown'))} | "
                f"median PQ {row['pq_median']:.3f}"
            ),
            (
                f"foreground {row['foreground_fraction']:.3f} | "
                f"nuclei {int(row['total_nuclei'])} | "
                f"mean area {row['mean_area']:.1f}"
            ),
            textwrap.fill(case.takeaway, width=34),
        ]
        metric_axis.text(
            0.02,
            0.02,
            "\n".join(summary_lines),
            transform=metric_axis.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.9, "pad": 3},
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def select_dominant_class_exemplars(
    frame: pd.DataFrame,
    *,
    class_name: str,
    n_examples: int = 3,
    projection_columns: tuple[str, str] = ("umap_1", "umap_2"),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[str]:
    from sklearn.cluster import KMeans

    subset = frame.loc[frame["dominant_class"].astype(str).eq(class_name)].copy()
    if subset.empty:
        return []

    if all(column in subset.columns for column in projection_columns) and len(subset) >= n_examples:
        coords = subset.loc[:, list(projection_columns)].apply(pd.to_numeric, errors="coerce")
        subset = subset.loc[coords.notna().all(axis=1)].copy()
        coords = coords.loc[subset.index]
        if len(subset) >= n_examples:
            kmeans = KMeans(n_clusters=n_examples, random_state=random_seed, n_init=10)
            labels = kmeans.fit_predict(coords.to_numpy())
            subset = subset.assign(_cluster=labels)
            selected_rows: list[pd.Series] = []
            for cluster_index, center in enumerate(kmeans.cluster_centers_):
                cluster_subset = subset.loc[subset["_cluster"].eq(cluster_index)].copy()
                cluster_coords = coords.loc[cluster_subset.index].to_numpy()
                distances = np.linalg.norm(cluster_coords - center, axis=1)
                cluster_subset = cluster_subset.assign(_distance=distances)
                cluster_subset = cluster_subset.sort_values(
                    ["_distance", "dominant_fraction", "foreground_fraction"],
                    ascending=[True, False, False],
                )
                selected_rows.append(cluster_subset.iloc[0])
            selected_frame = pd.DataFrame(selected_rows).sort_values("foreground_fraction")
            return selected_frame["sample_id"].astype(str).tolist()

    fallback_subset = subset.sort_values(
        ["dominant_fraction", "foreground_fraction", "total_nuclei"],
        ascending=[False, False, False],
    )
    return fallback_subset["sample_id"].astype(str).head(n_examples).tolist()


def plot_dominant_class_atlas(
    atlas_map: dict[str, list[str]],
    *,
    dataset_root: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    atlas_rows = [(class_name, sample_ids) for class_name, sample_ids in atlas_map.items() if sample_ids]
    if not atlas_rows:
        raise ValueError("plot_dominant_class_atlas requires at least one class with sample_ids.")

    nrows = len(atlas_rows)
    ncols = max(len(sample_ids) for _, sample_ids in atlas_rows)
    if figsize is None:
        figsize = (3.8 * ncols, 3.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)

    for row_index, (class_name, sample_ids) in enumerate(atlas_rows):
        display_name = class_name.replace("-", " ").title()
        class_color = DOMINANT_CLASS_COLORS.get(class_name, "#666666")
        for col_index in range(ncols):
            axis = axes[row_index, col_index]
            axis.set_xticks([])
            axis.set_yticks([])
            if col_index < len(sample_ids):
                rendered = render_view(sample_ids[col_index], ViewSpec(kind="image"), dataset_root=dataset_root)
                axis.imshow(rendered)
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(class_color)
                    spine.set_linewidth(2.0)
            else:
                axis.axis("off")

            if row_index == 0:
                axis.set_title(f"Example {col_index + 1}", fontsize=11)
            if col_index == 0:
                axis.set_ylabel(display_name, fontsize=11, color=class_color, labelpad=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def color_lookup_for_column(column_name: str) -> dict[str, str]:
    if column_name == "dominant_class":
        return DOMINANT_CLASS_COLORS
    if column_name in COLUMN_CATEGORY_COLORS:
        return COLUMN_CATEGORY_COLORS[column_name]
    if column_name.endswith(CONTINUOUS_CATEGORY_SUFFIX):
        return LEVEL_COLORS
    return STORY_LABEL_COLORS


def categories_for_column(frame: pd.DataFrame, column_name: str) -> list[str]:
    color_lookup = color_lookup_for_column(column_name)
    present = set(frame[column_name].dropna().astype(str))
    ordered = [label for label in color_lookup if label in present]
    extras = sorted(present.difference(ordered))
    return [*ordered, *extras]


def display_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def scatter_title_for_column(column_name: str) -> str:
    title_map = {
        "story_label": "UMAP colored by patch phenotype",
        "dominant_class": "UMAP colored by dominant class",
        "fraction_connective_level": "UMAP colored by connective-fraction level",
        "foreground_fraction_level": "UMAP colored by foreground fraction",
        "total_nuclei_level": "UMAP colored by total nuclei",
        "mean_area_level": "UMAP colored by mean nuclei area",
        "consensus_pq_tier": "UMAP colored by consensus PQ tier",
    }
    return title_map.get(column_name, f"UMAP colored by {column_name.replace('_', ' ')}")


def select_representative_umap_cases(
    frame: pd.DataFrame,
    *,
    label_order: list[str],
    label_column: str,
    case_config: dict[str, dict[str, str]],
    n_global_clusters: int = DEFAULT_UMAP_SELECTION_CLUSTERS,
) -> list[StoryCase]:
    from sklearn.cluster import KMeans

    required_columns = {
        label_column,
        "umap_1",
        "umap_2",
        "sample_id",
        "foreground_fraction",
        "total_nuclei",
        "dominant_class",
    }
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"select_representative_umap_cases missing required columns: {missing_columns}")

    working_frame = frame.copy()
    valid_cluster_rows = working_frame.loc[working_frame["umap_1"].notna() & working_frame["umap_2"].notna()].copy()
    cluster_center_map: dict[int, np.ndarray] = {}
    cluster_purity_table = pd.DataFrame()
    if len(valid_cluster_rows) >= 2:
        effective_clusters = max(1, min(n_global_clusters, len(valid_cluster_rows)))
        if effective_clusters > 1:
            kmeans = KMeans(n_clusters=effective_clusters, random_state=DEFAULT_RANDOM_SEED, n_init=20)
            cluster_labels = kmeans.fit_predict(valid_cluster_rows[["umap_1", "umap_2"]].to_numpy())
            valid_cluster_rows = valid_cluster_rows.assign(_global_cluster=cluster_labels)
            working_frame = working_frame.merge(
                valid_cluster_rows[["sample_id", "_global_cluster"]],
                on="sample_id",
                how="left",
            )
            cluster_center_map = {
                cluster_index: center for cluster_index, center in enumerate(kmeans.cluster_centers_)
            }
            cluster_counts = (
                valid_cluster_rows.groupby("_global_cluster")[label_column]
                .value_counts(normalize=True)
                .rename("purity")
                .reset_index()
                .pivot(index="_global_cluster", columns=label_column, values="purity")
                .fillna(0.0)
            )
            cluster_purity_table = cluster_counts
        else:
            working_frame["_global_cluster"] = 0
            cluster_center_map = {0: np.asarray([working_frame["umap_1"].median(), working_frame["umap_2"].median()])}

    cases: list[StoryCase] = []
    for label in label_order:
        subset = working_frame.loc[working_frame[label_column].astype(str).eq(label)].copy()
        if subset.empty:
            continue

        config = case_config.get(label, {})
        filtered = subset
        query = config.get("query")
        if isinstance(query, str) and query.strip():
            queried_subset = subset.query(query).copy()
            if not queried_subset.empty:
                filtered = queried_subset

        cluster_filtered = filtered
        if "_global_cluster" in filtered.columns and filtered["_global_cluster"].notna().any():
            cluster_summary = (
                filtered["_global_cluster"]
                .dropna()
                .astype(int)
                .value_counts()
                .rename_axis("_global_cluster")
                .reset_index(name="label_count")
            )
            if not cluster_summary.empty:
                purity_values = []
                for cluster_index in cluster_summary["_global_cluster"]:
                    purity = 0.0
                    if not cluster_purity_table.empty and label in cluster_purity_table.columns and cluster_index in cluster_purity_table.index:
                        purity = float(cluster_purity_table.loc[cluster_index, label])
                    purity_values.append(purity)
                cluster_summary["purity"] = purity_values
                cluster_summary["score"] = cluster_summary["label_count"] * cluster_summary["purity"]
                chosen_cluster = int(
                    cluster_summary.sort_values(["score", "label_count", "purity"], ascending=[False, False, False]).iloc[0]["_global_cluster"]
                )
                chosen_cluster_rows = filtered.loc[filtered["_global_cluster"].fillna(-1).astype(int).eq(chosen_cluster)].copy()
                if not chosen_cluster_rows.empty:
                    cluster_filtered = chosen_cluster_rows

        if "_global_cluster" in cluster_filtered.columns and cluster_filtered["_global_cluster"].notna().any():
            cluster_index = int(cluster_filtered["_global_cluster"].dropna().iloc[0])
            cluster_center = cluster_center_map.get(cluster_index)
            if cluster_center is None:
                center_x = cluster_filtered["umap_1"].median()
                center_y = cluster_filtered["umap_2"].median()
            else:
                center_x, center_y = float(cluster_center[0]), float(cluster_center[1])
        else:
            center_x = cluster_filtered["umap_1"].median()
            center_y = cluster_filtered["umap_2"].median()

        cluster_filtered["_distance"] = np.sqrt(
            (cluster_filtered["umap_1"] - center_x) ** 2 + (cluster_filtered["umap_2"] - center_y) ** 2
        )
        sort_columns = ["_distance"]
        ascending = [True]
        if "dominant_fraction" in cluster_filtered.columns:
            sort_columns.append("dominant_fraction")
            ascending.append(False)
        if "fraction_connective" in cluster_filtered.columns:
            sort_columns.append("fraction_connective")
            ascending.append(False)
        sort_columns.extend(["total_nuclei", "foreground_fraction"])
        ascending.extend([False, False])
        cluster_filtered = cluster_filtered.sort_values(sort_columns, ascending=ascending)

        row = cluster_filtered.iloc[0]
        title = str(config.get("title", label.replace("-", " ").title()))
        takeaway = str(config.get("takeaway", "A representative local neighborhood from the embedding manifold."))
        cases.append(StoryCase(sample_id=str(row["sample_id"]), title=title, takeaway=takeaway))
    return cases


def select_umap_story_cases(
    frame: pd.DataFrame,
    *,
    label_order: list[str] | None = None,
    label_column: str = "story_label",
) -> list[StoryCase]:
    if label_order is None:
        label_order = ["connective-rich", "epithelial-rich", "mixed", "lymphocyte-rich"]
    return select_representative_umap_cases(
        frame,
        label_order=label_order,
        label_column=label_column,
        case_config=UMAP_CASE_CONFIG,
    )


def select_umap_fraction_connective_cases(
    frame: pd.DataFrame,
    *,
    label_order: list[str] | None = None,
    label_column: str = "fraction_connective_level",
) -> list[StoryCase]:
    if label_order is None:
        label_order = ["low", "mid", "high", "Missing"]
    working_frame = add_quantile_category_column(frame, "fraction_connective")
    return select_representative_umap_cases(
        working_frame,
        label_order=label_order,
        label_column=label_column,
        case_config=UMAP_CONNECTIVE_LEVEL_CASE_CONFIG,
    )


def plot_umap_patch_neighborhoods(
    projection_frame: pd.DataFrame,
    cases: list[StoryCase],
    *,
    dataset_root: str | Path | None = None,
    color_column: str = "story_label",
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    show_scatter_title: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    if not cases:
        raise ValueError("plot_umap_patch_neighborhoods requires at least one selected story case.")

    required_columns = {"sample_id", "umap_1", "umap_2", color_column}
    missing_columns = [column for column in required_columns if column not in projection_frame.columns]
    if missing_columns:
        raise KeyError(f"plot_umap_patch_neighborhoods missing required columns: {missing_columns}")

    selected_ids = [case.sample_id for case in cases]
    selected_frame = projection_frame.set_index("sample_id").loc[selected_ids].reset_index()

    if figsize is None:
        figsize = (14.0, 8.0)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gridspec = fig.add_gridspec(2, 4, width_ratios=[1.8, 1.8, 1.0, 1.0])
    scatter_axis = fig.add_subplot(gridspec[:, :2])
    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = scatter_axis

    base_frame = projection_frame.loc[
        projection_frame["umap_1"].notna() & projection_frame["umap_2"].notna()
    ].copy()

    color_lookup = color_lookup_for_column(color_column)
    categories = categories_for_column(base_frame, color_column)
    for label in categories:
        subset = base_frame.loc[base_frame[color_column].astype(str).eq(label)]
        scatter_axis.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=11,
            alpha=0.65,
            linewidths=0,
            c=color_lookup.get(label, "#999999"),
            label=label,
        )

    for index, case in enumerate(cases, start=1):
        row = selected_frame.loc[selected_frame["sample_id"].astype(str).eq(case.sample_id)].iloc[0]
        scatter_axis.scatter(
            [row["umap_1"]],
            [row["umap_2"]],
            s=90,
            facecolors="white",
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
        )
        scatter_axis.annotate(
            str(index),
            (row["umap_1"], row["umap_2"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    if show_scatter_title:
        scatter_axis.set_title(scatter_title_for_column(color_column), fontsize=12)
    scatter_axis.set_xlabel("UMAP 1")
    scatter_axis.set_ylabel("UMAP 2")
    scatter_axis.grid(alpha=0.15)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_lookup.get(label, "#999999"),
            markeredgecolor=color_lookup.get(label, "#999999"),
            markersize=8.5,
            label=label,
        )
        for label in categories
    ]
    scatter_axis.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    patch_positions = [(0, 2), (0, 3), (1, 2), (1, 3)]
    for index, ((grid_row, grid_col), case) in enumerate(zip(patch_positions, cases), start=1):
        patch_axis = fig.add_subplot(gridspec[grid_row, grid_col])
        axes[grid_row, grid_col - 1] = patch_axis
        rendered = render_view(case.sample_id, ViewSpec(kind="image"), dataset_root=dataset_root)
        patch_row = selected_frame.loc[selected_frame["sample_id"].astype(str).eq(case.sample_id)].iloc[0]
        patch_color = color_lookup.get(str(patch_row.get(color_column, "")), "black")
        patch_axis.imshow(rendered)
        patch_axis.set_xticks([])
        patch_axis.set_yticks([])
        patch_axis.set_title(f"{index}. {case.title}", fontsize=10)
        for spine in patch_axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(patch_color)
            spine.set_linewidth(1.5)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def plot_umap_scatter_panel(
    frame: pd.DataFrame,
    color_column: str,
    *,
    axis: plt.Axes,
    title: str | None = None,
    point_size: float = 9.0,
    alpha: float = 0.68,
    legend_marker_size: float = 8.0,
) -> None:
    required_columns = {"umap_1", "umap_2", color_column}
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"plot_umap_scatter_panel missing required columns: {missing_columns}")

    subset = frame.loc[frame["umap_1"].notna() & frame["umap_2"].notna()].copy()
    if subset.empty:
        raise ValueError("No rows with non-null UMAP coordinates were available to plot.")

    color_lookup = color_lookup_for_column(color_column)
    categories = categories_for_column(subset, color_column)
    for label in categories:
        category_subset = subset.loc[subset[color_column].astype(str).eq(label)]
        axis.scatter(
            category_subset["umap_1"],
            category_subset["umap_2"],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            c=color_lookup.get(label, "#8e8e8e"),
            label=label,
        )

    axis.set_title(title or scatter_title_for_column(color_column), fontsize=11.5)
    axis.set_xlabel("UMAP 1", fontsize=9)
    axis.set_ylabel("UMAP 2", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.grid(alpha=0.14)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_lookup.get(label, "#8e8e8e"),
            markeredgecolor=color_lookup.get(label, "#8e8e8e"),
            markersize=legend_marker_size,
            label=display_label(label),
        )
        for label in categories
    ]
    axis.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.0,
    )


def plot_umap_scatter_grid(
    frame: pd.DataFrame,
    panels: list[tuple[str, str]],
    *,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if not panels:
        raise ValueError("plot_umap_scatter_grid requires at least one panel.")

    nrows = int(np.ceil(len(panels) / ncols))
    if figsize is None:
        figsize = (5.4 * ncols, 4.75 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    flat_axes = list(axes.flat)
    for axis, (color_column, title) in zip(flat_axes, panels):
        plot_umap_scatter_panel(frame, color_column, axis=axis, title=title)

    for axis in flat_axes[len(panels) :]:
        axis.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def save_figure(fig: plt.Figure, output_dir: str | Path, file_name: str, *, dpi: int = 200) -> Path:
    resolved_output_dir = ensure_directory(output_dir)
    output_path = resolved_output_dir / file_name
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
