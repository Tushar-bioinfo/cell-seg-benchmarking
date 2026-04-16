from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_MODEL_ID = "hf_hub:prov-gigapath/prov-gigapath"
DEFAULT_IMAGE_COLUMN_CANDIDATES = ("image_path", "source_image_path", "patch_image_path", "path")
DEFAULT_ID_COLUMN_CANDIDATES = ("patch_id", "sample_id", "unique_id", "image_id", "id")
OFFICIAL_TRANSFORM_CONFIG = {
    "resize": 256,
    "crop_size": 224,
    "interpolation": "bicubic",
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}

_SAFE_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RowRecord:
    """One resolved input row ready for dataset loading."""

    input_row_index: int
    embedding_id: str
    image_path: Path
    row_data: dict[str, Any]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not resolve repository root from the current script path.")


def resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    return Path(path_like).expanduser().resolve()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_output_tree(outdir: Path) -> None:
    """Delete only the output subtree managed by this workflow."""

    for child_name in ("embeddings", "metadata", "logs"):
        child = outdir / child_name
        if child.exists():
            shutil.rmtree(child)
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def configure_logger(log_path: Path) -> logging.Logger:
    ensure_directory(log_path.parent)
    logger = logging.getLogger(f"gigapath_embeddings.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def load_input_frame(input_csv: Path, limit: int | None = None) -> pd.DataFrame:
    dataframe = pd.read_csv(input_csv)
    if limit is not None and limit > 0:
        dataframe = dataframe.head(limit).copy()
    dataframe = dataframe.reset_index(drop=False).rename(columns={"index": "input_row_index"})
    return dataframe


def infer_column_name(
    columns: Sequence[str],
    preferred: str | None,
    candidates: Sequence[str],
    *,
    required: bool,
) -> str | None:
    if preferred:
        if preferred not in columns:
            if required:
                raise ValueError(f"Requested column {preferred!r} was not found. Available columns: {list(columns)}")
            return None
        return preferred

    for candidate in candidates:
        if candidate in columns:
            return candidate

    if required:
        raise ValueError(f"Could not infer a required column from candidates {list(candidates)}. Found: {list(columns)}")
    return None


def build_resolution_roots(
    *,
    input_csv: Path,
    repo_root: Path,
    path_base_dir: Path | None,
) -> list[Path]:
    candidates = [input_csv.parent, repo_root, Path.cwd()]
    if path_base_dir is not None:
        candidates.insert(0, path_base_dir)

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def resolve_existing_file(raw_value: Any, *, roots: Sequence[Path]) -> Path | None:
    if raw_value is None:
        return None

    raw_text = str(raw_value).strip()
    if not raw_text:
        return None

    candidate = Path(raw_text).expanduser()
    search_paths = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
    if not candidate.is_absolute():
        search_paths.append(candidate)

    checked: set[Path] = set()
    for search_path in search_paths:
        resolved = search_path.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def safe_token(text: str) -> str:
    sanitized = _SAFE_TOKEN_PATTERN.sub("_", text).strip("_")
    return sanitized or "row"


def derive_embedding_id(
    row_data: dict[str, Any],
    *,
    id_column: str | None,
    image_path: Path,
    input_row_index: int,
) -> str:
    if id_column is not None:
        raw_value = row_data.get(id_column)
        if raw_value is not None:
            raw_text = str(raw_value).strip()
            if raw_text:
                return safe_token(raw_text)

    return f"{safe_token(image_path.stem)}__row{input_row_index:07d}"


def build_failure_row(
    *,
    row_data: dict[str, Any],
    input_row_index: int,
    embedding_id: str | None,
    image_column: str,
    raw_image_value: Any,
    resolved_image_path: Path | None,
    failure_stage: str,
    error: str,
) -> dict[str, Any]:
    failure_row = dict(row_data)
    failure_row.update(
        {
            "input_row_index": int(input_row_index),
            "embedding_id": embedding_id or "",
            "image_column": image_column,
            "raw_image_path": "" if raw_image_value is None else str(raw_image_value),
            "resolved_image_path": "" if resolved_image_path is None else str(resolved_image_path),
            "failure_stage": failure_stage,
            "error": error,
        }
    )
    return failure_row


def prepare_records(
    dataframe: pd.DataFrame,
    *,
    image_column: str,
    id_column: str | None,
    roots: Sequence[Path],
) -> tuple[list[RowRecord], list[dict[str, Any]]]:
    records: list[RowRecord] = []
    failures: list[dict[str, Any]] = []

    for row in dataframe.to_dict(orient="records"):
        row_data = {str(key): normalize_scalar(value) for key, value in row.items()}
        input_row_index = int(row_data["input_row_index"])
        resolved_image_path = resolve_existing_file(row_data.get(image_column), roots=roots)
        embedding_id = ""

        if resolved_image_path is not None:
            embedding_id = derive_embedding_id(
                row_data,
                id_column=id_column,
                image_path=resolved_image_path,
                input_row_index=input_row_index,
            )

        if resolved_image_path is None:
            failures.append(
                build_failure_row(
                    row_data=row_data,
                    input_row_index=input_row_index,
                    embedding_id=embedding_id or f"row{input_row_index:07d}",
                    image_column=image_column,
                    raw_image_value=row_data.get(image_column),
                    resolved_image_path=None,
                    failure_stage="path_resolution",
                    error="Could not resolve an existing image file from the CSV row.",
                )
            )
            continue

        records.append(
            RowRecord(
                input_row_index=input_row_index,
                embedding_id=embedding_id,
                image_path=resolved_image_path,
                row_data=row_data,
            )
        )

    return records, failures


def append_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_directory(path.parent)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(path, mode="a", header=not path.exists(), index=False)


def read_processed_row_indices(*paths: Path) -> set[int]:
    processed: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        dataframe = pd.read_csv(path, usecols=["input_row_index"])
        processed.update(int(value) for value in dataframe["input_row_index"].dropna().tolist())
    return processed


def next_part_index(embeddings_dir: Path) -> int:
    if not embeddings_dir.exists():
        return 0

    highest = -1
    for path in embeddings_dir.glob("part-*.*"):
        stem = path.stem
        if not stem.startswith("part-"):
            continue
        try:
            highest = max(highest, int(stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return highest + 1


def relative_to(base: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target.resolve())


def finalize_sidecar_tables(
    *,
    metadata_csv: Path,
    metadata_parquet: Path,
    failures_csv: Path,
    failures_parquet: Path,
    logger: logging.Logger,
) -> None:
    if metadata_csv.exists():
        metadata_df = pd.read_csv(metadata_csv)
        metadata_df.to_parquet(metadata_parquet, index=False)
        logger.info("Wrote metadata parquet: %s", metadata_parquet)

    if failures_csv.exists():
        failures_df = pd.read_csv(failures_csv)
        failures_df.to_parquet(failures_parquet, index=False)
        logger.info("Wrote failures parquet: %s", failures_parquet)
