#!/usr/bin/env python3
"""Automatic offset/span calibration for CellViT JSON -> patch visualization.

This script searches candidate tile spans and local contour shifts across many
patches, in parallel, and recommends stable notebook settings.

It reports:
1) Per-patch best candidate.
2) Global candidate leaderboard.
3) Recommended settings for CellViT_JSON_Visualization_HPC.ipynb.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class CellEntry:
    bbox_xyxy: tuple[float, float, float, float]
    contour: np.ndarray


@dataclass
class PatchEntry:
    tile_id: str
    image_path: str
    x0: float
    y0: float


# Worker globals (set once per process)
G_CELLS: list[CellEntry] = []
G_BBOX_XMIN: np.ndarray | None = None
G_BBOX_YMIN: np.ndarray | None = None
G_BBOX_XMAX: np.ndarray | None = None
G_BBOX_YMAX: np.ndarray | None = None


def parse_xy_from_filename(path: Path) -> tuple[float, float]:
    m = re.search(r"(?P<x>-?\d+)x[_-](?P<y>-?\d+)y", path.stem.lower())
    if m is None:
        raise ValueError(f"Could not parse x/y from filename: {path.name}")
    return float(m.group("x")), float(m.group("y"))


def load_cells(json_path: Path) -> list[CellEntry]:
    obj = json.loads(json_path.read_text())
    out: list[CellEntry] = []
    for c in obj.get("cells", []):
        b = c.get("bbox")
        cont = c.get("contour")
        if b is None or cont is None or len(b) != 2:
            continue
        try:
            ymin, xmin = float(b[0][0]), float(b[0][1])
            ymax, xmax = float(b[1][0]), float(b[1][1])
            arr = np.asarray(cont, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
        except Exception:
            continue
        out.append(CellEntry((xmin, ymin, xmax, ymax), arr[:, :2]))
    return out


def _init_worker(cells_json: str) -> None:
    global G_CELLS, G_BBOX_XMIN, G_BBOX_YMIN, G_BBOX_XMAX, G_BBOX_YMAX

    cells = load_cells(Path(cells_json))
    if len(cells) == 0:
        raise RuntimeError(f"No valid cells found in {cells_json}")

    G_CELLS = cells
    G_BBOX_XMIN = np.fromiter((c.bbox_xyxy[0] for c in cells), dtype=np.float32)
    G_BBOX_YMIN = np.fromiter((c.bbox_xyxy[1] for c in cells), dtype=np.float32)
    G_BBOX_XMAX = np.fromiter((c.bbox_xyxy[2] for c in cells), dtype=np.float32)
    G_BBOX_YMAX = np.fromiter((c.bbox_xyxy[3] for c in cells), dtype=np.float32)


def intersecting_indices(x0: float, y0: float, span: float) -> np.ndarray:
    if G_BBOX_XMIN is None or G_BBOX_YMIN is None or G_BBOX_XMAX is None or G_BBOX_YMAX is None:
        raise RuntimeError("Worker globals are not initialized")
    x1, y1 = x0 + span, y0 + span
    inter = (G_BBOX_XMAX > x0) & (G_BBOX_XMIN < x1) & (G_BBOX_YMAX > y0) & (G_BBOX_YMIN < y1)
    return np.where(inter)[0]


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy)


def evaluate_candidate_fast(
    grad_mag: np.ndarray,
    contours_xy: Sequence[np.ndarray],
    x0: float,
    y0: float,
    span: float,
    shift_x: float,
    shift_y: float,
) -> tuple[float, int]:
    """Return (edge_score, kept_contours) without mask/overlay rasterization."""
    h, w = grad_mag.shape[:2]
    sx = float(w) / float(span)
    sy = float(h) / float(span)

    edge_sum = 0.0
    edge_n = 0
    kept = 0

    for c in contours_xy:
        if c.shape[0] < 2:
            continue
        x = (c[:, 0] - x0) * sx + float(shift_x)
        y = (c[:, 1] - y0) * sy + float(shift_y)

        inside = (x >= -1.0) & (x <= float(w)) & (y >= -1.0) & (y <= float(h))
        if not bool(np.any(inside)):
            continue
        kept += 1

        # Sample every other contour point for speed.
        xx = np.rint(x[::2]).astype(np.int32, copy=False)
        yy = np.rint(y[::2]).astype(np.int32, copy=False)
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not bool(np.any(valid)):
            continue
        edge_sum += float(grad_mag[yy[valid], xx[valid]].sum())
        edge_n += int(valid.sum())

    if edge_n == 0:
        return float("-inf"), kept
    return edge_sum / float(edge_n), kept


def _score_one_patch(
    patch: PatchEntry,
    candidates: Sequence[tuple[float, float, float]],
) -> dict:
    path = Path(patch.image_path)
    if not path.exists():
        return {"tile_id": patch.tile_id, "image_path": patch.image_path, "status": "missing_image"}

    try:
        image_rgb = np.array(Image.open(path).convert("RGB"))
    except Exception as e:
        return {"tile_id": patch.tile_id, "image_path": patch.image_path, "status": "read_error", "error": str(e)}

    gray = image_rgb.mean(axis=2).astype(np.float32, copy=False)
    grad_mag = sobel_magnitude(gray)

    span_to_idxs: dict[float, np.ndarray] = {}
    span_to_contours: dict[float, list[np.ndarray]] = {}
    for span, _, _ in candidates:
        if span in span_to_idxs:
            continue
        idxs = intersecting_indices(x0=patch.x0, y0=patch.y0, span=span)
        span_to_idxs[span] = idxs
        span_to_contours[span] = [G_CELLS[int(i)].contour for i in idxs.tolist()]

    # Skip this patch if no candidate has intersecting cells.
    if all(len(v) == 0 for v in span_to_contours.values()):
        return {"tile_id": patch.tile_id, "image_path": patch.image_path, "status": "no_cells_intersect"}

    scores = np.full((len(candidates),), fill_value=np.float32(-np.inf), dtype=np.float32)
    kept_counts = np.zeros((len(candidates),), dtype=np.int32)

    for i, (span, sx, sy) in enumerate(candidates):
        contours = span_to_contours[span]
        if len(contours) == 0:
            continue
        edge_score, kept = evaluate_candidate_fast(
            grad_mag=grad_mag,
            contours_xy=contours,
            x0=patch.x0,
            y0=patch.y0,
            span=span,
            shift_x=sx,
            shift_y=sy,
        )
        scores[i] = np.float32(edge_score)
        kept_counts[i] = int(kept)

    if not np.isfinite(scores).any():
        return {"tile_id": patch.tile_id, "image_path": patch.image_path, "status": "invalid_scores"}

    best_i = int(np.nanargmax(scores))
    finite_scores = scores[np.isfinite(scores)]
    second_score = float("-inf")
    if finite_scores.size >= 2:
        top2 = np.partition(finite_scores, -2)[-2:]
        second_score = float(np.min(top2))

    return {
        "tile_id": patch.tile_id,
        "image_path": patch.image_path,
        "x0": float(patch.x0),
        "y0": float(patch.y0),
        "status": "ok",
        "best_index": best_i,
        "best_score": float(scores[best_i]),
        "second_score": second_score,
        "margin": float(scores[best_i] - second_score) if np.isfinite(second_score) else float("nan"),
        "scores": scores.tolist(),
        "kept_counts": kept_counts.tolist(),
    }


def _resolve_dataset_image(path_value: str, image_root: Optional[Path]) -> Path:
    p = Path(str(path_value))
    if p.is_absolute() or image_root is None:
        return p
    return image_root / p


def load_patch_entries_from_dataset(
    dataset_csv: Path,
    image_root: Optional[Path],
    coord_mode: str,
    slide_id: Optional[str],
) -> list[PatchEntry]:
    df = pd.read_csv(dataset_csv)
    required = {"image", "tile_x", "tile_y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"dataset.csv missing columns: {sorted(missing)}")

    if slide_id is not None and "slide_id" in df.columns:
        df = df[df["slide_id"].astype(str) == str(slide_id)].copy()

    if "tile_id" not in df.columns:
        df["tile_id"] = df["image"].astype(str).map(lambda s: Path(s).stem)

    out: list[PatchEntry] = []
    for _, row in df.iterrows():
        p = _resolve_dataset_image(str(row["image"]), image_root=image_root)
        tx = float(row["tile_x"])
        ty = float(row["tile_y"])
        if coord_mode == "swap_xy":
            tx, ty = ty, tx
        out.append(PatchEntry(tile_id=str(row["tile_id"]), image_path=str(p), x0=tx, y0=ty))
    return out


def load_patch_entries_from_dir(patch_dir: Path) -> list[PatchEntry]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    out: list[PatchEntry] = []
    for p in sorted(patch_dir.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        try:
            x0, y0 = parse_xy_from_filename(p)
        except Exception:
            continue
        out.append(PatchEntry(tile_id=p.stem, image_path=str(p), x0=x0, y0=y0))
    return out


def sample_entries(entries: Sequence[PatchEntry], n: int, seed: int) -> list[PatchEntry]:
    if n <= 0 or n >= len(entries):
        return list(entries)
    rng = random.Random(seed)
    return rng.sample(list(entries), k=n)


def make_candidates(
    spans: Iterable[float],
    shift_min: int,
    shift_max: int,
    shift_step: int,
) -> list[tuple[float, float, float]]:
    shifts = list(range(int(shift_min), int(shift_max) + 1, int(shift_step)))
    out: list[tuple[float, float, float]] = []
    for span in spans:
        s = float(span)
        for sx in shifts:
            for sy in shifts:
                out.append((s, float(sx), float(sy)))
    return out


def aggregate_results(
    rows_ok: Sequence[dict],
    candidates: Sequence[tuple[float, float, float]],
) -> tuple[pd.DataFrame, int]:
    n = len(candidates)
    if len(rows_ok) == 0:
        return pd.DataFrame(columns=["span", "shift_x", "shift_y", "wins", "win_rate", "mean_score", "median_score", "mean_kept"]), -1

    score_mat = np.full((len(rows_ok), n), np.nan, dtype=np.float32)
    kept_mat = np.full((len(rows_ok), n), np.nan, dtype=np.float32)
    for i, r in enumerate(rows_ok):
        scores = np.asarray(r["scores"], dtype=np.float32)
        kept = np.asarray(r["kept_counts"], dtype=np.float32)
        score_mat[i, : min(n, scores.size)] = scores[:n]
        kept_mat[i, : min(n, kept.size)] = kept[:n]

    # Winner count per patch.
    wins = np.zeros((n,), dtype=np.int32)
    for i in range(score_mat.shape[0]):
        row = score_mat[i]
        if not np.isfinite(row).any():
            continue
        wins[int(np.nanargmax(row))] += 1

    mean_scores = np.nanmean(score_mat, axis=0)
    med_scores = np.nanmedian(score_mat, axis=0)
    mean_kept = np.nanmean(kept_mat, axis=0)
    win_rate = wins.astype(np.float32) / float(max(1, len(rows_ok)))

    summary = []
    for i, (span, sx, sy) in enumerate(candidates):
        summary.append(
            {
                "candidate_index": int(i),
                "span": float(span),
                "shift_x": float(sx),
                "shift_y": float(sy),
                "wins": int(wins[i]),
                "win_rate": float(win_rate[i]),
                "mean_score": float(mean_scores[i]) if np.isfinite(mean_scores[i]) else float("-inf"),
                "median_score": float(med_scores[i]) if np.isfinite(med_scores[i]) else float("-inf"),
                "mean_kept": float(mean_kept[i]) if np.isfinite(mean_kept[i]) else 0.0,
            }
        )

    df = pd.DataFrame(summary).sort_values(
        by=["wins", "mean_score", "median_score", "mean_kept"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best_idx = int(df.iloc[0]["candidate_index"]) if len(df) else -1
    return df, best_idx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-csv", type=Path, help="dataset.csv with image,tile_x,tile_y")
    src.add_argument("--patch-dir", type=Path, help="Patch directory (filenames must encode x/y)")

    ap.add_argument("--image-root", type=Path, default=None, help="Root prefix for relative dataset image paths")
    ap.add_argument("--slide-id", type=str, default=None, help="Optional slide_id filter when dataset.csv has mixed slides")
    ap.add_argument("--coord-mode", choices=["xy", "swap_xy"], default="xy", help="How to read dataset tile_x/tile_y")

    ap.add_argument("--cells-json", type=Path, required=True, help="CellViT cells.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory")

    ap.add_argument("--sample-size", type=int, default=24, help="Random sample size (<=0 means all)")
    ap.add_argument("--seed", type=int, default=13, help="Random seed for sampling")
    ap.add_argument("--workers", type=int, default=8, help="Number of processes")

    ap.add_argument("--spans", type=float, nargs="+", default=[256.0, 512.0, 1024.0], help="Candidate global tile spans")
    ap.add_argument("--shift-min", type=int, default=-24, help="Min local shift (patch pixels)")
    ap.add_argument("--shift-max", type=int, default=24, help="Max local shift (patch pixels)")
    ap.add_argument("--shift-step", type=int, default=8, help="Local shift step (patch pixels)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_csv is not None:
        entries = load_patch_entries_from_dataset(
            dataset_csv=args.dataset_csv,
            image_root=args.image_root,
            coord_mode=args.coord_mode,
            slide_id=args.slide_id,
        )
    else:
        entries = load_patch_entries_from_dir(args.patch_dir)

    if len(entries) == 0:
        raise RuntimeError("No valid patch entries found")

    # Keep only existing images.
    entries = [e for e in entries if Path(e.image_path).exists()]
    if len(entries) == 0:
        raise RuntimeError("No existing patch images found after path resolution")

    sampled = sample_entries(entries, n=int(args.sample_size), seed=int(args.seed))
    candidates = make_candidates(args.spans, args.shift_min, args.shift_max, args.shift_step)
    if len(candidates) == 0:
        raise RuntimeError("No candidates generated")

    workers = max(1, min(int(args.workers), len(sampled)))
    print(f"entries_total={len(entries)} | sampled={len(sampled)} | workers={workers}")
    print(f"candidates={len(candidates)} (spans={list(args.spans)}, shifts={args.shift_min}..{args.shift_max} step {args.shift_step})")

    rows_ok: list[dict] = []
    rows_fail: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(args.cells_json),),
    ) as ex:
        futs = [ex.submit(_score_one_patch, p, candidates) for p in sampled]
        done = 0
        for fut in as_completed(futs):
            done += 1
            row = fut.result()
            if row.get("status") == "ok":
                rows_ok.append(row)
            else:
                rows_fail.append(row)
            if done % max(1, math.ceil(len(futs) / 10)) == 0 or done == len(futs):
                print(f"progress: {done}/{len(futs)}")

    if len(rows_ok) == 0:
        fail_csv = out_dir / "failed.csv"
        pd.DataFrame(rows_fail).to_csv(fail_csv, index=False)
        raise RuntimeError(f"No valid scored patches. See {fail_csv}")

    # Per-patch best table.
    best_rows = []
    for r in rows_ok:
        i = int(r["best_index"])
        span, sx, sy = candidates[i]
        best_rows.append(
            {
                "tile_id": r["tile_id"],
                "image_path": r["image_path"],
                "x0": r["x0"],
                "y0": r["y0"],
                "best_span": span,
                "best_shift_x": sx,
                "best_shift_y": sy,
                "best_score": r["best_score"],
                "second_score": r["second_score"],
                "margin": r["margin"],
            }
        )
    best_df = pd.DataFrame(best_rows).sort_values(by=["best_score", "margin"], ascending=False).reset_index(drop=True)
    best_df.to_csv(out_dir / "per_patch_best.csv", index=False)

    summary_df, best_idx = aggregate_results(rows_ok, candidates)
    summary_df.to_csv(out_dir / "candidate_summary.csv", index=False)
    if len(rows_fail):
        pd.DataFrame(rows_fail).to_csv(out_dir / "failed.csv", index=False)

    if best_idx < 0:
        raise RuntimeError("Could not determine best candidate")

    best = summary_df.iloc[0]
    best_span = float(best["span"])
    best_shift_x = float(best["shift_x"])
    best_shift_y = float(best["shift_y"])
    rec = {
        "n_entries_total": int(len(entries)),
        "n_patches_sampled": int(len(sampled)),
        "n_patches_scored": int(len(rows_ok)),
        "n_patches_failed": int(len(rows_fail)),
        "coord_mode_used": str(args.coord_mode),
        "best_tile_span_override": best_span,
        "best_local_shift_x": best_shift_x,
        "best_local_shift_y": best_shift_y,
        "recommended_dataset_tile_offset_x": -best_shift_x,
        "recommended_dataset_tile_offset_y": -best_shift_y,
        "recommended_dataset_tile_span_mode": "override",
        "recommended_dataset_tile_span_override": best_span,
        "win_rate": float(best["win_rate"]),
        "wins": int(best["wins"]),
    }
    rec_path = out_dir / "recommended_config.json"
    rec_path.write_text(json.dumps(rec, indent=2))

    print("\nTop candidates:")
    print(summary_df.head(10)[["span", "shift_x", "shift_y", "wins", "win_rate", "mean_score"]].to_string(index=False))
    print("\nRecommended notebook settings:")
    print(f"DATASET_TILE_SPAN_MODE = 'override'")
    print(f"DATASET_TILE_SPAN_OVERRIDE = {int(best_span) if float(best_span).is_integer() else best_span}")
    print(f"DATASET_TILE_OFFSET_X = {rec['recommended_dataset_tile_offset_x']}")
    print(f"DATASET_TILE_OFFSET_Y = {rec['recommended_dataset_tile_offset_y']}")
    print(f"\nSaved: {rec_path}")


if __name__ == "__main__":
    main()
