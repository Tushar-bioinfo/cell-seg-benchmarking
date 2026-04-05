from __future__ import annotations

import json
import sys
import traceback
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


LOG_PATH = Path("tmp/jupyter-notebook/validation_steps.log")


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    repo_root = Path.cwd().resolve()
    log(f"repo_root={repo_root}")

    # System Python in this workspace does not ship with imageio, so inject a tiny
    # PNG-capable stub for validation. The notebook still depends on real imageio
    # in the project environment, matching the repo dependencies.
    imageio_module = types.ModuleType("imageio")
    imageio_v3_module = types.ModuleType("imageio.v3")

    def synthetic_imread(path: str | Path) -> np.ndarray:
        return np.asarray(Image.open(path))

    imageio_v3_module.imread = synthetic_imread
    imageio_module.v3 = imageio_v3_module
    sys.modules["imageio"] = imageio_module
    sys.modules["imageio.v3"] = imageio_v3_module
    log("installed imageio validation stub")

    synthetic_root = repo_root / "tmp" / "jupyter-notebook" / "synthetic_eval" / "benchmarking" / "monusac"
    evaluation_root = synthetic_root / "_evaluation"
    image_root = repo_root / "tmp" / "jupyter-notebook" / "synthetic_eval" / "data"
    plot_root = repo_root / "tmp" / "jupyter-notebook" / "synthetic_eval" / "plots"
    models = ["cellpose_sam", "cellsam", "cellvit_sam", "stardist"]

    for path in [evaluation_root, image_root, plot_root, *[synthetic_root / model for model in models]]:
        path.mkdir(parents=True, exist_ok=True)

    image_path = image_root / "sample_001_image.png"
    gt_mask_path = image_root / "sample_001_mask.png"

    height, width = 32, 32
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(30, 180, width, dtype=np.uint8)
    image[..., 1] = np.linspace(20, 220, height, dtype=np.uint8)[:, None]
    image[..., 2] = 120

    base_mask = np.zeros((height, width), dtype=np.uint16)
    base_mask[4:14, 5:15] = 1
    base_mask[18:28, 16:27] = 2

    Image.fromarray(image).save(image_path)
    Image.fromarray(base_mask, mode="I;16").save(gt_mask_path)
    log("wrote source image and gt mask")

    for index, model in enumerate(models, start=1):
        pred_dir = synthetic_root / model / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / "sample_001_mask.png"

        pred_mask = base_mask.copy()
        if model == "cellsam":
            pred_mask[4:14, 14:16] = 1
        elif model == "cellvit_sam":
            pred_mask[18:20, 16:27] = 0
        elif model == "stardist":
            pred_mask[8:12, 20:24] = 3

        Image.fromarray(pred_mask, mode="I;16").save(pred_path)

        predictions_table = pd.DataFrame(
            [
                {
                    "patch_id": "patch_0001",
                    "source_image_path": str(image_path),
                    "relative_image_path": "sample_001_image.png",
                    "mask_path": str(gt_mask_path),
                    "predicted_mask_path": str(pred_path),
                    "predicted_mask_relative_path": "predictions/sample_001_mask.png",
                    "predicted_mask_name": pred_path.name,
                }
            ]
        )
        predictions_table.to_csv(synthetic_root / model / "predictions.csv", index=False)

        evaluation_table = pd.DataFrame(
            [
                {
                    "model_name": model,
                    "patient_id": "patient_a",
                    "relative_mask_path": "sample_001_mask.png",
                    "match_key": "patch_0001",
                    "image_id": "patch_0001",
                    "status": "ok",
                    "error_message": "",
                    "gt_path": str(gt_mask_path),
                    "pred_path": str(pred_path),
                    "gt_relative_path": "sample_001_mask.png",
                    "pred_relative_path": "predictions/sample_001_mask.png",
                    "gt_file_name": gt_mask_path.name,
                    "pred_file_name": pred_path.name,
                    "pixel_dice": 0.90 - index * 0.01,
                    "instance_pq": 0.80 - index * 0.02,
                    "instance_rq": 0.85 - index * 0.02,
                    "pixel_precision": 0.92 - index * 0.01,
                }
            ]
        )
        evaluation_table.to_csv(evaluation_root / f"{model}_evaluation.csv", index=False)
    log("wrote synthetic model outputs")

    notebook = json.loads((repo_root / "scripts" / "benchmarking" / "monusac_evaluation_visualizer.ipynb").read_text())
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    log(f"loaded notebook with {len(code_cells)} code cells")

    namespace: dict[str, object] = {}
    exec(code_cells[0], namespace)
    log("executed config cell")
    namespace["DEFAULT_EVALUATION_ROOT"] = Path("tmp") / "jupyter-notebook" / "synthetic_eval" / "benchmarking" / "monusac" / "_evaluation"
    namespace["FALLBACK_EVALUATION_ROOTS"] = ()
    exec(code_cells[1], namespace)
    log("executed discovery cell")
    exec(code_cells[2], namespace)
    log("executed helper cell")
    exec(code_cells[3], namespace)
    log("executed selection cell")
    exec(code_cells[4], namespace)
    log("executed plotting cell")

    selected_rows = namespace["select_evaluation_rows"](namespace["evaluation_table"], 0)
    log(f"selected rows shape={selected_rows.shape}")
    plot_results = namespace["plot_selected_rows"](
        selected_rows,
        evaluation_table=namespace["evaluation_table"],
        prediction_root=namespace["PREDICTION_ROOT"],
        model_names=namespace["DEFAULT_MODEL_ORDER"],
        max_models=namespace["MAX_MODELS_PER_FIGURE"],
        status=namespace["DEFAULT_STATUS_FOR_PLOTTING"],
        save_directory=plot_root,
        save_dpi=120,
    )
    log(f"plot_results={len(plot_results)}")

    assert len(plot_results) == 1
    saved_path = Path(plot_results[0][1]["saved_path"])
    assert saved_path.is_file(), saved_path
    plt.close("all")
    log(f"saved_path={saved_path}")
    print("synthetic validation ok")
    print(saved_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("validation failed")
        log(traceback.format_exc())
        traceback.print_exc()
        raise
