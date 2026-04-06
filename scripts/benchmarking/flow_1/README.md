# Flow 1 Benchmarking Workflow

`flow_1` is now dataset-generic. It assumes the raw starting point is paired image/mask data where images are usually PNG and masks are PNG or TIFF, then standardizes the rest of the flow with reusable manifests.

Workflow order:

1. Start from paired images and masks in one root, separate subdirectories, or a CSV manifest.
2. Optional rescale or other preprocessing.
3. Optional tiling into fixed-size patches.
4. Model prediction.
5. Manifest-driven evaluation from each model's `predictions.csv`.

Supported raw-input patterns:

- One tree with paired files like `sample_image.png` and `sample_mask.tiff`
- Separate image and mask subdirectories
- A manifest CSV with image and mask path columns

Stage scripts:

- `rescale_dataset.py`
- `tile_dataset.py`
- `run_cellpose_sam.py`
- `run_cellsam.py`
- `run_cellvit_sam.py`
- `run_stardist.py`
- `run_all.py`
- `evaluate_predictions.py`
- `run_workflow.py`

Output contracts used across the flow:

- Dataset-level prep writes `dataset_manifest.csv`
- Tiling writes `all_patches_dataset.csv` and per-sample `dataset.csv`
- Inference writes one model folder per runner with `predictions.csv` and `failed.csv`
- Evaluation reads those `predictions.csv` files directly

Examples:

```bash
python scripts/benchmarking/flow_1/rescale_dataset.py --in data/benchmark_input
python scripts/benchmarking/flow_1/tile_dataset.py --in data/benchmark_input/rescaled --patch-size 256
pixi run python scripts/benchmarking/flow_1/run_all.py --in data/benchmark_input/rescaled/tiles_256
python scripts/benchmarking/flow_1/evaluate_predictions.py --in inference/benchmarking/benchmark_input
python scripts/benchmarking/flow_1/run_workflow.py --images-subdir images --masks-subdir masks
```
