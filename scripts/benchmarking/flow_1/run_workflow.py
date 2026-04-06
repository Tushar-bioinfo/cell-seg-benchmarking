from __future__ import annotations

from pathlib import Path
import argparse

from common import FLOW_ROOT, PROJECT_ROOT, append_flag, append_option, append_repeatable, log, python_command, resolve_path, run_command

DATASET_TAG = "benchmark_input"
DATA_ROOT = PROJECT_ROOT / "data" / DATASET_TAG
INPUT_MANIFEST = None
IMAGES_SUBDIR = None
MASKS_SUBDIR = None
PAIR_MODE = "suffix"
IMAGE_SUFFIX_TOKEN = "_image"
MASK_SUFFIX_TOKEN = "_mask"
IMAGE_EXTENSIONS = (".png",)
MASK_EXTENSIONS = (".png", ".tif", ".tiff")
RECURSIVE_INPUT = True
RESCALE_OUTPUT_ROOT = None
TILE_INPUT_ROOT = None
TILE_OUTPUT_ROOT = None
PREDICTION_INPUT_ROOT = PROJECT_ROOT / "data" / DATASET_TAG / "tiles_256"
PREDICTION_OUTPUT_ROOT = PROJECT_ROOT / "inference" / "benchmarking" / DATASET_TAG
EVALUATION_OUTPUT_DIR = None
PATCH_SIZE = 256
STRIDE = None
SOURCE_MAGNIFICATION = 40.0
TARGET_MAGNIFICATION = 20.0
MIN_INSTANCE_FRACTION = 0.25
MODELS_TO_RUN = ("cellpose_sam", "cellsam", "cellvit_sam", "stardist")
WORKERS = 8
RAM_LIMIT_GB = 24
GPU_SLOTS = 1
LIMIT = None
OVERWRITE = False
EVALUATION_THRESHOLD = 0.5
SAVE_COMBINED_CSV = False
COMBINED_CSV_NAME = "all_models_evaluation.csv"

FLOW_NAME = "flow_1.workflow"
RESCALE_SCRIPT = FLOW_ROOT / "rescale_dataset.py"
TILE_SCRIPT = FLOW_ROOT / "tile_dataset.py"
PREDICT_SCRIPT = FLOW_ROOT / "run_all.py"
EVALUATE_SCRIPT = FLOW_ROOT / "evaluate_predictions.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reusable benchmarking workflow in order: "
            "rescale -> tile -> predict -> evaluate."
        )
    )
    parser.add_argument("--in", "--data-root", dest="data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest", "--input-manifest", dest="input_manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--images-subdir", default=IMAGES_SUBDIR)
    parser.add_argument("--masks-subdir", default=MASKS_SUBDIR)
    parser.add_argument("--pair-mode", choices=("suffix", "stem"), default=PAIR_MODE)
    parser.add_argument("--image-suffix-token", default=IMAGE_SUFFIX_TOKEN)
    parser.add_argument("--mask-suffix-token", default=MASK_SUFFIX_TOKEN)
    parser.add_argument("--image-exts", nargs="+", default=list(IMAGE_EXTENSIONS))
    parser.add_argument("--mask-exts", nargs="+", default=list(MASK_EXTENSIONS))
    parser.add_argument("--non-recursive", action="store_true", default=not RECURSIVE_INPUT)
    parser.add_argument("--rescale-out", type=Path, default=RESCALE_OUTPUT_ROOT)
    parser.add_argument("--tile-in", type=Path, default=TILE_INPUT_ROOT)
    parser.add_argument("--tile-out", type=Path, default=TILE_OUTPUT_ROOT)
    parser.add_argument("--predict-in", type=Path, default=None)
    parser.add_argument("--predict-out", type=Path, default=PREDICTION_OUTPUT_ROOT)
    parser.add_argument("--eval-out", type=Path, default=EVALUATION_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--source-magnification", type=float, default=SOURCE_MAGNIFICATION)
    parser.add_argument("--target-magnification", type=float, default=TARGET_MAGNIFICATION)
    parser.add_argument("--min-instance-fraction", type=float, default=MIN_INSTANCE_FRACTION)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--ram-limit-gb", type=int, default=RAM_LIMIT_GB)
    parser.add_argument("--gpu-slots", type=int, default=GPU_SLOTS)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--models", nargs="+", default=list(MODELS_TO_RUN))
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--threshold", type=float, default=EVALUATION_THRESHOLD)
    parser.add_argument("--save-combined-csv", action="store_true", default=SAVE_COMBINED_CSV)
    parser.add_argument("--combined-csv-name", default=COMBINED_CSV_NAME)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--skip-rescale", action="store_true")
    parser.add_argument("--skip-tile", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser.parse_args()


def _default_rescale_output(data_root: Path) -> Path:
    return data_root / "rescaled"


def _default_tile_output(tile_input_root: Path, patch_size: int) -> Path:
    return tile_input_root / f"tiles_{patch_size}"


def _run_rescale(args: argparse.Namespace, data_root: Path) -> Path:
    output_root = resolve_path(args.rescale_out) or _default_rescale_output(data_root)
    command = python_command(RESCALE_SCRIPT)
    append_option(command, "--in", data_root)
    append_option(command, "--out", output_root)
    append_option(command, "--manifest", resolve_path(args.input_manifest))
    append_option(command, "--images-subdir", args.images_subdir)
    append_option(command, "--masks-subdir", args.masks_subdir)
    append_option(command, "--pair-mode", args.pair_mode)
    append_option(command, "--image-suffix-token", args.image_suffix_token)
    append_option(command, "--mask-suffix-token", args.mask_suffix_token)
    if args.image_exts:
        command.extend(["--image-exts", *args.image_exts])
    if args.mask_exts:
        command.extend(["--mask-exts", *args.mask_exts])
    append_flag(command, "--non-recursive", bool(args.non_recursive))
    append_option(command, "--source-magnification", args.source_magnification)
    append_option(command, "--target-magnification", args.target_magnification)
    append_option(command, "--min-instance-fraction", args.min_instance_fraction)
    append_repeatable(command, "--sample-id", args.sample_id)
    append_flag(command, "--overwrite", bool(args.overwrite))
    run_command(command, name=f"{FLOW_NAME}.rescale", cwd=PROJECT_ROOT)
    return output_root


def _run_tile(args: argparse.Namespace, tile_input_root: Path) -> Path:
    output_root = resolve_path(args.tile_out) or _default_tile_output(tile_input_root, args.patch_size)
    command = python_command(TILE_SCRIPT)
    append_option(command, "--in", tile_input_root)
    append_option(command, "--out", output_root)
    if args.skip_rescale:
        append_option(command, "--manifest", resolve_path(args.input_manifest))
    append_option(command, "--images-subdir", args.images_subdir)
    append_option(command, "--masks-subdir", args.masks_subdir)
    append_option(command, "--pair-mode", args.pair_mode)
    append_option(command, "--image-suffix-token", args.image_suffix_token)
    append_option(command, "--mask-suffix-token", args.mask_suffix_token)
    if args.image_exts:
        command.extend(["--image-exts", *args.image_exts])
    if args.mask_exts:
        command.extend(["--mask-exts", *args.mask_exts])
    append_flag(command, "--non-recursive", bool(args.non_recursive))
    append_option(command, "--patch-size", args.patch_size)
    append_option(command, "--stride", args.stride)
    append_repeatable(command, "--sample-id", args.sample_id)
    append_flag(command, "--overwrite", bool(args.overwrite))
    run_command(command, name=f"{FLOW_NAME}.tile", cwd=PROJECT_ROOT)
    return output_root


def _run_predict(args: argparse.Namespace, predict_input_root: Path) -> Path:
    output_root = resolve_path(args.predict_out)
    predict_manifest = resolve_path(args.input_manifest) if args.skip_tile else None
    command = python_command(PREDICT_SCRIPT)
    append_option(command, "--in", predict_input_root)
    append_option(command, "--out", output_root)
    append_option(command, "--manifest", predict_manifest)
    append_option(command, "--workers", args.workers)
    append_option(command, "--ram-limit-gb", args.ram_limit_gb)
    append_option(command, "--gpu-slots", args.gpu_slots)
    append_option(command, "--limit", args.limit)
    if args.models:
        command.extend(["--models", *args.models])
    append_flag(command, "--overwrite", bool(args.overwrite))
    run_command(command, name=f"{FLOW_NAME}.predict", cwd=PROJECT_ROOT)
    return output_root


def _run_evaluate(args: argparse.Namespace, prediction_root: Path) -> None:
    command = python_command(EVALUATE_SCRIPT)
    append_option(command, "--in", prediction_root)
    append_option(command, "--out", resolve_path(args.eval_out))
    append_option(command, "--threshold", args.threshold)
    if args.models:
        command.extend(["--model-names", *args.models])
    append_flag(command, "--save-combined-csv", bool(args.save_combined_csv))
    append_option(command, "--combined-csv-name", args.combined_csv_name)
    run_command(command, name=f"{FLOW_NAME}.evaluate", cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    log(FLOW_NAME, f"data_root={data_root}")
    log(
        FLOW_NAME,
        "input_pairing="
        f"pair_mode={args.pair_mode} images_subdir={args.images_subdir or '.'} "
        f"masks_subdir={args.masks_subdir or '.'} recursive={not args.non_recursive}",
    )
    log(
        FLOW_NAME,
        "rescale="
        f"{args.source_magnification:g}x->{args.target_magnification:g}x "
        f"min_instance_fraction={args.min_instance_fraction:.3f}",
    )
    log(
        FLOW_NAME,
        "steps="
        f"rescale={'no' if args.skip_rescale else 'yes'}, "
        f"tile={'no' if args.skip_tile else 'yes'}, "
        f"predict={'no' if args.skip_predict else 'yes'}, "
        f"evaluate={'no' if args.skip_evaluate else 'yes'}",
    )

    current_dataset_root = data_root
    if not args.skip_rescale:
        if not data_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {data_root}")
        current_dataset_root = _run_rescale(args, current_dataset_root)
    else:
        log(FLOW_NAME, "skipping rescale step")

    if not args.skip_tile:
        tile_input_root = resolve_path(args.tile_in) or current_dataset_root
        if not tile_input_root.exists():
            raise FileNotFoundError(f"Tile input root does not exist: {tile_input_root}")
        predict_input_root = _run_tile(args, tile_input_root)
    else:
        log(FLOW_NAME, "skipping tile step")
        predict_input_root = resolve_path(args.predict_in) or PREDICTION_INPUT_ROOT

    if not args.skip_predict:
        if args.predict_in is not None:
            predict_input_root = resolve_path(args.predict_in)
        if not predict_input_root.exists():
            raise FileNotFoundError(f"Prediction input root does not exist: {predict_input_root}")
        prediction_root = _run_predict(args, predict_input_root)
    else:
        log(FLOW_NAME, "skipping predict step")
        prediction_root = resolve_path(args.predict_out)

    if not args.skip_evaluate:
        _run_evaluate(args, prediction_root)
    else:
        log(FLOW_NAME, "skipping evaluate step")


if __name__ == "__main__":
    main()
