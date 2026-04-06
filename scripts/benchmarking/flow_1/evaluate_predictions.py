from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_TAG = "benchmark_input"
PREDICTION_ROOT = PROJECT_ROOT / "inference" / "benchmarking" / DATASET_TAG
OUTPUT_DIR = None
MODEL_NAMES = None
THRESHOLD = 0.5
SAVE_COMBINED_CSV = False
COMBINED_CSV_NAME = "all_models_evaluation.csv"

import argparse

from common import append_flag, append_option, log, python_command, resolve_path, run_command

FLOW_NAME = "flow_1.evaluate"
ORIGINAL_SCRIPT = PROJECT_ROOT / "scripts" / "benchmarking" / "monusac_segmentation_evaluation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reusable wrapper for the manifest-driven prediction evaluator. "
            "Adds consistent --in/--out aliases and clearer run logging."
        )
    )
    parser.add_argument("--in", "--prediction-root", "--pred-root", dest="prediction_root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-names", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--save-combined-csv", action="store_true", default=SAVE_COMBINED_CSV)
    parser.add_argument("--combined-csv-name", default=COMBINED_CSV_NAME)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = python_command(ORIGINAL_SCRIPT)
    append_option(command, "--prediction-root", resolve_path(args.prediction_root) or PREDICTION_ROOT)
    append_option(command, "--output-dir", resolve_path(args.output_dir))
    if args.model_names:
        command.extend(["--model-names", *args.model_names])
    append_option(command, "--threshold", args.threshold)
    append_flag(command, "--save-combined-csv", bool(args.save_combined_csv))
    append_option(command, "--combined-csv-name", args.combined_csv_name)
    return command


def main() -> None:
    args = parse_args()
    log(FLOW_NAME, f"script={ORIGINAL_SCRIPT}")
    log(FLOW_NAME, f"prediction_root={resolve_path(args.prediction_root) or PREDICTION_ROOT}")
    log(FLOW_NAME, f"output_dir={resolve_path(args.output_dir) if args.output_dir is not None else 'original default'}")
    log(FLOW_NAME, f"model_names={', '.join(args.model_names) if args.model_names else 'all discovered models'}")
    log(FLOW_NAME, f"threshold={args.threshold:g} save_combined_csv={bool(args.save_combined_csv)}")
    run_command(build_command(args), name=FLOW_NAME, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
