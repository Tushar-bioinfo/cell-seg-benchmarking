from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_TAG = "benchmark_input"
INPUT_DIR = PROJECT_ROOT / "data" / DATASET_TAG / "tiles_256"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "benchmarking" / DATASET_TAG / "stardist"
INPUT_MANIFEST = None
WORKERS = 8
RAM_LIMIT_GB = 24
GPU_INDEX = 0
LIMIT = None
OVERWRITE = False
RECURSIVE_INPUT = True

import argparse

from common import append_flag, append_option, log, python_command, resolve_path, run_command

FLOW_NAME = "flow_1.predict_stardist"
ORIGINAL_SCRIPT = PROJECT_ROOT / "scripts" / "02-inference" / "run_stardist.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow 1 wrapper for StarDist inference.")
    parser.add_argument("--in", "--input-dir", dest="input_dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", "--input-manifest", dest="input_manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--ram-limit-gb", type=float, default=RAM_LIMIT_GB)
    parser.add_argument("--gpu-index", type=int, default=GPU_INDEX)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--non-recursive", action="store_true", default=not RECURSIVE_INPUT)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = python_command(ORIGINAL_SCRIPT)
    append_option(command, "--input-dir", resolve_path(args.input_dir))
    append_option(command, "--output-dir", resolve_path(args.output_dir))
    append_option(command, "--input-manifest", resolve_path(args.input_manifest))
    append_option(command, "--workers", args.workers)
    append_option(command, "--ram-limit-gb", args.ram_limit_gb)
    append_option(command, "--gpu-index", args.gpu_index)
    append_option(command, "--limit", args.limit)
    append_flag(command, "--overwrite", bool(args.overwrite))
    append_flag(command, "--non-recursive", bool(args.non_recursive))
    return command


def main() -> None:
    args = parse_args()
    log(FLOW_NAME, f"script={ORIGINAL_SCRIPT}")
    log(FLOW_NAME, f"input_dir={resolve_path(args.input_dir)}")
    log(FLOW_NAME, f"output_dir={resolve_path(args.output_dir)}")
    log(FLOW_NAME, f"input_manifest={resolve_path(args.input_manifest) if args.input_manifest else 'auto'}")
    log(FLOW_NAME, f"workers={args.workers} ram_limit_gb={args.ram_limit_gb:g} gpu_index={args.gpu_index}")
    run_command(build_command(args), name=FLOW_NAME, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
