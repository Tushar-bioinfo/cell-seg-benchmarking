from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "Monusac" / "tiles_256"
OUTPUT_ROOT = PROJECT_ROOT / "inference" / "benchmarking" / "monusac"
INPUT_MANIFEST = None
WORKERS = 8
RAM_LIMIT_GB = 24
GPU_SLOTS = 1
OVERWRITE = False
LIMIT = None
MODELS_TO_RUN = ("cellpose_sam", "cellsam", "cellvit_sam", "stardist")
MODEL_RAM_GB = {
    "cellpose_sam": 6,
    "cellsam": 6,
    "cellvit_sam": 8,
    "stardist": 4,
}
MODEL_ENVS = {
    "cellpose_sam": "cellpose",
    "cellsam": "cellsam",
    "cellvit_sam": "cellvit",
    "stardist": "stardist",
}
MODEL_SCRIPTS = {
    "cellpose_sam": PROJECT_ROOT / "scripts" / "02-inference" / "run_cellpose_sam.py",
    "cellsam": PROJECT_ROOT / "scripts" / "02-inference" / "run_cellsam.py",
    "cellvit_sam": PROJECT_ROOT / "scripts" / "02-inference" / "run_cellvit_sam.py",
    "stardist": PROJECT_ROOT / "scripts" / "02-inference" / "run_stardist.py",
}
LOG_DIR = OUTPUT_ROOT / "_logs"

import argparse
import os
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace

from benchmark_inference_utils import ensure_directory


@dataclass(frozen=True)
class ModelJob:
    model_name: str
    pixi_environment: str
    script_path: Path
    input_dir: Path
    output_dir: Path
    input_manifest: Path | None
    workers: int
    ram_gb: int
    overwrite: bool
    limit: int | None
    log_path: Path
    gpu_device: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the four benchmarking inference scripts.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--input-manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--ram-limit-gb", type=int, default=RAM_LIMIT_GB)
    parser.add_argument("--gpu-slots", type=int, default=GPU_SLOTS)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--models", nargs="+", default=list(MODELS_TO_RUN), choices=sorted(MODEL_SCRIPTS))
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[ModelJob]:
    ensure_directory(args.output_root)
    log_dir = ensure_directory(LOG_DIR if args.output_root == OUTPUT_ROOT else args.output_root / "_logs")
    jobs: list[ModelJob] = []
    for model_name in args.models:
        jobs.append(
            ModelJob(
                model_name=model_name,
                pixi_environment=MODEL_ENVS[model_name],
                script_path=MODEL_SCRIPTS[model_name],
                input_dir=args.input_dir.resolve(),
                output_dir=(args.output_root / model_name).resolve(),
                input_manifest=args.input_manifest.resolve() if args.input_manifest else None,
                workers=args.workers,
                ram_gb=int(MODEL_RAM_GB[model_name]),
                overwrite=bool(args.overwrite),
                limit=args.limit,
                log_path=(log_dir / f"{model_name}.log").resolve(),
            )
        )
    return jobs


def launch_job(job: ModelJob) -> dict[str, str | int]:
    command = [
        "pixi",
        "run",
        "-e",
        job.pixi_environment,
        "python",
        str(job.script_path),
        "--input-dir",
        str(job.input_dir),
        "--output-dir",
        str(job.output_dir),
        "--workers",
        str(job.workers),
        "--ram-limit-gb",
        str(job.ram_gb),
        "--gpu-index",
        "0",
    ]
    if job.input_manifest is not None:
        command.extend(["--input-manifest", str(job.input_manifest)])
    if job.limit is not None:
        command.extend(["--limit", str(job.limit)])
    if job.overwrite:
        command.append("--overwrite")

    ensure_directory(job.log_path.parent)
    ensure_directory(job.output_dir)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if job.gpu_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_device)

    with open(job.log_path, "w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "model_name": job.model_name,
        "exit_code": int(result.returncode),
        "log_path": str(job.log_path),
        "output_dir": str(job.output_dir),
        "gpu_device": -1 if job.gpu_device is None else int(job.gpu_device),
    }


def run_scheduler(args: argparse.Namespace) -> None:
    jobs = build_jobs(args)
    if not jobs:
        raise ValueError("No models were selected.")

    available_ram = int(args.ram_limit_gb)
    available_gpu_devices = list(range(max(1, int(args.gpu_slots))))
    pending_jobs = list(jobs)
    running: dict[object, ModelJob] = {}
    finished_results: list[dict[str, str | int]] = []

    print(f"[run_all] input_dir={args.input_dir.resolve()}")
    print(f"[run_all] output_root={args.output_root.resolve()}")
    print(f"[run_all] workers={args.workers} total_ram_limit_gb={args.ram_limit_gb}")
    print(f"[run_all] gpu_slots={args.gpu_slots}")
    print(f"[run_all] models={', '.join(job.model_name for job in jobs)}")

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        while pending_jobs or running:
            launched_job = False
            for job in list(pending_jobs):
                if job.ram_gb > available_ram or not available_gpu_devices:
                    continue
                gpu_device = available_gpu_devices.pop(0)
                scheduled_job = replace(job, gpu_device=gpu_device)
                future = executor.submit(launch_job, scheduled_job)
                running[future] = scheduled_job
                pending_jobs.remove(job)
                available_ram -= job.ram_gb
                launched_job = True
                print(
                    f"[run_all] started {scheduled_job.model_name} "
                    f"(gpu={gpu_device}, ram={scheduled_job.ram_gb}GB, "
                    f"available_ram={available_ram}GB, log={scheduled_job.log_path})"
                )
            if not running:
                raise RuntimeError(
                    "No job could be scheduled with the current RAM/GPU limits. "
                    "Increase RAM_LIMIT_GB, increase GPU_SLOTS, or lower MODEL_RAM_GB."
                )
            if launched_job and pending_jobs:
                continue

            completed, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in completed:
                job = running.pop(future)
                available_ram += job.ram_gb
                if job.gpu_device is not None:
                    available_gpu_devices.append(job.gpu_device)
                    available_gpu_devices.sort()
                result = future.result()
                finished_results.append(result)
                status = "ok" if int(result["exit_code"]) == 0 else "failed"
                print(
                    f"[run_all] finished {job.model_name} status={status} "
                    f"exit_code={result['exit_code']} gpu={result['gpu_device']} log={result['log_path']}"
                )

    failures = [result for result in finished_results if int(result["exit_code"]) != 0]
    if failures:
        failed_models = ", ".join(str(result["model_name"]) for result in failures)
        raise SystemExit(f"One or more model runs failed: {failed_models}")

    print("[run_all] all model runs completed successfully")


def main() -> None:
    args = parse_args()
    run_scheduler(args)


if __name__ == "__main__":
    main()
