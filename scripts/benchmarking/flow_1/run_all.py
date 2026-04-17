from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import subprocess
import time
from typing import TextIO

from common import FLOW_ROOT, PROJECT_ROOT, ensure_directory, format_elapsed, log, resolve_path

DATASET_TAG = "benchmark_input"
INPUT_DIR = PROJECT_ROOT / "data" / DATASET_TAG / "tiles_256"
OUTPUT_ROOT = PROJECT_ROOT / "inference" / "benchmarking" / DATASET_TAG
INPUT_MANIFEST = None
WORKERS = 12
RAM_LIMIT_GB = 128
GPU_SLOTS = 1
OVERWRITE = False
LIMIT = None
MODELS_TO_RUN = ("cellpose_sam", "cellsam", "cellvit_sam", "stardist")
MODEL_RAM_GB = {
    "cellpose_sam": 16,
    "cellsam": 16,
    "cellvit_sam": 16,
    "stardist": 16,
}
MODEL_ENVS = {
    "cellpose_sam": "cellpose",
    "cellsam": "cellsam",
    "cellvit_sam": "cellvit",
    "stardist": "stardist",
}
MODEL_SCRIPTS = {
    "cellpose_sam": FLOW_ROOT / "run_cellpose_sam.py",
    "cellsam": FLOW_ROOT / "run_cellsam.py",
    "cellvit_sam": FLOW_ROOT / "run_cellvit_sam.py",
    "stardist": FLOW_ROOT / "run_stardist.py",
}

FLOW_NAME = "flow_1.predict_all"


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


@dataclass
class RunningJob:
    job: ModelJob
    process: subprocess.Popen[str]
    log_handle: TextIO
    gpu_device: int | None
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the four Flow 1 benchmarking inference wrappers.")
    parser.add_argument("--in", "--input-dir", dest="input_dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", "--output-root", dest="output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", "--input-manifest", dest="input_manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--ram-limit-gb", type=int, default=RAM_LIMIT_GB)
    parser.add_argument("--gpu-slots", type=int, default=GPU_SLOTS)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--models", nargs="+", default=list(MODELS_TO_RUN), choices=sorted(MODEL_SCRIPTS))
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[ModelJob]:
    output_root = resolve_path(args.output_root)
    input_dir = resolve_path(args.input_dir)
    input_manifest = resolve_path(args.input_manifest)
    ensure_directory(output_root)
    log_dir = ensure_directory(output_root / "_logs")

    jobs: list[ModelJob] = []
    for model_name in args.models:
        jobs.append(
            ModelJob(
                model_name=model_name,
                pixi_environment=MODEL_ENVS[model_name],
                script_path=MODEL_SCRIPTS[model_name],
                input_dir=input_dir,
                output_dir=(output_root / model_name).resolve(),
                input_manifest=input_manifest,
                workers=int(args.workers),
                ram_gb=int(MODEL_RAM_GB[model_name]),
                overwrite=bool(args.overwrite),
                limit=args.limit,
                log_path=(log_dir / f"{model_name}.log").resolve(),
            )
        )
    return jobs


def _job_command(job: ModelJob) -> list[str]:
    command = [
        "pixi",
        "run",
        "-e",
        job.pixi_environment,
        "python",
        str(job.script_path),
        "--in",
        str(job.input_dir),
        "--out",
        str(job.output_dir),
        "--workers",
        str(job.workers),
        "--ram-limit-gb",
        str(job.ram_gb),
        "--gpu-index",
        "0",
    ]
    if job.input_manifest is not None:
        command.extend(["--manifest", str(job.input_manifest)])
    if job.limit is not None:
        command.extend(["--limit", str(job.limit)])
    if job.overwrite:
        command.append("--overwrite")
    return command


def _start_job(job: ModelJob, gpu_device: int | None) -> RunningJob:
    ensure_directory(job.log_path.parent)
    ensure_directory(job.output_dir)
    log_handle = job.log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

    command = _job_command(job)
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(
        job=job,
        process=process,
        log_handle=log_handle,
        gpu_device=gpu_device,
        started_at=time.perf_counter(),
    )


def run_scheduler(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    jobs = build_jobs(args)
    if not jobs:
        raise ValueError("No models were selected.")

    available_ram = int(args.ram_limit_gb)
    available_gpu_devices = list(range(max(1, int(args.gpu_slots))))
    pending_jobs = list(jobs)
    running_jobs: list[RunningJob] = []
    failures: list[str] = []

    log(FLOW_NAME, f"input_dir={resolve_path(args.input_dir)}")
    log(FLOW_NAME, f"output_root={resolve_path(args.output_root)}")
    log(FLOW_NAME, f"input_manifest={resolve_path(args.input_manifest) if args.input_manifest else 'auto'}")
    log(FLOW_NAME, f"models={', '.join(args.models)}")
    log(FLOW_NAME, f"workers={args.workers} total_ram_limit_gb={args.ram_limit_gb} gpu_slots={args.gpu_slots}")

    while pending_jobs or running_jobs:
        launched_job = False
        for job in list(pending_jobs):
            if job.ram_gb > available_ram or not available_gpu_devices:
                continue

            gpu_device = available_gpu_devices.pop(0)
            running_job = _start_job(job, gpu_device=gpu_device)
            running_jobs.append(running_job)
            pending_jobs.remove(job)
            available_ram -= job.ram_gb
            launched_job = True

            log(
                FLOW_NAME,
                "started "
                f"{job.model_name} gpu={gpu_device} ram={job.ram_gb}GB "
                f"available_ram={available_ram}GB log={job.log_path}",
            )

        if not running_jobs and pending_jobs:
            raise RuntimeError(
                "No job could be scheduled with the current RAM/GPU limits. "
                "Increase --ram-limit-gb, increase --gpu-slots, or lower MODEL_RAM_GB."
            )

        if not running_jobs:
            break

        time.sleep(1.0 if launched_job else 2.0)

        for running_job in list(running_jobs):
            return_code = running_job.process.poll()
            if return_code is None:
                continue

            running_jobs.remove(running_job)
            running_job.log_handle.close()
            available_ram += running_job.job.ram_gb
            if running_job.gpu_device is not None:
                available_gpu_devices.append(running_job.gpu_device)
                available_gpu_devices.sort()

            elapsed = time.perf_counter() - running_job.started_at
            status = "ok" if return_code == 0 else "failed"
            log(
                FLOW_NAME,
                "finished "
                f"{running_job.job.model_name} status={status} exit_code={return_code} "
                f"gpu={running_job.gpu_device} elapsed={format_elapsed(elapsed)} "
                f"log={running_job.job.log_path}",
            )
            if return_code != 0:
                failures.append(running_job.job.model_name)

    total_elapsed = time.perf_counter() - started_at
    if failures:
        raise SystemExit(
            "One or more model runs failed: "
            + ", ".join(failures)
            + f" | elapsed={format_elapsed(total_elapsed)}"
        )

    log(FLOW_NAME, f"all model runs completed elapsed={format_elapsed(total_elapsed)}")


def main() -> None:
    args = parse_args()
    run_scheduler(args)


if __name__ == "__main__":
    main()
