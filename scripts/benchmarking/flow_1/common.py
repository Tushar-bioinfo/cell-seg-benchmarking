from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not resolve the repository root from the current script location.")


PROJECT_ROOT = find_repo_root(Path(__file__).resolve())
FLOW_ROOT = Path(__file__).resolve().parent


def resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    return Path(path_like).expanduser().resolve()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:d}m {remaining_seconds:0.1f}s"
    if minutes:
        return f"{minutes:d}m {remaining_seconds:0.1f}s"
    return f"{remaining_seconds:0.1f}s"


def log(name: str, message: str) -> None:
    print(f"[{name}] {message}")


def append_option(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def append_repeatable(command: list[str], flag: str, values: list[object] | tuple[object, ...] | None) -> None:
    if not values:
        return
    for value in values:
        command.extend([flag, str(value)])


def append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def python_command(script_path: Path) -> list[str]:
    return [sys.executable, str(script_path)]


def run_command(
    command: list[str],
    *,
    name: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log(name, f"command={shlex.join(command)}")
    started_at = time.perf_counter()
    try:
        subprocess.run(
            command,
            cwd=str(cwd or PROJECT_ROOT),
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - started_at
        log(name, f"failed exit_code={exc.returncode} elapsed={format_elapsed(elapsed)}")
        raise SystemExit(exc.returncode) from exc

    elapsed = time.perf_counter() - started_at
    log(name, f"completed elapsed={format_elapsed(elapsed)}")
