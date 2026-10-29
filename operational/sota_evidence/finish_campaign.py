"""Wait for CPU inputs, run bounded inference, and release our V13 pause on every exit."""

# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from operational.train40_system.durable_io import atomic_json


def read(path: Path) -> dict[str, Any]:
    """Read UTF-8 JSON written by either Python or PowerShell."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resume_v13(output: Path) -> None:
    """Remove only this campaign's own markers and resume the current IO recovery queue."""
    ownership = read(output / "V13_PAUSE_OWNERSHIP.json")
    run = Path(ownership["run"])
    paths = [Path(value) for value in ownership["paths"]]
    for path in paths:
        if path.read_text(encoding="utf-8") != ownership["marker"]:
            raise RuntimeError(f"Pause marker ownership changed: {path}")
    # Release the supervisor last, after both producer stop requests are gone.
    for path in reversed(paths):
        path.unlink()
    queues = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if "operational.rgb_port_io_recovery.queue" in command and "--run" in command:
            if Path(command[command.index("--run") + 1]).resolve() == run.resolve():
                queues.append(process.pid)
    if queues:
        pid = queues[0]
    else:
        repository = run.parents[1]
        env = dict(os.environ, PYTHONPATH=str(repository / "src") + os.pathsep + str(repository))
        with (
            (output / "v13_resume.stdout.log").open("ab") as stdout,
            (output / "v13_resume.stderr.log").open("ab") as stderr,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "operational.rgb_port_io_recovery.queue",
                    "resume",
                    "--run",
                    str(run),
                ],
                cwd=repository,
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        pid = process.pid
    atomic_json(
        output / "V13_RESUME_REQUEST.json",
        {
            "status": "RESUME_REQUESTED",
            "pid": pid,
            "utc": datetime.now(UTC).isoformat(),
            "module": "operational.rgb_port_io_recovery.queue",
            "owned_markers_removed": True,
        },
    )


def run(
    output: Path, cache: Path, campaign: Path, *, budget_seconds: float, prepare_pid: int
) -> None:
    """Keep all waiting on CPU and account for every GPU QA and inference reservation."""
    status: dict[str, Any] = {"status": "WAITING_FOR_CPU_INPUTS"}
    owner = output / "FINISH_OWNER.json"
    if owner.exists() and psutil.pid_exists(int(read(owner)["pid"])):
        raise RuntimeError("Another completion supervisor is active")
    atomic_json(owner, {"pid": os.getpid(), "created": psutil.Process().create_time()})
    deadline = time.monotonic() + 6 * 3600
    try:
        while True:
            prepared = output / "PREPARATION_RESULT.json"
            if prepared.exists() and read(prepared)["status"] == "COMPLETE":
                break
            if not psutil.pid_exists(prepare_pid):
                raise RuntimeError("CPU preparation exited before complete coverage")
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU preparation/download wait exceeded six hours")
            atomic_json(
                output / "CAMPAIGN_STATUS.json",
                {
                    **status,
                    "utc": datetime.now(UTC).isoformat(),
                },
            )
            time.sleep(30)
        used = sum(
            float(read(path)["seconds"])
            for pattern in (
                "GPU_QA_ACCOUNTING_*.json",
                "GPU_RESERVATION_*.json",
            )
            for path in output.glob(pattern)
        )
        failed = output / "GPU_QA_FIRST_FAILURE.json"
        if failed.exists():
            used += float(read(failed)["gpu_reservation_upper_bound_seconds"])
        remaining = budget_seconds - used
        if remaining <= 60:
            raise RuntimeError("No remaining authorized GPU reservation budget")
        atomic_json(
            output / "CAMPAIGN_STATUS.json",
            {
                "status": "GPU_INFERENCE",
                "remaining_gpu_budget_seconds": remaining,
                "utc": datetime.now(UTC).isoformat(),
            },
        )
        with (
            (output / "inference.stdout.log").open("ab") as stdout,
            (output / "inference.stderr.log").open("ab") as stderr,
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-u",
                    "-m",
                    "operational.sota_evidence.predict_test",
                    "--campaign",
                    str(campaign),
                    "--output",
                    str(output),
                    "--cache",
                    str(cache),
                    "--budget-seconds",
                    str(remaining),
                ],
                stdout=stdout,
                stderr=stderr,
                timeout=remaining + 120,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        if result.returncode:
            raise RuntimeError(f"Inference exited with code {result.returncode}; see inference log")
        status = read(output / "INFERENCE_RESULT.json")
    except Exception as error:
        status = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
        raise
    finally:
        atomic_json(
            output / "CAMPAIGN_STATUS.json",
            {
                **status,
                "utc": datetime.now(UTC).isoformat(),
            },
        )
        try:
            resume_v13(output)
        finally:
            owner.unlink(missing_ok=True)


def main() -> None:
    """Coordinate only the previously authorized local campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "cache", "campaign"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--prepare-pid", type=int, required=True)
    args = parser.parse_args()
    run(
        args.output,
        args.cache,
        args.campaign,
        budget_seconds=args.budget_seconds,
        prepare_pid=args.prepare_pid,
    )


if __name__ == "__main__":
    main()
