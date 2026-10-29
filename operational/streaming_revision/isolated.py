"""Temporarily suspend owned CPU preparation while a bounded pilot measures latency."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from operational.train40_system.durable_io import atomic_json


def main() -> None:
    """Own only this campaign's preparation tree; always resume it after child exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-pid", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.benchmark_args[:1] == ["--"]:
        args.benchmark_args = args.benchmark_args[1:]
    parent = psutil.Process(args.prepare_pid)
    if "operational.sota_evidence.prepare_parallel" not in parent.cmdline():
        raise ValueError("PID is not the authorized campaign CPU preparer")
    if not 0 < args.timeout <= 600:
        raise ValueError("bounded timeout required")
    paused: list[psutil.Process] = []
    record: dict = {"status": "STARTING", "processes": [], "timeout_s": args.timeout}
    atomic_json(args.receipt, record)
    child = None
    started = time.monotonic()
    try:
        # Suspend the parent before enumerating descendants to prevent new workers.
        parent.suspend()
        paused.append(parent)
        for process in parent.children(recursive=True):
            if "python" in process.name().lower():
                process.suspend()
                paused.append(process)
        record.update(
            status="PAUSED_FOR_BOUNDED_MEASUREMENT",
            processes=[
                {"pid": p.pid, "created": p.create_time(), "command": p.cmdline()} for p in paused
            ],
        )
        atomic_json(args.receipt, record)
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-u",
            "-m",
            "operational.streaming_revision.benchmark",
            *args.benchmark_args,
        ]
        child = subprocess.Popen(
            command,
            env={**os.environ, "STREAMING_CPU_ISOLATION_RECEIPT": str(args.receipt.resolve())},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            code = child.wait(timeout=20)
            record["timeout_reached"] = True
        record["returncode"] = code
    finally:
        if child is not None and child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired) as exc:
                record["child_cleanup_error"] = repr(exc)
        failures = []
        for process in reversed(paused):
            try:
                if process.is_running():
                    process.resume()
            except psutil.Error as exc:
                failures.append({"pid": process.pid, "error": repr(exc)})
        record.update(
            status="RESUMED" if not failures else "RESUME_ERROR",
            resume_errors=failures,
            elapsed_s=time.monotonic() - started,
        )
        atomic_json(args.receipt, record)
    if record.get("returncode", 1):
        raise SystemExit(record.get("returncode", 1))


if __name__ == "__main__":
    main()
