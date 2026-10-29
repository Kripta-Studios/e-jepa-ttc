"""Low-overhead external telemetry for an unchanged RGB training process."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def _commit() -> dict[str, int] | None:
    class Performance(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ],
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    value = Performance()
    value.cb = ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value), value.cb):
        return None
    return {
        "total": int(value.CommitTotal * value.PageSize),
        "limit": int(value.CommitLimit * value.PageSize),
        "peak": int(value.CommitPeak * value.PageSize),
    }


def _gpu() -> dict[str, float] | None:
    fields = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        values = [float(item.strip()) for item in result.stdout.strip().split(",")]
        return dict(zip(fields.split(","), values, strict=True))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Sample external counters without importing or attaching to PyTorch."""
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(args.pid)
    process.cpu_percent(None)
    start_wall, start_mono = datetime.now(UTC), time.perf_counter()
    overhead = 0.0
    samples = 0
    first_update = last_update = None
    deadline = start_mono + args.duration_seconds
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        while time.perf_counter() < deadline:
            sample_start = time.perf_counter()
            if not process.is_running():
                break
            journal = _json(args.journal)
            runtime = _json(args.runtime)
            memory = process.memory_info()
            io = process.io_counters()
            system_memory = psutil.virtual_memory()
            checkpoint = args.checkpoint
            row = {
                "utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - start_mono,
                "pid": args.pid,
                "process_cpu_percent": process.cpu_percent(None),
                "process_threads": process.num_threads(),
                "process_rss_bytes": memory.rss,
                "process_vms_bytes": memory.vms,
                "process_read_bytes": io.read_bytes,
                "process_write_bytes": io.write_bytes,
                "system_cpu_percent": psutil.cpu_percent(None),
                "system_ram_available_bytes": system_memory.available,
                "system_ram_percent": system_memory.percent,
                "system_commit": _commit(),
                "gpu": _gpu(),
                "journal": journal,
                "runtime_prefetch": None if runtime is None else runtime.get("prefetch"),
                "checkpoint": {
                    "exists": checkpoint.is_file(),
                    "size": checkpoint.stat().st_size if checkpoint.is_file() else None,
                    "mtime_ns": checkpoint.stat().st_mtime_ns if checkpoint.is_file() else None,
                },
            }
            if journal is not None:
                update = int(journal.get("completed_updates", -1))
                first_update = update if first_update is None else first_update
                last_update = update
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            samples += 1
            overhead += time.perf_counter() - sample_start
            remaining = args.interval_seconds - (time.perf_counter() - sample_start)
            if remaining > 0:
                time.sleep(remaining)
    end_wall = datetime.now(UTC)
    result = {
        "schema": "final_refit_external_runtime_monitor_v1",
        "status": "COMPLETE",
        "pid": args.pid,
        "start_utc": start_wall.isoformat(),
        "end_utc": end_wall.isoformat(),
        "elapsed_seconds": time.perf_counter() - start_mono,
        "interval_seconds": args.interval_seconds,
        "samples": samples,
        "monitor_sampling_seconds": overhead,
        "monitor_overhead_fraction": overhead / max(time.perf_counter() - start_mono, 1e-9),
        "first_completed_update": first_update,
        "last_completed_update": last_update,
        "observed_updates": None if first_update is None else last_update - first_update,
        "jsonl": str(output),
    }
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pid", type=int, required=True)
    result.add_argument("--journal", type=Path, required=True)
    result.add_argument("--runtime", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--duration-seconds", type=float, default=905.0)
    result.add_argument("--interval-seconds", type=float, default=5.0)
    return result


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
