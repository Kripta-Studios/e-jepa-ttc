"""Passive bounded telemetry: no training, inference, recursive scans or CUDA allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

DEADLINE = 1793145600.0
GPU_FIELDS = (
    "index,name,uuid,driver_version,utilization.gpu,utilization.memory,memory.used,"
    "memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem"
)
STATUS_PATHS = (
    "/workspace/ejepa_final_operations/FINAL_QUEUE_STATUS.json",
    "/workspace/ejepa_final_runs/final/TRAINING_EXPORT_STATUS.json",
    "/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json",
    "/workspace/ejepa_test12_probe/STATUS.json",
    "/workspace/ejepa_test12_probe/DOWNLOAD_RESERVATIONS.json",
    "/workspace/ejepa_test12_events/STATUS.json",
    "/workspace/ejepa_test12_rgb/CONTROLLER_STATUS.json",
)


def utc() -> str:
    return datetime.now(UTC).isoformat()


def read_status(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        if stat.st_size > 2 * 1024**2:
            return {"error": "STATUS_EXCEEDS_2_MIB_LIMIT", "bytes": stat.st_size}
        return {
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "age_seconds": time.time() - stat.st_mtime,
            "value": json.loads(path.read_text()),
        }
    except (OSError, ValueError) as error:
        return {"error": type(error).__name__}


def cgroup() -> dict[str, Any]:
    result = {}
    # Files belong to the container's mounted cgroup namespace; missing values stay explicit.
    for relative in (
        "cpu.max",
        "cpu.stat",
        "cpuset.cpus.effective",
        "memory.current",
        "memory.max",
        "memory.events",
        "memory.stat",
        "io.stat",
        "memory/memory.stat",
        "cpuacct/cpuacct.usage",
        "cpuacct/cpuacct.stat",
        "blkio/blkio.throttle.io_service_bytes",
        "cpuset/cpuset.cpus",
        "memory/memory.usage_in_bytes",
        "memory/memory.limit_in_bytes",
        "cpu/cpu.stat",
        "cpu/cpu.cfs_quota_us",
        "cpu/cpu.cfs_period_us",
    ):
        path = Path("/sys/fs/cgroup") / relative
        try:
            result[relative] = path.read_text().strip()
        except OSError:
            pass
    return result


def snapshot(paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "utc": utc(),
        "unix_seconds": time.time(),
        "monotonic_seconds": time.monotonic(),
        "host_cpu_percent": psutil.cpu_percent(),
        "host_cpu_times_seconds": psutil.cpu_times()._asdict(),
        "host_memory_bytes": psutil.virtual_memory()._asdict(),
        "container_cgroup": cgroup(),
        "network_counters_scope": "visible network namespace cumulative bytes",
        "network_counters": psutil.net_io_counters()._asdict(),
        "status": {str(path): read_status(path) for path in paths},
    }
    disk = psutil.disk_io_counters()
    result["visible_disk_counters"] = disk._asdict() if disk else None
    result["disk_counters_scope"] = "visible block devices; network volume IO may not appear"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=4,
        )
        result["gpu_fields"] = GPU_FIELDS.split(",")
        result["gpu_rows"] = [line.split(", ") for line in gpu.stdout.strip().splitlines()]
    except (OSError, subprocess.SubprocessError) as error:
        result["gpu_error"] = str(error)
    pids = {os.getpid()}
    for status in result["status"].values():
        value = status.get("value", {})
        if isinstance(value, dict):
            pid = value.get("pid")
            if isinstance(pid, int):
                pids.add(pid)
            running = value.get("running", {})
            if isinstance(running, dict):
                pids.update(pid for pid in running.values() if isinstance(pid, int))
    result["known_processes"] = {}
    for pid in pids:
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                result["known_processes"][str(pid)] = {
                    "created": process.create_time(),
                    "cpu_times": process.cpu_times()._asdict(),
                    "memory": process.memory_info()._asdict(),
                    "io": process.io_counters()._asdict(),
                    "threads": process.num_threads(),
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
            result["known_processes"][str(pid)] = {"error": type(error).__name__}
    return result


def publish(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=DEADLINE)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--status", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.interval < 10 or args.deadline > DEADLINE:
        raise ValueError("Sampling interval >=10s; deadline cannot exceed authorized 00:00 UTC")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [Path(value) for value in STATUS_PATHS] + args.status
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    with (args.output / "COLLECTOR.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = utc()
        name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        metadata = {
            "schema": "ejepa_passive_diagnostics_v1",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pid": os.getpid(),
            "started_utc": started,
            "deadline_utc": datetime.fromtimestamp(args.deadline, UTC).isoformat(),
            "interval_seconds": args.interval,
            "platform": platform.platform(),
            "cpu_count_logical_visible": psutil.cpu_count(),
            "claims": {
                "optimizer_updates": 0,
                "inference_executed": False,
                "gpu_kernel_latency_measured": False,
                "recursive_disk_scan": False,
            },
            "limitations": [
                "Host CPU/RAM may exceed container entitlement; use cgroup values.",
                "GPU samples describe device occupancy, not kernel latency.",
                "No attribution of shared device utilization to one process.",
                "Stage wall time includes CPU, IO, scheduling and GPU waits.",
            ],
            "status_paths": [str(path) for path in paths],
        }
        publish(args.output / f"{name}_metadata.json", metadata)
        samples = 0
        durations: list[float] = []
        psutil.cpu_percent()
        with (args.output / f"{name}_samples.jsonl").open("x", buffering=1) as stream:
            while time.time() < args.deadline and not stopping:
                before = time.monotonic()
                additional = read_status(args.output / "STATUS_SOURCES.json").get("value", [])
                if not isinstance(additional, list) or len(additional) > 64:
                    raise ValueError("Diagnostic status source list must contain at most 64 paths")
                dynamic = [
                    Path(item)
                    for item in additional
                    if isinstance(item, str) and item.startswith("/workspace/ejepa")
                ]
                value = snapshot(list(dict.fromkeys(paths + dynamic)))
                elapsed = time.monotonic() - before
                value["collector_wall_seconds"] = elapsed
                value["collector_wall_seconds_scope"] = "snapshot_only_excludes_output_writes_fsync"
                durations.append(elapsed)
                stream.write(json.dumps(value, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                samples += 1
                publish(
                    args.output / "COLLECTOR_STATUS.json",
                    {
                        "pid": os.getpid(),
                        "utc": utc(),
                        "status": "RUNNING",
                        "samples": samples,
                        "sample_file": str(args.output / f"{name}_samples.jsonl"),
                        "collector_wall_seconds_total": sum(durations),
                        "collector_wall_seconds_max": max(durations),
                        "deadline_utc": metadata["deadline_utc"],
                    },
                )
                delay = min(max(0, args.interval - elapsed), max(0, args.deadline - time.time()))
                until = time.monotonic() + delay
                while time.monotonic() < until and not stopping:
                    time.sleep(min(1, until - time.monotonic()))
        publish(
            args.output / f"{name}_closure.json",
            {
                "started_utc": started,
                "ended_utc": utc(),
                "samples": samples,
                "status": "SIGNALLED" if stopping else "DEADLINE",
                "optimizer_updates": 0,
                "collector_wall_seconds_total": sum(durations),
            },
        )


if __name__ == "__main__":
    main()
