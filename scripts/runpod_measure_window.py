"""Read-only, bounded telemetry for explicit Runpod training windows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--stop-run", action="append", default=[], type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1800 or args.interval < 1:
        raise ValueError("Measurement must be bounded to 30 minutes; interval >= 1 second")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    samples = []
    gpu_fields = (
        "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem"
    )
    processes: dict[int, psutil.Process] = {}
    with (args.output / "samples.jsonl").open("x", buffering=1) as stream:
        while True:
            elapsed = time.monotonic() - start
            sample = {"utc": datetime.now(UTC).isoformat(), "elapsed_s": elapsed, "fits": []}
            for fit_dir in args.fit_dir:
                progress = read_json(fit_dir / "PROGRESS.json")
                journal = read_json(fit_dir / "UPDATE_JOURNAL.json")
                entry = {"directory": str(fit_dir), "progress": progress, "journal": journal}
                pid = progress.get("pid")
                if pid:
                    try:
                        if pid not in processes:
                            processes[pid] = psutil.Process(pid)
                        proc = processes[pid]
                        with proc.oneshot():
                            entry["process"] = {
                                "pid": pid,
                                "cpu_percent": proc.cpu_percent(),
                                "cpu_times": proc.cpu_times()._asdict(),
                                "memory": proc.memory_info()._asdict(),
                                "io": proc.io_counters()._asdict(),
                                "threads": proc.num_threads(),
                                "status": proc.status(),
                            }
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                        entry["process_error"] = str(error)
                sample["fits"].append(entry)
            try:
                gpu = subprocess.run(
                    ["nvidia-smi", "--query-gpu=" + gpu_fields, "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                sample["gpu_fields"] = gpu_fields
                sample["gpu_values"] = gpu.stdout.strip()
                cuda = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=5,
                )
                sample["cuda_compute_pids"] = sorted(
                    int(line.strip()) for line in cuda.stdout.splitlines() if line.strip()
                )
            except (subprocess.SubprocessError, OSError) as error:
                sample["gpu_error"] = str(error)
            base = Path("/sys/fs/cgroup/memory")
            try:
                sample["cgroup_memory_bytes"] = int((base / "memory.usage_in_bytes").read_text())
                sample["cgroup_memory_stats"] = dict(
                    (key, int(value))
                    for key, value in (
                        line.split() for line in (base / "memory.stat").read_text().splitlines()
                    )
                )
            except (OSError, ValueError) as error:
                sample["cgroup_error"] = str(error)
            sample["host_cpu_percent"] = psutil.cpu_percent(percpu=True)
            stream.write(json.dumps(sample) + "\n")
            samples.append(sample)
            if elapsed >= args.seconds:
                break
            time.sleep(min(args.interval, max(0, args.seconds - (time.monotonic() - start))))
    result = {
        "schema": "ejepa_runpod_window_v1",
        "requested_seconds": args.seconds,
        "actual_seconds": samples[-1]["elapsed_s"] - samples[0]["elapsed_s"],
        "samples": len(samples),
        "fits": [],
        "gpu_utilization_is_sampled_not_kernel_timing": True,
        "host_cpu_is_not_pod_cpu_allocation": True,
    }
    for index, fit_dir in enumerate(args.fit_dir):
        first = samples[0]["fits"][index]["journal"]
        last = samples[-1]["fits"][index]["journal"]
        before, after = first.get("completed_updates"), last.get("completed_updates")
        delta = None if before is None or after is None else after - before
        result["fits"].append(
            {
                "directory": str(fit_dir),
                "before": before,
                "after": after,
                "updates": delta,
                "updates_per_min": None if delta is None else delta * 60 / result["actual_seconds"],
            }
        )
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    for run in args.stop_run:
        try:
            with (run / "STOP_REQUEST").open("x") as stop:
                stop.write("Bounded performance window completed; save at optimizer boundary.\n")
                stop.flush()
                os.fsync(stop.fileno())
        except FileExistsError:
            pass
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
