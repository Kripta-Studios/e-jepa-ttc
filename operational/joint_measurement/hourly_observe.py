"""Observe one RGB driver on local clock hours; never restart or modify training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

from operational.train40_system.durable_io import atomic_json
from operational.train40_system.hourly_supervisor import next_hour

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/joint_measurement_20261012"
OUT = RUN / "rgb/hourly"


def read_optional(path: Path) -> dict:
    """Read an atomic metadata file, preserving a missing file as an empty state."""
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def observe(pid: int, created: float, previous: dict | None) -> dict:
    """Verify process identity and sample cheap progress/resource metadata once."""
    now = datetime.now().astimezone()
    alive = False
    memory = None
    try:
        process = psutil.Process(pid)
        alive = abs(process.create_time() - created) < 0.001 and process.is_running()
        if alive:
            memory = process.memory_info().rss
    except psutil.NoSuchProcess:
        pass
    fits = {
        path.parent.name: read_optional(path)
        for path in sorted((RUN / "rgb/fits").glob("*/UPDATE_JOURNAL.json"))
    }
    evaluation = {
        path.relative_to(RUN).as_posix(): read_optional(path).get("rows")
        for path in sorted((RUN / "rgb/predictions").glob("*/*/COMPLETE.json"))
    }
    fragments = len(list((RUN / "rgb/predictions").glob("**/chunk_*.parquet")))
    # A fit's journal is the authoritative counter; no checkpoint tensor is loaded.
    signature = {
        "updates": {name: value["completed_updates"] for name, value in fits.items()},
        "evaluation_complete": evaluation,
        "prediction_fragments": fragments,
    }
    runner = read_optional(RUN / "rgb/RUNNER_STATE.json")
    status = "RUNNING"
    if not alive:
        status = (
            "RGB_READY_FOR_INTERPRETATION"
            if runner.get("status") == "RGB_READY_FOR_INTERPRETATION"
            else "ACTION_REQUIRED_DRIVER_NOT_LIVE"
        )
    elif previous and signature == previous["progress_signature"]:
        status = "REVIEW_NO_PROGRESS_SINCE_PREVIOUS_HOUR"
    ledger = read_optional(RUN / "GPU_BUDGET.json")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return {
        "local_time": now.isoformat(),
        "status": status,
        "driver_pid": pid,
        "driver_create_time": created,
        "driver_live_identity_verified": alive,
        "driver_rss_bytes": memory,
        "host_available_bytes": psutil.virtual_memory().available,
        "gpu_observation": gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip(),
        "budget_reserved_upper_s": sum(row["charged_s"] for row in ledger["entries"]),
        "budget_cap_s": ledger["cap_s"],
        "fits": fits,
        "runner": runner,
        "progress_signature": signature,
        "training_actions": [],
        "scope": "Process identity, counters and resources; no scientific selection or restart",
    }


def run(pid: int) -> None:
    """Use one observer lock and sleep until each local hour, without intermediate polls."""
    import msvcrt

    process = psutil.Process(pid)
    if (
        "operational.joint_measurement.rgb_run" not in process.cmdline()
        or Path(process.cwd()).resolve() != ROOT
    ):
        raise ValueError("The observed PID must be the actual V18 RGB driver")
    created = process.create_time()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "OBSERVER.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            target = next_hour(datetime.now().astimezone())
            atomic_json(
                OUT / "ARMED.json",
                {
                    "pid": os.getpid(),
                    "driver_pid": pid,
                    "driver_create_time": created,
                    "next_local_hour": target.isoformat(),
                    "training_actions": [],
                },
            )
            previous = None
            while True:
                while (delay := target.timestamp() - time.time()) > 0:
                    time.sleep(min(60, delay))
                result = observe(pid, created, previous)
                atomic_json(OUT / f"{target.strftime('%Y%m%dT%H%M%S%z')}.json", result)
                atomic_json(OUT / "LATEST.json", result)
                print(json.dumps(result), flush=True)
                if not result["driver_live_identity_verified"]:
                    return
                previous = result
                target = next_hour(datetime.now().astimezone())
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    run(parser.parse_args().pid)
