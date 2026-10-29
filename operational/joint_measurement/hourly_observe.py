"""Observe one registered branch on local clock hours; never modify training."""

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
JOBS = {
    "fusion_cost_millisecond": (
        "fusion/cost/raw_e2e_millisecond_v1",
        "fusion_cost_millisecond",
        "COMPLETE",
    ),
    "adaptation_evaluation": (
        "adaptive/A1_adaptation_depth",
        "adaptation_evaluate",
        "A1_READY_FOR_INTERPRETATION",
    ),
    "adaptation_depth": (
        "adaptive/A1_adaptation_depth",
        "adaptation_depth",
        "A1_ENDPOINTS_COMPLETE",
    ),
    "fusion_cost": (
        "fusion/cost/raw_e2e_historical_shape_v1",
        "fusion_cost_historical_shape",
        "COMPLETE",
    ),
    "rgb": ("rgb", "rgb_run", "RGB_READY_FOR_INTERPRETATION"),
    "geometry": ("adaptive/C1_geometry", "geometry_run", "C1_READY_FOR_INTERPRETATION"),
    "fusion_replication": (
        "replication/fusion_C1",
        "fusion_replication_train",
        "REPLICA_PRODUCERS_COMPLETE",
    ),
    "fusion_replication_pipeline": (
        "replication/fusion_C1",
        "fusion_replication_pipeline",
        "REPLICA_INPUTS_AND_HEADS_COMPLETE",
    ),
    "capacity": ("adaptive/D1_predictor_width", "capacity_train", "D1_ENDPOINT_COMPLETE"),
    "capacity_evaluation": (
        "adaptive/D1_predictor_width",
        "capacity_evaluate",
        "D1_READY_FOR_INTERPRETATION",
    ),
    "correspondence": (
        "adaptive/B1_correspondence",
        "correspondence_isolated",
        "B1_ENDPOINTS_COMPLETE",
    ),
    "correspondence_evaluation": (
        "adaptive/B1_correspondence",
        "correspondence_evaluate",
        "B1_READY_FOR_INTERPRETATION",
    ),
}
STATE_FILES = {
    "fusion_cost_millisecond": "COMPLETE.json",
    "adaptation_evaluation": "evidence/COMPLETE.json",
    "fusion_cost": "COMPLETE.json",
    "fusion_replication_pipeline": "PIPELINE_STATE.json",
    "capacity_evaluation": "evidence/COMPLETE.json",
    "correspondence_evaluation": "evidence/COMPLETE.json",
}
OBSERVATION_FOLDERS = {
    "adaptation_evaluation": "hourly_evaluation",
    "fusion_replication_pipeline": "hourly_pipeline",
    "capacity_evaluation": "hourly_evaluation",
    "correspondence_evaluation": "hourly_evaluation",
}
COST_JOBS = {"fusion_cost", "fusion_cost_millisecond"}


def read_optional(path: Path) -> dict:
    """Read an atomic metadata file, preserving a missing file as an empty state."""
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def observe(pid: int, created: float, previous: dict | None, *, job: str = "rgb") -> dict:
    """Verify process identity and sample cheap progress/resource metadata once."""
    relative, _, terminal = JOBS[job]
    directory = RUN / relative
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
        for path in sorted((directory / "fits").glob("*/UPDATE_JOURNAL.json"))
    }
    if job == "fusion_replication_pipeline":
        fits.update(
            {
                path.parent.name: read_optional(path)
                for path in sorted((directory / "head_fits").glob("*/UPDATE_JOURNAL.json"))
            }
        )
    evaluation = {
        path.relative_to(RUN).as_posix(): read_optional(path).get("rows")
        for path in sorted((directory / "predictions").glob("*/*/COMPLETE.json"))
    }
    fragments = len(list((directory / "predictions").glob("**/chunk_*.parquet")))
    # A fit's journal is the authoritative counter; no checkpoint tensor is loaded.
    signature = {
        "updates": {name: value["completed_updates"] for name, value in fits.items()},
        "evaluation_complete": evaluation,
        "prediction_fragments": fragments,
    }
    if job in COST_JOBS:
        signature["cost_rows"] = len(list(directory.glob("row_*.json")))
        signature["parity_failure"] = (directory / "PARITY_FAILURE.json").exists()
    if job == "fusion_replication_pipeline":
        signature["feature_fragments"] = len(list((directory / "inputs").glob("**/chunk_*.npz")))
        signature["feature_complete"] = {
            path.relative_to(directory).as_posix(): read_optional(path).get("rows")
            for path in sorted((directory / "inputs").glob("**/COMPLETE.json"))
        }
    state = STATE_FILES.get(job, "RUNNER_STATE.json")
    runner = read_optional(directory / state)
    status = "RUNNING"
    if not alive:
        status = terminal if runner.get("status") == terminal else "ACTION_REQUIRED_DRIVER_NOT_LIVE"
    elif previous and signature == previous["progress_signature"]:
        status = "REVIEW_NO_PROGRESS_SINCE_PREVIOUS_HOUR"
    if job in COST_JOBS and signature["parity_failure"]:
        status = "ACTION_REQUIRED_PARITY_FAILURE"
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
        "job": job,
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


def run(pid: int, *, job: str = "rgb") -> None:
    """Use one observer lock and sleep until each local hour, without intermediate polls."""
    import msvcrt

    relative, module, _ = JOBS[job]
    out = RUN / relative / OBSERVATION_FOLDERS.get(job, "hourly")
    process = psutil.Process(pid)
    if (
        not any(
            f"operational.joint_measurement.{name}" in process.cmdline()
            for name in (
                [module, "rgb_cached_run"]
                if job == "rgb"
                else [module, "geometry_capture_run"]
                if job == "geometry"
                else [
                    module,
                    "adaptation_runtime",
                    "adaptation_loss_graph_runtime",
                    "adaptation_observed_run",
                ]
                if job == "adaptation_depth"
                else [
                    module,
                    "correspondence_inference_graph",
                    "correspondence_prefetch",
                    "correspondence_export",
                ]
                if job == "correspondence_evaluation"
                else [module, "fusion_replication_isolated"]
                if job == "fusion_replication"
                else [module]
            )
        )
        or Path(process.cwd()).resolve() != ROOT
    ):
        raise ValueError(f"The observed PID must be the actual V18 {job} driver")
    created = process.create_time()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "OBSERVER.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            target = next_hour(datetime.now().astimezone())
            atomic_json(
                out / "ARMED.json",
                {
                    "pid": os.getpid(),
                    "job": job,
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
                result = observe(pid, created, previous, job=job)
                atomic_json(out / f"{target.strftime('%Y%m%dT%H%M%S%z')}.json", result)
                atomic_json(out / "LATEST.json", result)
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
    parser.add_argument("--job", choices=JOBS, default="rgb")
    args = parser.parse_args()
    run(args.pid, job=args.job)
