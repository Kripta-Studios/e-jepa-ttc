"""Coordinate the authorized balanced A5+C2F TRAIN40 parallel pilot."""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.parallel_runtime_v2 import (
    LEDGER_NAME,
    MODULE,
    REGISTRY_NAME,
    arm_journal,
    arm_target_complete,
    publish_upfront_reservation,
    read_object,
)

AUTHORIZATION_NAME = "PARALLEL_PILOT_AUTHORIZATION.json"
RESULT_NAME = "PARALLEL_PILOT_V2_RESULT.json"
PROGRESS_NAME = "PARALLEL_PILOT_V2_PROGRESS.json"


def verify_admission(output: Path) -> dict[str, Any]:
    """Require the explicit user authorization and published additive freeze."""
    authorization = read(output / AUTHORIZATION_NAME)
    if (
        authorization.get("allowed_concurrent_arms") != ["a5", "c2f"]
        or authorization.get("per_arm_scientific_quota") != {"a5": 1000, "c2f": 1000}
        or authorization.get("scientific_updates_in_measurement_total") != 2000
    ):
        raise ValueError("Parallel pilot authorization differs")
    freeze = read(output / "PARALLEL_PILOT_V2_FREEZE.json")
    verify_sources(freeze)
    if freeze["controller_sha256"] != digest(Path(__file__)):
        raise ValueError("Parallel controller differs from admission")
    if freeze["authorization_sha256"] != digest(output / AUTHORIZATION_NAME):
        raise ValueError("Parallel authorization differs from freeze")
    return freeze


def child_row(output: Path, token: str, arm: str, timeout: float = 120.0) -> dict[str, Any]:
    """Wait for and authenticate a child's exclusive ready receipt."""
    path = output / f"PARALLEL_{arm.upper()}_V2_READY.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            row = read_object(path)
            if row.get("token") == token and row.get("arm") == arm:
                process = psutil.Process(int(row["pid"]))
                if abs(process.create_time() - float(row["create_time"])) <= 0.001:
                    return row
        time.sleep(0.05)
    raise TimeoutError(f"Parallel {arm} child did not become ready")


def bind_launcher(row: dict[str, Any], process: subprocess.Popen[bytes]) -> dict[str, Any]:
    """Bind the Popen venv shim separately from the READY interpreter below it."""
    launcher = psutil.Process(process.pid)
    return {
        **row,
        "launcher": {"pid": launcher.pid, "create_time": launcher.create_time()},
    }


def wait_file(
    path: Path,
    token: str,
    process: subprocess.Popen[bytes],
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Wait for a token-bound barrier receipt."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            value = read_object(path)
            if value.get("token") == token:
                return value
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Parallel child exited before {path.name}: {code}")
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {path.name}")


def launch(output: Path, token: str, arm: str) -> subprocess.Popen[bytes]:
    """Launch one hidden arm with the same scoped CPU environment as frozen controllers."""
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--output",
        str(output),
        "--arm",
        arm,
        "--token",
        token,
    ]
    directory = output / "controller"
    directory.mkdir(exist_ok=True)
    log_path = directory / f"parallel_{arm}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    atomic_json(
        directory / f"parallel_{arm}_LATEST.json",
        {"pid": process.pid, "command": command, "log_path": str(log_path), "token": token},
    )
    return process


def gpu_preflight(output: Path) -> dict[str, int]:
    """Use prior admitted peaks and NVML CLI telemetry without initializing CUDA here."""
    a5 = read_object(output / "fits" / "a5_seed7" / "PROGRESS.json")
    c2f = read_object(output / "fits" / "c2f_seed7" / "PROGRESS.json")
    required = int(a5["peak_reserved_vram_bytes"]) + int(c2f["peak_reserved_vram_bytes"])
    required += 512 * 1024**2
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    first = query.stdout.strip().splitlines()[0]
    total_mib, free_mib = (int(part.strip()) for part in first.split(","))
    total, free = total_mib * 1024**2, free_mib * 1024**2
    c2f_margin = int(c2f["peak_reserved_vram_bytes"]) + 512 * 1024**2
    if required > 12 * 1024**3 or required > total or free < c2f_margin:
        raise RuntimeError("Parallel GPU admission lacks the admitted physical VRAM margin")
    return {"total_bytes": total, "free_bytes_before_c2f": free, "required_upper_bytes": required}


def gpu_postload() -> dict[str, int]:
    """Fail closed on aggregate device pressure after both models reach the barrier."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    total_mib, used_mib, free_mib = (
        int(part.strip()) for part in query.stdout.strip().splitlines()[0].split(",")
    )
    result = {
        "total_bytes": total_mib * 1024**2,
        "used_bytes": used_mib * 1024**2,
        "free_bytes": free_mib * 1024**2,
    }
    if result["used_bytes"] >= 12 * 1024**3:
        raise RuntimeError("Post-load aggregate GPU use exceeds the physical pilot cap")
    if result["free_bytes"] < 256 * 1024**2:
        raise RuntimeError("Post-load GPU reserve is below 256 MiB")
    return result


def final_arm_state(output: Path, arm: str, token: str) -> dict[str, Any]:
    """Bind one exact target checkpoint after normal quota exit."""
    if not arm_target_complete(output, arm, token):
        raise ValueError(f"{arm} did not finish its exact durable target")
    journal = arm_journal(output, arm)
    receipt_path = output / "fits" / f"{arm}_seed7" / "CHECKPOINT_RECEIPT.json"
    receipt = read_object(receipt_path)
    return {
        "committed_updates": int(journal["committed_updates"]),
        "durable_updates": int(journal["durable_updates"]),
        "recovery_upper": int(journal["recovery_upper"]),
        "pending_update_upper": int(journal["pending_update_upper"]),
        "checkpoint_sha256": receipt["sha256"],
        "checkpoint_receipt_sha256": digest(receipt_path),
    }


def request_peer_pause(output: Path, token: str, failed_arm: str) -> None:
    """Pause the surviving peer at its next optimizer boundary."""
    marker = output / "COORDINATION_PAUSE_REQUEST.json"
    if marker.exists():
        return
    atomic_json(
        marker,
        {
            "owner": f"parallel_pilot_{token}",
            "reason": f"parallel peer {failed_arm} exited before exact paired completion",
            "requested_utc": datetime.now(UTC).isoformat(),
        },
    )


def wait_for_paused_survivors(
    output: Path, token: str, processes: dict[str, subprocess.Popen[bytes]]
) -> None:
    """Keep the global lease until every live peer exits with a full boundary checkpoint."""
    survivors = {arm for arm, process in processes.items() if process.poll() is None}
    if not survivors:
        return
    request_peer_pause(output, token, "coordinator_failure")
    while any(processes[arm].poll() is None for arm in survivors):
        atomic_json(
            output / PROGRESS_NAME,
            {
                "status": "WAITING_FOR_PEER_CHECKPOINT",
                "token": token,
                "live_survivors": [
                    arm for arm in sorted(survivors) if processes[arm].poll() is None
                ],
            },
        )
        time.sleep(0.5)
    for arm in survivors:
        journal = arm_journal(output, arm)
        receipt_path = output / "fits" / f"{arm}_seed7" / "CHECKPOINT_RECEIPT.json"
        receipt = read_object(receipt_path)
        checkpoint = output / "fits" / f"{arm}_seed7" / "checkpoint_last.pt"
        if (
            int(journal["pending_update_upper"]) != 0
            or int(journal["committed_updates"]) != int(journal["durable_updates"])
            or int(receipt["committed_updates"]) != int(journal["committed_updates"])
            or digest(checkpoint) != receipt.get("sha256")
        ):
            raise RuntimeError(f"Paused {arm} peer lacks a full boundary checkpoint")


def run(output: Path) -> None:
    """Warm A5 alone, arm exact balanced baselines, then supervise both to full CPs."""
    verify_admission(output)
    if (output / "COORDINATION_PAUSE_REQUEST.json").exists() or (output / "STOP_REQUEST").exists():
        raise RuntimeError("A stale pause marker blocks parallel launch")
    token = secrets.token_hex(16)
    for name in (REGISTRY_NAME, LEDGER_NAME):
        if (output / name).exists():
            raise FileExistsError(f"Preserve prior parallel artifact: {name}")
    begun = time.perf_counter()
    begun_utc = datetime.now(UTC).isoformat()
    processes: dict[str, subprocess.Popen[bytes]] = {}
    children: dict[str, dict[str, Any]] = {}
    status = "STARTING_A5_WARMUP"
    try:
        processes["a5"] = launch(output, token, "a5")
        children["a5"] = bind_launcher(
            child_row(output, token, "a5"), processes["a5"]
        )
        registry = {
            "schema": "train40_parallel_pilot_v2_registry_v1",
            "token": token,
            "phase": "A5_WARMUP",
            "coordinator": {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "module": "operational.train40_system.parallel_controller_v2",
            },
            "children": children,
            "freeze_sha256": digest(output / "PARALLEL_PILOT_V2_FREEZE.json"),
            "authorization_sha256": digest(output / AUTHORIZATION_NAME),
        }
        atomic_json(output / REGISTRY_NAME, registry)
        wait_file(output / "PARALLEL_A5_V2_BARRIER.json", token, processes["a5"])
        gpu = gpu_preflight(output)

        processes["c2f"] = launch(output, token, "c2f")
        children["c2f"] = bind_launcher(
            child_row(output, token, "c2f"), processes["c2f"]
        )
        registry.update({"phase": "PAIR_STARTING", "children": children, "gpu_preflight": gpu})
        atomic_json(output / REGISTRY_NAME, registry)
        wait_file(output / "PARALLEL_C2F_V2_BARRIER.json", token, processes["c2f"])
        registry["gpu_postload"] = gpu_postload()
        registry["phase"] = "BOTH_AT_BARRIER"
        atomic_json(output / REGISTRY_NAME, registry)
        ledger = publish_upfront_reservation(output, token)
        armed_utc = datetime.now(UTC).isoformat()
        armed = time.perf_counter()
        status = "RUNNING"

        completed: dict[str, dict[str, Any]] = {}
        next_progress_at = 0.0
        while len(completed) < 2:
            state_changed = False
            for arm, process in processes.items():
                if arm in completed:
                    continue
                code = process.poll()
                if code is None:
                    continue
                if code != 0 or not arm_target_complete(output, arm, token):
                    request_peer_pause(output, token, arm)
                    raise RuntimeError(f"Parallel {arm} exited before exact target: {code}")
                completed[arm] = final_arm_state(output, arm, token)
                state_changed = True
                registry["children"][arm]["status"] = "TARGET_COMPLETE"
                registry["children"][arm]["completed_utc"] = datetime.now(UTC).isoformat()
                atomic_json(output / REGISTRY_NAME, registry)
            now = time.monotonic()
            if state_changed or now >= next_progress_at:
                atomic_json(
                    output / PROGRESS_NAME,
                    {
                        "status": status,
                        "token": token,
                        "arms": {
                            arm: {
                                "committed_updates": int(
                                    arm_journal(output, arm)["committed_updates"]
                                ),
                                "target_committed": int(ledger["arms"][arm]["target_committed"]),
                                "process_alive": process.poll() is None,
                            }
                            for arm, process in processes.items()
                        },
                        "elapsed_seconds": time.perf_counter() - armed,
                    },
                )
                next_progress_at = now + 5.0
            if len(completed) < 2:
                time.sleep(0.5)
        status = "COMPLETE"
        atomic_json(
            output / RESULT_NAME,
            {
                "schema": "train40_parallel_pilot_v2_result_v1",
                "status": status,
                "token": token,
                "started_utc": begun_utc,
                "armed_utc": armed_utc,
                "completed_utc": datetime.now(UTC).isoformat(),
                "startup_seconds": armed - begun,
                "measurement_seconds": time.perf_counter() - armed,
                "combined_scientific_updates": 2000,
                "additional_optimizer_updates": 0,
                "arms": completed,
                "ledger_sha256": digest(output / LEDGER_NAME),
                "freeze_sha256": digest(output / "PARALLEL_PILOT_V2_FREEZE.json"),
            },
        )
    except BaseException:
        status = "FAILED"
        wait_for_paused_survivors(output, token, processes)
        raise
    finally:
        atomic_json(
            output / PROGRESS_NAME,
            {"status": status, "token": token, "elapsed_seconds": time.perf_counter() - begun},
        )


def main() -> None:
    """Own the sole global campaign lease while both fit-local writers run."""
    parser = argparse.ArgumentParser(description=__doc__)
    from operational.train40_system.data_audit import OUTPUT

    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    with Lease(output):
        run(output)


if __name__ == "__main__":
    main()
