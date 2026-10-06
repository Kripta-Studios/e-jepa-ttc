"""Run one frozen TRAIN40 arm under an authenticated paired-pilot lease."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, cast

import psutil

from operational.efficient_context.common import Lease, digest
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.parallel_runtime_v3 import MODULE, REGISTRY_NAME, read_object


def verify_admission(output: Path) -> dict[str, Any]:
    """Bind this wrapper and its frozen host-fast predecessor."""
    from operational.train40_system.contracts import read, verify_sources

    freeze = read(output / "PARALLEL_PILOT_V3_FREEZE.json")
    verify_sources(freeze)
    if freeze["engine_sha256"] != digest(Path(__file__)):
        raise ValueError("Parallel v3 engine differs from admission")
    predecessor = output / "HOST_FAST_4_FREEZE.json"
    if digest(predecessor) != freeze["host_fast_4_freeze_sha256"]:
        raise ValueError("Host-fast-4 predecessor differs")
    verify_sources(read(predecessor))
    pilot_v1 = output / "PARALLEL_PILOT_FREEZE.json"
    if digest(pilot_v1) != freeze["parallel_pilot_v1_freeze_sha256"]:
        raise ValueError("Parallel-pilot-v1 predecessor differs")
    verify_sources(read(pilot_v1))
    return cast(dict[str, Any], freeze)


def wait_for_registry(output: Path, token: str, arm: str, timeout: float = 120.0) -> None:
    """Wait until the coordinator binds this exact PID and command identity."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = output / REGISTRY_NAME
        if path.is_file():
            registry = read_object(path)
            row = registry.get("children", {}).get(arm)
            if (
                registry.get("token") == token
                and isinstance(row, dict)
                and int(row.get("pid", -1)) == os.getpid()
                and abs(float(row.get("create_time", -1)) - psutil.Process().create_time())
                <= 0.001
                and row.get("module") == MODULE
            ):
                return
        time.sleep(0.05)
    raise TimeoutError("Coordinator did not authenticate parallel child")


def run(output: Path, arm: str, token: str) -> None:
    """Patch only the process-local monitor class and delegate frozen training."""
    if arm not in {"a5", "c2f"} or not token:
        raise ValueError("An admitted arm and nonempty pilot token are required")
    verify_admission(output)
    from operational.train40_system import engine_host_fast_4, resource_monitor
    from operational.train40_system.parallel_runtime_v3 import ParallelResourceMonitor

    original_monitor = resource_monitor.ResourceMonitor
    status = "STARTING"
    runtime = output / "fits" / f"{arm}_seed7" / "PARALLEL_PILOT_V3_RUNTIME.json"
    try:
        wait_for_registry(output, token, arm)
        os.environ["TRAIN40_PARALLEL_ARM"] = arm
        os.environ["TRAIN40_PARALLEL_TOKEN"] = token
        resource_monitor.ResourceMonitor = cast(Any, ParallelResourceMonitor)
        atomic_json(runtime, {"status": status, "token": token, "arm": arm})
        engine_host_fast_4.run(output, arm)
        status = "DELEGATE_RETURNED"
    except BaseException:
        status = "FAILED"
        raise
    finally:
        resource_monitor.ResourceMonitor = original_monitor
        atomic_json(runtime, {"status": status, "token": token, "arm": arm})


def main() -> None:
    """Acquire only the fit-local lease; the coordinator owns the global lease."""
    parser = argparse.ArgumentParser(description=__doc__)
    from operational.train40_system.data_audit import OUTPUT

    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    process = psutil.Process()
    atomic_json(
        output / f"PARALLEL_{args.arm.upper()}_V3_READY.json",
        {
            "token": args.token,
            "arm": args.arm,
            "module": MODULE,
            "pid": process.pid,
            "create_time": process.create_time(),
            "output": str(output),
        },
    )
    with Lease(output / "fits" / f"{args.arm}_seed7"):
        run(output, args.arm, args.token)


if __name__ == "__main__":
    main()
