"""Windows-spawn admission: keep resource snapshots fresh during long initialization."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.resource_monitor_receiver import ContinuousC2FResourceMonitor


def main() -> None:
    """Exercise more snapshots than the previous blocked pipe, without calling guard."""
    output = ROOT / "artifacts/train40_system_20261005"
    destination = output / "h8_transport_admission/H8_RECEIVER_LIVE_ADMISSION.json"
    report: dict[str, Any] = {
        "status": "RUNNING",
        "source_sha256": digest(Path(__file__).with_name("resource_monitor_receiver.py")),
        "admission_source_sha256": digest(Path(__file__)),
        "optimizer_updates": 0,
        "idle_without_guard_seconds": 145,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(destination, report)
    monitor = ContinuousC2FResourceMonitor(output)
    try:
        monitor.start()
        begin = time.monotonic()
        time.sleep(145)
        allowed, telemetry = monitor.guard(output)
        snapshot = monitor.snapshot()
        if (
            telemetry["monitor_age_seconds"] > 12
            or "RESOURCE_MONITOR_STALE" in telemetry["reasons"]
        ):
            raise RuntimeError("Monitor receiver left stale data")
        if snapshot["snapshots_received"] < 27:
            raise RuntimeError("Did not exceed original blocked-pipe snapshot count")
        receiver = snapshot["continuous_receiver"]
        if receiver["error"] is not None or not receiver["thread_alive"]:
            raise RuntimeError("Receiver lifecycle failed")
        if snapshot["latest"]["torch_imported"] is not False:
            raise RuntimeError("Independent monitor imported Torch")
        report.update(
            status="PASSED",
            elapsed_without_guard_seconds=time.monotonic() - begin,
            guard_allowed=allowed,
            guard=telemetry,
            snapshot=snapshot,
        )
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        monitor.close()
        report["closed_snapshot"] = monitor.snapshot()
        report["finished_utc"] = datetime.now(UTC).isoformat()
        atomic_json(destination, report)


if __name__ == "__main__":
    main()
