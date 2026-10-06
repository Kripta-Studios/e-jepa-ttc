"""Finish only the already reserved C2F pilot tail through the frozen serial engine."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.train40_system import resource_monitor


class TailMonitor(resource_monitor.ResourceMonitor):
    """Retain every serial resource check and stop before the reserved target."""

    def guard(self, output: Path) -> tuple[bool, dict[str, Any]]:
        """Read the original reservation without creating a new update allowance."""
        from operational.train40_system.parallel_runtime_v4 import read_object

        allowed, telemetry = super().guard(output)
        ledger = read_object(output / "PARALLEL_PILOT_V4_LEDGER.json")
        journal = read_object(output / "fits/c2f_seed7/UPDATE_JOURNAL.json")
        target = int(ledger["arms"]["c2f"]["target_committed"])
        if int(journal["committed_updates"]) >= target:
            return False, {
                **telemetry,
                "reasons": [*telemetry.get("reasons", []), "PARALLEL_ARM_TARGET_REACHED"],
            }
        return allowed, telemetry


def run(output: Path) -> None:
    """Validate preserved boundaries, run the outstanding tail and verify its full CP."""
    from operational.efficient_context.common import digest
    from operational.train40_system import engine_host_fast_4
    from operational.train40_system.contracts import verify_sources
    from operational.train40_system.durable_io import atomic_json
    from operational.train40_system.parallel_runtime_v4 import (
        arm_journal,
        arm_target_complete,
        read_object,
    )

    freeze = read_object(output / "PARALLEL_PILOT_V4_TAIL_FREEZE.json")
    verify_sources(freeze)
    if freeze["engine_sha256"] != digest(Path(__file__)):
        raise ValueError("Tail engine differs from admission")
    for name, checksum in freeze["evidence"].items():
        if digest(output / name) != checksum:
            raise ValueError(f"Tail evidence differs: {name}")
    ledger = read_object(output / "PARALLEL_PILOT_V4_LEDGER.json")
    interrupted = read_object(output / "PARALLEL_PILOT_V4_RESULT.json")
    if not (
        interrupted["status"] == "INTERRUPTED_NORMAL_PEER_EXIT_RACE"
        and interrupted["token"] == ledger["token"]
        and ledger["status"] == "ARMED"
        and ledger["arms"]["c2f"]["baseline_committed"] == 420
        and ledger["arms"]["c2f"]["target_committed"] == 1420
    ):
        raise ValueError("Only the preserved interrupted pilot can resume here")
    if not arm_target_complete(output, "a5", ledger["token"]):
        raise ValueError("A5 must already have its exact completed pilot checkpoint")
    journal = arm_journal(output, "c2f")
    baseline = int(journal["committed_updates"])
    target = int(ledger["arms"]["c2f"]["target_committed"])
    receipt = read_object(output / "fits/c2f_seed7/CHECKPOINT_RECEIPT.json")
    if not (
        baseline
        == interrupted["arms"]["c2f"]["committed_updates"]
        == journal["durable_updates"]
        == receipt["committed_updates"]
        and baseline == 920
        and target - baseline == 500
        and journal["pending_update_upper"] == 0
        and journal["recovery_upper"] == ledger["arms"]["c2f"]["baseline_recovery_upper"]
        and receipt["full_optimizer_scheduler_sampler_and_all_RNG"] is True
        and receipt["status"] == "PAUSED_RESOURCE"
        and receipt["recovery_upper"] == journal["recovery_upper"] == 0
        and digest(output / "fits/c2f_seed7/checkpoint_last.pt") == receipt["sha256"]
    ):
        raise ValueError("Tail baseline is not the preserved full checkpoint")
    started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    atomic_json(
        output / "PARALLEL_PILOT_V4_TAIL_STARTED.json",
        {
            "started_utc": started_utc,
            "baseline_committed": baseline,
            "target_committed": target,
            "freeze_sha256": digest(output / "PARALLEL_PILOT_V4_TAIL_FREEZE.json"),
            "original_reservation_sha256": digest(output / "PARALLEL_PILOT_V4_LEDGER.json"),
        },
    )
    original = resource_monitor.ResourceMonitor
    resource_monitor.ResourceMonitor = TailMonitor
    try:
        engine_host_fast_4.run(output, "c2f")
    finally:
        resource_monitor.ResourceMonitor = original
    if not arm_target_complete(output, "c2f", ledger["token"]):
        raise RuntimeError("Tail paused without reaching its original target; work preserved")
    atomic_json(
        output / "PARALLEL_PILOT_V4_TAIL_RESULT.json",
        {
            "status": "COMPLETE",
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds_including_restore": time.perf_counter() - started,
            "baseline_committed": baseline,
            "target_committed": target,
            "scientific_updates": target - baseline,
            "additional_recovery_updates": 0,
            "receipt": read_object(output / "fits/c2f_seed7/CHECKPOINT_RECEIPT.json"),
        },
    )


def main() -> None:
    """Own the sole global trainer lease while completing the outstanding C2F quota."""
    from operational.efficient_context.common import Lease
    from operational.train40_system.data_audit import OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve())


if __name__ == "__main__":
    main()
