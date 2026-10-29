"""Preserve the old trainer at a fresh durable checkpoint before exclusive GPU profiling."""

from __future__ import annotations

import shutil
import time
from datetime import UTC, datetime

import psutil

from operational.efficient_context.common import digest
from operational.train40_system.contracts import read
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def run() -> None:
    """Stop only verified owners, retaining every lost update in the existing journal."""
    output = OUTPUT.resolve()
    root_owner = read(output / "WRITER.lock")
    trainer = psutil.Process(root_owner["pid"])
    if abs(trainer.create_time() - root_owner["create_time"]) > 0.01:
        raise ValueError("Trainer owner PID was reused")
    if "operational.train40_system.engine" not in trainer.cmdline():
        raise ValueError("Pause applies only to the original exact engine module")
    owner = read(output / "controller/WRITER.lock")
    try:
        controller = psutil.Process(owner["pid"])
    except psutil.NoSuchProcess:
        previous = read(output / "COORDINATION_CONTROLLER_PAUSE.json")
        if previous["controller_pid"] != owner["pid"]:
            raise ValueError("Unknown controller pause") from None
        controller = None
    if controller is not None:
        if abs(controller.create_time() - owner["create_time"]) > 0.01:
            raise ValueError("Controller owner PID was reused")
        if "operational.train40_system.controller_sealed" not in controller.cmdline():
            raise ValueError("Preserve unrelated controllers")
    directory = output / "fits/a5_seed7"
    initial = read(directory / "CHECKPOINT_RECEIPT.json")["committed_updates"]
    if controller is not None:
        controller.kill()
        controller.wait(timeout=10)
    started = time.monotonic()
    try:
        while time.monotonic() - started < 240:
            try:
                receipt = read(directory / "CHECKPOINT_RECEIPT.json")
            except PermissionError:
                time.sleep(0.05)
                continue
            if receipt["committed_updates"] > initial:
                trainer.kill()
                trainer.wait(timeout=10)
                break
            if not trainer.is_running():
                raise RuntimeError("Trainer exited before a fresh checkpoint")
            time.sleep(0.05)
        else:
            raise TimeoutError("Fresh checkpoint wait expired; trainer remains preserved")
    finally:
        atomic_json(
            output / "COORDINATION_CONTROLLER_PAUSE.json",
            {
                "controller_pid": owner["pid"],
                "trainer_pid": root_owner["pid"],
                "checked_utc": datetime.now(UTC).isoformat(),
                "purpose": "Exclusive read-only GPU profiling and parity before optimized resume",
                "resume_commands": "RESUME_COMMANDS.json",
                "scientific_negative": False,
            },
        )
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    journal = read(directory / "UPDATE_JOURNAL.json")
    checkpoint = directory / "checkpoint_last.pt"
    if digest(checkpoint) != receipt["sha256"]:
        raise ValueError("Preserve checkpoint; receipt hash mismatch")
    snapshot = output / "coordination_admission/paused_checkpoint"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint_last.pt", "CHECKPOINT_RECEIPT.json", "UPDATE_JOURNAL.json"):
        shutil.copyfile(directory / name, snapshot / name)
    lost = max(0, journal["committed_updates"] - receipt["committed_updates"])
    lost += journal["pending_update_upper"]
    atomic_json(
        output / "COORDINATION_PAUSE_RECEIPT.json",
        {
            "status": "PRESERVED_FOR_PROFILE",
            "checked_utc": datetime.now(UTC).isoformat(),
            "trainer_pid": root_owner["pid"],
            "checkpoint_sha256": receipt["sha256"],
            "checkpoint_updates": receipt["committed_updates"],
            "journal_committed_updates": journal["committed_updates"],
            "pending_update_upper": journal["pending_update_upper"],
            "additional_recovery_upper_on_resume": lost,
            "journal_unchanged_recovery_accounted_by_DurableState_restore": True,
            "snapshot": str(snapshot),
            "scientific_negative": False,
        },
    )
    print(f"Preserved {receipt['committed_updates']} durable updates; recovery upper {lost}")


if __name__ == "__main__":
    run()
