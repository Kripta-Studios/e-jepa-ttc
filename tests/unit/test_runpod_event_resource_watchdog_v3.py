"""Zero-update preflight compatibility remains narrow and durable."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import runpod_event_resource_watchdog_v3 as overlay


def write(path: Path, value: object, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def paused(tmp_path: Path) -> tuple[Any, dict, dict, Path]:
    module = overlay.install_v3()
    run = tmp_path / "runs"
    directory = run / "rotations/4/fits/E_C2F"
    version = directory / "checkpoint_versions/checkpoint_000200.pt"
    checkpoint = directory / "checkpoint_last.pt"
    version.parent.mkdir(parents=True)
    version.write_bytes(b"checkpoint")
    checkpoint.write_bytes(b"checkpoint")
    checksum = hashlib.sha256(b"checkpoint").hexdigest()
    common = {"identity_sha256": "i" * 64, "execution_binding_sha256": "e" * 64}
    write(
        directory / "CHECKPOINT_RECEIPT.json",
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            **common,
            "status": "PAUSED_RESOURCE",
            "complete_state": True,
            "accumulation_index": 0,
            "recovery_upper": 0,
            "completed_updates": 200,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checksum,
        },
        1100,
    )
    write(
        directory / "CHECKPOINT_POINTER.json",
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            **common,
            "completed_updates": 200,
            "version": version.name,
            "checkpoint_sha256": checksum,
        },
        1100,
    )
    write(
        directory / "UPDATE_JOURNAL.json",
        {
            "schema": "rgb_port_update_journal_v1",
            **common,
            "completed_updates": 200,
            "durable_updates": 200,
            "pending_update_upper": 0,
            "recovery_upper": 0,
        },
        1100,
    )
    write(directory / "RESOURCE_PAUSE.json", {"manual": False, "completed_updates": 200}, 1100)
    write(directory / "PROGRESS.json", {"pid": 41, "completed_updates": 200}, 900)
    task = {"run": str(run), "rotation": 4, "fit": "E_C2F"}
    managed = {"pid": 42, "started_epoch": 1000}
    return module, task, managed, directory


def test_old_matching_progress_admits_zero_update_preflight(tmp_path: Path) -> None:
    module, task, managed, _ = paused(tmp_path)
    assert module.producer_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_DURABLE_ZERO_UPDATE_PREFLIGHT_PAUSE",
    )


@pytest.mark.parametrize("failure", ["newer", "updates", "pid", "manual"])
def test_progress_or_pause_mismatch_remains_blocked(tmp_path: Path, failure: str) -> None:
    module, task, managed, directory = paused(tmp_path)
    path = directory / ("RESOURCE_PAUSE.json" if failure == "manual" else "PROGRESS.json")
    value = json.loads(path.read_text())
    if failure == "newer":
        mtime = 1200
    else:
        mtime = path.stat().st_mtime
        if failure == "updates":
            value["completed_updates"] = 199
        elif failure == "pid":
            value["pid"] = 42
        else:
            value["manual"] = True
    write(path, value, mtime)
    assert module.producer_terminal(task, managed) == (
        "BLOCKED",
        "PROGRESS_PID_OR_FRESHNESS_MISMATCH",
    )
