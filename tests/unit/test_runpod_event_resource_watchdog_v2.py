"""Retry budgets reset only on a new exact durable EVENT producer boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import runpod_event_resource_watchdog_v2 as overlay


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def boundary(tmp_path: Path, updates: int = 200) -> tuple[dict, Path]:
    run = tmp_path / "runs"
    directory = run / "rotations/2/fits/E_A5"
    version = directory / f"checkpoint_versions/checkpoint_{updates:06d}.pt"
    checkpoint = directory / "checkpoint_last.pt"
    version.parent.mkdir(parents=True)
    version.write_bytes(b"durable checkpoint")
    checkpoint.write_bytes(version.read_bytes())
    checksum = hashlib.sha256(version.read_bytes()).hexdigest()
    common = {"identity_sha256": "i" * 64, "execution_binding_sha256": "e" * 64}
    write(
        directory / "CHECKPOINT_RECEIPT.json",
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            **common,
            "complete_state": True,
            "accumulation_index": 0,
            "recovery_upper": 0,
            "completed_updates": updates,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checksum,
        },
    )
    write(
        directory / "CHECKPOINT_POINTER.json",
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            **common,
            "completed_updates": updates,
            "version": version.name,
            "checkpoint_sha256": checksum,
        },
    )
    write(
        directory / "UPDATE_JOURNAL.json",
        {
            "schema": "rgb_port_update_journal_v1",
            **common,
            "completed_updates": updates,
            "durable_updates": updates,
            "pending_update_upper": 0,
            "recovery_upper": 0,
        },
    )
    return {
        "run": str(run),
        "rotation": 2,
        "fit": "E_A5",
        "log": str(tmp_path / "x.log"),
    }, directory


def test_equal_anchor_does_not_hash_or_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = overlay.install_v2()
    task, _ = boundary(tmp_path)
    monkeypatch.setattr(overlay, "digest", lambda path: (_ for _ in ()).throw(AssertionError()))
    assert overlay.durable_producer_updates(module, task, 200) is None


@pytest.mark.parametrize("mutation", ["pending", "hash", "accumulation"])
def test_inflight_or_mismatched_transaction_does_not_reset(tmp_path: Path, mutation: str) -> None:
    module = overlay.install_v2()
    task, directory = boundary(tmp_path)
    target = directory / (
        "UPDATE_JOURNAL.json" if mutation == "pending" else "CHECKPOINT_RECEIPT.json"
    )
    value = json.loads(target.read_text())
    if mutation == "pending":
        value["pending_update_upper"] = 1
    elif mutation == "hash":
        value["checkpoint_sha256"] = "0" * 64
    else:
        value["accumulation_index"] = 1
    write(target, value)
    assert overlay.durable_producer_updates(module, task, 100) is None


def test_durable_advance_resets_only_its_producer(tmp_path: Path) -> None:
    module = overlay.install_v2()
    task, _ = boundary(tmp_path)
    first = {"retries": 9, "retry_anchor_completed_updates": 100}
    other = {"retries": 7, "retry_anchor_completed_updates": 100}
    watcher = SimpleNamespace()
    module.Watchdog._observe_durable_retry_boundary(watcher, task, first)
    assert first["retries"] == 0
    assert first["retry_reset_previous_retries"] == 9
    assert first["retry_reset_from_anchor"] == 100
    assert first["retry_reset_current_durable"] == 200
    assert other["retries"] == 7


def test_live_journal_ahead_still_admits_sealed_durable_advance(tmp_path: Path) -> None:
    module = overlay.install_v2()
    task, directory = boundary(tmp_path, updates=200)
    journal_path = directory / "UPDATE_JOURNAL.json"
    journal = json.loads(journal_path.read_text())
    journal["completed_updates"] = 237
    write(journal_path, journal)
    current = {"retries": 11, "retry_anchor_completed_updates": 100}
    module.Watchdog._observe_durable_retry_boundary(SimpleNamespace(), task, current)
    assert current["retries"] == 0
    assert current["retry_reset_current_durable"] == 200


def test_migrated_state_sets_anchor_without_reset(tmp_path: Path) -> None:
    module = overlay.install_v2()
    task, _ = boundary(tmp_path)
    current = {"retries": 9}
    module.Watchdog._observe_durable_retry_boundary(SimpleNamespace(), task, current)
    assert current["retries"] == 9
    assert current["retry_anchor_completed_updates"] == 200


def test_twelve_failures_without_advance_remain_blocked(tmp_path: Path) -> None:
    module = overlay.install_v2()
    task, _ = boundary(tmp_path, updates=100)
    current = {"retries": 12, "retry_anchor_completed_updates": 100}
    watcher = SimpleNamespace(now=lambda: 1_000.0)
    module.Watchdog._restart(watcher, task, current)
    assert current["state"] == "BLOCKED"
    assert current["reason"] == "MAX_RETRIES_EXHAUSTED"


def test_downstream_never_borrows_producer_reset(tmp_path: Path) -> None:
    module = overlay.install_v2()
    task = {"run": str(tmp_path), "rotation": 2, "fit": module.HEAD}
    assert overlay.durable_producer_updates(module, task, 0) is None
