from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from scripts import runpod_event_resource_watchdog_v8 as v8


@pytest.fixture(scope="module")
def watchdog() -> Any:
    return v8.install_v8()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _case(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = tmp_path / "run"
    run.mkdir()
    task = {"run": str(run), "rotation": 2, "fit": "EVENT_DOWNSTREAM"}
    managed = {"started_epoch": time.time() - 10, "pid": 321, "start_ticks": 654}
    _write(
        run / "EVENT_DOWNSTREAM_PAUSE_R2.json",
        {
            "schema": "ejepa_event_downstream_pause_v1",
            "status": "PAUSED_RECOVERABLE",
            "rotation": 2,
            "reason": "resource reserve unavailable",
            "manual": False,
            "pause_classification": "AUTOMATIC_RESOURCE_PAUSE",
            "owner_pid": 321,
            "owner_start_ticks": 654,
            "resources": {"headroom": 1},
        },
    )
    stale = run / "rotations/2/EVENT_PIPELINE_RECEIPT.json"
    _write(stale, {"schema": "ejepa_cloud_event_downstream_queue_v1"})
    old = managed["started_epoch"] - 1
    os.utime(stale, (old, old))
    _head(run, managed)
    return task, managed


def _head(run: Path, managed: dict[str, Any], fit: str = "PAIR_E") -> None:
    root = run / f"rotations/2/fits/{fit}"
    checkpoint = root / "checkpoint.pt"
    version = root / "checkpoint_versions/1"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    version.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"same durable checkpoint")
    version.write_bytes(checkpoint.read_bytes())
    sha = v8.digest(checkpoint)
    common = {"identity_sha256": "i" * 64, "execution_binding_sha256": "b" * 64}
    _write(
        root / "CHECKPOINT_RECEIPT.json",
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "status": "PAUSED_RESOURCE",
            "complete_state": True,
            "accumulation_index": 0,
            "completed_updates": 7,
            "recovery_upper": 0,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha,
            **common,
        },
    )
    _write(
        root / "CHECKPOINT_POINTER.json",
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            "version": "1",
            "completed_updates": 7,
            "checkpoint_sha256": sha,
            **common,
        },
    )
    _write(
        root / "UPDATE_JOURNAL.json",
        {
            "schema": "rgb_port_update_journal_v1",
            "completed_updates": 7,
            "durable_updates": 7,
            "pending_update_upper": 0,
            "recovery_upper": 0,
            **common,
        },
    )
    _write(root / "RESOURCE_PAUSE.json", {"manual": False, "completed_updates": 7})
    _write(root / "PROGRESS.json", {"pid": managed["pid"], "completed_updates": 7})


def test_fresh_durable_head_pause_overrides_only_stale_pipeline(
    tmp_path: Path, watchdog: Any
) -> None:
    task, managed = _case(tmp_path)
    assert watchdog.downstream_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_DURABLE_HEAD_PAUSE_WITH_STALE_PIPELINE_RECEIPT",
    )


def test_manual_pause_is_rejected(tmp_path: Path, watchdog: Any) -> None:
    task, managed = _case(tmp_path)
    path = Path(task["run"]) / "EVENT_DOWNSTREAM_PAUSE_R2.json"
    value = json.loads(path.read_text())
    value["manual"] = True
    _write(path, value)
    assert watchdog.downstream_terminal(task, managed)[0] == "BLOCKED"


def test_stale_head_receipt_is_rejected(tmp_path: Path, watchdog: Any) -> None:
    task, managed = _case(tmp_path)
    path = Path(task["run"]) / "rotations/2/fits/PAIR_E/CHECKPOINT_RECEIPT.json"
    old = managed["started_epoch"] - 1
    os.utime(path, (old, old))
    assert watchdog.downstream_terminal(task, managed)[0] == "BLOCKED"


def test_reused_pid_start_identity_is_rejected(tmp_path: Path, watchdog: Any) -> None:
    task, managed = _case(tmp_path)
    managed["start_ticks"] = 999
    assert watchdog.downstream_terminal(task, managed)[0] == "BLOCKED"


@pytest.mark.parametrize("mutation", ["bad_hash", "pending", "recovery", "bad_progress_pid"])
def test_non_durable_head_is_rejected(tmp_path: Path, watchdog: Any, mutation: str) -> None:
    task, managed = _case(tmp_path)
    root = Path(task["run"]) / "rotations/2/fits/PAIR_E"
    target = {
        "bad_hash": root / "CHECKPOINT_RECEIPT.json",
        "pending": root / "UPDATE_JOURNAL.json",
        "recovery": root / "UPDATE_JOURNAL.json",
        "bad_progress_pid": root / "PROGRESS.json",
    }[mutation]
    value = json.loads(target.read_text())
    key, replacement = {
        "bad_hash": ("checkpoint_sha256", "0" * 64),
        "pending": ("pending_update_upper", 1),
        "recovery": ("recovery_upper", 1),
        "bad_progress_pid": ("pid", 999),
    }[mutation]
    value[key] = replacement
    _write(target, value)
    assert watchdog.downstream_terminal(task, managed)[0] == "BLOCKED"


def test_present_e_ctx_must_be_equally_durable(tmp_path: Path, watchdog: Any) -> None:
    task, managed = _case(tmp_path)
    _head(Path(task["run"]), managed, "E_CTX")
    progress = Path(task["run"]) / "rotations/2/fits/E_CTX/PROGRESS.json"
    value = json.loads(progress.read_text())
    value["pid"] = 999
    _write(progress, value)
    assert watchdog.downstream_terminal(task, managed)[0] == "BLOCKED"
