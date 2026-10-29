"""The Runpod watchdog only resumes fresh, durable automatic resource pauses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import runpod_resource_resume_watchdog_v1 as watchdog


def _write(path: Path, value: dict, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _producer(tmp_path: Path, *, started: float = 1_000, pid: int = 42) -> tuple[dict, dict, Path]:
    run = tmp_path / "r2_rgb_v3"
    task = {"run": str(run), "rotation": 2, "fit": "R_A5"}
    managed = {"pid": pid, "started_epoch": started}
    directory = run / "rotations/2/fits/R_A5"
    return task, managed, directory


def _durable_pause(directory: Path, *, mtime: float, pid: int) -> None:
    _write(
        directory / "CHECKPOINT_RECEIPT.json",
        {
            "status": "PAUSED_RESOURCE",
            "completed_updates": 40,
            "complete_state": True,
        },
        mtime,
    )
    _write(
        directory / "UPDATE_JOURNAL.json",
        {"completed_updates": 40, "durable_updates": 40, "pending_update_upper": 0},
        mtime,
    )
    _write(
        directory / "RESOURCE_PAUSE.json",
        {"manual": False, "completed_updates": 40},
        mtime,
    )
    _write(directory / "PROGRESS.json", {"pid": pid}, mtime)


def test_fresh_durable_producer_pause_is_restartable(tmp_path: Path) -> None:
    task, managed, directory = _producer(tmp_path)
    _durable_pause(directory, mtime=1_001, pid=42)
    assert watchdog.producer_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_DURABLE_RESOURCE_PAUSE",
    )


def test_stale_pause_from_a_prior_process_is_blocked(tmp_path: Path) -> None:
    task, managed, directory = _producer(tmp_path)
    _durable_pause(directory, mtime=999, pid=42)
    state, reason = watchdog.producer_terminal(task, managed)
    assert state == "BLOCKED"
    assert "STALE" in reason


def test_progress_must_belong_to_managed_pid(tmp_path: Path) -> None:
    task, managed, directory = _producer(tmp_path)
    _durable_pause(directory, mtime=1_001, pid=99)
    assert watchdog.producer_terminal(task, managed) == (
        "BLOCKED",
        "PROGRESS_PID_OR_FRESHNESS_MISMATCH",
    )


def test_manual_pause_is_never_restarted(tmp_path: Path) -> None:
    task, managed, directory = _producer(tmp_path)
    _durable_pause(directory, mtime=1_001, pid=42)
    _write(
        directory / "RESOURCE_PAUSE.json",
        {"manual": True, "completed_updates": 40},
        1_001,
    )
    assert watchdog.producer_terminal(task, managed) == (
        "BLOCKED",
        "NOT_AN_AUTOMATIC_RESOURCE_PAUSE",
    )


def test_head_pause_requires_fresh_pid_bound_pipeline_receipt(tmp_path: Path) -> None:
    run = tmp_path / "r3_rgb_downstream_v1"
    task = {"run": str(run), "rotation": 3, "fit": watchdog.HEAD}
    managed = {"pid": 77, "started_epoch": 2_000}
    _write(
        run / "RGB_DOWNSTREAM_PAUSE.json",
        {
            "status": "PAUSED_RECOVERABLE",
            "manual": False,
            "rotation": 3,
            "optimizer_state_durable_at_last_scientific_checkpoint": True,
        },
        2_001,
    )
    _write(run / "rotations/3/RGB_PIPELINE_RECEIPT.json", {"pid": 77}, 2_001)
    assert watchdog.head_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_DURABLE_RGB_RESOURCE_PAUSE",
    )
    _write(run / "rotations/3/RGB_PIPELINE_RECEIPT.json", {"pid": 78}, 2_001)
    assert watchdog.head_terminal(task, managed)[0] == "BLOCKED"


def test_head_accepts_old_sealed_pair_with_fresh_paused_context(tmp_path: Path) -> None:
    run = tmp_path / "r3_rgb_downstream_v1"
    task = {"run": str(run), "rotation": 3, "fit": watchdog.HEAD}
    managed = {"pid": 77, "started_epoch": 2_000}
    _write(
        run / "RGB_DOWNSTREAM_PAUSE.json",
        {
            "status": "PAUSED_RECOVERABLE",
            "manual": False,
            "rotation": 3,
            "optimizer_state_durable_at_last_scientific_checkpoint": True,
        },
        2_001,
    )
    _write(run / "rotations/3/RGB_PIPELINE_RECEIPT.json", {"pid": 77}, 2_001)
    pair = run / "rotations/3/fits/PAIR_R"
    checkpoint = pair / "checkpoint_last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed-pair")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt_path = pair / "CHECKPOINT_RECEIPT.json"
    receipt = {
        "status": "COMPLETE",
        "scientific_endpoint": True,
        "complete_state": True,
        "completed_updates": 100,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
    }
    _write(receipt_path, receipt, 1_500)
    _write(
        pair / "UPDATE_JOURNAL.json",
        {"completed_updates": 100, "durable_updates": 100, "pending_update_upper": 0},
        1_500,
    )
    _write(
        pair / "ENDPOINT_ADMISSION.json",
        {
            "schema": "final_refit_fold_endpoint_admission_v1",
            "status": "PASS",
            "scientific_endpoint": True,
            "rotation": 3,
            "fit_id": "PAIR_R",
            "updates": 100,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_receipt_path": str(receipt_path.resolve()),
            "checkpoint_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
        1_500,
    )
    context = run / "rotations/3/fits/R_CTX"
    _durable_pause(context, mtime=2_001, pid=77)
    assert watchdog.head_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_DURABLE_RGB_RESOURCE_PAUSE",
    )


def test_current_wrapper_may_differ_from_future_argv_but_must_be_hash_allowed(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.py"
    future = tmp_path / "future.py"
    current.write_text("# current\n", encoding="utf-8")
    future.write_text("# future\n", encoding="utf-8")
    current_sha = hashlib.sha256(current.read_bytes()).hexdigest()
    future_sha = hashlib.sha256(future.read_bytes()).hexdigest()
    bundle = "/workspace/ejepa_bundle"
    run = "/workspace/ejepa_technical/r2_rgb_v3"
    task = {
        "fit": "R_A5",
        "rotation": 2,
        "run": run,
        "argv": [
            "/opt/python",
            str(future),
            "--bundle",
            bundle,
            "--run",
            run,
            "--rotation",
            "2",
            "--fit",
            "R_A5",
            "--quota-entrypoint",
            "ram-a5",
        ],
        "adopt_entrypoint_sha256": [current_sha, future_sha],
    }
    process = {
        "state": "R",
        "argv": [
            "/opt/python",
            "-u",
            str(current),
            "--bundle",
            bundle,
            "--run",
            run,
            "--rotation",
            "2",
            "--fit",
            "R_A5",
        ],
    }
    assert watchdog.command_matches(task, process) is True
    task["adopt_entrypoint_sha256"] = [future_sha]
    assert watchdog.command_matches(task, process) is False


def test_proc_identity_records_start_time_and_cmdline_hash(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    fields = ["R", *(["0"] * 18), "500"]
    (process / "stat").write_text(f"123 (python worker) {' '.join(fields)}\n")
    (process / "cmdline").write_bytes(b"/opt/python\0/task.py\0")
    (proc / "stat").write_text("cpu 0 0 0 0\nbtime 1000\n")
    identity = watchdog.proc_identity(123, proc, clock_ticks=100)
    assert identity is not None
    assert identity["start_ticks"] == 500
    assert identity["started_epoch"] > 1000
    assert len(identity["cmdline_sha256"]) == 64


def test_previously_adopted_zombie_is_classified_from_fresh_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = object.__new__(watchdog.Watchdog)
    instance.proc_root = Path("/proc")  # type: ignore[reportAttributeAccessIssue]
    monkeypatch.setattr(
        watchdog,
        "proc_identity",
        lambda *_args: {"pid": 9, "state": "Z", "start_ticks": 123},
    )
    current = {"pid": 9, "start_ticks": 123, "started_epoch": 1_000}
    assert instance._adopt({"pid": 9}, current) is None
    assert current.get("state") != "BLOCKED"


def test_one_blocked_task_does_not_end_supervision_of_running_siblings() -> None:
    assert watchdog.campaign_terminal_code(["BLOCKED", "RUNNING", "COMPLETE"]) is None
    assert watchdog.campaign_terminal_code(["BLOCKED", "COMPLETE", "DEADLINE_EXPIRED"]) == 5
    assert watchdog.campaign_terminal_code(["COMPLETE", "DEADLINE_EXPIRED"]) == 0
