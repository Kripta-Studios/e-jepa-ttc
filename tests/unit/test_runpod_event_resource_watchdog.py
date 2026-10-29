"""The EVENT watchdog resumes only exact, durable, automatic Pod2 pauses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import runpod_event_resource_watchdog_v1 as watchdog


def _write(path: Path, value: dict, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _paused_checkpoint(directory: Path, *, mtime: float, pid: int) -> None:
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


def _producer(tmp_path: Path) -> tuple[dict, dict, Path]:
    run = tmp_path / "ejepa_event_runs"
    task = {
        "id": "R3_E_C2F",
        "run": str(run),
        "rotation": 3,
        "fit": "E_C2F",
        "assignment_sha256": "a" * 64,
    }
    managed = {"pid": 42, "started_epoch": 1_000}
    return task, managed, run / "rotations/3/fits/E_C2F"


def test_event_producer_fresh_durable_pause_is_restartable(tmp_path: Path) -> None:
    task, managed, directory = _producer(tmp_path)
    _paused_checkpoint(directory, mtime=1_001, pid=42)
    assert watchdog.producer_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_DURABLE_EVENT_RESOURCE_PAUSE",
    )


def test_progress_cannot_declare_event_producer_complete(tmp_path: Path) -> None:
    task, managed, directory = _producer(tmp_path)
    _sealed_pair(directory, rotation=3, fit="E_C2F")
    _write(
        directory / "PROGRESS.json",
        {"pid": 42, "completed_updates": 100, "status": "COMPLETE"},
        1_001,
    )
    assert watchdog.producer_terminal(task, managed) == (
        "BLOCKED",
        "EVENT_CLOUD_ADMISSION_MISSING",
    )


def test_only_admitted_cloud_endpoint_completes_event_producer(tmp_path: Path) -> None:
    task, managed, directory = _producer(tmp_path)
    _sealed_pair(directory, rotation=3, fit="E_C2F")
    receipt = json.loads((directory / "CHECKPOINT_RECEIPT.json").read_text())
    _write(
        directory / "SCIENTIFIC_CLOUD_ADMISSION.json",
        {
            "schema": "ejepa_event_scientific_cloud_admission_v1",
            "status": "ADMITTED",
            "rotation": 3,
            "fit": "E_C2F",
            "assignment_sha256": "a" * 64,
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "completed_updates": 100,
        },
        1_001,
    )
    assert watchdog.producer_terminal(task, managed) == (
        "COMPLETE",
        "EVENT_CLOUD_ADMITTED",
    )


def _sealed_pair(directory: Path, *, rotation: int, fit: str) -> None:
    checkpoint = directory / "checkpoint_last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"old sealed pair")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    receipt = {
        "status": "COMPLETE",
        "scientific_endpoint": True,
        "complete_state": True,
        "completed_updates": 100,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checksum,
    }
    _write(receipt_path, receipt, 900)
    _write(
        directory / "UPDATE_JOURNAL.json",
        {"completed_updates": 100, "durable_updates": 100, "pending_update_upper": 0},
        900,
    )
    _write(
        directory / "ENDPOINT_ADMISSION.json",
        {
            "schema": "final_refit_fold_endpoint_admission_v1",
            "status": "PASS",
            "scientific_endpoint": True,
            "rotation": rotation,
            "fit_id": fit,
            "updates": 100,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checksum,
            "checkpoint_receipt_path": str(receipt_path.resolve()),
            "checkpoint_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
        900,
    )


def _head(tmp_path: Path, *, manual: object = False) -> tuple[dict, dict]:
    run = tmp_path / "ejepa_event_runs"
    task = {"run": str(run), "rotation": 4, "fit": watchdog.HEAD}
    managed = {"pid": 77, "started_epoch": 1_000}
    _write(
        run / "EVENT_DOWNSTREAM_PAUSE_R4.json",
        {
            "schema": "ejepa_event_downstream_pause_v1",
            "status": "PAUSED_RECOVERABLE",
            "rotation": 4,
            "manual": manual,
        },
        1_001,
    )
    _write(
        run / "rotations/4/EVENT_PIPELINE_RECEIPT.json",
        {
            "schema": "ejepa_cloud_event_downstream_queue_v1",
            "rotation": 4,
            "pid": 77,
            "stage": "WAIT_E_C2F",
            "status": "WAITING_DEPENDENCY",
        },
        1_001,
    )
    _sealed_pair(run / "rotations/4/fits/PAIR_E", rotation=4, fit="PAIR_E")
    _paused_checkpoint(run / "rotations/4/fits/E_CTX", mtime=1_001, pid=77)
    return task, managed


def test_downstream_accepts_old_sealed_pair_and_fresh_paused_context(
    tmp_path: Path,
) -> None:
    task, managed = _head(tmp_path)
    assert watchdog.downstream_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_DURABLE_EVENT_DOWNSTREAM_RESOURCE_PAUSE",
    )


def test_downstream_pause_without_explicit_manual_false_is_blocked(tmp_path: Path) -> None:
    task, managed = _head(tmp_path, manual=None)
    assert watchdog.downstream_terminal(task, managed) == (
        "BLOCKED",
        "EVENT_PAUSE_NOT_AUTOMATIC",
    )


def test_event_oof_result_is_the_only_downstream_completion(tmp_path: Path) -> None:
    run = tmp_path / "ejepa_event_runs"
    task = {
        "run": str(run),
        "rotation": 2,
        "fit": watchdog.HEAD,
        "assignment_sha256": "a" * 64,
    }
    managed = {"pid": 88, "started_epoch": 2_000}
    path = run / "EVENT_DOWNSTREAM_RESULT_R2.json"
    _write(
        path,
        {
            "schema": "ejepa_cloud_event_downstream_queue_v1",
            "stage": "WAIT_RAW_EVENT",
            "status": "COMPLETE",
            "rotation": 2,
        },
        2_001,
    )
    assert watchdog.downstream_terminal(task, managed)[0] == "BLOCKED"
    migration_path = run / "EVENT_MIGRATION_BINDING.json"
    _write(
        migration_path,
        {"schema": "ejepa_event_linux_migration_binding_v1"},
        1_900,
    )
    admission_path = run / "EVENT_DOWNSTREAM_ADMISSION_R2.json"
    _write(
        admission_path,
        {
            "schema": "ejepa_event_downstream_admission_v1",
            "status": "PASS",
            "task": {"rotation": 2, "fit": watchdog.HEAD},
            "assignment_sha256": "a" * 64,
            "event_migration_binding_sha256": hashlib.sha256(
                migration_path.read_bytes()
            ).hexdigest(),
            "allowed_new_endpoints": ["PAIR_E", "E_CTX"],
            "allowed_generation_interface": "PHASE17_RAW_EVENT",
            "rgb_model_inference_authorized": False,
            "scientific_pipeline_edited": False,
        },
        1_900,
    )
    _sealed_pair(run / "rotations/2/fits/PAIR_E", rotation=2, fit="PAIR_E")
    _sealed_pair(run / "rotations/2/fits/E_CTX", rotation=2, fit="E_CTX")
    fragment = run / "rotations/2/generation/E_CTX/fragment.npz"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_bytes(b"event-oof")
    payload = {
        "schema": "ejepa_cloud_event_downstream_queue_v1",
        "stage": "EVENT_OOF",
        "status": "COMPLETE",
        "rotation": 2,
        "dependencies": {"E_A5": True, "E_C2F": True, "PAIR_E": True, "E_CTX": True},
        "emitted_interface": "PHASE17_RAW_EVENT",
        "phase20_context_emitted": False,
        "final_ttc_predictions_emitted": False,
        "authority": {
            "event_downstream_admission_sha256": hashlib.sha256(
                admission_path.read_bytes()
            ).hexdigest(),
            "assignment_sha256": "a" * 64,
            "event_migration_binding_sha256": hashlib.sha256(
                migration_path.read_bytes()
            ).hexdigest(),
        },
        "artifacts": [
            {
                "path": "rotations/2/generation/E_CTX/fragment.npz",
                "bytes": fragment.stat().st_size,
                "sha256": hashlib.sha256(fragment.read_bytes()).hexdigest(),
            }
        ],
    }
    result = {**payload, "receipt_sha256": watchdog.canonical_sha256(payload)}
    _write(
        path,
        result,
        2_001,
    )
    assert watchdog.downstream_terminal(task, managed) == (
        "COMPLETE",
        "EVENT_OOF_ADMITTED_COMPLETE",
    )

    fragment.write_bytes(b"tampered")
    assert watchdog.downstream_terminal(task, managed) == (
        "BLOCKED",
        "EVENT_RESULT_ARTIFACT_HASH_OR_SIZE_MISMATCH",
    )


def test_process_binding_is_exact_argv_not_option_similarity() -> None:
    argv = ["/opt/python", "/runtime/wrapper.py", "producer", "--rotation", "2"]
    task = {"adopted_argv": [argv]}
    process = {
        "state": "R",
        "argv": argv,
        "cmdline_sha256": watchdog.command_sha256(argv),
    }
    assert watchdog.command_matches(task, process)
    changed = [*argv, "--additional-updates", "8"]
    process.update(argv=changed, cmdline_sha256=watchdog.command_sha256(changed))
    assert not watchdog.command_matches(task, process)


def test_restart_identity_wait_tolerates_proc_exec_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = ["/opt/python", "/runtime/worker.py"]
    exact = {
        "pid": 93,
        "state": "R",
        "start_ticks": 555,
        "started_epoch": 2_000.0,
        "argv": argv,
        "cmdline_sha256": watchdog.command_sha256(argv),
    }
    observations = iter([None, exact])
    monkeypatch.setattr(watchdog, "proc_identity", lambda *_args: next(observations))
    ticks = iter([0.0, 0.01])
    identity, last = watchdog.wait_process_identity(
        93,
        {"adopted_argv": [argv]},
        Path("/proc"),
        monotonic=lambda: next(ticks),
        sleeper=lambda _seconds: None,
    )
    assert identity == exact
    assert last == exact


def test_adoption_binds_start_ticks_and_treats_zombie_as_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = object.__new__(watchdog.Watchdog)
    instance.proc_root = Path("/proc")  # type: ignore[reportAttributeAccessIssue]
    argv = ["/opt/python", "/runtime/worker.py"]
    task = {"adopted_argv": [argv]}
    current = {
        "pid": 91,
        "start_ticks": 123,
        "started_epoch": 1_000,
        "cmdline_sha256": watchdog.command_sha256(argv),
    }
    monkeypatch.setattr(
        watchdog,
        "proc_identity",
        lambda *_args: {
            "pid": 91,
            "state": "Z",
            "start_ticks": 123,
            "argv": [],
            "cmdline_sha256": watchdog.command_sha256([]),
        },
    )
    assert instance._adopt(task, current) is None
    assert current.get("state") != "BLOCKED"

    monkeypatch.setattr(
        watchdog,
        "proc_identity",
        lambda *_args: {
            "pid": 91,
            "state": "R",
            "start_ticks": 124,
            "argv": argv,
            "cmdline_sha256": watchdog.command_sha256(argv),
        },
    )
    assert instance._adopt(task, current) is not None
    assert current["reason"] == "PID_REUSED"


def test_blocked_task_does_not_stop_other_event_tasks() -> None:
    assert watchdog.campaign_terminal_code(["BLOCKED", "RUNNING", "COMPLETE"]) is None
    assert watchdog.campaign_terminal_code(["BLOCKED", "COMPLETE"]) == 5
