from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from operational.rgb_port import accounting
from scripts import runpod_event_queue_pause_v1 as pause_overlay
from scripts import runpod_event_resource_watchdog_v4 as watchdog_v4
from scripts import runpod_event_resource_watchdog_v6 as watchdog_v6
from scripts import runpod_event_resource_watchdog_v10 as watchdog_v10


@pytest.fixture(scope="module")
def watchdog() -> Any:
    return watchdog_v4.install_v4()


def _case(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run = tmp_path / "run"
    run.mkdir()
    task = {"run": str(run), "rotation": 2, "fit": "EVENT_DOWNSTREAM"}
    managed = {"started_epoch": time.time() - 10, "pid": 321, "start_ticks": 654}
    return task, managed, run / "EVENT_DOWNSTREAM_PAUSE_R2.json"


def _write_pause(path: Path, managed: dict[str, Any], **changes: Any) -> None:
    value = {
        "schema": "ejepa_event_downstream_pause_v1",
        "status": "PAUSED_RECOVERABLE",
        "rotation": 2,
        "reason": "resource reserve unavailable",
        "manual": False,
        "pause_classification": "AUTOMATIC_RESOURCE_PAUSE",
        "owner_pid": managed["pid"],
        "owner_start_ticks": managed["start_ticks"],
        "resources": {"effective_headroom_bytes": 1},
        **changes,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_fresh_exact_pre_head_automatic_pause_is_restartable(tmp_path: Path, watchdog: Any) -> None:
    task, managed, pause = _case(tmp_path)
    _write_pause(pause, managed)
    assert watchdog.downstream_terminal(task, managed) == (
        "RESTARTABLE",
        "FRESH_AUTOMATIC_PRE_HEAD_RESOURCE_PAUSE",
    )


def test_manual_pause_is_not_reclassified_or_restarted(tmp_path: Path, watchdog: Any) -> None:
    task, managed, pause = _case(tmp_path)
    _write_pause(
        pause,
        managed,
        reason="manual stop",
        manual=True,
        pause_classification="MANUAL_STOP",
    )
    state, _ = watchdog.downstream_terminal(task, managed)
    assert state == "BLOCKED"


@pytest.mark.parametrize(
    "changes",
    [{"owner_pid": 999}, {"owner_start_ticks": 999}, {"resources": None}],
)
def test_owner_identity_or_resource_mismatch_is_rejected(
    tmp_path: Path, watchdog: Any, changes: dict[str, Any]
) -> None:
    task, managed, pause = _case(tmp_path)
    _write_pause(pause, managed, **changes)
    state, _ = watchdog.downstream_terminal(task, managed)
    assert state == "BLOCKED"


def test_stale_pause_is_rejected(tmp_path: Path, watchdog: Any) -> None:
    task, managed, pause = _case(tmp_path)
    _write_pause(pause, managed)
    stale = managed["started_epoch"] - 1
    os.utime(pause, (stale, stale))
    state, _ = watchdog.downstream_terminal(task, managed)
    assert state == "BLOCKED"


@pytest.mark.parametrize("fit", ["PAIR_E", "E_CTX"])
def test_fresh_head_checkpoint_prevents_pre_head_restart(
    tmp_path: Path, watchdog: Any, fit: str
) -> None:
    task, managed, pause = _case(tmp_path)
    _write_pause(pause, managed)
    receipt = Path(task["run"]) / f"rotations/2/fits/{fit}/CHECKPOINT_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    state, _ = watchdog.downstream_terminal(task, managed)
    assert state == "BLOCKED"


def test_pause_overlay_does_not_auto_classify_manual_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[Path, dict[str, Any]]] = []

    def writer(path: Path, value: dict[str, Any]) -> None:
        observed.append((path, dict(value)))

    monkeypatch.setattr(accounting, "atomic_write_json", writer)
    pause_overlay.install()
    target = tmp_path / "EVENT_DOWNSTREAM_PAUSE_R2.json"
    accounting.atomic_write_json(
        target,
        {
            "schema": "ejepa_event_downstream_pause_v1",
            "status": "PAUSED_RECOVERABLE",
            "rotation": 2,
            "reason": "manual stop",
        },
    )
    assert observed == [
        (
            target,
            {
                "schema": "ejepa_event_downstream_pause_v1",
                "status": "PAUSED_RECOVERABLE",
                "rotation": 2,
                "reason": "manual stop",
            },
        )
    ]


@pytest.fixture(scope="module")
def strict_watchdog() -> Any:
    return watchdog_v6.install_v6()


def _exec_case(module: Any) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    waiter = ["/usr/bin/python3", "/ops/waiter.py", "--run", "/workspace/run"]
    queue = ["/usr/bin/python3", "/ops/queue.py", "--run", "/workspace/run"]
    task = {
        "fit": module.HEAD,
        "run": "/workspace/run",
        "rotation": 2,
        "adopted_argv": [waiter, queue],
    }
    current = {
        "pid": 123,
        "start_ticks": 456,
        "cmdline_sha256": module.command_sha256(waiter),
    }
    process = {
        "pid": 123,
        "state": "S",
        "start_ticks": 456,
        "argv": queue,
        "cmdline_sha256": module.command_sha256(queue),
    }
    instance = object.__new__(module.Watchdog)
    instance.proc_root = Path("/proc")
    return instance, task, current, process


def test_exact_waiter_to_queue_exec_transition_is_adopted(
    strict_watchdog: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance, task, current, process = _exec_case(strict_watchdog)
    monkeypatch.setattr(strict_watchdog, "proc_identity", lambda *_args: process)
    observed = strict_watchdog.Watchdog._adopt(instance, task, current)
    assert observed == process
    assert current["state"] == "RUNNING"
    assert current["reason"] == "ADOPTED_AFTER_ADMITTED_EXEC_TRANSITION"
    assert current["cmdline_sha256"] == process["cmdline_sha256"]


@pytest.mark.parametrize("failure", ["fit", "prior", "argv", "pid", "start"])
def test_exec_transition_rejects_any_identity_expansion(
    strict_watchdog: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    instance, task, current, process = _exec_case(strict_watchdog)
    if failure == "fit":
        task["fit"] = "E_A5"
    elif failure == "prior":
        current["cmdline_sha256"] = "0" * 64
    elif failure == "argv":
        process["argv"] = ["/usr/bin/python3", "/ops/unbound.py"]
        process["cmdline_sha256"] = strict_watchdog.command_sha256(process["argv"])
    elif failure == "pid":
        process["pid"] = 999
    else:
        process["start_ticks"] = 999
    monkeypatch.setattr(strict_watchdog, "proc_identity", lambda *_args: process)
    strict_watchdog.Watchdog._adopt(instance, task, current)
    assert current["state"] == "BLOCKED"


def test_admitted_exec_exit_between_polls_reaches_terminal_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = watchdog_v10.install_v10()
    instance, task, current, _process = _exec_case(module)
    queue = task["adopted_argv"][1]
    current.update(
        cmdline_sha256=module.command_sha256(queue),
        state="RUNNING",
        reason="ADOPTED_AFTER_ADMITTED_EXEC_TRANSITION",
    )
    previous = dict(current)
    monkeypatch.setattr(module, "proc_identity", lambda *_args: None)
    assert module.Watchdog._adopt(instance, task, current) is None
    assert current == previous
