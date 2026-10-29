from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import pytest

from scripts import runpod_event_resource_watchdog_v9 as v9


@pytest.fixture()
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, dict[str, Any]]:
    module = v9.install_v9()
    instance = object.__new__(module.Watchdog)
    task_ids = [v9.DEFERRED, *v9.CANDIDATES]
    tasks = [
        {
            "id": task_id,
            "fit": task_id.split("_", 1)[1],
            "rotation": int(task_id[1]),
            "run": str(tmp_path / "run"),
        }
        for task_id in task_ids
    ]
    instance.config = {
        "tasks": tasks,
        "authority": {"global_deadline_utc": "2030-01-01T00:00:00+00:00"},
    }
    instance.state = {
        "tasks": {
            task_id: {
                "state": "RUNNING" if task_id != v9.DEFERRED else "BLOCKED",
                "pid": 10,
                "start_ticks": 20,
                "started_epoch": 1.0,
                "retries": 3,
            }
            for task_id in task_ids
        }
    }
    instance.status_path = tmp_path / "WATCHDOG_STATUS.json"
    instance.now = lambda: 1_800_000_000.0
    instance._initial = MethodType(lambda _self, _task: {}, instance)
    instance._adopt = MethodType(lambda _self, _task, _current: {}, instance)
    instance._publish = MethodType(lambda self: self.state, instance)
    instance._restart = MethodType(
        lambda _self, _task, current: current.update(state="RESTARTED", pid=99, start_ticks=100),
        instance,
    )
    monkeypatch.setattr(module, "stop_paths", lambda _task: [])
    result = {"candidate_complete": False, "deferred_durable": True}

    def terminal(task: dict[str, Any], _current: dict[str, Any]) -> tuple[str, str]:
        if task["id"] == v9.DEFERRED:
            return (
                ("RESTARTABLE", "FRESH_DURABLE_EVENT_RESOURCE_PAUSE")
                if result["deferred_durable"]
                else ("BLOCKED", "CHECKPOINT_TRANSACTION_NOT_DURABLE")
            )
        return (
            ("COMPLETE", "EVENT_CLOUD_ADMITTED")
            if result["candidate_complete"]
            else ("BLOCKED", "NOT_COMPLETE")
        )

    monkeypatch.setattr(module, "producer_terminal", terminal)
    return module, instance, result


def test_waits_without_restart_until_candidate_complete(
    case: tuple[Any, Any, dict[str, Any]],
) -> None:
    _module, instance, _result = case
    value = instance.step()
    deferred = value["tasks"][v9.DEFERRED]
    assert (deferred["state"], deferred["reason"], deferred["retries"]) == (
        "WAITING_CAPACITY",
        "CAPACITY_WAIT",
        3,
    )


def test_releases_on_first_scientifically_complete_candidate(
    case: tuple[Any, Any, dict[str, Any]], tmp_path: Path
) -> None:
    _module, instance, result = case
    candidate = v9.CANDIDATES[2]
    instance.state["tasks"][candidate]["state"] = "COMPLETE"
    result["candidate_complete"] = True
    value = instance.step()
    deferred = value["tasks"][v9.DEFERRED]
    assert deferred["state"] == "RESTARTED"
    assert deferred["capacity_released_by"] == candidate
    assert deferred["retries"] == 3
    assert (tmp_path / "EVENT_CAPACITY_RELEASE_RECEIPT.json").is_file()


def test_bad_deferred_boundary_never_restarts(case: tuple[Any, Any, dict[str, Any]]) -> None:
    _module, instance, result = case
    instance.state["tasks"][v9.CANDIDATES[0]]["state"] = "COMPLETE"
    result.update(candidate_complete=True, deferred_durable=False)
    value = instance.step()
    assert value["tasks"][v9.DEFERRED]["state"] == "BLOCKED"


def test_deadline_prevents_release(case: tuple[Any, Any, dict[str, Any]]) -> None:
    _module, instance, result = case
    instance.config["authority"]["global_deadline_utc"] = "2020-01-01T00:00:00+00:00"
    instance.state["tasks"][v9.CANDIDATES[0]]["state"] = "COMPLETE"
    result["candidate_complete"] = True
    value = instance.step()
    assert value["tasks"][v9.DEFERRED]["state"] == "DEADLINE_EXPIRED"


def test_stop_prevents_release(
    case: tuple[Any, Any, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, instance, result = case
    stop = tmp_path / "STOP_REQUEST"
    stop.write_text("manual", encoding="utf-8")
    monkeypatch.setattr(module, "stop_paths", lambda _task: [stop])
    instance.state["tasks"][v9.CANDIDATES[0]]["state"] = "COMPLETE"
    result["candidate_complete"] = True
    value = instance.step()
    assert value["tasks"][v9.DEFERRED] == {
        **value["tasks"][v9.DEFERRED],
        "state": "BLOCKED",
        "reason": "STOP_REQUEST_PRESENT",
    }
