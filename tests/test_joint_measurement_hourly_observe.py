"""Hourly branch observation must not confuse old counters or reused process IDs."""

from types import SimpleNamespace

import pytest

from operational.joint_measurement import hourly_observe as observer
from operational.train40_system.durable_io import atomic_json


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setattr(observer, "RUN", tmp_path)
    process = SimpleNamespace(
        create_time=lambda: 100.0,
        is_running=lambda: True,
        memory_info=lambda: SimpleNamespace(rss=123),
    )
    monkeypatch.setattr(observer.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(
        observer.psutil, "virtual_memory", lambda: SimpleNamespace(available=456)
    )
    monkeypatch.setattr(
        observer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="test GPU", stderr=""),
    )
    atomic_json(tmp_path / "GPU_BUDGET.json", {"entries": [{"charged_s": 7}], "cap_s": 100})
    return tmp_path


def test_geometry_uses_only_its_own_progress(environment):
    atomic_json(
        environment / "rgb/fits/RGB/UPDATE_JOURNAL.json", {"completed_updates": 12000}
    )
    directory = environment / "adaptive/C1_geometry"
    atomic_json(directory / "fits/GEOMETRY/UPDATE_JOURNAL.json", {"completed_updates": 100})
    atomic_json(directory / "predictions/H/GEOMETRY/COMPLETE.json", {"rows": 18422})
    result = observer.observe(7, 100, None, job="geometry")
    assert result["driver_live_identity_verified"]
    assert result["progress_signature"]["updates"] == {"GEOMETRY": 100}
    assert list(result["progress_signature"]["evaluation_complete"].values()) == [18422]
    assert result["training_actions"] == []
    assert result["budget_reserved_upper_s"] == 7


def test_reused_pid_is_not_a_live_driver(environment):
    result = observer.observe(7, 99, None, job="geometry")
    assert result["status"] == "ACTION_REQUIRED_DRIVER_NOT_LIVE"
    assert not result["driver_live_identity_verified"]
    assert result["driver_rss_bytes"] is None
    assert result["training_actions"] == []


@pytest.mark.parametrize("job", ["rgb", "geometry"])
def test_terminal_receipt_never_terminates_a_live_process(environment, job):
    relative, _, terminal = observer.JOBS[job]
    atomic_json(environment / relative / "RUNNER_STATE.json", {"status": terminal})
    assert observer.observe(7, 100, None, job=job)["status"] == "RUNNING"
    assert observer.observe(7, 99, None, job=job)["status"] == terminal


def test_default_rgb_and_unchanged_hour_are_preserved(environment):
    atomic_json(environment / "rgb/fits/RGB/UPDATE_JOURNAL.json", {"completed_updates": 500})
    first = observer.observe(7, 100, None)
    assert first["job"] == "rgb"
    assert first["progress_signature"]["updates"] == {"RGB": 500}
    second = observer.observe(7, 100, first)
    assert second["status"] == "REVIEW_NO_PROGRESS_SINCE_PREVIOUS_HOUR"
    assert second["training_actions"] == []
