"""Exited download processes must not crash the bounded RGB queue."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import psutil

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "rgb_v2", Path(__file__).parents[2] / "scripts/runpod_test12_rgb_controller_v2.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_owned_child_reaped_before_process_inspection(monkeypatch):
    child = Mock()
    child.poll.return_value = 0
    factory = Mock(side_effect=AssertionError("No psutil for reaped child"))
    monkeypatch.setattr(module.psutil, "Process", factory)
    children = {1: child}
    assert not module.worker_alive(1, Path("/worker"), "s", children)
    assert children == {}


def test_zombie_status_is_not_running(monkeypatch):
    process = Mock()
    process.status.return_value = psutil.STATUS_ZOMBIE
    monkeypatch.setattr(module.psutil, "Process", Mock(return_value=process))
    assert not module.worker_alive(1, Path("/worker"), "s", {})
    process.cmdline.assert_not_called()


def test_process_disappearing_during_cmdline_is_not_running(monkeypatch):
    process = Mock()
    process.status.return_value = psutil.STATUS_RUNNING
    process.cmdline.side_effect = psutil.NoSuchProcess(1)
    monkeypatch.setattr(module.psutil, "Process", Mock(return_value=process))
    assert not module.worker_alive(1, Path("/worker"), "s", {})


def test_cmdline_zombie_race_is_not_running(monkeypatch):
    process = Mock()
    process.status.return_value = psutil.STATUS_RUNNING
    process.cmdline.side_effect = psutil.ZombieProcess(1)
    monkeypatch.setattr(module.psutil, "Process", Mock(return_value=process))
    assert not module.worker_alive(1, Path("/worker"), "s", {})


def test_resource_resume_does_not_admit_integrity_failure():
    assert module.recoverable_stop(
        {
            "status": "STOPPED_REQUIRES_REVIEW",
            "error": "RuntimeError: Quota reserve unavailable: 0 < 1",
        }
    )
    assert not module.recoverable_stop(
        {
            "status": "STOPPED_REQUIRES_REVIEW",
            "error": "ValueError: Frozen RGB media identity failed",
        }
    )
