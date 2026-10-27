"""The V2 PID identity overlay is Unicode-safe and fail closed."""

from __future__ import annotations

from types import SimpleNamespace

import psutil
import pytest

from scripts import runpod_rgb_durable_phase_worker_v2 as worker


def test_accented_argv_matches_and_reuses_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [r"C:\Users\Álvaro Schwiedop\.venv\python.exe", "worker.py"]
    process = SimpleNamespace(
        is_running=lambda: True,
        status=lambda: psutil.STATUS_RUNNING,
        cmdline=lambda: list(argv),
    )
    monkeypatch.setattr(worker.psutil, "Process", lambda pid: process)
    assert worker.process_matches(3352, argv)


def test_exact_argv_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(
        is_running=lambda: True,
        status=lambda: psutil.STATUS_RUNNING,
        cmdline=lambda: ["python", "other.py"],
    )
    monkeypatch.setattr(worker.psutil, "Process", lambda pid: process)
    assert not worker.process_matches(3352, ["python", "worker.py"])


def test_missing_pid_is_not_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(_pid: int) -> None:
        raise psutil.NoSuchProcess(4)

    monkeypatch.setattr(worker.psutil, "Process", denied)
    assert not worker.process_matches(4, ["python"])


def test_inaccessible_pid_blocks_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(_pid: int) -> None:
        raise psutil.AccessDenied(4)

    monkeypatch.setattr(worker.psutil, "Process", denied)
    with pytest.raises(RuntimeError, match="Cannot prove"):
        worker.process_matches(4, ["python"])


def test_zombie_is_not_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(
        is_running=lambda: True,
        status=lambda: psutil.STATUS_ZOMBIE,
        cmdline=lambda: ["python"],
    )
    monkeypatch.setattr(worker.psutil, "Process", lambda pid: process)
    assert not worker.process_matches(4, ["python"])


def test_frozen_v1_is_loaded_with_only_identity_replaced() -> None:
    module = worker.load_v1()
    assert module.process_matches is worker.process_matches
    assert module.CONFIG_SCHEMA == "ejepa_rgb_durable_phase_worker_config_v1"
