"""CPU-only tests for the bounded TRAIN40 process scan wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from operational.train40_system import models, process_scan


class AccessDeniedError(Exception):
    pass


class NoSuchProcessError(Exception):
    pass


class ZombieProcessError(Exception):
    pass


@dataclass
class FakeProcess:
    pid: int
    name: str = "python.exe"
    command: tuple[str, ...] = ()
    failure: type[Exception] | None = None
    cmdline_calls: int = 0

    @property
    def info(self) -> dict[str, str]:
        return {"name": self.name}

    def cmdline(self) -> list[str]:
        self.cmdline_calls += 1
        if self.failure is not None:
            raise self.failure()
        return list(self.command)

    def parents(self) -> list[FakeProcess]:
        return []

    def children(self, *, recursive: bool) -> list[FakeProcess]:
        assert recursive
        return []


class FakePsutil:
    AccessDenied = AccessDeniedError
    NoSuchProcess = NoSuchProcessError
    ZombieProcess = ZombieProcessError

    def __init__(self, current: FakeProcess, candidates: list[FakeProcess]) -> None:
        self.current = current
        self.candidates = candidates
        self.attrs: list[list[str]] = []

    def Process(self) -> FakeProcess:  # noqa: N802
        return self.current

    def process_iter(self, attrs: list[str]) -> list[FakeProcess]:
        self.attrs.append(attrs)
        return self.candidates


@pytest.mark.parametrize(
    "command",
    [
        ("python", "-m", "operational.train40_system.models"),
        ("python", "-m", "operational.train40_system.teacher"),
        ("python", "-m", "operational.efficient_context.garl_train"),
        ("python", "-m", "operational.efficient_context.garl_heads"),
        ("python", "-m", "operational.efficient_context.run", "train"),
        ("python", "train_worker", "stage73"),
    ],
)
def test_legacy_heavy_classification_is_preserved(command: tuple[str, ...]) -> None:
    assert process_scan._is_heavy_command(command)


@pytest.mark.parametrize(
    "module",
    [
        "operational.train40_system.engine",
        "operational.train40_system.engine_overlap",
        "operational.train40_system.engine_profiled",
        "operational.train40_system.engine_relation_reuse",
        "operational.train40_system.engine_idle_wait",
        "operational.train40_system.engine_adaptive_cache",
        "operational.train40_system.engine_future_admitted",
        "operational.train40_system.graph_launch_admission",
    ],
)
def test_admitted_engine_and_graph_probe_modules_are_heavy(module: str) -> None:
    assert process_scan._is_heavy_command(("python", "-m", module, "--arm", "a5"))


def test_scan_filters_name_before_cmdline_and_excludes_related_processes() -> None:
    current = FakeProcess(10)
    child = FakeProcess(11, command=("python", "-m", "operational.train40_system.engine"))
    current.children = lambda recursive: [child]  # type: ignore[method-assign]
    non_python = FakeProcess(12, name="firefox.exe", command=("private",))
    heavy = FakeProcess(
        13, command=("python", "-m", "operational.train40_system.engine_adaptive_cache")
    )
    inaccessible = FakeProcess(14, failure=AccessDeniedError)
    fake = FakePsutil(current, [current, child, non_python, heavy, inaccessible])

    others, counts = process_scan._scan(
        fake, (AccessDeniedError, NoSuchProcessError, ZombieProcessError)
    )

    assert others == [13]
    assert fake.attrs == [["name"]]
    assert current.cmdline_calls == child.cmdline_calls == non_python.cmdline_calls == 0
    assert heavy.cmdline_calls == inaccessible.cmdline_calls == 1
    assert counts == {
        "processes_seen": 5,
        "python_cmdline_reads": 2,
        "skipped_non_python": 1,
        "cmdline_read_failures": 1,
    }


def test_wrapper_uses_five_second_cache_and_delegates_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan_calls = 0
    delegate_calls = 0
    clock = iter((10.0, 10.1, 10.2, 12.0, 12.1, 12.2))

    def scan() -> tuple[list[int], dict[str, int]]:
        nonlocal scan_calls
        scan_calls += 1
        return [321], {
            "processes_seen": 4,
            "python_cmdline_reads": 1,
            "skipped_non_python": 3,
            "cmdline_read_failures": 0,
        }

    def delegate(output: Path) -> tuple[bool, dict[str, Any]]:
        nonlocal delegate_calls
        delegate_calls += 1
        assert output == tmp_path
        assert models._heavy_scan_at > 0
        return True, {"reasons": [], "other_heavy_pids": models._other_heavy_pids}

    monkeypatch.setattr(process_scan, "_scan", scan)
    monkeypatch.setattr(process_scan, "_ORIGINAL_RESOURCE_GUARD", delegate)
    monkeypatch.setattr(process_scan.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(process_scan.time, "perf_counter", lambda: 1.0)
    monkeypatch.setattr(models, "_heavy_scan_at", float("-inf"))
    monkeypatch.setattr(models, "_other_heavy_pids", [])

    first = process_scan.resource_guard(tmp_path)
    second = process_scan.resource_guard(tmp_path)

    assert first == second == (True, {"reasons": [], "other_heavy_pids": [321]})
    assert scan_calls == 1
    assert delegate_calls == 2
    assert models._heavy_scan_at == 10.1


def test_wrapper_restores_cadence_when_budget_delegate_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(process_scan, "_scan", lambda: ([], {name: 0 for name in (
        "processes_seen", "python_cmdline_reads", "skipped_non_python", "cmdline_read_failures"
    )}))
    monkeypatch.setattr(process_scan.time, "monotonic", lambda: 20.0)
    monkeypatch.setattr(models, "_heavy_scan_at", 19.0)

    def fail(_output: Path) -> tuple[bool, dict]:
        raise RuntimeError("budget guard failed")

    monkeypatch.setattr(process_scan, "_ORIGINAL_RESOURCE_GUARD", fail)
    with pytest.raises(RuntimeError, match="budget guard failed"):
        process_scan.resource_guard(tmp_path)
    assert models._heavy_scan_at == 19.0
