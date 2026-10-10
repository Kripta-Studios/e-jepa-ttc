"""Filtering before RSS preserves resource decisions and fresh membership."""

from types import SimpleNamespace

import psutil
import pytest

from operational.rgb_port import train_producers as training
from operational.rgb_port_rgb_loader import guards


class Process:
    """Model psutil as_dict's AccessDenied default and process_iter disappearance."""

    def __init__(self, pid, command, rss=100, failure=None):
        self.pid = pid
        self.command = command
        self.rss = rss
        self.failure = failure
        self.info = {}
        self.memory_calls = 0

    def as_dict(self, attrs):
        result = {}
        for name in attrs:
            if name == "pid":
                result[name] = self.pid
            elif name == "cmdline":
                if self.failure == "cmd_gone":
                    raise psutil.NoSuchProcess(self.pid)
                result[name] = None if self.failure == "cmd_denied" else self.command
            elif name == "memory_info":
                self.memory_calls += 1
                if self.failure == "memory_gone":
                    raise psutil.NoSuchProcess(self.pid)
                result[name] = (
                    None if self.failure == "memory_denied" else SimpleNamespace(rss=self.rss)
                )
        return result


def _environment(
    monkeypatch, processes, *, available=8 * 1024**3, commit=8 * 1024**3, disk=100_000_000_000
):
    def process_iter(attrs):
        for process in processes:
            try:
                process.info = process.as_dict(attrs)
            except psutil.NoSuchProcess:
                continue
            yield process

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(psutil, "disk_usage", lambda path: SimpleNamespace(free=disk))
    monkeypatch.setattr(training, "_windows_commit_headroom", lambda: commit)


def test_guard_identical_snapshot_and_skips_only_unrelated_memory(monkeypatch, tmp_path):
    processes = [
        Process(1, ["python", "operational.rgb_port_rgb_loader.producer"]),
        Process(2, ["pytest", "operational.rgb_port"]),
        Process(3, ["firefox"]),
        Process(4, ["python", "OPERATIONAL.RGB_PORT.queue"], rss=200),
    ]
    _environment(monkeypatch, processes)
    original = training._resource_guard(tmp_path, checkpoint_reservation=1024**3)
    calls = [process.memory_calls for process in processes]
    actual = guards.resource_guard(tmp_path, checkpoint_reservation=1024**3)
    assert actual == original
    assert calls == [1, 1, 1, 1]
    assert [
        process.memory_calls - count for process, count in zip(processes, calls, strict=True)
    ] == [
        1,
        0,
        0,
        1,
    ]


@pytest.mark.parametrize(
    "changes,rss",
    [
        ({"available": 2 * 1024**3 - 1}, 100),
        ({"available": 2 * 1024**3}, 100),
        ({"commit": 3 * 1024**3 - 1}, 100),
        ({"commit": 3 * 1024**3}, 100),
        ({"commit": None}, 100),
        ({"disk": 10_000_000_000 + 1024**3 - 1}, 100),
        ({"disk": 10_000_000_000 + 1024**3}, 100),
        ({}, 23_000_000_000),
        ({}, 23_000_000_001),
    ],
)
def test_guard_preserves_every_original_threshold(monkeypatch, tmp_path, changes, rss):
    _environment(monkeypatch, [Process(1, ["operational.rgb_port"], rss=rss)], **changes)
    assert guards.resource_guard(tmp_path, checkpoint_reservation=1024**3) == (
        training._resource_guard(tmp_path, checkpoint_reservation=1024**3)
    )


@pytest.mark.parametrize("failure", ["cmd_denied", "cmd_gone", "memory_gone"])
def test_guard_preserves_unreadable_or_disappeared_process_handling(monkeypatch, tmp_path, failure):
    _environment(monkeypatch, [Process(1, ["operational.rgb_port"], failure=failure)])
    assert guards.resource_guard(tmp_path, checkpoint_reservation=1024**3) == (
        training._resource_guard(tmp_path, checkpoint_reservation=1024**3)
    )


def test_guard_preserves_original_inaccessible_selected_rss_failure(monkeypatch, tmp_path):
    # as_dict maps AccessDenied to None. Preserve this historical fail-closed
    # AttributeError rather than silently excluding an unmeasurable project PID.
    _environment(monkeypatch, [Process(1, ["operational.rgb_port"], failure="memory_denied")])
    for function in (training._resource_guard, guards.resource_guard):
        with pytest.raises(AttributeError, match="rss"):
            function(tmp_path, checkpoint_reservation=1024**3)


def test_guard_requeries_command_and_membership_every_call(monkeypatch, tmp_path):
    process = Process(1, ["unrelated"])
    _environment(monkeypatch, [process])
    assert guards.resource_guard(tmp_path, checkpoint_reservation=0)[1]["project_processes"] == []
    process.command = ["operational.rgb_port"]
    assert guards.resource_guard(tmp_path, checkpoint_reservation=0)[1]["project_processes"] == [
        {"pid": 1, "rss": 100}
    ]
    process.command = ["pytest", "operational.rgb_port"]
    assert guards.resource_guard(tmp_path, checkpoint_reservation=0)[1]["project_processes"] == []
