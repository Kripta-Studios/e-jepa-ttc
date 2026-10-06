"""CPU-only admission tests for bounded profiling of real TRAIN40 commits."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from operational.efficient_context.common import ROOT, digest
from operational.train40_system import (
    controller_profiled,
    engine_profiled,
    execution_profile_freeze,
)
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.durable_io import atomic_json


class FakeProfiler:
    def __init__(self, callback: Any = None, *, fail_step: int | None = None) -> None:
        self.callback = callback
        self.fail_step = fail_step
        self.started = 0
        self.steps = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def step(self) -> None:
        self.steps += 1
        if self.steps == self.fail_step:
            raise RuntimeError("profile step failed")
        if self.steps == engine_profiled.PROFILE_UPDATES and self.callback is not None:
            self.callback(self)

    def stop(self) -> None:
        self.stopped += 1


def _run_with_fake_profiler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    engine: Any,
    *,
    fail_step: int | None = None,
) -> tuple[FakeProfiler, list[int], Any]:
    (tmp_path / "EXECUTION_PROFILE_FREEZE.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine_profiled, "_verify_execution", lambda output: {})
    monkeypatch.setattr(engine_profiled, "_profile_directory", lambda output, arm: tmp_path)
    monkeypatch.setattr(
        engine_profiled,
        "_trace_callback",
        lambda output, directory, observation: lambda p: None,
    )
    calls: list[int] = []

    def original(state: Any) -> None:
        calls.append(id(state))

    monkeypatch.setattr(DurableState, "commit_update", original)
    original_method = DurableState.commit_update
    fake: FakeProfiler | None = None

    def profile(**kwargs: Any) -> FakeProfiler:
        nonlocal fake
        fake = FakeProfiler(kwargs["on_trace_ready"], fail_step=fail_step)
        return fake

    monkeypatch.setattr(engine_profiled.torch.profiler, "profile", profile)
    monkeypatch.setattr(engine_profiled.engine_overlap, "run", engine)
    engine_profiled.run(tmp_path, "a5")
    assert fake is not None
    return fake, calls, original_method


def test_wrapper_calls_original_commit_once_and_restores_after_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def engine(output: Path, arm: str) -> None:
        state = SimpleNamespace(committed=0)
        for committed in range(1, engine_profiled.PROFILE_UPDATES + 2):
            state.committed = committed
            DurableState.commit_update(state)  # type: ignore[arg-type]

    profiler, calls, original = _run_with_fake_profiler(monkeypatch, tmp_path, engine)
    assert len(calls) == engine_profiled.PROFILE_UPDATES + 1
    assert profiler.started == 1
    assert profiler.steps == engine_profiled.PROFILE_UPDATES
    assert profiler.stopped == 1
    assert DurableState.commit_update is original


def test_profile_failure_unpatches_and_training_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[str] = []

    def engine(output: Path, arm: str) -> None:
        state = SimpleNamespace(committed=0)
        for committed in range(1, 4):
            state.committed = committed
            DurableState.commit_update(state)  # type: ignore[arg-type]
            observed.append("continued")

    profiler, calls, original = _run_with_fake_profiler(
        monkeypatch, tmp_path, engine, fail_step=2
    )
    assert len(calls) == 3
    assert observed == ["continued"] * 3
    assert profiler.steps == 2
    assert profiler.stopped == 1
    assert DurableState.commit_update is original
    failure = json.loads((tmp_path / "FULL_STEP_PROFILE_FAILURE.json").read_text("utf-8"))
    assert failure["training_continues_without_profiler"] is True
    assert failure["additional_optimizer_updates"] == 0


def test_engine_exception_restores_class_and_stops_profiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def engine(output: Path, arm: str) -> None:
        DurableState.commit_update(SimpleNamespace(committed=1))  # type: ignore[arg-type]
        raise RuntimeError("trainer failed")

    with pytest.raises(RuntimeError, match="trainer failed"):
        _run_with_fake_profiler(monkeypatch, tmp_path, engine)


def test_trace_callback_writes_relative_trace_tables_and_sha_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = ROOT / "artifacts" / "profile_callback_cpu_test"
    directory = output / "fits/a5_seed7/full_step_profile"
    directory.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []

    class Averages:
        def table(self, *, sort_by: str, row_limit: int) -> str:
            return f"{sort_by}:{row_limit}"

    class Profiler:
        def export_chrome_trace(self, path: str) -> None:
            exported.append(path)
            Path(path).write_text('{"traceEvents": []}', encoding="utf-8")

        def key_averages(self) -> Averages:
            return Averages()

    try:
        monkeypatch.chdir(ROOT)
        observation = {
            "checkpoint_start_committed": 14200,
            "scheduled_commit_range": [14201, 14227],
            "active_trace_commit_range": [14223, 14227],
        }
        engine_profiled._trace_callback(output, directory, observation)(Profiler())
        assert exported == [
            str((directory / "FULL_STEP_TRACE.json").relative_to(ROOT))
        ]
        manifest = json.loads(
            (directory / "FULL_STEP_PROFILE_MANIFEST.json").read_text(encoding="utf-8")
        )
        assert manifest["additional_optimizer_updates"] == 0
        assert manifest["contains_raw_tensor_payloads"] is False
        assert manifest["commit_observation"] == observation
        assert len(manifest["artifacts"]) == 3
        for artifact in manifest["artifacts"]:
            path = output / artifact["path"]
            assert artifact["sha256"] == digest(path)
    finally:
        for path in sorted(directory.glob("*"), reverse=True):
            path.unlink()
        directory.rmdir()
        directory.parent.rmdir()
        directory.parent.parent.rmdir()
        output.rmdir()


def test_controller_routes_only_exact_a5_c2f_engine_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, str, str, list[str]]] = []

    def raw_launch(output: Path, tag: str, module: str, arguments: list[str]) -> Any:
        calls.append((output, tag, module, arguments))
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(controller_profiled.controller_overlap, "_launch", raw_launch)
    monkeypatch.setattr(controller_profiled, "_launch", raw_launch)
    controller_profiled.launch(
        tmp_path, "a5", "operational.train40_system.engine", ["--arm", "a5"]
    )
    controller_profiled.launch(tmp_path, "teacher", "some.other.module", [])
    assert [call[2] for call in calls] == [
        "operational.train40_system.engine_profiled",
        "some.other.module",
    ]
    with pytest.raises(ValueError, match="engine task"):
        controller_profiled.launch(
            tmp_path, "a5", "operational.train40_system.engine", ["--arm", "c2f"]
        )


def test_controller_composes_pause_safe_run_and_restores_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "COORDINATION_FREEZE.json",
        "CONTROLLER_PAUSE_SAFE_FREEZE.json",
        "EXECUTION_PROFILE_FREEZE.json",
    ):
        atomic_json(tmp_path / name, {"files": []})
    original = controller_profiled.controller_overlap.launch
    observed: list[Any] = []

    def run(output: Path, raw: Path, teacher: Path) -> None:
        observed.extend([output, raw, teacher, controller_profiled.controller_overlap.launch])

    monkeypatch.setattr(controller_profiled.controller_pause_safe, "run", run)
    raw, teacher = tmp_path / "raw", tmp_path / "teacher"
    controller_profiled.run(tmp_path, raw, teacher)
    assert observed[:3] == [tmp_path, raw, teacher]
    assert observed[3] is controller_profiled.launch
    assert controller_profiled.controller_overlap.launch is original


def test_freeze_requires_cpu_qa_and_binds_both_published_freezes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, marker in execution_profile_freeze.QA_CHECKS.items():
        (tmp_path / name).write_text(marker, encoding="utf-8")
    coordination = tmp_path / "COORDINATION_FREEZE.json"
    pause = tmp_path / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    atomic_json(coordination, {"files": [], "trainer_sha256": "trainer"})
    atomic_json(pause, {"files": [], "source_sha256": "pause"})
    monkeypatch.setattr(execution_profile_freeze, "dependency_files", lambda seeds: [])
    execution_profile_freeze.run(tmp_path)
    frozen = json.loads((tmp_path / "EXECUTION_PROFILE_FREEZE.json").read_text("utf-8"))
    assert frozen["coordination_freeze_sha256"] == digest(coordination)
    assert frozen["controller_pause_safe_freeze_sha256"] == digest(pause)
    assert frozen["additional_optimizer_updates"] == 0
    assert frozen["instrumentation_removed_after_updates"] == 27
    (tmp_path / "TRAIN40_PROFILED_RUFF.txt").write_text("failed", encoding="utf-8")
    (tmp_path / "EXECUTION_PROFILE_FREEZE.json").unlink()
    with pytest.raises(ValueError, match="QA failed"):
        execution_profile_freeze.run(tmp_path)
