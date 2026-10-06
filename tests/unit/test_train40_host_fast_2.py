"""CPU integration contracts for additive host-pipeline routing and cleanup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from operational.train40_system import controller_host_fast_2 as controller
from operational.train40_system import engine_host_fast_2 as engine


def test_admission_rejects_changed_qa_receipt(tmp_path: Path) -> None:
    from operational.efficient_context.common import digest
    from operational.train40_system.durable_io import atomic_json

    qa = tmp_path / "QA.txt"
    qa.write_text("passed", encoding="utf-8")
    atomic_json(
        tmp_path / "HOST_FAST_2_FREEZE.json",
        {
            "files": [],
            "engine_sha256": digest(Path(engine.__file__)),
            "predecessor_freezes": {},
            "QA": {qa.name: digest(qa)},
        },
    )
    engine.verify_admission(tmp_path)
    qa.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="QA receipt differs"):
        engine.verify_admission(tmp_path)


def test_exact_producer_routing_and_argument_rejection(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(controller.controller_overlap, "_launch", lambda *a: calls.append(a))
    for arm in ("a5", "c2f"):
        controller.launch(tmp_path, arm, "operational.train40_system.engine", ["--arm", arm])
        assert calls[-1][2] == "operational.train40_system.engine_host_fast_2"
        assert calls[-1][3] == ["--arm", arm]
    with pytest.raises(ValueError, match="Unexpected"):
        controller.launch(tmp_path, "a5", "operational.train40_system.engine", ["--arm", "c2f"])


def test_nonproducer_route_unchanged(monkeypatch, tmp_path: Path) -> None:
    args = (
        tmp_path,
        "h8_features",
        "operational.train40_system.history_features",
        ["--kind", "H8"],
    )
    calls: list[tuple] = []
    monkeypatch.setattr(controller, "_launch", lambda *a: calls.append(a))
    controller.launch(*args)
    assert calls == [args]


def test_existing_process_probe_adopts_host_fast_2_without_duplicate_launch(
    monkeypatch,
) -> None:
    from operational.train40_system import controller as original_controller

    process = SimpleNamespace(
        info={
            "name": "python.exe",
            "cmdline": ["python.exe", "-m", "operational.train40_system.engine_host_fast_2"],
        }
    )
    monkeypatch.setattr(original_controller.psutil, "process_iter", lambda _: [process])
    assert controller.controller_overlap.active("operational.train40_system.engine")


def test_planned_prefetch_preserves_order_and_rejects_pending_epoch() -> None:
    hints, takes = [], []

    class FakePrefetch:
        def __init__(self) -> None:
            self.order = None
            self.pending = []
            self.source = SimpleNamespace(prepare_order=lambda *a: hints.append(a))

        def take(self, *args):
            takes.append(args)
            return "ticket"

    wrapper = engine.planned_prefetch_type(FakePrefetch)()
    order = [3, 2, 1, 0]
    assert wrapper.take(order, 0, 2) == "ticket"
    assert hints == takes == [(order, 0, 2)]
    wrapper.order, wrapper.pending = order, [object()]
    with pytest.raises(ValueError, match="unconsumed"):
        wrapper.take([9], 0, 1)
    assert len(hints) == 1


@pytest.mark.parametrize("arm", ["a5", "c2f"])
@pytest.mark.parametrize("fail_stage", ["start", "delegate", "none"])
def test_bindings_and_children_restored_on_every_exit(
    monkeypatch, tmp_path: Path, arm: str, fail_stage: str
) -> None:
    from operational.train40_system import (
        engine_graph_replay,
        engine_graph_replay_scalar_safe,
        engine_overlap,
        process_inputs_fast_2,
        resource_monitor,
    )
    from operational.train40_system.checkpoint import DurableState

    target = engine_graph_replay if arm == "a5" else engine_overlap
    delegate = engine_graph_replay_scalar_safe if arm == "a5" else engine_overlap
    names = ("SealedInputs", "DevicePrefetch", "resource_guard")
    before = {name: getattr(target, name) for name in names}
    original_save = DurableState.save
    original_scan = getattr(target, "scan_snapshot", None)
    events = []

    class Monitor:
        def __init__(self, output):
            pass

        def start(self):
            events.append("monitor_start")
            if fail_stage == "start":
                raise RuntimeError("test failure")

        def guard(self, output):
            return True, {}

        def snapshot(self):
            return {}

        def close(self):
            events.append("monitor_close")

    class Inputs:
        def __init__(self, output):
            pass

        def snapshot(self):
            return {}

        def close(self):
            events.append("input_close")

    def delegated(output, actual_arm):
        assert actual_arm == arm
        assert target.resource_guard(output) == (True, {})
        target.SealedInputs(output)
        events.append("delegate")
        if fail_stage == "delegate":
            raise RuntimeError("test failure")

    (tmp_path / "HOST_FAST_2_FREEZE.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine, "verify_admission", lambda _: {})
    monkeypatch.setattr(resource_monitor, "ResourceMonitor", Monitor)
    monkeypatch.setattr(process_inputs_fast_2, "ProcessInputs", Inputs)
    monkeypatch.setattr(delegate, "run", delegated)
    if fail_stage == "none":
        engine.run(tmp_path, arm)
    else:
        with pytest.raises(RuntimeError, match="test failure"):
            engine.run(tmp_path, arm)
    assert {name: getattr(target, name) for name in names} == before
    assert DurableState.save is original_save
    assert getattr(target, "scan_snapshot", None) is original_scan
    assert events[-1] == "monitor_close"
    if fail_stage != "start":
        assert "input_close" in events


def test_controller_restores_route_on_failure(monkeypatch, tmp_path: Path) -> None:
    original = controller.controller_overlap.launch
    monkeypatch.setattr(controller, "verify_admission", lambda _: {})

    def failure(*args: Any) -> None:
        assert controller.controller_overlap.launch is controller.launch
        raise RuntimeError("controller failure")

    monkeypatch.setattr(controller.controller_pause_safe, "run", failure)
    with pytest.raises(RuntimeError, match="controller failure"):
        controller.run(tmp_path, tmp_path, tmp_path)
    assert controller.controller_overlap.launch is original


def test_engine_variant_is_exact_name_substitution() -> None:
    root = Path(__file__).parents[2] / "operational/train40_system"
    base = (root / "engine_host_fast_collate.py").read_text(encoding="utf-8")
    candidate = (root / "engine_host_fast_2.py").read_text(encoding="utf-8")
    reconstructed = candidate.replace("Host fast 2", "Host pipeline")
    reconstructed = reconstructed.replace("HOST_FAST_2", "HOST_FAST_COLLATE")
    reconstructed = reconstructed.replace("host_fast_2", "host_fast_collate")
    reconstructed = reconstructed.replace(
        "process_inputs_fast_2", "process_inputs_fast_collate"
    )
    assert reconstructed == base
