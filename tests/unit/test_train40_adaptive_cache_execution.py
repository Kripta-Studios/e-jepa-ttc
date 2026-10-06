"""Cache-only routing keeps original training and checkpoint lifecycle semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from operational.train40_system import (
    adaptive_cache,
    controller_adaptive_cache,
    engine_adaptive_cache,
    engine_overlap,
)
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import read


@pytest.mark.parametrize("arm", ["a5", "c2f"])
def test_only_original_exact_fit_route_changes(monkeypatch, tmp_path: Path, arm: str) -> None:
    calls = []
    monkeypatch.setattr(
        controller_adaptive_cache.controller_overlap, "_launch", lambda *args: calls.append(args)
    )
    controller_adaptive_cache.launch(
        tmp_path, arm, "operational.train40_system.engine", ["--arm", arm]
    )
    assert calls == [
        (tmp_path, arm, "operational.train40_system.engine_adaptive_cache", ["--arm", arm])
    ]
    with pytest.raises(ValueError, match="Unexpected"):
        controller_adaptive_cache.launch(
            tmp_path, arm, "operational.train40_system.engine", ["--arm", arm, "--batch-size", "64"]
        )


def test_unrelated_task_delegates_unchanged(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(controller_adaptive_cache, "_launch", lambda *args: calls.append(args))
    arguments = ["--kind", "H8"]
    controller_adaptive_cache.launch(tmp_path, "h8_features", "other.module", arguments)
    assert calls == [(tmp_path, "h8_features", "other.module", arguments)]


def test_installed_inputs_restores_bindings_after_failure(monkeypatch, tmp_path: Path) -> None:
    class FakeInputs:
        def __init__(self, output: Path) -> None:
            self.output = output

        def snapshot(self) -> dict:
            return {"cache_bytes": 12, "reads": 1}

    saves = []
    monkeypatch.setattr(adaptive_cache, "AdaptiveSealedInputs", FakeInputs)
    monkeypatch.setattr(DurableState, "save", lambda *args, **kwargs: saves.append((args, kwargs)))
    original_inputs, original_save = engine_overlap.SealedInputs, DurableState.save
    state = cast(Any, SimpleNamespace(committed=123, durable=123))
    with pytest.raises(RuntimeError, match="injected"):
        with engine_adaptive_cache.installed_inputs(tmp_path) as instances:
            engine_overlap.SealedInputs(tmp_path)
            cast(Any, DurableState.save)(state, "model", status="RUNNING")
            assert len(instances) == 1
            raise RuntimeError("injected")
    assert engine_overlap.SealedInputs is original_inputs
    assert DurableState.save is original_save
    assert len(saves) == 1
    assert saves[0] == ((state, "model"), {"status": "RUNNING"})
    assert read(tmp_path / "ADAPTIVE_CACHE_COUNTERS.json")["durable_updates"] == 123


def test_failed_original_save_does_not_publish_telemetry(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise OSError("checkpoint publication failed")

    monkeypatch.setattr(DurableState, "save", fail)
    with pytest.raises(OSError, match="checkpoint"):
        with engine_adaptive_cache.installed_inputs(tmp_path):
            cast(Any, DurableState.save)(SimpleNamespace(committed=1, durable=0))
    assert not (tmp_path / "ADAPTIVE_CACHE_COUNTERS.json").exists()
    assert DurableState.save is fail
