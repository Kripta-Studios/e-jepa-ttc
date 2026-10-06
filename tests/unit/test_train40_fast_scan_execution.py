"""The process-scan wrapper preserves all budget decisions and durable saves."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from operational.efficient_context.common import digest
from operational.train40_system import controller_fast_scan, engine_fast_scan, engine_overlap
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import read
from operational.train40_system.durable_io import atomic_json


def test_guard_decision_and_single_durable_save_are_preserved(monkeypatch, tmp_path: Path):
    from operational.train40_system import process_scan

    cache_freeze = tmp_path / "ADAPTIVE_CACHE_FREEZE.json"
    cache_freeze.write_text("{}", encoding="utf-8")
    atomic_json(
        tmp_path / "FAST_PROCESS_SCAN_FREEZE.json",
        {
            "files": [],
            "engine_sha256": digest(Path(engine_fast_scan.__file__)),
            "adaptive_cache_freeze_sha256": digest(cache_freeze),
        },
    )
    directory = tmp_path / "fits/a5_seed7"
    directory.mkdir(parents=True)
    decision = (False, {"reasons": ["PHYSICAL_UPDATE_CAP"]})
    monkeypatch.setattr(process_scan, "resource_guard", lambda _: decision)
    monkeypatch.setattr(process_scan, "snapshot", lambda: {"scans": 1})
    saves = []
    monkeypatch.setattr(DurableState, "save", lambda *args, **kwargs: saves.append((args, kwargs)))
    original_guard, original_save = engine_overlap.resource_guard, DurableState.save
    state = cast(Any, SimpleNamespace(committed=12, durable=12))

    def delegated(output, arm):
        assert output == tmp_path and arm == "a5"
        assert engine_overlap.resource_guard(output) is decision
        cast(Any, DurableState.save)(state, "model", status="PAUSED_RESOURCE")
        raise RuntimeError("after durable save")

    monkeypatch.setattr(engine_fast_scan.engine_adaptive_cache, "run", delegated)
    with pytest.raises(RuntimeError, match="after durable"):
        engine_fast_scan.run(tmp_path, "a5")
    assert saves == [((state, "model"), {"status": "PAUSED_RESOURCE"})]
    assert engine_overlap.resource_guard is original_guard
    assert DurableState.save is original_save
    counters = read(directory / "FAST_PROCESS_SCAN_COUNTERS.json")
    assert counters["guard_calls"] == 1
    assert counters["durable_updates"] == 12


@pytest.mark.parametrize("arm", ["a5", "c2f"])
def test_exact_producer_route_only(monkeypatch, tmp_path: Path, arm: str):
    calls = []
    monkeypatch.setattr(
        controller_fast_scan.controller_overlap, "_launch", lambda *args: calls.append(args)
    )
    controller_fast_scan.launch(tmp_path, arm, "operational.train40_system.engine", ["--arm", arm])
    assert calls == [(tmp_path, arm, "operational.train40_system.engine_fast_scan", ["--arm", arm])]
    with pytest.raises(ValueError, match="Unexpected"):
        controller_fast_scan.launch(tmp_path, "other", "operational.train40_system.engine", [])


def test_non_producer_route_is_unchanged(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(controller_fast_scan, "_launch", lambda *args: calls.append(args))
    controller_fast_scan.launch(tmp_path, "features", "existing.route", ["--kind", "H8"])
    assert calls == [(tmp_path, "features", "existing.route", ["--kind", "H8"])]
