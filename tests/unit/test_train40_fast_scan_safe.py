"""Regression checks for unconditional fast-scan wrapper restoration."""

from pathlib import Path
from typing import Any, cast

import pytest

from operational.efficient_context.common import digest
from operational.train40_system import (
    engine_fast_scan,
    engine_fast_scan_safe,
    engine_overlap,
)
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.durable_io import atomic_json


def _write_graph_freeze(output: Path) -> None:
    atomic_json(
        output / "GRAPH_REPLAY_FREEZE.json",
        {
            "files": [],
            "fast_scan_safe_engine_sha256": digest(Path(engine_fast_scan_safe.__file__)),
        },
    )


def test_early_sidecar_failure_restores_both_wrappers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_freeze = tmp_path / "ADAPTIVE_CACHE_FREEZE.json"
    atomic_json(cache_freeze, {})
    atomic_json(
        tmp_path / "FAST_PROCESS_SCAN_FREEZE.json",
        {
            "files": [],
            "engine_sha256": digest(Path(engine_fast_scan.__file__)),
            "adaptive_cache_freeze_sha256": digest(cache_freeze),
        },
    )
    _write_graph_freeze(tmp_path)
    original_guard = engine_overlap.resource_guard
    original_save = cast(Any, DurableState.save)

    def fail_sidecar(*_args: object, **_kwargs: object) -> None:
        raise OSError("sidecar write failed")

    monkeypatch.setattr(engine_fast_scan, "atomic_json", fail_sidecar)
    with pytest.raises(OSError, match="sidecar write failed"):
        engine_fast_scan_safe.run(tmp_path, "a5")

    assert engine_overlap.resource_guard is original_guard
    assert DurableState.save is original_save


def test_success_delegates_without_an_extra_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_graph_freeze(tmp_path)
    calls: list[tuple[Path, str]] = []
    saves = 0

    def save(*_args: object, **_kwargs: object) -> None:
        nonlocal saves
        saves += 1

    def delegated(output: Path, arm: str) -> None:
        calls.append((output, arm))

    monkeypatch.setattr(DurableState, "save", save)
    monkeypatch.setattr(engine_fast_scan, "run", delegated)
    original_guard = engine_overlap.resource_guard

    assert engine_fast_scan_safe.run(tmp_path, "c2f") is None
    assert calls == [(tmp_path, "c2f")]
    assert saves == 0
    assert engine_overlap.resource_guard is original_guard
    assert DurableState.save is save
