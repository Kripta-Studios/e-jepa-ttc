"""Read-only cache advice fails closed on changed file identity."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cache_controller",
    Path(__file__).resolve().parents[2] / "scripts/runpod_test12_cache_reclaim.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_advice_preserves_source_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    path.write_bytes(b"unchanged")
    calls = []
    item = {"path": str(path), "stat": module.identity(path.stat())}
    module.advise(item, lambda *args: calls.append(args))
    assert path.read_bytes() == b"unchanged"
    assert calls[0][1:3] == (0, 0)


def test_changed_stat_rejected_before_advice(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    path.write_bytes(b"original")
    item = {"path": str(path), "stat": module.identity(path.stat())}
    path.write_bytes(b"changed size")
    calls = []
    with pytest.raises(ValueError, match="changed before"):
        module.advise(item, lambda *args: calls.append(args))
    assert not calls


def test_changed_stat_during_advice_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    path.write_bytes(b"original")
    item = {"path": str(path), "stat": module.identity(path.stat())}
    with pytest.raises(ValueError, match="changed during"):
        module.advise(item, lambda *_args: path.write_bytes(b"changed size"))
