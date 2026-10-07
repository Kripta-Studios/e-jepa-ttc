"""Tests for direct-cursor H8 resume and deferred canonical assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from operational.train40_system import h8_resume_direct as direct


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("status", ["RUNNING", "PAUSED_PRESERVED", "COMPLETE"])
def test_cursor_accepts_durable_status_and_verifies_only_last_prefix(
    tmp_path: Path, status: str
) -> None:
    directory = tmp_path / "h8_feature_fragments"
    binding = {"source_sha256": "canonical"}
    binding_path = directory / "BINDING.json"
    _json(binding_path, binding)
    for row in range(3):
        (directory / f"query_{row:05d}.npz").write_bytes(f"row-{row}".encode())
    last = directory / "query_00002.npz"
    _json(
        last.with_suffix(".json"),
        {"sha256": _digest(last), "binding_sha256": _digest(binding_path)},
    )
    _json(
        tmp_path / "H8_FEATURE_PROGRESS.json",
        {"status": status, "completed_rows": 3, "total_rows": 5},
    )
    digested: list[Path] = []

    def digest(path: Path) -> str:
        digested.append(path)
        return _digest(path)

    kernel = SimpleNamespace(read=lambda path: json.loads(path.read_text()), digest=digest)
    assert direct._resume_cursor(
        tmp_path, directory, binding_path, binding, kernel=kernel, total_rows=5
    ) == 3
    assert digested == [binding_path, last]


@pytest.mark.parametrize(
    ("cursor", "status"),
    [(-1, "PAUSED_PRESERVED"), (6, "PAUSED_PRESERVED"), (3, "FAILED"), (True, "RUNNING")],
)
def test_cursor_rejects_invalid_bounds_or_status(
    tmp_path: Path, cursor: object, status: str
) -> None:
    directory = tmp_path / "h8_feature_fragments"
    directory.mkdir()
    binding_path = directory / "BINDING.json"
    binding = {"source_sha256": "canonical"}
    _json(binding_path, binding)
    _json(
        tmp_path / "H8_FEATURE_PROGRESS.json",
        {"status": status, "completed_rows": cursor, "total_rows": 5},
    )
    kernel = SimpleNamespace(read=lambda path: json.loads(path.read_text()), digest=_digest)
    with pytest.raises(ValueError, match="cursor"):
        direct._resume_cursor(
            tmp_path, directory, binding_path, binding, kernel=kernel, total_rows=5
        )


def _fragment(path: Path, features: np.ndarray, valid: np.ndarray) -> None:
    np.savez_compressed(path, features=features, valid=valid)


def test_deferred_assembly_matches_canonical_first_occurrence_order(tmp_path: Path) -> None:
    directory = tmp_path / "h8_feature_fragments"
    directory.mkdir()
    valid = np.ones(8, dtype=bool)
    first = np.arange(16, dtype=np.float32).reshape(8, 2)
    second = (100 + np.arange(16, dtype=np.float32)).reshape(8, 2)
    second[0] = first[1]
    _fragment(directory / "query_00000.npz", first, valid)
    _fragment(directory / "query_00001.npz", second, valid)
    histories = [
        ["a", "shared", None, None, None, None, None, None],
        ["shared", "c", None, None, None, None, None, None],
    ]
    jobs = [
        {"valid": valid, "anchor": 400_000, "available": 450_000},
        {"valid": valid, "anchor": 500_000, "available": 550_000},
    ]
    publications: list[tuple[str, dict[str, Any]]] = []
    kernel = SimpleNamespace(
        np=np,
        resource_guard=lambda _output: (True, {}),
        atomic_json=lambda path, value: publications.append((path.name, value)),
    )
    result = direct._assemble_fragments(
        tmp_path, directory, histories, jobs, kernel=kernel
    )
    assert result is not None
    features, history, anchors, available = result
    np.testing.assert_array_equal(features, np.stack((first[0], first[1], second[1])))
    np.testing.assert_array_equal(
        history,
        [[0, 1, -1, -1, -1, -1, -1, -1], [1, 2, -1, -1, -1, -1, -1, -1]],
    )
    np.testing.assert_array_equal(anchors, [50_000, 100_000, 200_000])
    np.testing.assert_array_equal(available, [450_000, 450_000, 550_000])
    assert publications[-1][1] == {"status": "RUNNING", "completed_rows": 2, "total_rows": 2}


def test_assembly_guards_every_256_rows_and_pauses_separately(tmp_path: Path) -> None:
    directory = tmp_path / "h8_feature_fragments"
    directory.mkdir()
    valid = np.ones(8, dtype=bool)
    histories: list[list[str | None]] = []
    jobs: list[dict[str, Any]] = []
    for row in range(257):
        _fragment(directory / f"query_{row:05d}.npz", np.full((8, 1), row, np.float32), valid)
        histories.append([f"{row}:{slot}" for slot in range(8)])
        jobs.append({"valid": valid, "anchor": row * 50_000, "available": row})
    guards = 0
    publications: list[tuple[str, dict[str, Any]]] = []

    def guard(_output: Path) -> tuple[bool, dict[str, Any]]:
        nonlocal guards
        guards += 1
        return guards == 1, {"reason": "bounded"}

    kernel = SimpleNamespace(
        np=np,
        resource_guard=guard,
        atomic_json=lambda path, value: publications.append((path.name, value)),
    )
    assert direct._assemble_fragments(
        tmp_path, directory, histories, jobs, kernel=kernel
    ) is None
    assert guards == 2
    assert publications[-1] == (
        "H8_ASSEMBLY_PROGRESS.json",
        {"status": "PAUSED", "completed_rows": 256, "total_rows": 257},
    )
    assert not any(name == "H8_FEATURE_PROGRESS.json" for name, _ in publications)


def test_duplicate_feature_mismatch_remains_fatal(tmp_path: Path) -> None:
    directory = tmp_path / "h8_feature_fragments"
    directory.mkdir()
    valid = np.ones(8, dtype=bool)
    _fragment(directory / "query_00000.npz", np.zeros((8, 1), np.float32), valid)
    _fragment(directory / "query_00001.npz", np.ones((8, 1), np.float32), valid)
    histories: list[list[str | None]] = [["same"] * 8, ["same"] * 8]
    jobs = [
        {"valid": valid, "anchor": 0, "available": 0},
        {"valid": valid, "anchor": 0, "available": 0},
    ]
    kernel = SimpleNamespace(
        np=np,
        resource_guard=lambda _output: (True, {}),
        atomic_json=lambda _path, _value: None,
    )
    with pytest.raises(ValueError, match="differing expert features"):
        direct._assemble_fragments(tmp_path, directory, histories, jobs, kernel=kernel)
