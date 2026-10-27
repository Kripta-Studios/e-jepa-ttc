from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import runpod_event_resource_watchdog_v7 as v7


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _store(tmp_path: Path, parts: int = 1) -> tuple[SimpleNamespace, dict, Path]:
    run = tmp_path / "run"
    root = run / "rotations/2/caches/PAIR_E.parts"
    root.mkdir(parents=True)
    binding = {
        "schema": "scientific_fold_pair133_v1",
        "parents": {"E_A5": "a" * 64},
        "rows_sha256": "b" * 64,
        "precision": "float32",
    }
    (root / "BINDING.json").write_text(json.dumps(binding), encoding="utf-8")
    module = SimpleNamespace(HEAD="EVENT_DOWNSTREAM", read=_read)
    task = {"fit": module.HEAD, "run": str(run), "rotation": 2}
    for index in range(parts):
        _part(root, index, v7.canonical(binding))
    return module, task, root


def _part(root: Path, index: int, binding_sha: str) -> None:
    path = root / f"part_{index:06d}.npz"
    np.savez_compressed(path, features=np.full((32, 133), index, dtype=np.float32))
    receipt = {
        "binding_sha256": binding_sha,
        "sha256": v7.digest(path),
        "rows": 32,
        "optimizer_updates": 0,
    }
    path.with_suffix(".json").write_text(json.dumps(receipt), encoding="utf-8")


def test_fragment_anchor_advances_only_on_new_verified_part(tmp_path: Path) -> None:
    module, task, root = _store(tmp_path)
    first = v7.durable_pair_fragments(module, task, None)
    assert first is not None and first["parts"] == 1
    assert v7.durable_pair_fragments(module, task, first) is None
    binding_sha = v7.canonical(_read(root / "BINDING.json"))
    _part(root, 1, binding_sha)
    second = v7.durable_pair_fragments(module, task, first)
    assert second is not None and second["parts"] == 2 and second["rows"] == 64


def test_fragment_corruption_never_authorizes_reset(tmp_path: Path) -> None:
    module, task, root = _store(tmp_path)
    anchor = v7.durable_pair_fragments(module, task, None)
    assert anchor is not None
    (root / "part_000000.npz").write_bytes(b"corrupt")
    assert v7.durable_pair_fragments(module, task, anchor) is None


def test_fragment_retrogression_never_authorizes_reset(tmp_path: Path) -> None:
    module, task, root = _store(tmp_path, parts=2)
    anchor = v7.durable_pair_fragments(module, task, None)
    assert anchor is not None
    (root / "part_000001.npz").unlink()
    (root / "part_000001.json").unlink()
    assert v7.durable_pair_fragments(module, task, anchor) is None


def test_missing_binding_never_authorizes_reset(tmp_path: Path) -> None:
    module, task, root = _store(tmp_path)
    (root / "BINDING.json").unlink()
    assert v7.durable_pair_fragments(module, task, None) is None


def test_retry_reset_initializes_without_reset_then_requires_progress(tmp_path: Path) -> None:
    installed = v7.install_v7()
    _, task, root = _store(tmp_path)
    current = {"retries": 5}
    installed.Watchdog._observe_downstream_fragment_retry_boundary(task, current)
    assert current["retries"] == 5
    assert current["retry_anchor_pair_fragments"]["parts"] == 1
    installed.Watchdog._observe_downstream_fragment_retry_boundary(task, current)
    assert current["retries"] == 5
    _part(root, 1, v7.canonical(_read(root / "BINDING.json")))
    installed.Watchdog._observe_downstream_fragment_retry_boundary(task, current)
    assert current["retries"] == 0
    assert current["retry_reset_previous_retries"] == 5
