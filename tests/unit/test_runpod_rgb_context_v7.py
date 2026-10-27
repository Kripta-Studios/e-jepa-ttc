"""V7 caches only stable PASS admissions and invalidates every file identity change."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import runpod_rgb_context_v5 as v5
from scripts import runpod_rgb_context_v6 as v6
from scripts import runpod_rgb_context_v7 as v7
from scripts import runpod_rgb_recovery_v3 as v3
from scripts.runpod_rgb_context_v7 import AdmissionCache, canonical


def _case(tmp_path: Path):
    media = tmp_path / "media"
    tar_path = media / "data/train/a.tar"
    tar_path.parent.mkdir(parents=True)
    tar_path.write_bytes(b"tar")
    marker = tar_path.with_suffix(".tar.json")
    marker.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"tars": [{"relative": "data/train/a.tar"}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    scope = {
        "plan_path": str(plan),
        "manifest_path": str(manifest),
        "plan_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    config = {
        "media_root": str(media),
        "authority_sha256": "authority",
        "scope_map": {"R3_refiner": scope},
    }
    return config, tar_path, marker


def test_initial_full_coverage_then_stat_only_cache_hit(tmp_path: Path) -> None:
    config, _, _ = _case(tmp_path)
    calls = 0

    def verify(_config, rotation, role):
        nonlocal calls
        calls += 1
        return {"status": "PASS", "rotation": rotation, "role": role}

    cache = AdmissionCache(verify)
    first = cache.verify(config, 3, "refiner")
    second = cache.verify(config, 3, "refiner")
    assert first is not None and first["status"] == "PASS"
    assert second is not None and second["status"] == "PASS"
    assert calls == 1


def test_replaced_tar_inode_invalidates_cached_admission(tmp_path: Path) -> None:
    config, tar_path, _ = _case(tmp_path)
    cache = AdmissionCache(lambda *_args: {"status": "PASS"})
    cache.verify(config, 3, "refiner")
    replacement = tar_path.with_name("replacement.tar")
    replacement.write_bytes(b"tar")
    os.replace(replacement, tar_path)
    with pytest.raises(ValueError, match="stat barrier changed"):
        cache.verify(config, 3, "refiner")


def test_mutated_marker_invalidates_cached_admission(tmp_path: Path) -> None:
    config, _, marker = _case(tmp_path)
    cache = AdmissionCache(lambda *_args: {"status": "PASS"})
    cache.verify(config, 3, "refiner")
    marker.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="stat barrier changed"):
        cache.verify(config, 3, "refiner")


def test_scope_member_role_or_authority_mutation_invalidates(tmp_path: Path) -> None:
    config, _, _ = _case(tmp_path)
    cache = AdmissionCache(lambda *_args: {"status": "PASS"})
    cache.verify(config, 3, "refiner")
    config["scope_map"]["R3_refiner"]["roles"] = ["generation"]
    with pytest.raises(ValueError, match="authority/scope changed"):
        cache.verify(config, 3, "refiner")
    assert canonical(config["scope_map"]["R3_refiner"]) != cache.entries[(3, "refiner")][
        "scope_evidence_sha256"
    ]


def test_none_is_never_cached_and_gains_no_false_benefit(tmp_path: Path) -> None:
    config, _, _ = _case(tmp_path)
    calls = 0

    def waiting(*_args):
        nonlocal calls
        calls += 1
        return None

    cache = AdmissionCache(waiting)
    assert cache.verify(config, 3, "refiner") is None
    assert cache.verify(config, 3, "refiner") is None
    assert calls == 2
    assert cache.entries == {}


def test_real_v3_v5_v6_v7_composition_executes_dummy_queue_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_materializer = object()
    replacement = object()
    materialize = ModuleType("operational.final_refit.materialize")
    features = ModuleType("operational.final_refit.scientific_oof_features")
    materialize.materialize_rgb_observations = original_materializer  # type: ignore[attr-defined]
    features.materialize_rgb_observations = original_materializer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, materialize.__name__, materialize)
    monkeypatch.setitem(sys.modules, features.__name__, features)
    import operational.final_refit as package

    monkeypatch.setattr(package, "materialize", materialize, raising=False)
    monkeypatch.setattr(package, "scientific_oof_features", features, raising=False)
    monkeypatch.setattr(
        v5,
        "load_repair",
        lambda _path: SimpleNamespace(patched_materializer=lambda: replacement),
    )
    routed: list[tuple[Path, int, Path]] = []

    @contextmanager
    def fake_routed(run: Path, rotation: int, config: Path):
        routed.append((run, rotation, config))
        yield

    monkeypatch.setattr(v6, "routed_runtime", fake_routed)
    queue_calls: list[bool] = []
    receipts: list[tuple[tuple, dict]] = []

    def queue(run, rotation, external_guard):
        queue_calls.append(True)
        assert materialize.materialize_rgb_observations is replacement
        external_guard()
        return {"status": "COMPLETE"}

    module = SimpleNamespace(
        run_rgb_downstream=queue,
        _receipt=lambda *args, **kwargs: receipts.append((args, kwargs)),
    )
    original_v3 = v3.install_queue_heartbeat
    original_v5 = v5.install_queue_heartbeat
    media_config, repair_path = tmp_path / "media.json", tmp_path / "repair.py"

    with v7.compatible_v6_composed_v5_hook(v6, v5, media_config):
        with v5.composed_v3_hook(v3, repair_path):
            v3.install_queue_heartbeat(module, clock=lambda: 0.0)
        result = module.run_rgb_downstream(
            tmp_path, 3, external_guard=lambda: None
        )

    assert result == {"status": "COMPLETE"}
    assert queue_calls == [True]
    assert len(receipts) == 1
    assert routed == [(tmp_path, 3, media_config)]
    assert materialize.materialize_rgb_observations is original_materializer
    assert v3.install_queue_heartbeat is original_v3
    assert v5.install_queue_heartbeat is original_v5
