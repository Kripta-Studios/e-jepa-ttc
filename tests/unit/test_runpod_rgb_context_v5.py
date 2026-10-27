"""RGB V5 scopes the exact metadata repair without changing V3 behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from operational import oof_rgb_metadata_fix as fix
from scripts import runpod_rgb_context_v5 as v5


def _fake_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, ModuleType, object]:
    original = object()
    materialize = ModuleType("operational.final_refit.materialize")
    features = ModuleType("operational.final_refit.scientific_oof_features")
    materialize.materialize_rgb_observations = original  # type: ignore[attr-defined]
    features.materialize_rgb_observations = original  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, materialize.__name__, materialize)
    monkeypatch.setitem(sys.modules, features.__name__, features)
    import operational.final_refit as package

    monkeypatch.setattr(package, "materialize", materialize, raising=False)
    monkeypatch.setattr(package, "scientific_oof_features", features, raising=False)
    return materialize, features, original


def test_audited_comparator_preserves_array_shape_dtype_and_nested_exactness() -> None:
    left = {
        "sensor": np.array([[1, 2]], dtype=np.int16),
        "nested": {"roi": [1, (2, 3)]},
    }
    same = {
        "sensor": np.array([[1, 2]], dtype=np.int16),
        "nested": {"roi": [1, (2, 3)]},
    }
    wrong_dtype = {**same, "sensor": same["sensor"].astype(np.int32)}
    wrong_shape = {**same, "sensor": same["sensor"].reshape(2, 1)}
    assert fix.metadata_equal(left, same)
    assert not fix.metadata_equal(left, wrong_dtype)
    assert not fix.metadata_equal(left, wrong_shape)


def test_only_three_query_fields_may_differ() -> None:
    base = {
        "sample_token": "old",
        "query_time_us": 100,
        "query_age_us": 0,
        "sensor": "left",
        "roi": np.array([1, 2], dtype=np.int64),
        "group_id": "g1",
    }
    query_changed = {
        **base,
        "sample_token": "new",
        "query_time_us": 200,
        "query_age_us": 100,
    }
    assert fix.observation_metadata_equal(base, query_changed)
    assert not fix.observation_metadata_equal(base, {**query_changed, "group_id": "g2"})
    assert not fix.observation_metadata_equal(
        base, {**query_changed, "roi": np.array([1, 3], dtype=np.int64)}
    )


def test_patch_scopes_both_aliases_and_restores_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize, scientific_oof_features, original = _fake_consumers(monkeypatch)
    replacement = object()
    repair = SimpleNamespace(patched_materializer=lambda: replacement)
    with pytest.raises(RuntimeError, match="inside queue"):
        with v5.patched_materializer_context(repair):
            assert materialize.materialize_rgb_observations is replacement
            assert scientific_oof_features.materialize_rgb_observations is replacement
            raise RuntimeError("inside queue")
    assert materialize.materialize_rgb_observations is original
    assert scientific_oof_features.materialize_rgb_observations is original


def test_nested_patch_context_restores_each_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize, scientific_oof_features, original = _fake_consumers(monkeypatch)
    outer, inner = object(), object()
    with v5.patched_materializer_context(
        SimpleNamespace(patched_materializer=lambda: outer)
    ):
        assert materialize.materialize_rgb_observations is outer
        with v5.patched_materializer_context(
            SimpleNamespace(patched_materializer=lambda: inner)
        ):
            assert materialize.materialize_rgb_observations is inner
            assert scientific_oof_features.materialize_rgb_observations is inner
        assert materialize.materialize_rgb_observations is outer
    assert materialize.materialize_rgb_observations is original


def test_v3_heartbeat_clock_guard_frequency_and_result_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    materialize, _, original = _fake_consumers(monkeypatch)
    times = iter((0.0, 10.0, 31.0))
    receipts: list[tuple[tuple, dict]] = []
    guards: list[bool] = []
    replacement = object()

    def queue(run, rotation, external_guard):
        assert materialize.materialize_rgb_observations is replacement
        external_guard()
        external_guard()
        return {"stage": "R_CTX", "status": "COMPLETE"}

    module = SimpleNamespace(
        run_rgb_downstream=queue,
        _receipt=lambda *args, **kwargs: receipts.append((args, kwargs)),
    )
    repair = SimpleNamespace(patched_materializer=lambda: replacement)
    monkeypatch.setattr(v5, "load_repair", lambda _path: repair)
    v5.install_queue_heartbeat(
        module, clock=lambda: next(times), repair_path=tmp_path / "repair.py"
    )
    result = module.run_rgb_downstream(
        tmp_path, 3, external_guard=lambda: guards.append(True)
    )
    assert materialize.materialize_rgb_observations is original
    assert len(receipts) == 2
    assert len(guards) == 3
    assert result == {"stage": "R_CTX", "status": "COMPLETE"}


def test_main_hook_composition_calls_captured_v3_once_and_restores_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    materialize, _, original_materializer = _fake_consumers(monkeypatch)
    replacement = object()
    calls: list[object] = []

    def original_v3_hook(module, clock):
        calls.append(clock)

    v3 = SimpleNamespace(install_queue_heartbeat=original_v3_hook)
    module = SimpleNamespace(
        run_rgb_downstream=lambda: materialize.materialize_rgb_observations,
    )
    repair = SimpleNamespace(patched_materializer=lambda: replacement)
    monkeypatch.setattr(v5, "load_repair", lambda _path: repair)

    def clock() -> float:
        return 12.0

    with v5.composed_v3_hook(v3, tmp_path / "repair.py"):
        installed = v3.install_queue_heartbeat
        assert installed is not original_v3_hook
        installed(module, clock=clock)
        assert module.run_rgb_downstream() is replacement
        assert materialize.materialize_rgb_observations is original_materializer

    assert calls == [clock]
    assert v3.install_queue_heartbeat is original_v3_hook
    assert materialize.materialize_rgb_observations is original_materializer


def test_v5_sidecar_binds_all_interfaces_without_pixel_or_recipe_claims(
    tmp_path: Path,
) -> None:
    source, v3, repair, audit = (
        tmp_path / "v5.py",
        tmp_path / "v3.py",
        tmp_path / "repair.py",
        tmp_path / "audit.json",
    )
    for path, value in ((source, b"v5"), (v3, b"v3"), (repair, b"repair"), (audit, b"audit")):
        path.write_bytes(value)
    (tmp_path / "MIGRATION_BINDING.json").write_bytes(b"migration")
    v5.write_sidecar(tmp_path, source, v3, repair, audit)
    value = json.loads((tmp_path / "RGB_OPERATIONAL_RECOVERY_V5.json").read_text())
    assert value["source_sha256"] == v5.digest(source)
    assert value["metadata_repair_sha256"] == v5.digest(repair)
    assert value["excluded_query_reference_fields"] == v5.QUERY_FIELDS
    assert value["rgb_pixels_changed"] is False
    assert value["scientific_recipes_changed"] is False
