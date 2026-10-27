from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from operational.final_refit import materialize, scientific_oof_features
from operational.oof_rgb_metadata_fix import (
    metadata_equal,
    observation_metadata_equal,
    patched_materializer,
    repaired_rgb_metadata,
)


def test_object_arrays_are_compared_recursively():
    left = np.empty(2, dtype=object)
    left[:] = [np.array([1, 2]), np.array([3, 4])]
    assert metadata_equal(left, deepcopy(left))
    right = deepcopy(left)
    right[1][0] = 9
    assert not metadata_equal(left, right)


def test_reused_sensor_observation_keeps_distinct_query_references():
    left = {
        "anchor_us": 100,
        "roi": np.array([1, 2, 3, 4]),
        "sample_token": "q1",
        "query_time_us": 110,
        "query_age_us": 10,
    }
    right = {**deepcopy(left), "sample_token": "q2", "query_time_us": 120, "query_age_us": 20}
    assert observation_metadata_equal(left, right)
    assert right["query_time_us"] == 120
    assert left["query_time_us"] == 110
    assert not observation_metadata_equal(left, {**right, "anchor_us": 101})


def test_nested_arrays_equal_and_inputs_unchanged():
    left = {"roi": np.array([1.0, 2.0, 3.0, 4.0]), "members": [{"t": np.int64(4)}]}
    right = deepcopy(left)
    assert metadata_equal(left, right)
    np.testing.assert_array_equal(left["roi"], [1, 2, 3, 4])


@pytest.mark.parametrize(
    "right",
    [
        np.array([1.0, 2.0, 3.0, 5.0]),
        np.array([[1.0, 2.0, 3.0, 4.0]]),
        np.array([1, 2, 3, 4]),
        np.array([1.0, 2.0, 3.0, np.nan]),
    ],
)
def test_array_conflicts_rejected(right):
    assert not metadata_equal(np.array([1.0, 2.0, 3.0, 4.0]), right)


@pytest.mark.parametrize("key,value", [("group", "other"), ("time", 5), ("extra", True)])
def test_scalar_and_missing_key_conflicts_rejected(key, value):
    old = {"group": "g", "time": 4}
    new = {**old, key: value}
    assert not metadata_equal(old, new)


class ReachedStoreError(Exception):
    pass


def exercise(monkeypatch, tmp_path, function_factory, *, arrays=True, conflict=False):
    item = {
        "observation_id": "same",
        "roi": np.array([1.0, 2.0, 3.0, 4.0]) if arrays else [1, 2, 3, 4],
    }
    source = SimpleNamespace(
        population_size=2,
        row=lambda index: {"group_id": "g", "historical_role": "P"},
        history=lambda index: [
            {
                **deepcopy(item),
                **({"roi": np.array([0.0, 2.0, 3.0, 4.0])} if conflict and index else {}),
            }
        ],
    )
    captured = []

    def store(path, binding):
        captured.append(binding)
        raise ReachedStoreError

    monkeypatch.setattr(materialize, "FragmentStore", store)
    function = function_factory()
    function(
        source,
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        tmp_path / "o.npz",
        tmp_path / "h.npz",
        parents={},
    )
    return captured


def test_real_materializer_reproduces_failure(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="truth value of an array"):
        exercise(monkeypatch, tmp_path, lambda: materialize.materialize_rgb_observations)


def test_repair_accepts_duplicate_arrays(monkeypatch, tmp_path):
    with pytest.raises(ReachedStoreError):
        exercise(monkeypatch, tmp_path, patched_materializer)


def test_repair_still_rejects_conflicting_observations(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="conflicting group or sensor metadata"):
        exercise(monkeypatch, tmp_path, patched_materializer, conflict=True)


@pytest.mark.parametrize("patched", [False, True])
def test_original_supported_lists_preserved(monkeypatch, tmp_path, patched):
    with pytest.raises(ReachedStoreError):
        exercise(
            monkeypatch,
            tmp_path,
            patched_materializer if patched else lambda: materialize.materialize_rgb_observations,
            arrays=False,
        )


def test_overlay_restores_aliases_after_failure():
    old = materialize.materialize_rgb_observations
    alias = scientific_oof_features.materialize_rgb_observations
    with pytest.raises(RuntimeError), repaired_rgb_metadata():
        assert (
            materialize.materialize_rgb_observations
            is scientific_oof_features.materialize_rgb_observations
        )
        assert materialize.materialize_rgb_observations is not old
        raise RuntimeError("test")
    assert materialize.materialize_rgb_observations is old
    assert scientific_oof_features.materialize_rgb_observations is alias


def test_materialization_keeps_query_clocks_and_resumes(monkeypatch, tmp_path):
    """One sensor observation feeds two queries without merging their clocks."""
    calls = []
    source = SimpleNamespace(
        population_size=2,
        datasets={"P": SimpleNamespace(eap_root=tmp_path)},
        row=lambda i: {
            "group_id": "g",
            "historical_role": "P",
            "sample_token": f"q{i}",
            "query_time_us": 300000 + i * 100000,
        },
        history=lambda i: [
            {
                "observation_id": "shared",
                "anchor_us": 200000,
                "producer_available_us": np.array([100000, 200000]),
                "producer_sensor_times_us": np.array([100000, 200000]),
                "producer_members": ["a", "b"],
                "roi": np.array([0, 0, 4, 4]),
                "sample_token": f"q{i}",
                "query_time_us": 300000 + i * 100000,
                "query_age_us": 100000 + i * 100000,
            }
        ],
    )
    monkeypatch.setattr(
        materialize, "TarFrameReader", lambda _: SimpleNamespace(close=lambda: None)
    )

    def decode(item, **kwargs):
        calls.append(item["observation_id"])
        return {"rgb": np.ones((2, 3, 4, 4), np.float32), "delta_t_s": np.array([0.1], np.float32)}

    monkeypatch.setattr(materialize, "decode_query", decode)

    def features(model, sensor, delta):
        batch = len(sensor)
        return {
            "support": torch.ones(batch, 2),
            "token128": torch.ones(batch, 128),
            **{
                name: torch.ones(batch) * 0.1
                for name in ("flow", "margin", "log_variance", "point_phase")
            },
        }

    monkeypatch.setattr(materialize, "producer_features", features)
    monkeypatch.setattr(
        materialize, "pair_point_phase", lambda model, features: torch.ones(len(features)) * 0.1
    )
    fn = patched_materializer()
    args = (
        source,
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        tmp_path / "obs.npz",
        tmp_path / "hist.npz",
    )
    receipt = fn(*args, parents={"R_A5": "a" * 64}, include_targets=False)
    assert calls == ["shared"]
    with np.load(tmp_path / "hist.npz") as hist:
        assert hist["sample_token"].tolist() == ["q0", "q1"]
        assert "target_phase" not in hist.files
        for i in range(2):
            np.testing.assert_array_equal(
                hist["timing"][i, -1:],
                materialize.clock_features([200000], [200000], 300000 + i * 100000),
            )
        assert not np.array_equal(hist["timing"][0], hist["timing"][1])
    assert fn(*args, parents={"R_A5": "a" * 64}, include_targets=False) == receipt
    assert calls == ["shared"]


def test_unicode_cache_resume_preserves_original_receipt(tmp_path):
    path = tmp_path / "Álvaro" / "cache.npz"
    path.parent.mkdir()
    values = {
        "features": np.ones((2, 2), np.float32),
        "sample_token": np.array(["q1", "q2"]),
        "group_id": np.array(["g", "g"]),
        "role": np.array(["refiner", "refiner"]),
    }
    options = dict(
        schema="final_refit_context_cache_v1",
        role="refiner",
        parents={"E_A5": "a" * 64},
        rows_sha="b" * 64,
    )
    first = scientific_oof_features.seal_role_cache(path, values, **options)
    before = path.with_suffix(".json").read_bytes()
    with pytest.raises(ValueError, match="receipt or ancestry"):
        scientific_oof_features.seal_role_cache(path, values, **options)
    with repaired_rgb_metadata():
        assert scientific_oof_features.seal_role_cache(path, values, **options) == first
        with pytest.raises(ValueError, match="receipt or ancestry"):
            scientific_oof_features.seal_role_cache(
                path, values, **{**options, "parents": {"E_A5": "c" * 64}}
            )
    assert path.with_suffix(".json").read_bytes() == before
