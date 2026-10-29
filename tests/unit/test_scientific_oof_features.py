import json

import numpy as np
import pandas as pd
import pytest
import torch

import operational.final_refit.scientific_oof_features as feature_module
from operational.final_refit.recipe import canonical_sha256
from operational.final_refit.scientific_oof_features import (
    _seal_raw_phase17,
    arrays,
    normalize_histories,
    physical_phase20_to_state340,
    rgb_raw,
    seal_role_cache,
    unique_normalizer,
)
from operational.final_refit.training import PhaseOnlyFinal


def test_physical_adapter_matches_original_head_values_and_gradients():
    torch.manual_seed(7)
    model = PhaseOnlyFinal()
    original = torch.randn(9, 340)
    physical = torch.cat((original[:, 160:170], original[:, 330:340]), -1).numpy()
    adapted = torch.from_numpy(physical_phase20_to_state340(physical))
    widths = torch.randn(9, 2)
    a, b = model(original, widths), model(adapted, widths)
    for key in a:
        assert torch.equal(a[key], b[key])
    ga = torch.autograd.grad(
        a["raw_location"].sum(), tuple(model.parameters()), retain_graph=True, allow_unused=True
    )
    gb = torch.autograd.grad(b["raw_location"].sum(), tuple(model.parameters()), allow_unused=True)
    for x, y in zip(ga, gb, strict=True):
        assert (x is None and y is None) or torch.equal(x, y)
    assert torch.count_nonzero(adapted[:, :160]) == 0
    assert torch.count_nonzero(adapted[:, 170:330]) == 0


def raw():
    features = np.zeros((2, 2, 17), dtype=np.float32)
    features[0, 1] = 1
    features[1] = np.stack((np.ones(17), np.full(17, 3)))
    return {
        "features": features,
        "valid": np.array([[False, True], [True, True]]),
        "observation_id": np.array([["", "a"], ["a", "b"]]),
        "group_id": np.array(["training", "training"]),
        "target_phase": np.array([99.0, -88.0]),
    }


def test_normalizer_deduplicates_and_does_not_use_targets():
    value = raw()
    n = unique_normalizer(value, expected_groups={"training"})
    assert n["unique_observations"] == 2 and n["mean"] == [2.0] * 17
    value["target_phase"][:] = np.nan
    assert unique_normalizer(value, expected_groups={"training"}) == n
    normalized = normalize_histories(value, np.asarray(n["mean"]), np.asarray(n["scale"]))
    assert np.count_nonzero(normalized[0, 0]) == 0
    assert np.array_equal(normalized[0, 1], np.full(17, -1.0))


def test_generation_group_cannot_enter_refiner_normalizer():
    value = raw()
    value["group_id"][1] = "heldout"
    with pytest.raises(ValueError, match="different group"):
        unique_normalizer(value, expected_groups={"training"})


def test_reused_observation_id_with_different_feature_is_rejected():
    value = raw()
    value["features"][1, 0, 4] = 10
    with pytest.raises(ValueError, match="conflicting"):
        unique_normalizer(value, expected_groups={"training"})


def test_normalizer_can_bind_exact_fold_role_identity():
    value = raw()
    value["sample_token"] = np.asarray(["q0", "q1"])
    value["source_ordinal"] = np.asarray([4, 9], dtype=np.int64)
    identity = {
        "population": 2,
        "group_ids": ["training"],
        "sample_token_sha256": canonical_sha256(["q0", "q1"]),
        "ordered_native_ordinal_sha256": canonical_sha256(["4", "9"]),
    }
    normalizer = unique_normalizer(
        value,
        expected_groups={"training"},
        expected_role_identity=identity,
    )
    assert normalizer["role_identity"] == identity
    changed = dict(identity)
    changed["sample_token_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact fold role"):
        unique_normalizer(
            value,
            expected_groups={"training"},
            expected_role_identity=changed,
        )


def _role_values() -> dict[str, np.ndarray]:
    return {
        "features": np.asarray([[1.0], [2.0]], dtype=np.float32),
        "sample_token": np.asarray(["q0", "q1"]),
        "group_id": np.asarray(["g0", "g1"]),
        "role": np.asarray(["refiner", "refiner"]),
    }


def test_role_cache_replay_rejects_stale_arrays_and_repairs_only_exact_orphan(tmp_path):
    path = tmp_path / "cache.npz"
    parents = {"parent": "a" * 64}
    values = _role_values()
    receipt = seal_role_cache(
        path,
        values,
        schema="test_role_cache_v1",
        role="refiner",
        parents=parents,
        rows_sha="b" * 64,
        details={"normalizer_fit_role": "refiner"},
        normalizers={"phase": "c" * 64},
    )
    assert receipt["native_train40_rows_sha256"] == "b" * 64
    path.with_suffix(".json").unlink()
    repaired = seal_role_cache(
        path,
        values,
        schema="test_role_cache_v1",
        role="refiner",
        parents=parents,
        rows_sha="b" * 64,
        details={"normalizer_fit_role": "refiner"},
        normalizers={"phase": "c" * 64},
    )
    assert repaired == json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    changed = _role_values()
    changed["features"] = changed["features"].copy()
    changed["features"][0, 0] = 99
    with pytest.raises(ValueError, match="arrays changed"):
        seal_role_cache(
            path,
            changed,
            schema="test_role_cache_v1",
            role="refiner",
            parents=parents,
            rows_sha="b" * 64,
        )


def test_raw_phase17_replay_binds_rows_parents_and_exact_arrays(tmp_path):
    rows = pd.DataFrame(
        {"sample_token": ["q0", "q1"], "group_id": ["g0", "g1"], "source_ordinal": [2, 7]}
    )
    values = {
        "features": np.zeros((2, 1, 17), dtype=np.float32),
        "valid": np.ones((2, 1), dtype=np.bool_),
        "timing": np.zeros((2, 1, 4), dtype=np.float32),
        "observation_id": np.asarray([["o0"], ["o1"]]),
        "expert_phase": np.zeros((2, 3), dtype=np.float32),
        "sample_token": np.asarray(["q0", "q1"]),
        "group_id": np.asarray(["g0", "g1"]),
        "source_ordinal": np.asarray([2, 7], dtype=np.int64),
    }
    path = tmp_path / "raw_phase17.npz"
    parents = {"R_A5": "d" * 64}
    _seal_raw_phase17(
        path,
        values,
        rows,
        modality="rgb",
        parents=parents,
        provenance={
            "history_arrays_sha256": "e" * 64,
            "observations_arrays_sha256": "f" * 64,
        },
    )
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt["schema"] == "scientific_oof_raw_phase17_v1"
    assert receipt["targets_read"] is True
    assert receipt["targets_used_as_features"] is False
    assert arrays(path)["source_ordinal"].tolist() == [2, 7]
    path.with_suffix(".json").unlink()
    _seal_raw_phase17(
        path,
        values,
        rows,
        modality="rgb",
        parents=parents,
        provenance={
            "history_arrays_sha256": "e" * 64,
            "observations_arrays_sha256": "f" * 64,
        },
    )
    stale = dict(values)
    stale["features"] = stale["features"].copy()
    stale["features"][1, 0, 3] = 1
    with pytest.raises(ValueError, match="artifact changed"):
        _seal_raw_phase17(
            path,
            stale,
            rows,
            modality="rgb",
            parents=parents,
            provenance={
                "history_arrays_sha256": "e" * 64,
                "observations_arrays_sha256": "f" * 64,
            },
        )
    with pytest.raises(ValueError, match="receipt or ancestry"):
        _seal_raw_phase17(
            path,
            values,
            rows,
            modality="rgb",
            parents={"R_A5": "f" * 64},
            provenance={
                "history_arrays_sha256": "e" * 64,
                "observations_arrays_sha256": "f" * 64,
            },
        )


def test_rgb_raw_fast_reuse_verifies_receipt_and_orphan_replay(tmp_path, monkeypatch):
    rows = pd.DataFrame({"sample_token": ["q0"], "group_id": ["g0"], "source_ordinal": [0]})

    class Source:
        datasets = {}

        @staticmethod
        def row(index):
            assert index == 0
            return {"sample_token": "q0", "group_id": "g0", "query_time_us": 100}

        @staticmethod
        def history(index):
            assert index == 0
            return []

    generated = np.ones((1, 17), dtype=np.float32)
    workspaces = []

    def materialize(_source, _a5, _c2f, _pair, observations, history, **_kwargs):
        workspaces.append(observations.parent)
        np.savez_compressed(
            observations,
            observation_id=np.asarray(["o0"]),
            group_id=np.asarray(["g0"]),
            anchor_us=np.asarray([100], dtype=np.int64),
            available_us=np.asarray([100], dtype=np.int64),
            input_span_us=np.asarray([0], dtype=np.int64),
            features=generated,
        )
        np.savez_compressed(
            history,
            observation_id=np.asarray([["o0"]]),
            timing=np.zeros((1, 1, 4), dtype=np.float32),
            valid=np.ones((1, 1), dtype=np.bool_),
            expert_phase=np.zeros((1, 3), dtype=np.float32),
            sample_token=np.asarray(["q0"]),
            group_id=np.asarray(["g0"]),
        )

    monkeypatch.setattr(feature_module, "materialize_rgb_observations", materialize)
    models = {"R_A5": object(), "R_C2F": object(), "PAIR_R": object()}
    parents = {"R_A5": "a" * 64}
    first = rgb_raw(
        Source(), rows, models, tmp_path, device="cpu", parents=parents, guard=lambda: None
    )
    second = rgb_raw(
        Source(), rows, models, tmp_path, device="cpu", parents=parents, guard=lambda: None
    )
    assert first == second == tmp_path / "raw_phase17.npz"
    assert workspaces == [tmp_path]
    receipt_path = first.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["parents"]["R_A5"] = "b" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt, ancestry, or physical hash"):
        rgb_raw(Source(), rows, models, tmp_path, device="cpu", parents=parents, guard=lambda: None)
    receipt["parents"]["R_A5"] = "a" * 64
    receipt["receipt_payload_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    valid_receipt = dict(receipt)
    receipt["output_sha256"] = "c" * 64
    receipt["receipt_payload_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt, ancestry, or physical hash"):
        rgb_raw(Source(), rows, models, tmp_path, device="cpu", parents=parents, guard=lambda: None)
    receipt_path.write_text(json.dumps(valid_receipt), encoding="utf-8")
    changed_rows = rows.assign(sample_token=["other"])
    with pytest.raises(ValueError, match="exact role rows or layout"):
        rgb_raw(
            Source(),
            changed_rows,
            models,
            tmp_path,
            device="cpu",
            parents=parents,
            guard=lambda: None,
        )
    receipt_path.unlink()
    generated[0, 0] = 2
    with pytest.raises(ValueError, match="artifact changed"):
        rgb_raw(
            Source(),
            rows,
            models,
            tmp_path,
            device="cpu",
            parents=parents,
            guard=lambda: None,
        )
    assert workspaces[-1] != tmp_path
