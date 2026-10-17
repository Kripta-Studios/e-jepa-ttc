from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from operational.decision_recovery import sign_probe


def _features(rows: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, sign_probe.FEATURE_COUNT))
    label = (values[:, 0] - 0.4 * values[:, 1] > 0).astype(np.int64)
    return values, label


def test_feature_schema_is_the_native_194_contract() -> None:
    assert sign_probe.FEATURE_COUNT == 194
    assert len(sign_probe.FEATURE_NAMES) == 194
    assert sign_probe.FEATURE_NAMES[:2] == ("spatial_mean_000", "spatial_mean_001")
    assert sign_probe.FEATURE_NAMES[95:98] == (
        "spatial_mean_095",
        "spatial_max_000",
        "spatial_max_001",
    )


def test_frozen_contract_excludes_targets_and_updates() -> None:
    contract = sign_probe._contracts()
    assert contract["features"]["targets_in_cache"] is False
    assert contract["extract"]["batch"] == 4
    assert contract["extract"]["gpu_scope_seconds"] == 900
    assert contract["extract"]["parity"]["exact_expected"] is True
    assert contract["metrics"]["bootstrap_unit"] == "acquisition_group"
    assert contract["flags"] == {
        "system_promotion": False,
        "official_confirmation": False,
        "SOTA_claim": False,
        "encoder_optimizer_updates": 0,
    }


def test_extraction_applies_native_runtime_setup_before_model_load(monkeypatch) -> None:
    from operational.joint_measurement import train

    seeds = []
    monkeypatch.setattr(train, "setup", seeds.append)
    sign_probe._setup_extraction_runtime()
    assert seeds == [sign_probe.SEED]
    source = inspect.getsource(sign_probe._extract_registered)
    assert source.index("_setup_extraction_runtime()") < source.index("_load_model(record).cuda()")


def test_ttc_replay_uses_the_same_cuda_backend_as_extraction() -> None:
    assert sign_probe.DIRECTORY.name == "sign_probe_v2"
    replay_source = inspect.getsource(sign_probe._phase_to_ttc_on_extraction_device)
    assert "torch.from_numpy(raw_phase).cuda()" in replay_source
    assert ").cpu().numpy().copy()" in replay_source
    extraction_source = inspect.getsource(sign_probe._extract_registered)
    assert "_phase_to_ttc_on_extraction_device(raw_array)" in extraction_source
    assert "phase_to_ttc(torch.from_numpy(raw_array)).numpy()" not in extraction_source


def test_parity_allows_only_bounded_sign_preserving_drift() -> None:
    expected_raw = np.array([-0.2, 0.3], dtype=np.float32)
    expected_ttc = np.array([-5.0, 4.0], dtype=np.float32)
    exact = sign_probe._parity(expected_raw, expected_ttc, expected_raw, expected_ttc)
    assert exact["status"] == "EXACT"

    drifted_raw = expected_raw + np.array([1e-6, -1e-6], dtype=np.float32)
    drifted_ttc = expected_ttc + np.array([1e-6, -1e-6], dtype=np.float32)
    tolerated = sign_probe._parity(drifted_raw, drifted_ttc, expected_raw, expected_ttc)
    assert tolerated["status"] == "TOLERATED_SIGN_PRESERVING"
    sign_probe._validate_parity(tolerated)

    with pytest.raises(ValueError, match="historical FULL"):
        sign_probe._parity(
            np.array([0.0, 0.3], dtype=np.float32),
            expected_ttc,
            expected_raw,
            expected_ttc,
        )


def test_chunk_replay_rejects_target_fields_and_reordered_tokens(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sign_probe.common, "RUN", tmp_path)
    path = tmp_path / "chunk_00000.npz"
    tokens = np.array(["first", "second"])
    features = np.zeros((2, sign_probe.FEATURE_COUNT), dtype=np.float32)
    raw = np.array([-0.2, 0.3], dtype=np.float32)
    prediction = np.array([-5.0, 4.0], dtype=np.float32)
    parity = sign_probe._parity(raw, prediction, raw, prediction)
    sign_probe._write_chunk(
        path,
        binding="binding",
        begin=0,
        stop=2,
        tokens=tokens,
        features=features,
        raw_phase=raw,
        prediction_ttc=prediction,
        parity=parity,
    )

    np.savez_compressed(
        path,
        sample_token=tokens,
        features=features,
        raw_phase=raw,
        prediction_ttc=prediction,
        target_ttc=np.ones(2, dtype=np.float32),
    )
    receipt = sign_probe.common.read_json(path.with_suffix(".json"))
    receipt["sha256"] = sign_probe.common.sha(path)
    sign_probe.common.write_json(path.with_suffix(".json"), receipt)
    with pytest.raises(ValueError, match="Unexpected"):
        sign_probe._read_chunk(path, binding="binding", begin=0, stop=2, tokens=tokens)

    sign_probe._write_chunk(
        path,
        binding="binding",
        begin=0,
        stop=2,
        tokens=tokens[::-1],
        features=features,
        raw_phase=raw,
        prediction_ttc=prediction,
        parity=parity,
    )
    with pytest.raises(ValueError, match="Invalid"):
        sign_probe._read_chunk(path, binding="binding", begin=0, stop=2, tokens=tokens)
    assert sign_probe.FEATURE_NAMES[-2:] == (
        "native_delta_t_s",
        "log_native_delta_t_s",
    )


def test_hash_bound_pandas_object_tokens_recover_to_unicode_without_mutation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sign_probe.common, "RUN", tmp_path)
    directory = tmp_path / "sign_probe_v2"
    monkeypatch.setattr(sign_probe, "DIRECTORY", directory)
    amendment = directory / "TECHNICAL_RECOVERY.json"
    monkeypatch.setattr(sign_probe, "AMENDMENT", amendment)
    path = directory / "features/P/chunk_00000.npz"
    path.parent.mkdir(parents=True)
    tokens = pd.Series(["first", "secoñd"], dtype=object).to_numpy()
    assert tokens.dtype == np.dtype(object)
    features = np.zeros((2, sign_probe.FEATURE_COUNT), dtype=np.float32)
    raw = np.array([-0.2, 0.3], dtype=np.float32)
    prediction = np.array([-5.0, 4.0], dtype=np.float32)
    np.savez_compressed(
        path,
        sample_token=tokens,
        features=features,
        raw_phase=raw,
        prediction_ttc=prediction,
    )
    original_sha = sign_probe.common.sha(path)
    sign_probe.common.write_json(
        path.with_suffix(".json"),
        {
            "binding": "binding",
            "start": 0,
            "stop": 2,
            "sha256": original_sha,
            "parity": sign_probe._parity(raw, prediction, raw, prediction),
        },
    )
    sign_probe.common.write_json(
        amendment,
        {
            "status": "reviewed-test-fixture",
            "original_P_chunks": {
                path.name: original_sha,
                path.with_suffix(".json").name: sign_probe.common.sha(path.with_suffix(".json")),
            },
        },
    )

    recovered = sign_probe._read_chunk(
        path,
        binding="binding",
        begin=0,
        stop=2,
        tokens=np.asarray(tokens, dtype=str),
    )
    assert recovered["sample_token"].dtype.kind == "U"
    assert recovered["sample_token"].tolist() == ["first", "secoñd"]
    assert recovered["_legacy_sample_token"] is True
    assert sign_probe.common.sha(path) == original_sha
    with np.load(path, allow_pickle=False) as stored:
        with pytest.raises(ValueError, match="Object arrays"):
            stored["sample_token"]


def test_new_chunks_store_pandas_object_tokens_as_pickle_free_unicode(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sign_probe.common, "RUN", tmp_path)
    path = tmp_path / "chunk_00000.npz"
    tokens = pd.Series(["first", "secoñd"], dtype=object).to_numpy()
    features = np.zeros((2, sign_probe.FEATURE_COUNT), dtype=np.float32)
    raw = np.array([-0.2, 0.3], dtype=np.float32)
    prediction = np.array([-5.0, 4.0], dtype=np.float32)
    sign_probe._write_chunk(
        path,
        binding="binding",
        begin=0,
        stop=2,
        tokens=tokens,
        features=features,
        raw_phase=raw,
        prediction_ttc=prediction,
        parity=sign_probe._parity(raw, prediction, raw, prediction),
    )
    with np.load(path, allow_pickle=False) as stored:
        loaded = stored["sample_token"]
    assert loaded.dtype.kind == "U"
    assert loaded.tolist() == ["first", "secoñd"]


def test_fixed_logistic_round_trip_is_finite_and_informative() -> None:
    values, label = _features(240)
    model = sign_probe._fit_logistic(values, label)
    probability = sign_probe._predict_probability(model, values)
    assert probability.shape == (240,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert np.mean((probability >= 0.5) == label) > 0.95
    assert model["contract"] == sign_probe.LOGISTIC
    assert model["fit_role"] == "P"
    assert model["system_promotion"] is False


def test_normalizer_and_fit_use_only_supplied_training_rows() -> None:
    train, label = _features(160)
    held_out = np.full((20, sign_probe.FEATURE_COUNT), 10_000.0)
    model = sign_probe._fit_logistic(train, label)
    np.testing.assert_allclose(model["mean"], train.mean(axis=0), rtol=0, atol=1e-12)
    assert not np.allclose(model["mean"], np.concatenate((train, held_out)).mean(axis=0))


def test_serialized_model_rejects_contract_or_shape_drift() -> None:
    values, label = _features(100)
    model = sign_probe._fit_logistic(values, label)
    with pytest.raises(ValueError, match="contract"):
        sign_probe._predict_probability(
            {**model, "contract": {**model["contract"], "C": 2}}, values
        )
    with pytest.raises(ValueError, match="serialized"):
        sign_probe._predict_probability(model, values[:, :-1])


def test_metrics_report_conditional_errors_calibration_and_empty_bins() -> None:
    label = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.7, 0.8, 0.4])
    result = sign_probe._metrics(label, probability)
    assert result["negative_error"] == 0.5
    assert result["positive_error"] == 0.5
    assert result["balanced_error"] == 0.5
    assert len(result["calibration_bins"]) == 10
    assert sum(row["rows"] for row in result["calibration_bins"]) == 4
    assert result["brier"] == pytest.approx(0.225)


def test_group_metrics_do_not_pool_acquisition_groups() -> None:
    label = np.array([0, 1, 0, 1, 0, 1])
    probability = np.array([0.1, 0.8, 0.7, 0.6, 0.2, 0.3])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    result = sign_probe._group_metrics(label, probability, groups)
    assert set(result) == {"a", "b", "c"}
    assert all(value["rows"] == 2 for value in result.values())


def test_contingency_separates_rescues_and_damages_by_true_sign() -> None:
    label = np.array([False, False, True, True])
    baseline = np.array([True, False, False, True])
    candidate = np.array([False, True, True, True])
    result = sign_probe._contingency(label, baseline, candidate)
    assert result["negative"] == {
        "rows": 2,
        "both_correct": 0,
        "rescued_by_S1": 1,
        "damaged_by_S1": 1,
        "both_wrong": 0,
    }
    assert result["positive"] == {
        "rows": 2,
        "both_correct": 1,
        "rescued_by_S1": 1,
        "damaged_by_S1": 0,
        "both_wrong": 0,
    }


def test_group_bootstrap_is_deterministic_and_keeps_groups_as_units() -> None:
    label = np.tile(np.array([False, True]), 12)
    groups = np.repeat(np.array(["a", "b", "c", "d"]), 6)
    baseline = np.tile(np.array([True, False]), 12)
    candidate = label.copy()
    first = sign_probe._bootstrap_balanced_error(label, baseline, candidate, groups)
    second = sign_probe._bootstrap_balanced_error(label, baseline, candidate, groups)
    assert first == second
    assert first["unit"] == "acquisition_group"
    assert first["refitted_per_bootstrap"] is False
    assert first["ci95_if_all_defined"][1] < 0


def test_fit_rejects_missing_class_and_nonfinite_features() -> None:
    values, label = _features(40)
    with pytest.raises(ValueError, match="binary"):
        sign_probe._fit_logistic(values, np.ones_like(label))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sign_probe._fit_logistic(values, label)


def test_completed_fit_replay_is_idempotent_and_does_not_reopen_features(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sign_probe.common, "RUN", tmp_path)
    monkeypatch.setattr(sign_probe, "DIRECTORY", tmp_path)
    registration = tmp_path / "REGISTRATION.json"
    registration.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sign_probe, "REGISTRATION", registration)
    (tmp_path / "RESULT.json").write_text("{}", encoding="utf-8")
    record = {"flags": {"encoder_optimizer_updates": 0}}
    result = {"status": "COMPLETE_DIAGNOSTIC_NOT_CANDIDATE"}
    monkeypatch.setattr(sign_probe, "verify_registration", lambda: record)
    monkeypatch.setattr(sign_probe, "_validate_result_files", lambda value: result)
    monkeypatch.setattr(sign_probe, "_admit_result", lambda value: {"replayed": value})

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("completed replay reopened P/H features")

    monkeypatch.setattr(sign_probe, "_load_features", fail_if_loaded)
    assert sign_probe._fit_and_evaluate_registered() == {"replayed": record}
    assert sign_probe.common.read_json(tmp_path / "COMPLETE.json") == {
        "status": result["status"],
        "registration_sha256": sign_probe.common.sha(registration),
        "result_sha256": sign_probe.common.sha(tmp_path / "RESULT.json"),
        "encoder_optimizer_updates": 0,
    }
