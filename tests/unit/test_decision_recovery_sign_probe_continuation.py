from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from operational.decision_recovery import sign_probe
from operational.decision_recovery import sign_probe_continuation as continuation


def _features(rows: int, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, sign_probe.FEATURE_COUNT))
    label = (values[:, 0] - 0.5 * values[:, 1] > 0).astype(np.int64)
    return values, label


def test_continuation_changes_only_iteration_limit() -> None:
    expected = {**sign_probe.LOGISTIC, "max_iter": 10_000}
    assert continuation.CONTRACT == expected
    assert continuation.MAX_ITER == 10_000
    assert sign_probe.LOGISTIC["max_iter"] == 2_000


def test_continuation_fit_is_finite_and_records_both_limits() -> None:
    values, label = _features(300)
    model = continuation._fit_logistic(values, label)
    probability = continuation._predict_probability(model, values)
    assert np.isfinite(probability).all()
    assert np.mean((probability >= 0.5) == label) > 0.95
    assert model["contract"] == continuation.CONTRACT
    assert model["prior_failed_limit"] == 2_000
    assert 0 < model["logistic_fit_iterations"] < 10_000
    assert model["encoder_optimizer_updates"] == 0


def test_run_requires_registration_before_fit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(continuation, "DIRECTORY", tmp_path)
    monkeypatch.setattr(continuation, "REGISTRATION", tmp_path / "REGISTRATION.json")
    with pytest.raises(ValueError, match="Register"):
        continuation.run()


def test_h_is_loaded_only_after_both_p_fits_in_source_order() -> None:
    source = inspect.getsource(continuation._run_registered)
    first_fit = source.index('model = _fit_logistic(p["features"], p_label)')
    second_fit = source.index('shuffled = _fit_logistic(p["features"], shuffled_label)')
    h_load = source.index('h = sign_probe._load_features("H"')
    assert first_fit < second_fit < h_load


def test_fit_rejects_invalid_population() -> None:
    values, label = _features(30)
    with pytest.raises(ValueError, match="binary"):
        continuation._fit_logistic(values, np.ones_like(label))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        continuation._fit_logistic(values, label)


def test_shuffled_limit_records_completed_s1_without_scoring_h(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(continuation.common, "RUN", tmp_path)
    monkeypatch.setattr(continuation, "DIRECTORY", tmp_path)
    registration = tmp_path / "REGISTRATION.json"
    registration.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(continuation, "REGISTRATION", registration)
    monkeypatch.setattr(continuation, "registered", lambda: {"flags": {}})
    features, label = _features(30)

    def load_features(role, _record):
        if role != "P":
            raise AssertionError("H cache was loaded for scoring after a failed P control fit")
        return {"features": features}

    def load_population(role, _arms):
        if role != "P":
            raise AssertionError("H rows were opened for scoring after a failed P control fit")
        return {"FULL": pd.DataFrame({"target_ttc": np.where(label > 0, 1.0, -1.0)})}

    calls = 0

    def fit(_features, _label):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"logistic_fit_iterations": 2_345}
        raise continuation.FitLimitReachedError(10_000, ["iteration limit"])

    monkeypatch.setattr(continuation.sign_probe, "_load_features", load_features)
    monkeypatch.setattr(continuation.common, "load_population", load_population)
    monkeypatch.setattr(continuation, "_fit_logistic", fit)
    result = continuation._run_registered()
    assert result["status"] == "FAILED_CLOSED_AT_CONTINUATION_LIMIT"
    assert result["failed_fit"] == "SHUFFLED"
    assert result["completed_fit_iterations"] == {"S1": 2_345}
    assert result["observed_iterations"] == 10_000
    assert result["H_cache_bound_during_registration"] is True
    assert result["H_evaluation_started"] is False
    assert result["H_scores_created"] is False
