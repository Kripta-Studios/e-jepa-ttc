"""Fixed baseline inference never reads supervision or normalization."""

from types import SimpleNamespace

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.fixed_baselines import predict_fixed_baselines


class InputOnly(SimpleNamespace):
    def __getattribute__(self, name):
        if name in {"gather", "target_phase", "mass", "normalizer"}:
            raise AssertionError("baseline accessed supervised/training state")
        return super().__getattribute__(name)


@pytest.fixture
def source():
    features = np.zeros((8, 17), np.float32)
    features[:, 8:11] = 0.02
    return InputOnly(
        features=features,
        anchor_us=np.arange(8, dtype=np.int64) * 50000,
        available_us=np.arange(8, dtype=np.int64) * 50000 + 1000,
        history=np.array([[-1] * 7 + [0], list(range(8))], np.int64),
        identity_sha256="a" * 64,
        length=8,
        control="NONE",
        zero_latent=False,
        population=2,
    )


def run(source, **overrides):
    arguments = dict(
        expected_source_sha256="a" * 64,
        validate_prerequisites=lambda: None,
        resource_ok=lambda: True,
    )
    arguments.update(overrides)
    return predict_fixed_baselines(source, **arguments)


def test_constant_phase_and_cold_start_without_supervision(source):
    results = run(source)
    assert set(results) == {"CURRENT_MEDIAN", "EWMA_0P3S_H8"}
    for result in results.values():
        assert result["prediction_phase"].shape == (2,)
        assert np.isfinite(result["prediction_ttc_s"]).all()
        assert result["prediction_ttc_s"].dtype == np.float64
    np.testing.assert_allclose(
        results["CURRENT_MEDIAN"]["prediction_phase"], results["EWMA_0P3S_H8"]["prediction_phase"]
    )


def test_history_changes_only_smoother_not_current_median(source):
    original = run(source)
    source.features[1:7, 8:11] = 0.06
    changed = run(source)
    np.testing.assert_array_equal(
        original["CURRENT_MEDIAN"]["prediction_phase"],
        changed["CURRENT_MEDIAN"]["prediction_phase"],
    )
    assert (
        changed["EWMA_0P3S_H8"]["prediction_phase"][1]
        > original["EWMA_0P3S_H8"]["prediction_phase"][1]
    )


def test_resource_pause_returns_no_partial_predictions(source):
    with pytest.raises(InterruptedError):
        run(source, resource_ok=lambda: False)


@pytest.mark.parametrize("fault", ["identity", "future", "current_missing", "control"])
def test_invalid_source_rejected(source, fault):
    if fault == "identity":
        source.identity_sha256 = "b" * 64
    elif fault == "future":
        source.available_us[1] = 999999
    elif fault == "current_missing":
        source.history[0, -1] = -1
    else:
        source.control = "REPEAT_CURRENT"
    with pytest.raises(ValueError):
        run(source)
