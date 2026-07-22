from __future__ import annotations

import numpy as np

from e_jepa_ttc.evaluation.calibration import (
    fit_conformal_interval,
    fit_temperature_scaler,
    interval_metrics,
)
from e_jepa_ttc.evaluation.object_ttc import (
    binary_risk_metrics,
    garl_ttc_metrics,
    object_ttc_metrics,
)


def test_garl_metrics_are_zero_for_exact_predictions() -> None:
    target = np.asarray([-5.0, 1.0, 4.0, 8.0])
    metrics = garl_ttc_metrics(target, target.copy())

    assert metrics["weighted_mid"] == 0.0
    assert metrics["weighted_rte_pct"] == 0.0
    assert all(value["failure_ratio"] == 0.0 for value in metrics["bins"].values())


def test_risk_metrics_rank_perfect_predictions() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = binary_risk_metrics(labels, probabilities)

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["f1_at_0_5"] == 1.0


def test_conformal_intervals_and_temperature_are_calibration_only() -> None:
    target = np.asarray([1.0, 2.0, 3.0, 4.0])
    mean = np.asarray([1.1, 1.8, 3.2, 3.8])
    standard_deviation = np.full(4, 0.1)
    calibrator = fit_conformal_interval(target, mean, standard_deviation, coverage=0.75)
    lower, upper = calibrator.interval(mean, standard_deviation)
    metrics = interval_metrics(target, lower, upper)

    assert calibrator.calibration_count == 4
    assert metrics["coverage"] >= 0.75
    scaler = fit_temperature_scaler(
        np.asarray([-2.0, -1.0, 1.0, 2.0]),
        np.asarray([0, 0, 1, 1]),
    )
    calibrated = scaler.probabilities(np.asarray([-2.0, 2.0]))
    assert 0.0 < calibrated[0] < 0.5 < calibrated[1] < 1.0


def test_combined_object_metrics_reports_all_risk_horizons() -> None:
    target = np.asarray([0.4, 0.8, 1.5, 3.0, 5.0, -2.0])
    probability = np.full((target.size, 4), 0.5)
    metrics = object_ttc_metrics(target, target + 0.1, probability)

    assert set(metrics) == {"regression", "garl_ttc", "risk"}
    assert set(metrics["risk"]) == {"0.5", "1.0", "2.0", "4.0"}
    assert np.isfinite(metrics["regression"]["signed_log1p_mae"])
