from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from e_jepa_ttc.evaluation.garl_ttc_protocol import signed_garl_metrics
from e_jepa_ttc.models.collision_clock_math import (
    benchmark_phase_to_ttc,
    neutral_raw_phase,
    phase_lower_bound,
    ttc_to_benchmark_phase,
)


def test_positive_and_negative_ttc_phase_round_trip() -> None:
    values = torch.tensor([-9.0, -1.0, 0.2, 1.0, 9.0], dtype=torch.float64)
    phase, valid = ttc_to_benchmark_phase(values, metric_delta_t_s=0.1)
    assert bool(valid.all())
    reconstructed = benchmark_phase_to_ttc(
        phase,
        metric_delta_t_s=0.1,
        clip_seconds=60.0,
    )
    torch.testing.assert_close(reconstructed, values, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("value", [0.0, 0.05, 0.1, float("inf"), float("nan")])
def test_phase_rejects_invalid_domain(value: float) -> None:
    phase, valid = ttc_to_benchmark_phase(
        torch.tensor([value], dtype=torch.float64),
        metric_delta_t_s=0.1,
    )
    assert not bool(valid.item())
    assert math.isnan(float(phase.item()))


def test_t2_start_anchor_matches_interval_identity_but_not_same_end_anchor() -> None:
    delta = 0.1
    ttc_at_t2 = torch.tensor([1.0], dtype=torch.float64)
    ttc_at_t1 = ttc_at_t2 + delta
    start_phase, valid = ttc_to_benchmark_phase(ttc_at_t1, metric_delta_t_s=delta)
    assert bool(valid.item())
    interval_phase = torch.log1p(delta / ttc_at_t2)
    torch.testing.assert_close(start_phase, interval_phase)
    same_end_phase, _ = ttc_to_benchmark_phase(ttc_at_t2, metric_delta_t_s=delta)
    assert not torch.allclose(same_end_phase, interval_phase)


def test_phase_difference_is_exact_canonical_mid() -> None:
    target = np.array([-4.0, 0.5, 2.0, 8.0], dtype=np.float64)
    prediction = np.array([-3.0, 0.6, 2.5, 7.0], dtype=np.float64)
    target_phase, _ = ttc_to_benchmark_phase(torch.from_numpy(target), metric_delta_t_s=0.1)
    prediction_phase, _ = ttc_to_benchmark_phase(
        torch.from_numpy(prediction), metric_delta_t_s=0.1
    )
    per_row = 1.0e4 * (target_phase - prediction_phase).abs().numpy()
    for index, (truth, estimate) in enumerate(zip(target, prediction, strict=True)):
        metrics = signed_garl_metrics(np.array([truth]), np.array([estimate]))
        bucket = next(payload for payload in metrics["bins"].values() if payload["count"] == 1)
        assert float(bucket["mid"]) == pytest.approx(per_row[index], rel=1.0e-12)


def test_neutral_initialization_derives_exact_zero_phase() -> None:
    lower = phase_lower_bound(
        metric_delta_t_s=0.1,
        minimum_abs_prediction_ttc_s=0.1,
    )
    raw = neutral_raw_phase(
        metric_delta_t_s=0.1,
        minimum_abs_prediction_ttc_s=0.1,
    )
    phase = lower + torch.nn.functional.softplus(torch.tensor(raw, dtype=torch.float64))
    assert float(phase) == pytest.approx(0.0, abs=1.0e-15)
