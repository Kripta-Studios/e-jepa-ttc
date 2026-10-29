from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.models.object_event_v4_19 import (
    ObjectEventV419Config,
    antisymmetric_correspondence_scores,
    dense_flow_scores,
    local_correlation_flow,
)
from e_jepa_ttc.training.object_event_v4_19 import (
    apply_score_calibration,
    equal_physics_consensus,
    fit_score_calibration,
    prediction_from_score,
)


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[..., 1:] = x[..., :-1]
    return y


def test_local_correlation_recovers_rightward_shift() -> None:
    torch.manual_seed(4)
    first = torch.randn(1, 16, 9, 9)
    second = _shift_right(first)
    fx, fy, confidence = local_correlation_flow(first, second, radius=2, temperature=0.02)
    assert float(fx[:, 2:-2, 2:-2].mean()) > 0.8
    assert abs(float(fy[:, 2:-2, 2:-2].mean())) < 0.2
    assert float(confidence.mean()) > 0.05


def test_translation_has_near_zero_divergence_and_radial_slope() -> None:
    flow_x = torch.ones(2, 11, 11)
    flow_y = 0.5 * torch.ones_like(flow_x)
    fg = torch.ones_like(flow_x)
    conf = torch.ones_like(flow_x)
    divergence, radial, translation = dense_flow_scores(
        flow_x, flow_y, fg, conf, foreground_floor=0.05, confidence_floor=0.05
    )
    assert float(divergence.abs().max()) < 1.0e-6
    assert float(radial.abs().max()) < 1.0e-6
    assert float(translation.min()) > 1.0


def test_affine_expansion_has_positive_dense_scores() -> None:
    size = 13
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    cx = cy = (size - 1) / 2
    flow_x = (0.1 * (x - cx))[None].float()
    flow_y = (0.1 * (y - cy))[None].float()
    fg = torch.ones_like(flow_x)
    conf = torch.ones_like(flow_x)
    divergence, radial, _ = dense_flow_scores(
        flow_x, flow_y, fg, conf, foreground_floor=0.05, confidence_floor=0.05
    )
    assert float(divergence[0]) > 0.15
    assert float(radial[0]) > 0.08


def test_endpoint_swap_antisymmetry_on_identical_inputs_is_zero() -> None:
    torch.manual_seed(7)
    features = torch.randn(2, 8, 7, 7)
    foreground = torch.sigmoid(torch.randn(2, 14, 14))
    config = ObjectEventV419Config(search_radius=2, correlation_temperature=0.05)
    divergence, radial, _ = antisymmetric_correspondence_scores(
        features, features, foreground, foreground, config
    )
    assert float(divergence.abs().max()) < 1.0e-6
    assert float(radial.abs().max()) < 1.0e-6


def test_train_only_calibration_orients_without_centering() -> None:
    score = np.asarray([-3.0, -2.0, 1.0, 2.0])
    target = -score
    calibration = fit_score_calibration(score, target, minimum_scale=1.0e-3)
    transformed = apply_score_calibration(score, calibration)
    assert calibration.orientation == -1.0
    assert transformed[0] > 0.0 and transformed[-1] < 0.0
    assert abs(float(apply_score_calibration(np.asarray([0.0]), calibration)[0])) == 0.0


def test_consensus_and_prediction_use_zero_threshold_only() -> None:
    div = np.asarray([2.0, -2.0, 0.5])
    radial = np.asarray([1.0, -1.0, -0.25])
    score = equal_physics_consensus(div, radial)
    pred = prediction_from_score(score, np.asarray([0.2, 0.3, 0.4]))
    assert np.allclose(score, [1.5, -1.5, 0.125])
    assert np.allclose(pred, [0.2, -0.3, 0.4])
