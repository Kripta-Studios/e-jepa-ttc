from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.evaluation.collision_clock_runner import _prediction_coordinates
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.models.collision_clock_features import event_sensor_support
from e_jepa_ttc.models.collision_clock_ttc import (
    CollisionClockConfig,
    CollisionClockTTCOutput,
    X0A5Replay,
    X0PairDirectPhase,
)


def _source() -> CausalScaleTTC:
    return CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=12,
            hidden_dim=8,
            geometry_dim=8,
            residual_depth=1,
            dropout=0.0,
        )
    )


def test_pair_is_frozen_geometry_infused_readout_and_a5_replay_is_exact() -> None:
    torch.manual_seed(7)
    source = _source()
    pair = X0PairDirectPhase(
        source,
        CollisionClockConfig(
            encoder_hidden_dim=8,
            encoder_token_dim=8,
            residual_depth=1,
            dropout=0.0,
            clock_hidden_dim=8,
            feature_source="a5_pair",
            motion_feature_mode="embedded_a5",
        ),
    )
    replay = X0A5Replay(source)
    assert all(not parameter.requires_grad for parameter in pair.source_a5.parameters())
    assert any(parameter.requires_grad for parameter in pair.clock_head.parameters())
    inputs = torch.randn(1, 3, 12, 16, 16)
    delta = torch.full((1, 2), 0.05)
    pair_output = pair(inputs, delta)
    source_output = source(inputs, delta)
    replay_output = replay(inputs, delta)
    torch.testing.assert_close(
        pair_output.diagnostics["source_a5_ttc_seconds"], source_output.ttc_mean_seconds
    )
    torch.testing.assert_close(replay_output.ttc_mean_seconds, source_output.ttc_mean_seconds)
    phase, inverse, raw, clipped = _prediction_coordinates(replay_output, delta_t_s=0.1)
    assert all(value.shape == (1,) for value in (phase, inverse, raw, clipped))
    assert bool(pair_output.diagnostics["pair_is_geometry_infused"].all())


def test_typed_output_export_rebuilds_derived_coordinates_in_float64() -> None:
    phase_tensor = torch.tensor([0.12345679], dtype=torch.float32)
    finite_placeholder = torch.tensor([1.0], dtype=torch.float32)
    output = CollisionClockTTCOutput(
        benchmark_phase_mean=phase_tensor,
        predicted_ttc_raw=finite_placeholder,
        predicted_ttc_clipped=finite_placeholder,
        is_clip_saturated=torch.tensor([False]),
        ttc_mean_seconds=finite_placeholder,
        inverse_ttc_mean=finite_placeholder,
        known_mask=torch.tensor([True]),
        sensor_support=finite_placeholder,
        global_clock_token=finite_placeholder[:, None],
        diagnostics={},
    )
    phase, inverse, raw, clipped = _prediction_coordinates(output, delta_t_s=0.1)
    expected_inverse = -np.expm1(-phase) / 0.1
    expected_raw = np.reciprocal(expected_inverse)
    assert np.array_equal(inverse, expected_inverse)
    assert np.array_equal(raw, expected_raw)
    assert np.array_equal(clipped, np.clip(expected_raw, -60.0, 60.0))


def test_public_event_support_matches_historical_semantics() -> None:
    source = _source()
    values = torch.zeros(2, 12, 8, 8)
    values[0, :, :4] = 1.0
    torch.testing.assert_close(event_sensor_support(values), source._sensor_support(values))
