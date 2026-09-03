from __future__ import annotations

import torch

from e_jepa_ttc.evaluation.collision_clock_runner import _prediction_coordinates
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.models.collision_clock_features import event_sensor_support
from e_jepa_ttc.models.collision_clock_ttc import (
    CollisionClockConfig,
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


def test_public_event_support_matches_historical_semantics() -> None:
    source = _source()
    values = torch.zeros(2, 12, 8, 8)
    values[0, :, :4] = 1.0
    torch.testing.assert_close(event_sensor_support(values), source._sensor_support(values))
