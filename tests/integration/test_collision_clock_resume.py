from __future__ import annotations

from pathlib import Path

import torch

from e_jepa_ttc.models.collision_clock_features import (
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
)
from e_jepa_ttc.models.collision_clock_ttc import CollisionClockConfig, X0HeightBypassDirectPhase
from e_jepa_ttc.training.collision_clock_eap import (
    CollisionClockBatch,
    CollisionClockTrainingConfig,
    train_collision_clock_updates,
)


def _model() -> X0HeightBypassDirectPhase:
    config = CollisionClockConfig(
        encoder_hidden_dim=8,
        encoder_token_dim=4,
        residual_depth=1,
        clock_hidden_dim=4,
        dropout=0.05,
        motion_feature_mode="global_uniform",
    )
    encoder = HeightBypassEndpointEncoder(
        HeightBypassEncoderConfig(
            in_channels=12,
            hidden_dim=8,
            token_dim=4,
            residual_depth=1,
            dropout=0.05,
        )
    )
    return X0HeightBypassDirectPhase(encoder, config)


def _batches() -> list[CollisionClockBatch]:
    generator = torch.Generator().manual_seed(99)
    return [
        CollisionClockBatch(
            inputs=torch.randn(1, 3, 12, 16, 16, generator=generator),
            delta_t_s=torch.full((1, 2), 0.05),
            target_ttc_seconds=torch.tensor([1.0 + index]),
            sample_tokens=(f"token-{index}",),
        )
        for index in range(2)
    ]


def test_continuous_n_equals_k_load_n_minus_k(tmp_path: Path) -> None:
    config = CollisionClockTrainingConfig(arm_id="X0-DYN-U", planned_updates=4)
    torch.manual_seed(7)
    continuous = _model()
    continuous_result = train_collision_clock_updates(
        continuous,
        _batches(),
        config=config,
        checkpoint_path=tmp_path / "continuous.pt",
    )

    torch.manual_seed(7)
    partial = _model()
    train_collision_clock_updates(
        partial,
        _batches(),
        config=config,
        checkpoint_path=tmp_path / "resumed.pt",
        stop_after_updates=2,
    )
    torch.manual_seed(1234)
    resumed = _model()
    resumed_result = train_collision_clock_updates(
        resumed,
        _batches(),
        config=config,
        checkpoint_path=tmp_path / "resumed.pt",
        resume=True,
    )
    assert continuous_result.losses == resumed_result.losses
    assert continuous_result.batch_schedule_sha256 == resumed_result.batch_schedule_sha256
    for name, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0.0, atol=0.0)
