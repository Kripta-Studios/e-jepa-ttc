from __future__ import annotations

import torch

from e_jepa_ttc.losses.level_dynamics_jepa import LevelDynamicsLossConfig, ObjectiveArm
from e_jepa_ttc.models.dense_level_dynamics_jepa import DenseLevelDynamicsConfig
from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHRConfig
from e_jepa_ttc.training.eap_highres_jepa import (
    EAPHighResJEPATrainer,
    EAPHighResJEPATrainerConfig,
    LabelFreeBatch,
)


def test_tiny_synthetic_level_objective_overfits() -> None:
    torch.manual_seed(123)
    model = DenseLevelDynamicsConfig(
        encoder=EJEPATubeletLHRConfig(
            in_channels=2,
            embed_dim=16,
            patch_size=4,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=True,
        ),
        projection_dim=16,
        predictor_dim=16,
        predictor_layers=1,
        predictor_heads=4,
        predictor_mlp_ratio=2,
        patch_query_chunk_size=4,
        max_temporal_steps=2,
        max_patches=16,
        max_horizons=1,
    )
    trainer = EAPHighResJEPATrainer(
        EAPHighResJEPATrainerConfig(
            model=model,
            loss=LevelDynamicsLossConfig(objective=ObjectiveArm.LEVEL),
            learning_rate=5e-3,
            total_updates=16,
        )
    )
    context = torch.randn(1, 2, 2, 8, 8)
    batch = LabelFreeBatch(
        context_events=context,
        future_events=context[:, None].clone(),
        horizon_delta_t_s=torch.tensor([[0.1]]),
        sequence_ids=("synthetic",),
        track_ids=("track",),
        reference_timestamps_s=torch.tensor([0.0]),
        future_timestamps_s=torch.tensor([[0.1]]),
    )

    losses = [trainer.train_step(batch)["loss"] for _ in range(16)]

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert losses[-1] < losses[0] * 0.7
