from __future__ import annotations

import copy

import pytest
import torch

from e_jepa_ttc.losses.level_dynamics_jepa import LevelDynamicsLossConfig, ObjectiveArm
from e_jepa_ttc.models.dense_level_dynamics_jepa import (
    DenseLevelDynamicsConfig,
    DenseLevelDynamicsJEPA,
)
from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHRConfig
from e_jepa_ttc.training.eap_highres_jepa import (
    LABEL_FAMILY_PROVENANCE_FIELDS,
    EAPHighResJEPATrainer,
    EAPHighResJEPATrainerConfig,
    LabelFreeBatch,
    LabelFreeManifestProvenance,
)


def _config() -> EAPHighResJEPATrainerConfig:
    return EAPHighResJEPATrainerConfig(
        model=DenseLevelDynamicsConfig(
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
            max_horizons=3,
            ema_total_updates=4,
        ),
        loss=LevelDynamicsLossConfig(
            objective=ObjectiveArm.LEVEL_DYNAMICS_NCE_RESIDUAL_VISREG,
            nce_positive_tolerance_s=1e-5,
            visreg_projections=4,
        ),
        learning_rate=1e-3,
        total_updates=4,
        seed=17,
    )


def _batch(seed: int) -> LabelFreeBatch:
    generator = torch.Generator().manual_seed(seed)
    context = torch.randn(1, 2, 2, 8, 8, generator=generator)
    future = torch.randn(1, 3, 2, 2, 8, 8, generator=generator)
    return LabelFreeBatch(
        context_events=context,
        future_events=future,
        horizon_delta_t_s=torch.tensor([[0.1, 0.2, 0.3]]),
        sequence_ids=("synthetic",),
        track_ids=("track",),
        reference_timestamps_s=torch.tensor([0.0]),
        future_timestamps_s=torch.tensor([[0.1, 0.2, 0.3]]),
    )


def _provenance() -> LabelFreeManifestProvenance:
    return LabelFreeManifestProvenance(
        matched_manifest_hash="a" * 64,
        dataset_hashes={"synthetic": "b" * 64},
        split_hash="c" * 64,
        sampler_order_hash="d" * 64,
        selection_rule="synthetic_fixed_order_v1",
        label_family_provenance={key: False for key in LABEL_FAMILY_PROVENANCE_FIELDS},
    )


def test_resume_matches_uninterrupted_synthetic_updates(tmp_path) -> None:
    torch.manual_seed(41)
    config = _config()
    initialized = DenseLevelDynamicsJEPA(config.model)
    initial_state = copy.deepcopy(initialized.state_dict())
    batches = [_batch(index) for index in range(4)]

    uninterrupted_model = DenseLevelDynamicsJEPA(config.model)
    uninterrupted_model.load_state_dict(initial_state)
    uninterrupted = EAPHighResJEPATrainer(config, model=uninterrupted_model)
    uninterrupted_losses = [uninterrupted.train_step(batch)["loss"] for batch in batches]

    interrupted_model = DenseLevelDynamicsJEPA(config.model)
    interrupted_model.load_state_dict(initial_state)
    interrupted = EAPHighResJEPATrainer(config, model=interrupted_model)
    interrupted_losses = [interrupted.train_step(batch)["loss"] for batch in batches[:2]]
    checkpoint = tmp_path / "resume.pt"
    interrupted.save_checkpoint(checkpoint, _provenance())

    resumed = EAPHighResJEPATrainer(config)
    resumed.load_checkpoint(checkpoint)
    resumed_losses = [resumed.train_step(batch)["loss"] for batch in batches[2:]]

    assert uninterrupted_losses[:2] == interrupted_losses
    assert uninterrupted_losses[2:] == resumed_losses
    assert uninterrupted.update_count == resumed.update_count == 4
    for key, value in uninterrupted.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key

    with pytest.raises(RuntimeError, match="total update budget is already exhausted"):
        resumed.train_batches([_batch(99)])
