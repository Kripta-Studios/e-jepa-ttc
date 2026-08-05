from __future__ import annotations

import torch

from e_jepa_ttc.data.object_signed_expansion import (
    ObjectSignedExpansionBatch,
    collate_object_signed_expansion,
)
from e_jepa_ttc.models.object_signed_expansion import (
    ObjectCentricSignedExpansionTTC,
    ObjectSignedExpansionConfig,
)
from e_jepa_ttc.training.object_signed_expansion import (
    ObjectSignedExpansionLossConfig,
    object_signed_expansion_loss,
    targets_from_batch,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "jepa_event_roi": torch.randn(2, 3, 32, 32),
            "garl_delta_t_s": 0.1,
            "observable_motion": torch.zeros(18),
            "jepa_context_motion": torch.zeros(18),
            "precontext_motion_valid": True,
                "jepa_pair_valid": True,
            "garl_visible_heights_px": torch.tensor([90.0, 94.0]),
            "ttc_s": 2.4,
            "sequence_id": "train-a",
            "sample_token": "a",
            "track_id": "1",
        },
        {
            "jepa_event_roi": torch.randn(2, 3, 32, 32),
            "garl_delta_t_s": 0.1,
            "observable_motion": torch.zeros(18),
            "jepa_context_motion": torch.zeros(18),
            "precontext_motion_valid": False,
                "jepa_pair_valid": True,
            "garl_visible_heights_px": torch.tensor([94.0, 90.0]),
            "ttc_s": -2.4,
            "sequence_id": "train-b",
            "sample_token": "b",
            "track_id": "2",
        },
    ]


def _model() -> ObjectCentricSignedExpansionTTC:
    return ObjectCentricSignedExpansionTTC(
        ObjectSignedExpansionConfig(
            in_channels=3,
            embed_dim=24,
            patch_size=8,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
            adapter_hidden_dim=24,
            motion_hidden_dim=16,
            predictor_hidden_dim=32,
            pair_hidden_dim=32,
        )
    )


def test_collate_model_inputs_exclude_supervision() -> None:
    batch = collate_object_signed_expansion(_records(), event_field="jepa_event_roi")
    assert isinstance(batch, ObjectSignedExpansionBatch)
    assert set(batch.model_inputs()) == {
        "events",
        "delta_t_s",
        "observable_motion",
        "jepa_context_motion",
        "precontext_motion_valid",
    }
    assert "target_ttc_s" not in batch.model_inputs()
    assert "visible_heights_px" not in batch.model_inputs()


def test_targets_are_signed_continuous_expansion() -> None:
    batch = collate_object_signed_expansion(_records(), event_field="jepa_event_roi")
    targets = targets_from_batch(batch, ObjectSignedExpansionLossConfig())
    assert targets.signed_expansion[0] > 0
    assert targets.signed_expansion[1] < 0
    assert targets.sample_weights[1] > targets.sample_weights[0]


def test_full_loss_is_finite_and_backpropagates() -> None:
    batch = collate_object_signed_expansion(_records(), event_field="jepa_event_roi")
    model = _model()
    output = model(**batch.model_inputs())
    result = object_signed_expansion_loss(
        output,
        batch,
        epoch=4,
        config=ObjectSignedExpansionLossConfig(),
    )
    assert torch.isfinite(result.total)
    assert "latent_forward" in result.components
    assert "visible_log_ratio" in result.components
    result.total.backward()
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.latent_predictor[-1].weight.grad is not None
