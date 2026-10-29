from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_7 import ObjectEventV47Config
from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48, ObjectEventV48Config


def _model() -> ObjectEventTTCV48:
    return ObjectEventTTCV48(
        ObjectEventV41Config(dropout=0.0),
        ObjectEventV47Config(mask_size=32, decoder_hidden_dim=32),
        ObjectEventV48Config(
            motion_hidden_dim=32,
            motion_refine_dim=32,
            field_size=32,
            freeze_foreground=True,
        ),
    )


def test_v48_forward_is_event_only_and_has_dense_outputs() -> None:
    model = _model()
    output = model(torch.randn(2, 3, 12, 64, 64))
    assert output.expansion.shape == (2,)
    assert output.pooled_log_eta.shape == (2,)
    assert output.local_log_eta.shape == (2, 32, 32)
    assert output.confidence_logits.shape == (2, 32, 32)
    assert output.foreground_probabilities.shape == (2, 3, 32, 32)
    assert torch.isfinite(output.expansion).all()
    assert torch.isfinite(output.aggregation_weights).all()


def test_v48_freezes_v47_foreground_but_trains_motion_head() -> None:
    model = _model()
    assert not any(parameter.requires_grad for parameter in model.foreground_model.parameters())
    assert any(parameter.requires_grad for parameter in model.temporal_projection.parameters())
    assert any(parameter.requires_grad for parameter in model.field_head.parameters())


def test_v48_zero_events_start_near_zero_expansion() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model(torch.zeros(3, 3, 12, 64, 64))
    assert float(output.expansion.abs().max()) < 1.0e-4
