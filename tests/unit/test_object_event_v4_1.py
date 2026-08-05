from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventTTCV41, ObjectEventV41Config
from e_jepa_ttc.training.object_event_v4_1 import (
    ObjectEventV41LossConfig,
    object_event_v4_1_loss,
)


def _config() -> ObjectEventV41Config:
    return ObjectEventV41Config(
        input_size=16,
        stem_dim=16,
        embed_dim=24,
        spatial_grid=2,
        encoded_hidden_dim=48,
        activity_hidden_dim=24,
        dropout=0.0,
    )


def test_v41_prediction_is_event_only_direct_and_not_hard_coded_odd() -> None:
    torch.manual_seed(7)
    model = ObjectEventTTCV41(_config()).eval()
    events = torch.randn(4, 3, 12, 16, 16)
    output = model(events)
    reversed_output = model(torch.flip(events, dims=(1,)))
    assert output.expansion.shape == (4,)
    assert output.endpoint_embeddings.shape == (4, 3, 24)
    assert torch.allclose(output.reverse_expansion, reversed_output.expansion, atol=1.0e-6)
    # Reversal is measured, not imposed by construction.
    assert float((output.expansion + output.reverse_expansion).detach().abs().max()) > 1.0e-8


def test_v41_loss_has_finite_event_gradient_without_ttc_loss() -> None:
    torch.manual_seed(13)
    model = ObjectEventTTCV41(_config())
    events = torch.randn(8, 3, 12, 16, 16, requires_grad=True)
    delta_t = torch.full((8,), 0.1)
    target_ttc = torch.tensor((0.6, 0.8, 1.2, 2.0, -0.7, -1.0, -1.5, -3.0))
    output = model(events)
    loss = object_event_v4_1_loss(
        output,
        delta_t,
        target_ttc,
        step=1,
        config=ObjectEventV41LossConfig(sign_temperature=0.5),
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert events.grad is not None
    assert float(events.grad.abs().sum()) > 0.0
    assert "reversal" in loss.components
    assert "ttc" not in " ".join(loss.components)


def test_v41_zero_events_are_not_an_external_geometry_shortcut() -> None:
    torch.manual_seed(17)
    model = ObjectEventTTCV41(_config()).eval()
    zero = torch.zeros(6, 3, 12, 16, 16)
    output = model(zero).expansion
    # Every zero sample is identical; there is no sample-specific box/motion input.
    assert float(output.detach().std(unbiased=False)) < 1.0e-10
