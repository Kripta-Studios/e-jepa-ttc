from __future__ import annotations

import torch

from e_jepa_ttc.models.multimodal import ObjectEventRGBFusion, multimodal_ttc_loss
from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig


def test_multimodal_fusion_shapes_and_detached_rgb_distillation() -> None:
    config = ObjectJEPAConfig(
        in_channels=4,
        embedding_dim=16,
        feature_dim=16,
        predictor_depth=1,
        predictor_heads=4,
        dropout=0.0,
        pre_cropped_events=True,
    )
    model = ObjectEventRGBFusion(ObjectCentricEventJEPA(config))
    events = torch.randn(2, 3, 4, 16, 16)
    rgb = torch.randint(0, 256, (2, 3, 3, 24, 24), dtype=torch.uint8)
    boxes = torch.tensor(
        [[[[0.2, 0.2, 0.8, 0.8]]] * 3] * 2,
        dtype=torch.float32,
    )
    mask = torch.ones(2, 3, 1, dtype=torch.bool)

    output = model(events, rgb, boxes, mask)
    output.rgb_tokens.retain_grad()
    losses = multimodal_ttc_loss(output, torch.tensor([[2.0], [4.0]]))
    losses["rgb_to_event_distillation"].backward()

    assert output.inverse_ttc_mean.shape == (2, 1)
    assert output.risk_logits.shape == (2, 1, 4)
    assert torch.all((output.fusion_gate >= 0) & (output.fusion_gate <= 1))
    assert output.rgb_tokens.grad is None
    event_gradients = [
        parameter.grad
        for parameter in model.event_model.context_encoder.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None for gradient in event_gradients)
