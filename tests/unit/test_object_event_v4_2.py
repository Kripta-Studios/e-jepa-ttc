import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_2 import ObjectEventTTCV42


def test_v42_is_encoded_only_and_event_differentiable() -> None:
    model = ObjectEventTTCV42(
        ObjectEventV41Config(
            input_size=32,
            stem_dim=16,
            embed_dim=16,
            spatial_grid=2,
            encoded_hidden_dim=32,
            activity_hidden_dim=16,
        )
    )
    events = torch.randn(4, 3, 12, 32, 32, requires_grad=True)
    output = model(events)
    assert output.expansion.shape == (4,)
    assert output.reverse_expansion.shape == (4,)
    output.expansion.square().mean().backward()
    assert events.grad is not None
    assert float(events.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.activity_head.parameters())
    assert model.branch_scale.grad is None
