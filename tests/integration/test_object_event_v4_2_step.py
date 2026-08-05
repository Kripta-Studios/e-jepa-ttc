import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_2 import ObjectEventTTCV42
from e_jepa_ttc.training.object_event_v4_2 import (
    ObjectEventV42LossConfig,
    object_event_v4_2_loss,
)


def test_v42_optimisation_step_is_finite() -> None:
    torch.manual_seed(7)
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
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    events = torch.randn(8, 3, 12, 32, 32)
    delta_t = torch.full((8,), 0.1)
    target_ttc = torch.tensor([2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0])
    output = model(events)
    loss = object_event_v4_2_loss(
        output,
        delta_t,
        target_ttc,
        epoch=1,
        config=ObjectEventV42LossConfig(),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    assert torch.isfinite(loss.total)
    assert float(loss.total.detach()) > 0.0
