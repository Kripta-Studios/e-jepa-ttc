from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models.object_event_v4_22 import (
    ObjectEventV422Config,
    configure_partial_geometry_unfreeze,
    select_tail_parameter_names,
)
from e_jepa_ttc.training.object_event_v4_22 import (
    ObjectEventV422LossConfig,
    encoder_pseudoflow_loss,
    masked_flow_loss,
    relative_parameter_anchor,
)


class TinyGeometry(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.Conv2d(4, 4, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class MotionConfig:
    freeze_foreground: bool = True


class TinyForeground(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.geometry_encoder = TinyGeometry()
        self.decoder = nn.Conv2d(4, 4, 1)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.foreground_model = TinyForeground()
        self.motion_config = MotionConfig()


def test_tail_parameter_selection_is_deterministic() -> None:
    module = TinyGeometry()
    names = [name for name, _ in module.named_parameters()]
    assert select_tail_parameter_names(module, 2) == names[-2:]


def test_partial_unfreeze_only_enables_tail_geometry_parameters() -> None:
    backbone = TinyBackbone()
    selected = configure_partial_geometry_unfreeze(backbone, 2)
    trainable = [name for name, p in backbone.foreground_model.geometry_encoder.named_parameters() if p.requires_grad]
    assert trainable == selected
    assert backbone.motion_config.freeze_foreground is False
    assert not any(p.requires_grad for p in backbone.foreground_model.decoder.parameters())


def test_relative_anchor_is_zero_at_initial_state() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    initial = {"p": p.detach().clone()}
    value = relative_parameter_anchor({"p": p}, initial, epsilon=1.0e-6)
    assert float(value.detach()) == 0.0


def test_relative_anchor_increases_after_drift() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    initial = {"p": p.detach().clone()}
    with torch.no_grad():
        p.add_(0.2)
    value = relative_parameter_anchor({"p": p}, initial, epsilon=1.0e-6)
    assert float(value.detach()) > 0.0


def test_masked_flow_loss_zero_for_exact_target() -> None:
    target = torch.randn(2, 2, 8, 8)
    mask = torch.ones(2, 8, 8)
    value = masked_flow_loss(target, target, mask, beta=0.1, epsilon=1.0e-6)
    assert torch.allclose(value, torch.zeros_like(value))


def test_encoder_pseudoflow_loss_backpropagates_to_flow() -> None:
    forward = torch.zeros(2, 2, 8, 8, requires_grad=True)
    reverse = torch.zeros(2, 2, 8, 8, requires_grad=True)
    target_f = torch.zeros_like(forward)
    target_r = torch.zeros_like(reverse)
    target_f[:, 0] = 0.2
    target_r[:, 0] = -0.2
    mask = torch.ones(2, 8, 8)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    initial = {"p": parameter.detach().clone()}
    loss, parts = encoder_pseudoflow_loss(
        forward, reverse, target_f, mask, target_r, mask,
        {"p": parameter}, initial, config=ObjectEventV422LossConfig(),
    )
    loss.backward()
    assert forward.grad is not None and float(forward.grad.abs().sum()) > 0.0
    assert reverse.grad is not None and float(reverse.grad.abs().sum()) > 0.0
    assert set(parts) == {"flow", "divergence", "vertical_scale", "encoder_anchor"}


def test_config_rejects_invalid_radius() -> None:
    try:
        ObjectEventV422Config(search_radius=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_vertical_log_scale_recovers_affine_height_change() -> None:
    from e_jepa_ttc.training.object_event_v4_22 import vertical_log_scale_from_flow

    h, w = 9, 7
    y = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    scale = 1.25
    flow = torch.zeros(1, 2, h, w)
    flow[:, 1] = (scale - 1.0) * (y - (h - 1) / 2.0)
    mask = torch.ones(1, h, w)
    estimate = vertical_log_scale_from_flow(flow, mask, epsilon=1.0e-6)
    assert torch.allclose(estimate, torch.tensor([scale]).log(), atol=1.0e-5)


def test_vertical_log_scale_changes_sign_for_inverse_scale() -> None:
    from e_jepa_ttc.training.object_event_v4_22 import vertical_log_scale_from_flow

    h, w = 9, 7
    y = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    scale = 0.8
    flow = torch.zeros(1, 2, h, w)
    flow[:, 1] = (scale - 1.0) * (y - (h - 1) / 2.0)
    mask = torch.ones(1, h, w)
    estimate = vertical_log_scale_from_flow(flow, mask, epsilon=1.0e-6)
    assert float(estimate) < 0.0
