from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models.object_event_v4_23 import (
    configure_joint_geometry_ttc_unfreeze,
    geometry_parameters,
    motion_parameters,
    named_trainable_parameters,
)
from e_jepa_ttc.training.object_event_v4_23 import ObjectEventV423JointLossConfig, combine_joint_losses


class TinyGeometry(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1))


class TinyForeground(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.geometry_encoder = TinyGeometry()
        self.decoder = nn.Conv2d(4, 1, 1)
        self.refine = nn.Conv2d(1, 1, 1)


@dataclass(frozen=True)
class MotionConfig:
    freeze_foreground: bool = True


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.foreground_model = TinyForeground()
        self.temporal_projection = nn.Conv2d(4, 4, 1)
        self.field_head = nn.Conv2d(4, 2, 1)
        self.other = nn.Linear(4, 4)
        self.motion_config = MotionConfig()


def test_joint_unfreeze_selects_only_geometry_tail_and_motion_head() -> None:
    model = TinyBackbone()
    selected = configure_joint_geometry_ttc_unfreeze(model, 2)
    trainable = named_trainable_parameters(model)
    assert selected["geometry"] == ["net.1.weight", "net.1.bias"]
    assert "foreground_model.geometry_encoder.net.1.weight" in trainable
    assert any(name.startswith("temporal_projection.") for name in trainable)
    assert any(name.startswith("field_head.") for name in trainable)
    assert not any(name.startswith("foreground_model.decoder.") for name in trainable)
    assert not any(name.startswith("other.") for name in trainable)
    assert model.motion_config.freeze_foreground is False


def test_geometry_and_motion_parameter_groups_do_not_overlap() -> None:
    model = TinyBackbone()
    configure_joint_geometry_ttc_unfreeze(model, 2)
    geometry = {id(p) for p in geometry_parameters(model)}
    motion = {id(p) for p in motion_parameters(model)}
    assert geometry
    assert motion
    assert geometry.isdisjoint(motion)


def test_combined_loss_uses_requested_weights() -> None:
    cfg = ObjectEventV423JointLossConfig(ttc_weight=2.0, geometry_auxiliary_weight=0.5, trainable_anchor_weight=0.25)
    value = combine_joint_losses(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(4.0), config=cfg)
    assert torch.allclose(value, torch.tensor(4.0))


def test_combined_loss_backpropagates_to_all_terms() -> None:
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    c = torch.tensor(3.0, requires_grad=True)
    value = combine_joint_losses(a, b, c, config=ObjectEventV423JointLossConfig())
    value.backward()
    assert a.grad is not None and float(a.grad) > 0.0
    assert b.grad is not None and float(b.grad) > 0.0
    assert c.grad is not None and float(c.grad) > 0.0


def test_config_rejects_negative_weight() -> None:
    try:
        ObjectEventV423JointLossConfig(geometry_auxiliary_weight=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_config_rejects_nonpositive_epsilon() -> None:
    try:
        ObjectEventV423JointLossConfig(epsilon=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
