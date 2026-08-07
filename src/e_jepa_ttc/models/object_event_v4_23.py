"""Trainability controls for Object Event TTC v4.23 joint geometry + TTC."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from torch import nn

from e_jepa_ttc.models.object_event_v4_22 import select_tail_parameter_names


def configure_joint_geometry_ttc_unfreeze(backbone: nn.Module, geometry_tail_count: int) -> dict[str, list[str]]:
    """Train only the geometry tail and the existing v4.8 motion predictor.

    V4.8 normally runs the foreground model under no-grad. Switching
    ``freeze_foreground`` off is required for gradients to reach the geometry
    encoder, but all foreground parameters except the selected geometry tail
    remain frozen explicitly.
    """
    if not hasattr(backbone, "foreground_model") or not hasattr(backbone, "motion_config"):
        raise TypeError("expected an ObjectEventTTCV48-like backbone")
    if not hasattr(backbone, "temporal_projection") or not hasattr(backbone, "field_head"):
        raise TypeError("backbone lacks v4.8 motion modules")
    geometry = backbone.foreground_model.geometry_encoder
    selected_geometry = select_tail_parameter_names(geometry, geometry_tail_count)
    backbone.motion_config = replace(backbone.motion_config, freeze_foreground=False)
    backbone.requires_grad_(False)
    selected_set = set(selected_geometry)
    for name, parameter in geometry.named_parameters():
        parameter.requires_grad_(name in selected_set)
    backbone.temporal_projection.requires_grad_(True)
    backbone.field_head.requires_grad_(True)
    backbone.eval()
    return {
        "geometry": selected_geometry,
        "temporal_projection": [name for name, _ in backbone.temporal_projection.named_parameters()],
        "field_head": [name for name, _ in backbone.field_head.named_parameters()],
    }


def geometry_parameters(backbone: nn.Module) -> Iterable[nn.Parameter]:
    geometry = backbone.foreground_model.geometry_encoder
    return (parameter for parameter in geometry.parameters() if parameter.requires_grad)


def motion_parameters(backbone: nn.Module) -> Iterable[nn.Parameter]:
    modules = (backbone.temporal_projection, backbone.field_head)
    return (parameter for module in modules for parameter in module.parameters() if parameter.requires_grad)


def named_trainable_parameters(backbone: nn.Module) -> dict[str, nn.Parameter]:
    return {name: parameter for name, parameter in backbone.named_parameters() if parameter.requires_grad}


__all__ = [
    "configure_joint_geometry_ttc_unfreeze",
    "geometry_parameters",
    "motion_parameters",
    "named_trainable_parameters",
]
