"""Partial-unfreeze controls for Object Event TTC v4.22."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ObjectEventV422Config:
    search_radius: int = 4
    correlation_temperature: float = 0.07
    foreground_floor: float = 0.05
    confidence_floor: float = 0.05
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.search_radius <= 0:
            raise ValueError("search_radius must be positive")
        if min(
            self.correlation_temperature,
            self.foreground_floor,
            self.confidence_floor,
            self.epsilon,
        ) <= 0.0:
            raise ValueError("v4.22 numerical controls must be positive")


def select_tail_parameter_names(module: nn.Module, count: int) -> list[str]:
    """Return the final `count` trainable parameter tensors in registration order."""
    if count <= 0:
        raise ValueError("count must be positive")
    names = [name for name, _ in module.named_parameters()]
    if not names:
        raise ValueError("module has no parameters")
    if count > len(names):
        raise ValueError(f"requested {count} tensors but module only has {len(names)}")
    return names[-count:]


def configure_partial_geometry_unfreeze(backbone: nn.Module, count: int) -> list[str]:
    """Allow gradients through v4.8 geometry maps while training only a tail subset.

    The v4.8 helper places `_foreground_and_features` under no-grad when
    `motion_config.freeze_foreground` is true. We switch that control off, keep
    the entire backbone frozen, and then selectively enable only the final
    geometry-encoder parameter tensors. The model remains in eval mode, so no
    other training-time behaviour changes.
    """
    if not hasattr(backbone, "foreground_model") or not hasattr(backbone, "motion_config"):
        raise TypeError("expected an ObjectEventTTCV48-like backbone")
    geometry = backbone.foreground_model.geometry_encoder
    selected = select_tail_parameter_names(geometry, count)
    backbone.motion_config = replace(backbone.motion_config, freeze_foreground=False)
    backbone.requires_grad_(False)
    selected_set = set(selected)
    for name, parameter in geometry.named_parameters():
        parameter.requires_grad_(name in selected_set)
    backbone.eval()
    return selected


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "ObjectEventV422Config",
    "configure_partial_geometry_unfreeze",
    "select_tail_parameter_names",
    "trainable_parameters",
]
