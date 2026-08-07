"""Combined objective for Object Event TTC v4.23."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObjectEventV423JointLossConfig:
    ttc_weight: float = 1.0
    geometry_auxiliary_weight: float = 0.25
    trainable_anchor_weight: float = 0.005
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(self.ttc_weight, self.geometry_auxiliary_weight, self.trainable_anchor_weight) < 0.0:
            raise ValueError("v4.23 loss weights must be non-negative")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


def combine_joint_losses(
    ttc_loss: torch.Tensor,
    geometry_loss: torch.Tensor,
    anchor_loss: torch.Tensor,
    *,
    config: ObjectEventV423JointLossConfig,
) -> torch.Tensor:
    return (
        config.ttc_weight * ttc_loss
        + config.geometry_auxiliary_weight * geometry_loss
        + config.trainable_anchor_weight * anchor_loss
    )


__all__ = ["ObjectEventV423JointLossConfig", "combine_joint_losses"]
