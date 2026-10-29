"""Stop-gradient EMA target encoder update."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def update_target_encoder(
    target: nn.Module,
    online: nn.Module,
    momentum: float,
) -> None:
    """Update target parameters with ``target = m*target + (1-m)*online``."""

    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must lie in [0,1).")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("Target and online modules have different parameter names.")
    for name, target_parameter in target_parameters.items():
        target_parameter.mul_(momentum).add_(online_parameters[name], alpha=1.0 - momentum)
    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError("Target and online modules have different buffer names.")
    for name, target_buffer in target_buffers.items():
        target_buffer.copy_(online_buffers[name])


__all__ = ["update_target_encoder"]
