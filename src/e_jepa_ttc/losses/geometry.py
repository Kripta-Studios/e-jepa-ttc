"""Optional geometry consistency loss, kept separate from the main JEPA loss."""

from __future__ import annotations

import torch


def expansion_consistency_loss(
    inverse_ttc: torch.Tensor,
    expansion_rate: torch.Tensor,
) -> torch.Tensor:
    """Penalize a non-positive expansion signal only when TTC is approaching."""

    if inverse_ttc.shape != expansion_rate.shape:
        raise ValueError("inverse_ttc and expansion_rate must have equal shapes.")
    approaching = inverse_ttc.detach() > 0
    penalty = torch.relu(-expansion_rate)
    if not bool(approaching.any()):
        return penalty.sum() * 0.0
    return penalty[approaching].mean()


__all__ = ["expansion_consistency_loss"]
