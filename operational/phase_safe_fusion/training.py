"""Training-only expert reliability supervision; no target inputs at inference."""

import torch

from operational.phase_safe_fusion.model import SafeFusion


def gate_risk(
    model: SafeFusion,
    native: torch.Tensor,
    geometry: torch.Tensor,
    quality: torch.Tensor,
    target: torch.Tensor,
    seconds_scale: torch.Tensor,
) -> torch.Tensor:
    """Expected bounded expert error gives a gradient for a wrong sign choice."""
    experts = torch.cat((native, geometry[:, None]), 1)
    available = torch.isfinite(experts) & (experts.abs() > 0)
    available[:, 3] = available[:, 3] & torch.isfinite(quality).all(1)
    safe = torch.where(available, experts, torch.zeros_like(experts)).clamp(-60, 60)
    x = (
        (model.inputs(native, geometry, quality) - model.features.center) / model.features.scale
    ).clamp(-8, 8)
    logits = model.features.gate(x).masked_fill(~available, -10000)
    weights = torch.softmax(logits, 1) * available.to(logits.dtype)
    error = (safe - target[:, None]).abs()
    cost = (error / target[:, None].abs().clamp(min=0.2) + error / seconds_scale).clamp(max=10)
    history = torch.isfinite(quality).all(1) & (quality[:, 9] > 0) & (quality[:, 11] <= 0.15)
    return torch.where(history, (weights * cost).sum(1), torch.zeros_like(target))
