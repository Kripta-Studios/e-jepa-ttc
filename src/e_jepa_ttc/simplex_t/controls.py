"""Predeclared label-free controls. Never choose a control from evaluation scores."""

from __future__ import annotations

import torch
from torch import Tensor

from .phase import emitted_phase


def history_control(x: Tensor, valid: Tensor, kind: str) -> Tensor:
    """Timing/mask stay fixed. Current features are always untouched.

    shuffle reverses all available PRIOR observations deterministically. A two-
    observation history has only one prior and cannot be perturbed: log this.
    repeat duplicates current features at each valid slot (metadata stays real).
    """
    if x.ndim != 3 or valid.shape != x.shape[:2] or kind not in {"shuffle", "repeat"}:
        raise ValueError("invalid control input")
    result = x.clone()
    for b in range(len(x)):
        ids = valid[b].nonzero().flatten()
        if len(ids) == 0 or ids[-1] != x.shape[1] - 1:
            raise ValueError("current observation must be valid")
        if kind == "repeat":
            result[b, ids] = x[b, -1]
        elif len(ids) > 2:
            result[b, ids[:-1]] = x[b, ids[:-1].flip(0)]
    return result


def ewma_phase(
    history_experts: Tensor, anchor_lags: Tensor, valid: Tensor, time_constant: float = 0.3
) -> Tensor:
    """Fixed causal smoother of expert-median PHASE, not signed TTC averaging."""
    if history_experts.shape[:2] != valid.shape or history_experts.shape[-1] != 3:
        raise ValueError("EWMA shape mismatch")
    if anchor_lags.shape != valid.shape or time_constant != 0.3 or (anchor_lags[valid] < 0).any():
        raise ValueError("invalid EWMA time setting")
    phase = (
        torch.where(valid[..., None], history_experts, torch.zeros_like(history_experts))
        .median(-1)
        .values
    )
    weights = torch.exp(-anchor_lags / time_constant) * valid
    if not (weights.sum(-1) > 0).all():
        raise ValueError("empty EWMA")
    return emitted_phase((weights * phase).sum(-1) / weights.sum(-1))
