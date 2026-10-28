"""Signed benchmark phase and finite-output contract, compatible with Stage70.

A residual is NOT constrained to the current experts' interval. Only the
registered physical/numerical output support is enforced. Zero phase maps +60s.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

MIN_PHASE = -math.log(2.0) + 1e-6
MAX_PHASE = 8.0


def ttc_to_phase(ttc: Tensor) -> Tensor:
    if not torch.isfinite(ttc).all() or not ((ttc < 0) | (ttc > 0.1)).all():
        raise ValueError("TTC must be finite with T<0 or T>0.1 seconds")
    return -torch.log1p(-0.1 / ttc)


def phase_to_ttc(phase: Tensor) -> Tensor:
    if not torch.isfinite(phase).all():
        raise ValueError("nonfinite phase")
    phi = phase.clamp(MIN_PHASE, MAX_PHASE)
    q = -torch.expm1(-phi) / 0.1
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    return sign / q.abs().clamp_min(1.0 / 60.0)


def emitted_phase(phase: Tensor) -> Tensor:
    return ttc_to_phase(phase_to_ttc(phase))


def relative_cost_targets(experts: Tensor, truth: Tensor) -> Tensor:
    """Expected-cost regression targets in MiD/100, A5 reference cost zero."""
    if experts.shape != (len(truth), 3) or truth.ndim != 1:
        raise ValueError("expect [B,3] expert phases and [B] target phases")
    if not torch.isfinite(experts).all() or not torch.isfinite(truth).all():
        raise ValueError("nonfinite cost inputs")
    losses = 100.0 * (experts - truth[:, None]).abs()
    return losses[:, 1:] - losses[:, :1]


def pinball(prediction: Tensor, target: Tensor, level: float) -> Tensor:
    if not 0 < level < 1:
        raise ValueError("quantile level must be in (0,1)")
    error = target - prediction
    return torch.maximum(level * error, (level - 1) * error)
