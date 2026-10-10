"""Reference of the existing signed-phase convention. Integrate canonical repo version."""
from __future__ import annotations
import math
import torch
from torch import Tensor
MIN_PHASE = -math.log(2.0) + 1e-6
MAX_PHASE = 8.0

def ttc_to_phase(ttc: Tensor) -> Tensor:
    if not bool(torch.isfinite(ttc).all()) or not bool(((ttc < 0) | (ttc > 0.1)).all()):
        raise ValueError('TTC must be finite, negative or above 0.1 s')
    return -torch.log1p(-0.1 / ttc)

def phase_to_ttc(phase: Tensor) -> Tensor:
    if not bool(torch.isfinite(phase).all()):
        raise ValueError('Nonfinite phase')
    q = -torch.expm1(-phase.clamp(MIN_PHASE, MAX_PHASE)) / 0.1
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    return sign / q.abs().clamp_min(1.0 / 60.0)

def emitted_phase(phase: Tensor) -> Tensor:
    return ttc_to_phase(phase_to_ttc(phase))

def validate_initial_prior(prior: float) -> None:
    if not math.isfinite(prior) or not MIN_PHASE < prior < MAX_PHASE:
        raise ValueError('Invalid TRAIN phase prior')
    q = -math.expm1(-prior) / 0.1
    if abs(q) <= 1.0/60.0:
        raise ValueError('Prior is in the flat TTC-cap band; never initialize new direct head at zero')

def pinball(prediction: Tensor, truth: Tensor, q: float) -> Tensor:
    if not 0 < q < 1:
        raise ValueError('Invalid quantile')
    error = truth - prediction
    return torch.maximum(q * error, (q - 1) * error)

def per_row_loss(out: dict[str, Tensor], target_phase: Tensor) -> Tensor:
    if target_phase.shape != out['point_phase'].shape:
        raise ValueError('Target shape mismatch')
    point = (out['point_phase'] - target_phase).abs() / .03
    quant = (pinball(out['q10'], target_phase, .1) + pinball(out['q90'], target_phase, .9))/.03
    return point + .1*quant

def weighted_loss(out: dict[str, Tensor], target: Tensor, mass: Tensor, population: int) -> Tensor:
    if mass.shape != target.shape or population <= 0 or not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
        raise ValueError('Invalid global training mass')
    return (population * mass * per_row_loss(out, target)).mean()
