"""One isolated loss intervention, retaining native architecture and emission.

L0 is the existing loss exactly. L1 adds a dimensionless symmetric relative-TTC
SmoothL1 term. It does NOT replace phase, quantiles, costs, sign policy or cap.
The extra term cannot fix a zero-gradient decoder saturation; audit this rather
than inserting an undeclared straight-through gradient.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.nn import functional as F
from e_jepa_ttc.simplex_t.model import training_loss
from e_jepa_ttc.simplex_t.phase import phase_to_ttc


def range_transfer_loss(
    output: dict[str, Tensor], truth_phase: Tensor, experts: Tensor,
    mass: Tensor, population: int, *, relative_weight: float = 0.0
) -> Tensor:
    """Uniform-query sampling; use global mass, never batch renormalization.

    The protocol fixes relative_weight=0 or 0.25 before training. The target TTC
    is reconstructed from the same target phase, without output clipping.
    This helper is tested code, not a trainer/admission or lineage validator.
    """
    if not math.isfinite(relative_weight) or relative_weight < 0:
        raise ValueError('relative_weight must be finite and nonnegative')
    base = training_loss(output, truth_phase, experts, mass, population)
    if relative_weight == 0:
        return base
    if not torch.isfinite(truth_phase).all() or (truth_phase == 0).any():
        raise ValueError('Finite nonzero target phase required for finite TTC labels')
    target_ttc = -0.1 / torch.expm1(-truth_phase)
    predicted_ttc = phase_to_ttc(output['raw_location'])
    if not torch.isfinite(target_ttc).all() or not torch.isfinite(predicted_ttc).all():
        raise FloatingPointError('Nonfinite canonical TTC in added objective')
    residual = (predicted_ttc - target_ttc) / target_ttc.abs().clamp_min(0.5)
    per_row = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none')
    return base + relative_weight * (population * mass * per_row).mean()
