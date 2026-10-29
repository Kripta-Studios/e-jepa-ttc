"""Preregistered phase transport, without optimizer updates or target inputs."""

import torch
from torch import Tensor

from e_jepa_ttc.simplex_t.phase import emitted_phase


def transport_ewma(phases: Tensor, ages: Tensor, valid: Tensor) -> tuple[Tensor, dict]:
    """Transport raw median phases under constant velocity, then aggregate in phase."""
    if phases.shape != ages.shape or valid.shape != ages.shape or phases.ndim != 2:
        raise ValueError("matching [B,8] phase/age/mask required")
    if phases.shape[1] != 8 or not valid[:, -1].all():
        raise ValueError("eight slots and valid present required")
    if not torch.isfinite(phases[valid]).all() or (ages[valid] < 0).any():
        raise ValueError("finite phases and nonnegative ages required")
    p = phases.double()
    q = -torch.expm1(-p) / 0.1
    denom = 1 - ages.double() * q
    # Positive estimates crossing contact/domain are discarded as TERMS only.
    accepted = valid & ((q <= 0) | ((denom > 0) & (q / denom < 10)))
    if not accepted[:, -1].all():
        raise ValueError("present term outside canonical physical domain")
    transported_q = torch.where(accepted, q / denom, 0)
    transported_phase = -torch.log1p(-0.1 * transported_q)
    weights = torch.exp(-ages.double() / 0.3) * accepted
    point = emitted_phase((weights * transported_phase).sum(1) / weights.sum(1))
    return point, {
        "rejected_terms": int((valid & ~accepted).sum()),
        "valid_terms": int(valid.sum()),
        "queries": len(point),
        "zero_phase_terms": int(((p == 0) & valid).sum()),
        "optimizer_updates": 0,
    }
