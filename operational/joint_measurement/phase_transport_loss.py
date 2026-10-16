"""Identical ordered phase arithmetic with batched host diagnostics and guards."""

from __future__ import annotations

import torch

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE

from .geometry_capture import install_capture
from .geometry_model import GeometricPairMeasurement


def phase_terms(raw: torch.Tensor, truth: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Return the original terms and combined domain guard without host materialization."""
    if raw.ndim != 1 or raw.shape != truth.shape:
        raise ValueError("Aligned one-dimensional raw phases and TTC targets required")
    target = -torch.log1p(-0.1 / truth)
    phi = raw.clamp(MIN_PHASE, MAX_PHASE)
    q = -torch.expm1(-phi) / 0.1
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    ttc = sign / q.abs().clamp_min(1.0 / 60.0)
    emitted = -torch.log1p(-0.1 / ttc)
    valid = (
        torch.isfinite(truth).all()
        & ((truth < 0) | (truth > 0.1)).all()
        & torch.isfinite(target).all()
        & torch.isfinite(raw).all()
        & torch.isfinite(ttc).all()
        & ((ttc < 0) | (ttc > 0.1)).all()
    )
    return {
        "emitted": (emitted - target).abs() / 0.03,
        "raw": 0.1 * (raw - target).abs() / 0.03,
        "raw_emitted_gap": (raw - emitted).abs(),
    }, valid


def backward(
    model: GeometricPairMeasurement, batches: list, weights: torch.Tensor, population: int
) -> dict:
    """Keep each microbatch backward and Python-double diagnostic summation order."""
    if not batches or weights is None:
        raise ValueError("Nonempty microbatches and original population masses required")
    first = batches[0]
    if not getattr(model, "_v18_capture_active", False):
        install_capture(model, first.events, first.delta_t_s.reshape(len(first.events)))
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    size = sum(len(batch.events) for batch in batches)
    if weights.shape != (size,):
        raise ValueError("One mass per sampled query required")
    start, diagnostics = 0, []
    for batch in batches:
        n = len(batch.events)
        raw = model(batch.events, batch.delta_t_s.reshape(n))["raw_phase"]
        terms, valid = phase_terms(raw, batch.target_ttc_s)
        scaled = population * weights[start : start + n].to(raw.device)
        loss = ((terms["emitted"] + terms["raw"]) * scaled).sum() / size
        if not bool(valid & torch.isfinite(loss)):
            raise FloatingPointError("Original phase domain/finite loss guard failed")
        loss.backward()
        diagnostics.append(
            torch.stack(
                [
                    loss.detach(),
                    *[(value.detach() * scaled).sum() / size for value in terms.values()],
                ]
            )
        )
        start += n
    names = ("loss", "emitted", "raw", "raw_emitted_gap")
    result = dict.fromkeys(names, 0.0)
    for values in torch.stack(diagnostics).cpu().tolist():
        for name, value in zip(names, values, strict=True):
            result[name] += value
    return result
