"""Opt-in differentiable research blocks; auxiliary losses are training-only."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc


def attenuate_correction(
    rgb_raw: Tensor,
    original_raw: Tensor,
    original_widths: Tensor,
    weight: Tensor,
) -> dict[str, Tensor]:
    """O1 point intervention, retaining widths without claiming interval calibration.

    Use captured raw locations, never inverted saturated TTC. Endpoints return
    original tensor values exactly, including near the emission boundary.
    """
    if rgb_raw.ndim != 1 or original_raw.shape != rgb_raw.shape or weight.shape != rgb_raw.shape:
        raise ValueError("Expected matching [B] raw locations and attenuation")
    if original_widths.shape != (len(rgb_raw), 2):
        raise ValueError("Expected positive phase widths [B,2]")
    if not all(torch.isfinite(t).all() for t in (rgb_raw, original_raw, original_widths, weight)):
        raise ValueError("Nonfinite fusion input")
    if (original_widths < 0).any() or ((weight < 0) | (weight > 1)).any():
        raise ValueError("Widths must be nonnegative and attenuation in [0,1]")
    raw = rgb_raw + weight * (original_raw - rgb_raw)
    raw = torch.where(weight == 1, original_raw, torch.where(weight == 0, rgb_raw, raw))
    return {
        "raw_location": raw,
        "ttc": phase_to_ttc(raw),
        "point_phase": emitted_phase(raw),
        "q10": (raw - original_widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
        "q90": (raw + original_widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
        "widths_uncalibrated": original_widths,
    }


class CorrectionGate(nn.Module):
    """Small O1 gate; six standardized inference indicators plus missingness masks."""

    def __init__(self, indicators: int = 6) -> None:
        super().__init__()
        if not 1 <= indicators <= 6:
            raise ValueError("Screen permits at most six new indicators")
        self.indicators = indicators
        self.linear = nn.Linear(2 * indicators, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, values: Tensor, available: Tensor) -> Tensor:
        """Missing indicators have zero imputation and an explicit boolean mask."""
        if (
            values.ndim != 2
            or values.shape[1] != self.indicators
            or available.shape != values.shape
        ):
            raise ValueError("Gate indicator shape mismatch")
        if available.dtype != torch.bool or not torch.isfinite(values[available]).all():
            raise ValueError("Invalid gate availability/values")
        filled = torch.where(available, values, torch.zeros_like(values))
        return (
            self.linear(torch.cat((filled, available.to(values.dtype)), -1)).squeeze(-1).sigmoid()
        )


def sensor_increment(
    q: Tensor,
    xy: Tensor,
    gradient: Tensor,
    dt: Tensor,
    translation: Tensor,
    illumination_rate: Tensor,
) -> Tensor:
    """First-order local log-intensity increment tied directly to q.

    Coordinates and gradient share a fixed sensor frame. Texture and calibrated
    thresholds are external anchors, not unconstrained learned per-pixel fields.
    This local linearization is not a full timestamp likelihood or RAW simulator.
    """
    if xy.ndim != 3 or xy.shape[-1] != 2 or gradient.shape != xy.shape:
        raise ValueError("Expected coordinates and log-texture gradient [B,N,2]")
    batch, count = xy.shape[:2]
    if q.shape != (batch,) or dt.shape != (batch, count) or translation.shape != (batch, 2):
        raise ValueError("Sensor motion/time shapes differ")
    if illumination_rate.shape != (batch,):
        raise ValueError("Illumination is one scalar per patch, not a free pixel field")
    if not all(
        torch.isfinite(t).all() for t in (q, xy, gradient, dt, translation, illumination_rate)
    ):
        raise ValueError("Nonfinite sensor state")
    if (dt <= 0).any():
        raise ValueError("Sensor intervals must be positive")
    flow = q[:, None, None] * xy + translation[:, None, :]
    return dt * (illumination_rate[:, None] - (gradient * flow).sum(-1))


def sensor_interval_loss(
    predicted: Tensor,
    positive_count: Tensor,
    negative_count: Tensor,
    c_positive: float,
    c_negative: float,
    endpoint_tolerance: Tensor,
    visible: Tensor,
    covered: Tensor,
    reference_known: Tensor,
    *,
    minimum_coverage: float = 0.25,
) -> tuple[Tensor, dict[str, int | float]]:
    """Robust threshold-band loss on observed intervals with an anchored reference.

    Dropout/dead/refractory intervals must be uncovered. Silence is not used as
    q=0 supervision. Arbitrary interval endpoint residuals need explicit tolerance.
    """
    tensors = (
        positive_count,
        negative_count,
        endpoint_tolerance,
        visible,
        covered,
        reference_known,
    )
    if predicted.numel() == 0 or any(t.shape != predicted.shape for t in tensors):
        raise ValueError("Sensor interval arrays must be nonempty and aligned")
    if not all(t.dtype == torch.bool for t in (visible, covered, reference_known)):
        raise ValueError("Sensor admission masks must be boolean")
    if not all(math.isfinite(c) and c > 0 for c in (c_positive, c_negative)):
        raise ValueError("Use positive fixed calibrated polarity thresholds")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("Minimum coverage must be in (0,1]")
    if not all(torch.isfinite(t).all() for t in (predicted, *tensors[:3])):
        raise ValueError("Nonfinite sensor loss input")
    if any((t < 0).any() for t in tensors[:3]):
        raise ValueError("Counts and endpoint tolerances must be nonnegative")
    if any((t != t.round()).any() for t in (positive_count, negative_count)):
        raise ValueError("Event counts must be integral")
    valid = visible & covered & reference_known & ((positive_count + negative_count) > 0)
    admitted = int(valid.sum())
    coverage = admitted / predicted.numel()
    if coverage < minimum_coverage:
        raise ValueError("Insufficient anchored observed-event coverage; not a zero loss")
    observed = c_positive * positive_count - c_negative * negative_count
    residual = (predicted - observed).abs().sub(endpoint_tolerance).clamp_min(0)[valid]
    loss = functional.huber_loss(residual, torch.zeros_like(residual), delta=0.1)
    return loss, {"required": predicted.numel(), "admitted": admitted, "coverage": coverage}


class TimeConditionedPredictor(nn.Module):
    """Conditional JEPA building block, only admitted after the producer diagnostic gate."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 2:
            raise ValueError("Embedding dimension must be at least two")
        self.network = nn.Sequential(
            nn.Linear(dimension + 3, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
        )

    def forward(self, context: Tensor, horizon_s: Tensor) -> Tensor:
        """Condition on real elapsed seconds; future embeddings are targets only."""
        if context.ndim != 2 or horizon_s.shape != (len(context),):
            raise ValueError("Expected context[B,D] and horizon[B]")
        if (
            not torch.isfinite(context).all()
            or not torch.isfinite(horizon_s).all()
            or (horizon_s <= 0).any()
        ):
            raise ValueError("Nonfinite context or invalid future horizon")
        t = horizon_s[:, None]
        return self.network(torch.cat((context, t, torch.sin(t), torch.cos(t)), -1))


def anchored_predictive_loss(
    prediction: Tensor,
    target_embedding: Tensor,
    predicted_observable: Tensor,
    observed_future: Tensor,
    valid: Tensor,
    *,
    observable_weight: float = 1.0,
) -> Tensor:
    """JEPA auxiliary with stop-gradient target and independent physical observations.

    The caller owns EMA, collapse monitoring and task loss. No TTC labels enter
    this function; a supervised teacher must still be declared in lineage.
    """
    if prediction.ndim != 2 or prediction.shape != target_embedding.shape:
        raise ValueError("Embedding shapes differ")
    if (
        predicted_observable.shape != (len(prediction),)
        or observed_future.shape != predicted_observable.shape
    ):
        raise ValueError("Observable shape mismatch")
    if valid.shape != predicted_observable.shape or valid.dtype != torch.bool or not valid.any():
        raise ValueError("Require independent valid future observations")
    if not math.isfinite(observable_weight) or observable_weight <= 0:
        raise ValueError("Physical anchor must have a positive finite weight")
    if not all(
        torch.isfinite(t[valid]).all()
        for t in (prediction, target_embedding, predicted_observable, observed_future)
    ):
        raise ValueError("Nonfinite predictive target")
    latent = 1 - functional.cosine_similarity(prediction[valid], target_embedding[valid].detach())
    physical = functional.huber_loss(predicted_observable[valid], observed_future[valid].detach())
    return latent.mean() + observable_weight * physical
