"""Objective terms for the preregistered Dense Level--Dynamics JEPA arms.

All target-facing functions are intentionally explicit about masks and detached
targets.  The module contains no TTC, geometry, category, box, RGB, or cross-track
shortcut objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch.nn import functional


class ObjectiveArm(StrEnum):
    """The only four objective arms approved for the initial pilot."""

    LEVEL = "level"
    LEVEL_TEMPORAL_RESIDUAL = "level+temporal_residual"
    LEVEL_DYNAMICS_NCE = "level+dynamics_nce"
    LEVEL_DYNAMICS_NCE_RESIDUAL_VISREG = "level+dynamics_nce+residual_visreg"

    @property
    def uses_temporal_residual(self) -> bool:
        """Return whether this arm optimizes the target-only residual prediction."""

        return self is ObjectiveArm.LEVEL_TEMPORAL_RESIDUAL

    @property
    def uses_dynamics_nce(self) -> bool:
        """Return whether this arm optimizes within-track dynamics NCE."""

        return self in {
            ObjectiveArm.LEVEL_DYNAMICS_NCE,
            ObjectiveArm.LEVEL_DYNAMICS_NCE_RESIDUAL_VISREG,
        }

    @property
    def uses_residual_visreg(self) -> bool:
        """Return whether this arm optimizes dynamics-only residual VISReg."""

        return self is ObjectiveArm.LEVEL_DYNAMICS_NCE_RESIDUAL_VISREG


@dataclass(frozen=True)
class LevelDynamicsLossConfig:
    """Weights and validity gates shared by the four fixed objective arms."""

    objective: ObjectiveArm | str = ObjectiveArm.LEVEL
    level_weight: float = 1.0
    temporal_residual_weight: float = 1.0
    dynamics_nce_weight: float = 1.0
    residual_visreg_weight: float = 0.04
    nce_temperature: float = 0.12
    nce_exclusion_window_s: float = 0.02
    nce_positive_tolerance_s: float = 1e-6
    nce_min_valid_anchor_fraction: float = 0.8
    nce_min_negatives: int = 2
    visreg_projections: int = 8
    visreg_temperature: float = 0.12

    def __post_init__(self) -> None:
        objective = ObjectiveArm(self.objective)
        object.__setattr__(self, "objective", objective)
        weights = (
            self.level_weight,
            self.temporal_residual_weight,
            self.dynamics_nce_weight,
            self.residual_visreg_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Level-Dynamics objective weights must be non-negative.")
        if self.level_weight <= 0.0:
            raise ValueError("All approved objective arms retain a positive dense level loss.")
        if self.nce_temperature <= 0.0 or self.visreg_temperature <= 0.0:
            raise ValueError("NCE and VISReg temperatures must be positive.")
        if self.nce_exclusion_window_s < 0.0 or self.nce_positive_tolerance_s < 0.0:
            raise ValueError("NCE temporal windows must be non-negative.")
        if not 0.0 <= self.nce_min_valid_anchor_fraction <= 1.0:
            raise ValueError("nce_min_valid_anchor_fraction must lie in [0,1].")
        if self.nce_min_negatives < 2:
            raise ValueError("NCE requires at least two same-track negatives by contract.")
        if self.visreg_projections <= 0:
            raise ValueError("visreg_projections must be positive.")


@dataclass
class TemporalResidualTarget:
    """Detached target-only residual and its exact aligned-patch validity mask."""

    tokens: torch.Tensor
    valid_mask: torch.Tensor
    raw_norm: torch.Tensor


@dataclass
class WithinTrackNCEOutput:
    """NCE loss plus masks/coverage diagnostics needed by the trainer health gate."""

    loss: torch.Tensor
    anchor_input_valid: torch.Tensor
    valid_anchor_mask: torch.Tensor
    positive_weights: torch.Tensor
    candidate_mask: torch.Tensor
    negative_mask: torch.Tensor
    negatives_per_anchor: torch.Tensor
    valid_anchor_fraction: torch.Tensor
    mean_negatives_per_valid_anchor: torch.Tensor
    logits: torch.Tensor


@dataclass
class ResidualVISRegOutput:
    """Dynamics-only VISReg loss and differentiable component diagnostics."""

    loss: torch.Tensor
    variance_loss: torch.Tensor
    covariance_loss: torch.Tensor
    active_token_count: torch.Tensor


@dataclass
class LevelDynamicsObjectiveOutput:
    """Weighted total loss plus the four arm-specific components."""

    loss: torch.Tensor
    level_loss: torch.Tensor
    temporal_residual_loss: torch.Tensor
    dynamics_nce_loss: torch.Tensor
    residual_visreg_loss: torch.Tensor
    residual_target: TemporalResidualTarget | None
    nce: WithinTrackNCEOutput | None
    visreg: ResidualVISRegOutput | None


def _zero_from(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _require_equal_shape(prediction: torch.Tensor, target: torch.Tensor, name: str) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} prediction/target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}."
        )
    if prediction.ndim < 2:
        raise ValueError(f"{name} tensors must have a feature dimension.")


def dense_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute dense cosine loss, L2-normalizing only at the final loss boundary."""

    _require_equal_shape(prediction, target, "dense cosine")
    if valid_mask.shape != prediction.shape[:-1] or valid_mask.dtype != torch.bool:
        raise ValueError(
            "valid_mask must be bool and match all prediction axes except feature dim."
        )
    if not bool(valid_mask.any()):
        return _zero_from(prediction)
    normalized_prediction = functional.normalize(prediction[valid_mask], dim=-1)
    normalized_target = functional.normalize(target.detach()[valid_mask], dim=-1)
    return (1.0 - (normalized_prediction * normalized_target).sum(dim=-1)).mean()


def build_temporal_residual_target(
    reference_target_dynamics: torch.Tensor,
    future_target_dynamics: torch.Tensor,
    reference_valid_patch_mask: torch.Tensor,
    future_valid_patch_mask: torch.Tensor,
) -> TemporalResidualTarget:
    """Construct exactly ``stopgrad(LN(future) - LN(reference))`` on aligned patches."""

    if reference_target_dynamics.ndim != 3 or future_target_dynamics.ndim != 4:
        raise ValueError("reference/future target dynamics must be [B,P,D] and [B,H,P,D].")
    batch, patches, dim = reference_target_dynamics.shape
    if future_target_dynamics.shape[0] != batch or future_target_dynamics.shape[2:] != (
        patches,
        dim,
    ):
        raise ValueError("Future target dynamics must align with reference [B,P,D] axes.")
    horizons = future_target_dynamics.shape[1]
    if (
        reference_valid_patch_mask.shape != (batch, patches)
        or reference_valid_patch_mask.dtype != torch.bool
    ):
        raise ValueError("reference_valid_patch_mask must be bool [B,P].")
    if (
        future_valid_patch_mask.shape != (batch, horizons, patches)
        or future_valid_patch_mask.dtype != torch.bool
    ):
        raise ValueError("future_valid_patch_mask must be bool [B,H,P].")
    reference = functional.layer_norm(reference_target_dynamics, (dim,))
    future = functional.layer_norm(future_target_dynamics, (dim,))
    residual = (future - reference.unsqueeze(1)).detach()
    valid = future_valid_patch_mask & reference_valid_patch_mask.unsqueeze(1)
    residual = residual.masked_fill(~valid.unsqueeze(-1), 0.0)
    raw_norm = residual.norm(dim=-1).detach()
    return TemporalResidualTarget(tokens=residual, valid_mask=valid, raw_norm=raw_norm)


def temporal_residual_cosine_loss(
    predicted_residual: torch.Tensor,
    residual_target: TemporalResidualTarget,
) -> torch.Tensor:
    """Compare direct residual predictions to the detached target-only residual."""

    return dense_cosine_loss(predicted_residual, residual_target.tokens, residual_target.valid_mask)


def build_horizon_positive_weights(
    anchor_reference_timestamps_s: torch.Tensor,
    candidate_timestamps_s: torch.Tensor,
    horizon_delta_t_s: torch.Tensor,
    anchor_sequence_ids: Sequence[str],
    candidate_sequence_ids: Sequence[str],
    anchor_track_ids: Sequence[str],
    candidate_track_ids: Sequence[str],
    *,
    tolerance_s: float,
    distance_weighting: bool = True,
) -> torch.Tensor:
    """Return a multiple-positive matrix for correct same-track future-horizon matches."""

    if tolerance_s < 0.0:
        raise ValueError("tolerance_s must be non-negative.")
    if anchor_reference_timestamps_s.ndim != 1 or candidate_timestamps_s.ndim != 1:
        raise ValueError("NCE timestamps must be one-dimensional.")
    anchors = anchor_reference_timestamps_s.numel()
    candidates = candidate_timestamps_s.numel()
    if horizon_delta_t_s.shape != (anchors,):
        raise ValueError("horizon_delta_t_s must have one value per NCE anchor.")
    if not (
        len(anchor_sequence_ids) == len(anchor_track_ids) == anchors
        and len(candidate_sequence_ids) == len(candidate_track_ids) == candidates
    ):
        raise ValueError("NCE identity metadata must align with timestamp tensors.")
    same_track = torch.tensor(
        [
            [
                anchor_sequence_ids[anchor] == candidate_sequence_ids[candidate]
                and anchor_track_ids[anchor] == candidate_track_ids[candidate]
                for candidate in range(candidates)
            ]
            for anchor in range(anchors)
        ],
        device=anchor_reference_timestamps_s.device,
        dtype=torch.bool,
    )
    desired_future = anchor_reference_timestamps_s + horizon_delta_t_s
    distance = (candidate_timestamps_s.unsqueeze(0) - desired_future.unsqueeze(1)).abs()
    matches = same_track & (distance <= tolerance_s)
    if distance_weighting:
        scale = max(tolerance_s, torch.finfo(distance.dtype).eps)
        weights = 1.0 / (1.0 + distance / scale)
    else:
        weights = torch.ones_like(distance)
    return weights * matches.to(weights.dtype)


def within_track_nce_loss(
    predicted_embeddings: torch.Tensor,
    candidate_target_embeddings: torch.Tensor,
    positive_weights: torch.Tensor,
    anchor_sequence_ids: Sequence[str],
    candidate_sequence_ids: Sequence[str],
    anchor_track_ids: Sequence[str],
    candidate_track_ids: Sequence[str],
    anchor_desired_future_timestamps_s: torch.Tensor,
    candidate_timestamps_s: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    candidate_valid: torch.Tensor | None = None,
    anchor_valid: torch.Tensor | None = None,
    exclusion_window_s: float,
    temperature: float,
    min_negatives: int = 2,
) -> WithinTrackNCEOutput:
    """Compute multiple-positive NCE with strict same-sequence/track negatives only.

    ``anchor_desired_future_timestamps_s`` is the requested future endpoint for
    each anchor.  The exclusion window is centered on that endpoint, never on the
    reference/context timestamp.  The caller's candidate mask may remove negative
    candidates, but it cannot remove explicitly weighted positives.  Tracks without
    enough valid negatives, or anchors without valid context support, are reported
    and excluded; they never borrow candidates from another track.
    """

    if predicted_embeddings.ndim != 2 or candidate_target_embeddings.ndim != 2:
        raise ValueError("NCE embeddings must have shapes [A,D] and [C,D].")
    anchors, dim = predicted_embeddings.shape
    candidates, candidate_dim = candidate_target_embeddings.shape
    if dim != candidate_dim:
        raise ValueError("NCE prediction and candidate feature dimensions must match.")
    if positive_weights.shape != (anchors, candidates):
        raise ValueError("positive_weights must have shape [A,C].")
    if anchor_desired_future_timestamps_s.shape != (anchors,) or candidate_timestamps_s.shape != (
        candidates,
    ):
        raise ValueError(
            "NCE desired-future/candidate timestamps must align with anchors/candidates."
        )
    if not (
        len(anchor_sequence_ids) == len(anchor_track_ids) == anchors
        and len(candidate_sequence_ids) == len(candidate_track_ids) == candidates
    ):
        raise ValueError("NCE identity metadata must align with anchors/candidates.")
    if temperature <= 0.0 or exclusion_window_s < 0.0 or min_negatives < 0:
        raise ValueError("Invalid NCE temperature, exclusion window or minimum negatives.")
    if (positive_weights < 0).any():
        raise ValueError("positive_weights must be non-negative.")
    if candidate_mask is None:
        candidate_mask = torch.ones(
            anchors,
            candidates,
            dtype=torch.bool,
            device=predicted_embeddings.device,
        )
    if candidate_mask.shape != (anchors, candidates) or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be bool [A,C].")
    if candidate_valid is None:
        candidate_valid = torch.ones(
            candidates, dtype=torch.bool, device=predicted_embeddings.device
        )
    if candidate_valid.shape != (candidates,) or candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be bool [C].")
    if anchor_valid is None:
        anchor_valid = torch.ones(anchors, dtype=torch.bool, device=predicted_embeddings.device)
    if anchor_valid.shape != (anchors,) or anchor_valid.dtype != torch.bool:
        raise ValueError("anchor_valid must be bool [A].")

    same_track = torch.tensor(
        [
            [
                anchor_sequence_ids[anchor] == candidate_sequence_ids[candidate]
                and anchor_track_ids[anchor] == candidate_track_ids[candidate]
                for candidate in range(candidates)
            ]
            for anchor in range(anchors)
        ],
        device=predicted_embeddings.device,
        dtype=torch.bool,
    )
    positive = positive_weights.to(
        device=predicted_embeddings.device, dtype=predicted_embeddings.dtype
    )
    positive = positive * candidate_valid.unsqueeze(0).to(positive.dtype)
    if bool(((positive > 0.0) & ~same_track).any()):
        raise ValueError("NCE positives must share both sequence and track identity.")
    temporal_distance = (
        candidate_timestamps_s.to(predicted_embeddings.device).unsqueeze(0)
        - anchor_desired_future_timestamps_s.to(predicted_embeddings.device).unsqueeze(1)
    ).abs()
    outside_exclusion = temporal_distance > exclusion_window_s
    negative_mask = (
        same_track
        & outside_exclusion
        & candidate_mask
        & candidate_valid.unsqueeze(0)
        & ~(positive > 0.0)
    )
    # Positive mass is deliberately OR-ed after candidate/exclusion filtering.  This
    # is the guard against an exclusion mask degenerating NCE into no positive term.
    final_candidate_mask = negative_mask | (positive > 0.0)
    negatives_per_anchor = negative_mask.sum(dim=1)
    positive_mass = positive.sum(dim=1)
    valid_anchor_mask = (
        anchor_valid.to(predicted_embeddings.device)
        & (positive_mass > 0.0)
        & (negatives_per_anchor >= min_negatives)
    )
    normalized_prediction = functional.normalize(predicted_embeddings, dim=-1)
    normalized_candidates = functional.normalize(candidate_target_embeddings.detach(), dim=-1)
    logits = normalized_prediction @ normalized_candidates.transpose(0, 1) / temperature
    if bool(valid_anchor_mask.any()):
        negative_inf = torch.finfo(logits.dtype).min
        denominator = torch.logsumexp(
            logits.masked_fill(~final_candidate_mask, negative_inf), dim=1
        )
        numerator = torch.logsumexp(
            (logits + positive.clamp_min(torch.finfo(logits.dtype).tiny).log()).masked_fill(
                ~(positive > 0.0), negative_inf
            ),
            dim=1,
        )
        loss = (denominator[valid_anchor_mask] - numerator[valid_anchor_mask]).mean()
    else:
        loss = _zero_from(predicted_embeddings)
    valid_fraction = valid_anchor_mask.to(predicted_embeddings.dtype).mean()
    mean_negatives = (
        negatives_per_anchor[valid_anchor_mask].to(predicted_embeddings.dtype).mean()
        if bool(valid_anchor_mask.any())
        else predicted_embeddings.new_zeros(())
    )
    return WithinTrackNCEOutput(
        loss=loss,
        anchor_input_valid=anchor_valid.detach(),
        valid_anchor_mask=valid_anchor_mask,
        positive_weights=positive,
        candidate_mask=final_candidate_mask,
        negative_mask=negative_mask,
        negatives_per_anchor=negatives_per_anchor,
        valid_anchor_fraction=valid_fraction.detach(),
        mean_negatives_per_valid_anchor=mean_negatives.detach(),
        logits=logits,
    )


def validate_nce_preflight(
    nce: WithinTrackNCEOutput,
    *,
    minimum_valid_anchor_fraction: float = 0.8,
    minimum_negatives: int = 2,
) -> None:
    """Fail before an optimizer step if the approved NCE coverage contract is unmet."""

    valid_fraction = float(nce.valid_anchor_fraction.cpu())
    if valid_fraction < minimum_valid_anchor_fraction:
        raise RuntimeError(
            "NCE preflight failed: valid-anchor fraction "
            f"{valid_fraction:.3f} is below required {minimum_valid_anchor_fraction:.3f}."
        )
    if bool(nce.valid_anchor_mask.any()):
        minimum_observed = int(nce.negatives_per_anchor[nce.valid_anchor_mask].min().cpu())
    else:
        minimum_observed = 0
    if minimum_observed < minimum_negatives:
        raise RuntimeError(
            "NCE preflight failed: minimum valid-anchor negatives "
            f"{minimum_observed} is below required {minimum_negatives}."
        )


def masked_mean_time(dynamics_tokens: torch.Tensor, valid_patch_mask: torch.Tensor) -> torch.Tensor:
    """Return a per-sample, per-patch temporal mean without introducing label state."""

    if dynamics_tokens.ndim != 4 or valid_patch_mask.shape != dynamics_tokens.shape[:3]:
        raise ValueError("dynamics_tokens/mask must be [B,T,P,D] and [B,T,P].")
    weights = valid_patch_mask.to(dynamics_tokens.dtype).unsqueeze(-1)
    return (dynamics_tokens * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)


def residual_visreg_loss(
    dynamics_tokens: torch.Tensor,
    valid_patch_mask: torch.Tensor,
    *,
    generator: torch.Generator,
    projections: int,
    temperature: float,
) -> ResidualVISRegOutput:
    """Apply finite differentiable VISReg only to time-centred dynamics tokens."""

    if projections <= 0 or temperature <= 0.0:
        raise ValueError("VISReg projections and temperature must be positive.")
    centred = dynamics_tokens - masked_mean_time(dynamics_tokens, valid_patch_mask)
    active = centred[valid_patch_mask]
    active_count = torch.tensor(active.shape[0], device=centred.device, dtype=torch.int64)
    if active.shape[0] < 2:
        zero = _zero_from(dynamics_tokens)
        return ResidualVISRegOutput(zero, zero, zero, active_count)
    projection = torch.randn(
        active.shape[1],
        projections,
        device=active.device,
        dtype=active.dtype,
        generator=generator,
    )
    projection = functional.normalize(projection, dim=0)
    projected = active @ projection / temperature
    standard_deviation = projected.std(dim=0, unbiased=False)
    variance_loss = functional.relu(1.0 - standard_deviation).mean()
    covariance = (projected.transpose(0, 1) @ projected) / max(projected.shape[0] - 1, 1)
    diagonal = torch.diagonal(covariance)
    covariance_loss = (covariance - torch.diag(diagonal)).square().mean()
    return ResidualVISRegOutput(
        loss=variance_loss + covariance_loss,
        variance_loss=variance_loss,
        covariance_loss=covariance_loss,
        active_token_count=active_count,
    )


__all__ = [
    "LevelDynamicsLossConfig",
    "LevelDynamicsObjectiveOutput",
    "ObjectiveArm",
    "ResidualVISRegOutput",
    "TemporalResidualTarget",
    "WithinTrackNCEOutput",
    "build_horizon_positive_weights",
    "build_temporal_residual_target",
    "dense_cosine_loss",
    "masked_mean_time",
    "residual_visreg_loss",
    "temporal_residual_cosine_loss",
    "validate_nce_preflight",
    "within_track_nce_loss",
]
