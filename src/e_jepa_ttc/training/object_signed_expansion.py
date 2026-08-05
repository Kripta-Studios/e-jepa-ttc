"""Objectives for continuous signed expansion with JEPA dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.data.object_signed_expansion import ObjectSignedExpansionBatch
from e_jepa_ttc.models.object_signed_expansion import ObjectSignedExpansionOutput


@dataclass(frozen=True)
class ObjectSignedExpansionLossConfig:
    """Auditable geometry + latent-prediction objective."""

    geometry_warmup_epochs: int = 3
    expansion_weight: float = 4.0
    visible_ratio_weight: float = 2.0
    official_ratio_weight: float = 2.0
    geometry_prior_weight: float = 0.5
    sign_margin_weight: float = 0.5
    ttc_aux_weight: float = 0.05
    latent_prediction_weight: float = 0.2
    latent_variance_weight: float = 0.02
    activity_reconstruction_weight: float = 0.05
    ordered_swap_weight: float = 0.05
    negative_sample_weight: float = 2.0
    crucial_sample_weight: float = 1.5
    small_sample_weight: float = 1.0
    large_sample_weight: float = 1.0
    expansion_beta: float = 0.01
    ratio_beta: float = 0.01
    ttc_aux_beta: float = 0.05
    sign_temperature: float = 0.02
    geometry_disagreement_scale: float = 0.02
    variance_floor: float = 0.25

    def __post_init__(self) -> None:
        if self.geometry_warmup_epochs < 0:
            raise ValueError("geometry_warmup_epochs must be non-negative")
        weights = (
            self.expansion_weight,
            self.visible_ratio_weight,
            self.official_ratio_weight,
            self.geometry_prior_weight,
            self.sign_margin_weight,
            self.ttc_aux_weight,
            self.latent_prediction_weight,
            self.latent_variance_weight,
            self.activity_reconstruction_weight,
            self.ordered_swap_weight,
            self.negative_sample_weight,
            self.crucial_sample_weight,
            self.small_sample_weight,
            self.large_sample_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("Loss weights must be non-negative")
        if min(self.expansion_beta, self.ratio_beta, self.ttc_aux_beta) <= 0.0:
            raise ValueError("Smooth-L1 betas must be positive")
        if self.sign_temperature <= 0.0:
            raise ValueError("sign_temperature must be positive")
        if self.geometry_disagreement_scale <= 0.0:
            raise ValueError("geometry_disagreement_scale must be positive")
        if not 0.0 <= self.variance_floor < 1.0:
            raise ValueError("variance_floor must lie in [0,1)")


@dataclass
class ObjectSignedExpansionTargets:
    signed_expansion: torch.Tensor
    signed_inverse_ttc: torch.Tensor
    official_log_ratio: torch.Tensor
    visible_log_ratio: torch.Tensor
    visible_expansion: torch.Tensor
    sample_weights: torch.Tensor
    target_sign: torch.Tensor


@dataclass
class ObjectSignedExpansionLossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    phase: str


def curriculum_phase(
    epoch: int,
    config: ObjectSignedExpansionLossConfig,
) -> str:
    if epoch <= 0:
        raise ValueError("epoch must be one-indexed and positive")
    if epoch <= config.geometry_warmup_epochs:
        return "geometry_jepa_warmup"
    return "full_signed_expansion"


def signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return value.sign() * torch.log1p(value.abs())


def _sample_weights(
    ttc: torch.Tensor,
    config: ObjectSignedExpansionLossConfig,
) -> torch.Tensor:
    weights = torch.full_like(ttc, config.large_sample_weight)
    weights = torch.where(
        (ttc > 0.0) & (ttc <= 3.0),
        torch.full_like(ttc, config.crucial_sample_weight),
        weights,
    )
    weights = torch.where(
        (ttc > 3.0) & (ttc <= 6.0),
        torch.full_like(ttc, config.small_sample_weight),
        weights,
    )
    weights = torch.where(
        ttc < 0.0,
        torch.full_like(ttc, config.negative_sample_weight),
        weights,
    )
    return weights / weights.mean().clamp_min(1.0e-6)


def targets_from_batch(
    batch: ObjectSignedExpansionBatch,
    config: ObjectSignedExpansionLossConfig,
) -> ObjectSignedExpansionTargets:
    if not torch.isfinite(batch.target_ttc_s).all():
        raise ValueError("TTC targets must be finite")
    if bool((batch.target_ttc_s == 0).any()):
        raise ValueError("TTC targets must be non-zero")
    expansion = batch.delta_t_s / batch.target_ttc_s
    if bool((expansion >= 1.0).any()):
        raise ValueError("Official TTC lies outside the LHR domain")
    official_ratio = torch.log1p(-expansion)
    visible_ratio = (
        torch.log(batch.visible_heights_px[:, 0].clamp_min(1.0e-6))
        - torch.log(batch.visible_heights_px[:, 1].clamp_min(1.0e-6))
    )
    visible_expansion = 1.0 - torch.exp(visible_ratio)
    return ObjectSignedExpansionTargets(
        signed_expansion=expansion,
        signed_inverse_ttc=batch.target_ttc_s.reciprocal(),
        official_log_ratio=official_ratio,
        visible_log_ratio=visible_ratio,
        visible_expansion=visible_expansion,
        sample_weights=_sample_weights(batch.target_ttc_s, config),
        target_sign=batch.target_ttc_s.sign(),
    )


def _weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    losses = functional.smooth_l1_loss(
        prediction,
        target,
        beta=beta,
        reduction="none",
    )
    return (losses * weights).sum() / weights.sum().clamp_min(1.0e-6)


def _latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target = target.detach()
    cosine = 1.0 - functional.cosine_similarity(prediction, target, dim=-1)
    prediction_norm = functional.layer_norm(prediction, prediction.shape[-1:])
    target_norm = functional.layer_norm(target, target.shape[-1:])
    mse = (prediction_norm - target_norm).square().mean(dim=-1)
    return (cosine + 0.25 * mse).mean()


def _variance_floor_loss(
    embeddings: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    if embeddings.ndim != 3:
        raise ValueError("endpoint embeddings must have shape [B,T,D]")
    flattened = embeddings.reshape(-1, embeddings.shape[-1])
    std = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1.0e-4)
    return functional.relu(floor - std).mean()


def object_signed_expansion_loss(
    output: ObjectSignedExpansionOutput,
    batch: ObjectSignedExpansionBatch,
    *,
    epoch: int,
    config: ObjectSignedExpansionLossConfig,
) -> ObjectSignedExpansionLossResult:
    """Optimize a threshold-free physical coordinate and JEPA dynamics."""

    phase = curriculum_phase(epoch, config)
    targets = targets_from_batch(batch, config)
    components: dict[str, torch.Tensor] = {}

    geometry_disagreement = (
        targets.visible_expansion - targets.signed_expansion
    ).abs()
    geometry_confidence = torch.exp(
        -geometry_disagreement / config.geometry_disagreement_scale
    ).clamp_min(0.1)
    expansion_weights = targets.sample_weights * geometry_confidence

    components["signed_expansion"] = _weighted_smooth_l1(
        output.signed_expansion,
        targets.signed_expansion,
        expansion_weights,
        beta=config.expansion_beta,
    )
    components["geometry_prior_mae"] = _weighted_smooth_l1(
        output.geometry_prior_expansion,
        targets.signed_expansion,
        targets.sample_weights,
        beta=config.expansion_beta,
    )
    components["geometry_residual_regularization"] = (
        output.learned_residual_expansion.abs()
        * targets.sample_weights
        * geometry_confidence
    ).sum() / (targets.sample_weights * geometry_confidence).sum().clamp_min(1.0e-6)
    sign_margin = (
        -targets.target_sign
        * output.signed_expansion
        / config.sign_temperature
    )
    components["sign_margin"] = (
        functional.softplus(sign_margin) * targets.sample_weights
    ).sum() / targets.sample_weights.sum().clamp_min(1.0e-6)

    components["latent_forward"] = _latent_prediction_loss(
        output.predicted_second_embedding,
        output.target_endpoint_embeddings[:, 1],
    )
    components["latent_reverse"] = _latent_prediction_loss(
        output.predicted_first_embedding,
        output.target_endpoint_embeddings[:, 0],
    )
    components["latent_variance"] = _variance_floor_loss(
        output.target_endpoint_embeddings,
        config.variance_floor,
    )
    valid = output.features.valid_patch_mask.to(output.activity_logits.dtype)
    activity_loss = functional.binary_cross_entropy_with_logits(
        output.activity_logits,
        output.activity_targets,
        reduction="none",
    )
    components["activity_reconstruction"] = (
        activity_loss * valid
    ).sum() / valid.sum().clamp_min(1.0)
    components["ordered_swap"] = (
        torch.tanh(output.ordered_score_forward)
        + torch.tanh(output.ordered_score_reverse)
    ).abs().mean()

    total = (
        config.expansion_weight * components["signed_expansion"]
        + config.geometry_prior_weight * components["geometry_residual_regularization"]
        + config.sign_margin_weight * components["sign_margin"]
        + config.latent_prediction_weight
        * 0.5
        * (components["latent_forward"] + components["latent_reverse"])
        + config.latent_variance_weight * components["latent_variance"]
        + config.activity_reconstruction_weight
        * components["activity_reconstruction"]
        + config.ordered_swap_weight * components["ordered_swap"]
    )

    if phase == "full_signed_expansion":
        components["official_log_ratio"] = _weighted_smooth_l1(
            output.log_height_ratio,
            targets.official_log_ratio,
            targets.sample_weights,
            beta=config.ratio_beta,
        )
        components["visible_log_ratio"] = _weighted_smooth_l1(
            output.log_height_ratio,
            targets.visible_log_ratio,
            targets.sample_weights * geometry_confidence,
            beta=config.ratio_beta,
        )
        components["signed_ttc_aux"] = _weighted_smooth_l1(
            signed_log1p(output.ttc_mean_seconds),
            signed_log1p(batch.target_ttc_s),
            targets.sample_weights,
            beta=config.ttc_aux_beta,
        )
        total = (
            total
            + config.official_ratio_weight * components["official_log_ratio"]
            + config.visible_ratio_weight * components["visible_log_ratio"]
            + config.ttc_aux_weight * components["signed_ttc_aux"]
        )

    components["target_expansion_mae"] = (
        output.signed_expansion.detach() - targets.signed_expansion.detach()
    ).abs().mean()
    components["target_ratio_mae"] = (
        output.log_height_ratio.detach() - targets.official_log_ratio.detach()
    ).abs().mean()
    components["official_geometry_disagreement"] = geometry_disagreement.detach().mean()
    components["learned_residual_abs_mean"] = (
        output.learned_residual_expansion.detach().abs().mean()
    )
    return ObjectSignedExpansionLossResult(
        total=total,
        components=components,
        phase=phase,
    )


__all__ = [
    "ObjectSignedExpansionLossConfig",
    "ObjectSignedExpansionLossResult",
    "ObjectSignedExpansionTargets",
    "curriculum_phase",
    "object_signed_expansion_loss",
    "signed_log1p",
    "targets_from_batch",
]
