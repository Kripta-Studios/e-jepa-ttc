"""MiD-aligned paired-reversal objective for Object Event TTC v4.5.

V4.4 showed that post-hoc analytic geometry carries a weak event-only signal but
cannot repair the held-out sign bias.  V4.5 therefore leaves the validated v4.2
architecture unchanged and changes only the supervision:

* optimise the paper-compatible log-eta error directly;
* balance the four official eAP TTC ranges inside each batch;
* supervise the time-reversed clip with the exact reciprocal scale relation;
* regularise reciprocity in log-eta space rather than forcing ``g_rev = -g``.

No RGB, boxes, observable motion, validation labels, or sequence identifiers are
consumed by this loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_2 import ObjectEventV42Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion


@dataclass(frozen=True)
class ObjectEventV45LossConfig:
    expansion_weight: float = 1.5
    official_mid_weight: float = 4.0
    correlation_weight: float = 0.5
    ranking_weight: float = 0.10
    sign_weight: float = 0.75
    sign_margin_weight: float = 0.25
    reverse_expansion_weight: float = 1.5
    reverse_mid_weight: float = 3.0
    reverse_sign_weight: float = 0.75
    reciprocal_consistency_weight: float = 0.50
    variance_weight: float = 0.05
    smooth_l1_beta: float = 0.005
    sign_temperature: float = 0.20
    sign_margin: float = 0.0025
    max_abs_expansion: float = 0.25
    crucial_weight: float = 0.5
    small_weight: float = 0.3
    large_weight: float = 0.1
    negative_weight: float = 0.1

    def __post_init__(self) -> None:
        weights = (
            self.expansion_weight,
            self.official_mid_weight,
            self.correlation_weight,
            self.ranking_weight,
            self.sign_weight,
            self.sign_margin_weight,
            self.reverse_expansion_weight,
            self.reverse_mid_weight,
            self.reverse_sign_weight,
            self.reciprocal_consistency_weight,
            self.variance_weight,
            self.crucial_weight,
            self.small_weight,
            self.large_weight,
            self.negative_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v4.5 loss weights must be non-negative")
        if self.smooth_l1_beta <= 0.0 or self.sign_temperature <= 0.0:
            raise ValueError("v4.5 loss scales must be positive")
        if self.sign_margin < 0.0:
            raise ValueError("sign_margin must be non-negative")
        if not 0.0 < self.max_abs_expansion < 1.0:
            raise ValueError("max_abs_expansion must lie in (0,1)")
        if sum(
            (
                self.crucial_weight,
                self.small_weight,
                self.large_weight,
                self.negative_weight,
            )
        ) <= 0.0:
            raise ValueError("at least one eAP range weight must be positive")


@dataclass
class ObjectEventV45LossOutput:
    total: torch.Tensor
    target: torch.Tensor
    reverse_target: torch.Tensor
    components: dict[str, torch.Tensor]


def _bounded_expansion(values: torch.Tensor, maximum: float) -> torch.Tensor:
    limit = maximum * 0.999
    return values.clamp(-limit, limit)


def log_eta(expansion: torch.Tensor, *, maximum: float) -> torch.Tensor:
    """Return ``log(eta)`` for ``eta = 1 - g`` with finite gradients."""

    bounded = _bounded_expansion(expansion.float(), maximum)
    return torch.log1p(-bounded)


def reciprocal_reverse_target(
    forward_expansion: torch.Tensor,
    *,
    maximum: float,
) -> torch.Tensor:
    """Exact expansion target after reversing the temporal order.

    If ``eta = 1 - g`` is the forward scale ratio, temporal reversal has
    ``eta_rev = 1 / eta`` and therefore ``g_rev = 1 - 1 / (1 - g)``.
    """

    bounded = _bounded_expansion(forward_expansion.float(), maximum)
    reverse = 1.0 - torch.reciprocal(1.0 - bounded)
    return _bounded_expansion(reverse, maximum).to(forward_expansion.dtype)


def _range_masks(target_ttc_s: torch.Tensor) -> tuple[torch.Tensor, ...]:
    ttc = target_ttc_s.float()
    return (
        (ttc > 0.0) & (ttc <= 3.0),
        (ttc > 3.0) & (ttc <= 6.0),
        (ttc > 6.0) & (ttc <= 10.0),
        (ttc >= -10.0) & (ttc < 0.0),
    )


def range_balanced_log_eta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_ttc_s: torch.Tensor,
    *,
    maximum: float,
    range_weights: tuple[float, float, float, float],
) -> torch.Tensor:
    """Differentiable counterpart of the weighted eAP MiD objective.

    Each present TTC range contributes its official weight times its own mean
    absolute log-eta error.  We renormalise only when a mini-batch lacks one or
    more ranges so the loss scale does not depend on random batch composition.
    """

    if not (prediction.shape == target.shape == target_ttc_s.shape):
        raise ValueError("prediction, target and target_ttc_s must have equal shapes")
    error = (log_eta(prediction, maximum=maximum) - log_eta(target, maximum=maximum)).abs()
    numerator = error.new_zeros(())
    denominator = error.new_zeros(())
    for mask, weight in zip(_range_masks(target_ttc_s), range_weights, strict=True):
        if weight <= 0.0 or not bool(mask.any()):
            continue
        numerator = numerator + float(weight) * error[mask].mean()
        denominator = denominator + float(weight)
    if float(denominator.detach().cpu()) <= 0.0:
        return error.mean()
    return numerator / denominator


def _balanced_sign_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target >= 0.0
    negative = ~positive
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        terms.append(functional.softplus(-logits[positive]).mean())
    if bool(negative.any()):
        terms.append(functional.softplus(logits[negative]).mean())
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).mean()


def _balanced_sign_margin(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    positive = target >= 0.0
    negative = ~positive
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        terms.append(functional.relu(margin - prediction[positive]).mean())
    if bool(negative.any()):
        terms.append(functional.relu(margin + prediction[negative]).mean())
    if not terms:
        return prediction.new_zeros(())
    return torch.stack(terms).mean()


def _ranking(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    pred_delta = prediction[:, None] - prediction[None, :]
    target_delta = target[:, None] - target[None, :]
    mask = ~torch.eye(prediction.shape[0], dtype=torch.bool, device=prediction.device)
    return functional.smooth_l1_loss(pred_delta[mask], target_delta[mask], beta=beta)


def reciprocal_log_eta_error(
    forward_prediction: torch.Tensor,
    reverse_prediction: torch.Tensor,
    *,
    maximum: float,
) -> torch.Tensor:
    """Exact reciprocal consistency: ``log eta_fwd + log eta_rev = 0``."""

    return (
        log_eta(forward_prediction, maximum=maximum)
        + log_eta(reverse_prediction, maximum=maximum)
    ).abs().mean()


def object_event_v4_5_loss(
    output: ObjectEventV42Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    *,
    config: ObjectEventV45LossConfig,
) -> ObjectEventV45LossOutput:
    target = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    reverse_target = reciprocal_reverse_target(target, maximum=config.max_abs_expansion)
    range_weights = (
        config.crucial_weight,
        config.small_weight,
        config.large_weight,
        config.negative_weight,
    )
    forward_logits = output.raw_score / config.sign_temperature
    reverse_logits = output.reverse_raw_score / config.sign_temperature
    components = {
        "expansion": functional.smooth_l1_loss(
            output.expansion, target, beta=config.smooth_l1_beta
        ),
        "official_mid": range_balanced_log_eta_loss(
            output.expansion,
            target,
            target_ttc_s,
            maximum=config.max_abs_expansion,
            range_weights=range_weights,
        ),
        "correlation": 1.0 - pearson_torch(output.expansion, target),
        "ranking": _ranking(output.expansion, target, config.smooth_l1_beta),
        "sign": _balanced_sign_bce(forward_logits, target),
        "sign_margin": _balanced_sign_margin(
            output.expansion, target, margin=config.sign_margin
        ),
        "reverse_expansion": functional.smooth_l1_loss(
            output.reverse_expansion, reverse_target, beta=config.smooth_l1_beta
        ),
        "reverse_mid": (
            log_eta(output.reverse_expansion, maximum=config.max_abs_expansion)
            - log_eta(reverse_target, maximum=config.max_abs_expansion)
        ).abs().mean(),
        "reverse_sign": _balanced_sign_bce(reverse_logits, reverse_target),
        "reciprocal_consistency": reciprocal_log_eta_error(
            output.expansion,
            output.reverse_expansion,
            maximum=config.max_abs_expansion,
        ),
        "variance": (
            output.expansion.float().std(unbiased=False)
            - target.float().std(unbiased=False)
        ).abs(),
    }
    total = (
        config.expansion_weight * components["expansion"]
        + config.official_mid_weight * components["official_mid"]
        + config.correlation_weight * components["correlation"]
        + config.ranking_weight * components["ranking"]
        + config.sign_weight * components["sign"]
        + config.sign_margin_weight * components["sign_margin"]
        + config.reverse_expansion_weight * components["reverse_expansion"]
        + config.reverse_mid_weight * components["reverse_mid"]
        + config.reverse_sign_weight * components["reverse_sign"]
        + config.reciprocal_consistency_weight * components["reciprocal_consistency"]
        + config.variance_weight * components["variance"]
    )
    return ObjectEventV45LossOutput(
        total=total,
        target=target,
        reverse_target=reverse_target,
        components=components,
    )


__all__ = [
    "ObjectEventV45LossConfig",
    "ObjectEventV45LossOutput",
    "log_eta",
    "object_event_v4_5_loss",
    "range_balanced_log_eta_loss",
    "reciprocal_log_eta_error",
    "reciprocal_reverse_target",
]
