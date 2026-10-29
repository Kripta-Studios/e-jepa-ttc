"""Exact native downstream readout algebra over frozen vector inputs.

The body below retains the admitted native forward tail verbatim except dense
features, which have only a frozen DINO contribution cached separately.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCOutput,
    _AntisymmetricResidual,
    blend_current_inverse_ttc,
    finite_ttc_from_inverse,
    log_ratio_to_inverse_ttc,
)
from e_jepa_ttc.models.local_transport import TRANSPORT_FEATURE_NAMES

from .operators import ResidualSpec, ordered_residual


class EqualSlopeResidual(_AntisymmetricResidual):
    """Same scorer and ordered inputs; explicit serialized equal-slope operator."""

    def __init__(self, native: _AntisymmetricResidual, spec: ResidualSpec) -> None:
        nn.Module.__init__(self)
        self.scorer = native.scorer
        self.transport_dim = native.transport_dim
        self.spec = spec

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        forward_transport: torch.Tensor | None = None,
        reverse_transport: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fwd = [previous, current, current - previous]
        rev = [current, previous, previous - current]
        if self.transport_dim:
            if forward_transport is None or reverse_transport is None:
                raise ValueError("Missing frozen transport")
            fwd.append(forward_transport)
            rev.append(reverse_transport)
        return ordered_residual(
            self.scorer(torch.cat(fwd, -1)).squeeze(-1),
            self.scorer(torch.cat(rev, -1)).squeeze(-1),
            self.spec,
        )


def wrap(model: CausalScaleTTC, spec: ResidualSpec, *, reset: bool) -> CausalScaleTTC:
    """Freeze everything except the scorer; reset both arms identically."""
    model.eval().requires_grad_(False)
    model.residual = EqualSlopeResidual(model.residual, spec)
    if reset:
        last = model.residual.scorer[-1]
        if not isinstance(last, nn.Linear) or last.bias is None:
            raise TypeError("Scorer must end with a biased Linear")
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
    model.residual.scorer.requires_grad_(True)
    return model


class VectorReadout(nn.Module):
    """Frozen modules retain native names so checkpoints load through the wrapper."""

    def __init__(self, model: CausalScaleTTC) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    @property
    def pair_projector(self) -> nn.Module:
        return self.model.pair_projector

    @property
    def uncertainty_head(self) -> nn.Module:
        return self.model.uncertainty_head

    @property
    def log_ratio_variance_offset(self) -> torch.Tensor:
        return self.model.log_ratio_variance_offset

    @property
    def auxiliary_inverse_ttc_head(self) -> nn.Module:
        return self.model.auxiliary_inverse_ttc_head

    def forward(self, cache: dict[str, torch.Tensor]) -> CausalScaleTTCOutput:
        if any(k.startswith(("target_", "dino_")) for k in cache):
            raise ValueError("Training targets are forbidden in inference inputs")
        geometry_tokens = cache["geometry_tokens"]
        previous, current = geometry_tokens[:, :-1], geometry_tokens[:, 1:]
        transport_tokens = cache["forward_transport"]
        reverse_transport_tokens = cache["reverse_transport"]
        residual = self.model.residual(
            previous, current, transport_tokens, reverse_transport_tokens
        )
        analytic = cache["analytic"]
        delta_t_s = cache["delta"]
        sensor_support = cache["sensor_support"]
        visible_height = cache["visible_height"]
        observation = SimpleNamespace(
            **{
                key: cache[key]
                for key in (
                    "height_normalized",
                    "width_normalized",
                    "centroid_y_normalized",
                    "centroid_x_normalized",
                    "foreground_fraction",
                )
            }
        )
        raw_foreground_ratio = cache["raw_foreground_ratio"]
        raw_foreground_width_ratio = cache["raw_foreground_width_ratio"]
        log_height_correction = cache["log_height_correction"]
        transport_raw_features = cache["transport_raw_features"]
        temporal_gate_weights = None
        control = SimpleNamespace(
            temporal_blend="trained", residual_enabled=True, transport_enabled=True
        )
        batch, steps = sensor_support.shape
        foreground_logits = geometry_tokens.new_zeros((batch, steps, 1, 1, 1))
        pair_input_values = [
            previous,
            current,
            current - previous,
            transport_tokens,
            reverse_transport_tokens,
        ]
        log_ratio = analytic + residual
        pair_tokens = self.pair_projector(torch.cat(pair_input_values, dim=-1))
        pair_support = torch.minimum(sensor_support[:, :-1], sensor_support[:, 1:])
        symmetric = torch.cat(
            [previous + current, (current - previous).abs(), pair_support.unsqueeze(-1)],
            dim=-1,
        )
        log_ratio_log_variance = (
            self.uncertainty_head(symmetric)
            .squeeze(-1)
            .add(self.log_ratio_variance_offset)
            .clamp(
                self.config.log_ratio_log_variance_min,
                self.config.log_ratio_log_variance_max,
            )
        )
        pair_inverse_ttc = log_ratio_to_inverse_ttc(log_ratio, delta_t_s)
        minimum_inverse = 1.0 / self.config.ttc_clip_seconds
        pair_ttc = finite_ttc_from_inverse(
            pair_inverse_ttc,
            minimum_abs_inverse_ttc=minimum_inverse,
            clip_seconds=self.config.ttc_clip_seconds,
        )
        pair_known = (
            (log_ratio.abs() >= self.config.min_abs_log_ratio)
            & (pair_support >= self.config.min_sensor_support)
            & torch.isfinite(log_ratio)
        )
        current_dt = delta_t_s[:, -1]
        replay_blend_weight = {
            "trained": self.config.temporal_inverse_ttc_blend,
            "current_only": 1.0,
            "previous_only": 0.0,
            "neutral": 0.5,
        }[control.temporal_blend]
        inverse_ttc, temporal_blend_used = blend_current_inverse_ttc(
            pair_inverse_ttc,
            pair_known,
            current_dt,
            current_pair_weight=replay_blend_weight,
        )
        current_ratio = torch.log1p((current_dt * inverse_ttc).clamp_min(-1.0 + 1.0e-6))
        effective_log_ratio = torch.cat(
            (log_ratio[:, :-1], current_ratio.unsqueeze(-1)),
            dim=1,
        )
        ttc = finite_ttc_from_inverse(
            inverse_ttc,
            minimum_abs_inverse_ttc=minimum_inverse,
            clip_seconds=self.config.ttc_clip_seconds,
        )
        known = pair_known[:, -1] & torch.isfinite(current_ratio)
        current_log_variance = log_ratio_log_variance[:, -1]
        inverse_log_variance = (
            current_log_variance + 2.0 * current_ratio.clamp(-12.0, 12.0) - 2.0 * current_dt.log()
        ).clamp(-16.0, 16.0)
        denominator = torch.expm1(current_ratio.clamp(-12.0, 12.0)).abs().clamp_min(1.0e-6)
        ttc_log_variance = (
            current_log_variance
            + 2.0 * current_dt.log()
            + 2.0 * current_ratio.clamp(-12.0, 12.0)
            - 4.0 * denominator.log()
        ).clamp(-16.0, 16.0)
        inverse_std = torch.exp(0.5 * inverse_log_variance).clamp_min(1.0e-6)
        thresholds = inverse_ttc.new_tensor(self.config.risk_thresholds_s)
        standardized = (inverse_ttc[:, None] - thresholds.reciprocal()[None, :]) / inverse_std[
            :, None
        ]
        risk_probability = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        risk_probability = risk_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        collision_logits = torch.logit(risk_probability)
        collision_logits = torch.where(
            known[:, None],
            collision_logits,
            torch.zeros_like(collision_logits),
        )
        auxiliary_inverse_ttc = self.auxiliary_inverse_ttc_head(pair_tokens).squeeze(-1)
        endpoint_dense_features = None
        diagnostics = {
            "foreground_centroid_y": observation.centroid_y_normalized,
            "foreground_centroid_x": observation.centroid_x_normalized,
            "foreground_fraction": observation.foreground_fraction,
            "raw_foreground_height_normalized": observation.height_normalized,
            "raw_foreground_width_normalized": observation.width_normalized,
            "raw_foreground_log_height_ratio": raw_foreground_ratio,
            "raw_foreground_log_width_ratio": raw_foreground_width_ratio,
            "log_height_correction": log_height_correction,
            "pair_sensor_support": pair_support,
            "pair_known": pair_known,
            "temporal_blend_used": temporal_blend_used,
            "foreground_temporal_smoothing": foreground_logits.new_full(
                (batch,), self.config.foreground_temporal_smoothing
            ),
            "replay_residual_enabled": foreground_logits.new_full(
                (batch,), float(control.residual_enabled)
            ),
            "replay_transport_enabled": foreground_logits.new_full(
                (batch,), float(control.transport_enabled)
            ),
            "replay_current_pair_weight": foreground_logits.new_full((batch,), replay_blend_weight),
        }
        if temporal_gate_weights is not None:
            diagnostics["temporal_channel_gate_weights"] = temporal_gate_weights
            gate_entropy = -(
                temporal_gate_weights.clamp_min(1.0e-8)
                * temporal_gate_weights.clamp_min(1.0e-8).log()
            ).sum(dim=2)
            diagnostics["temporal_channel_gate_entropy"] = gate_entropy
        if transport_raw_features is not None:
            for index, name in enumerate(TRANSPORT_FEATURE_NAMES):
                diagnostics[f"transport_{name}"] = transport_raw_features[..., index]

        return CausalScaleTTCOutput(
            ttc_mean_seconds=ttc,
            ttc_log_variance=ttc_log_variance,
            inverse_ttc_mean=inverse_ttc,
            inverse_ttc_log_variance=inverse_log_variance,
            collision_logits=collision_logits,
            known_mask=known,
            log_height_ratio=effective_log_ratio,
            pair_log_height_ratio=log_ratio,
            analytic_log_height_ratio=analytic,
            residual_log_height_ratio=residual,
            log_ratio_log_variance=log_ratio_log_variance,
            pair_ttc_seconds=pair_ttc,
            pair_inverse_ttc=pair_inverse_ttc,
            visible_height_normalized=visible_height,
            visible_width_normalized=observation.width_normalized,
            foreground_logits=foreground_logits,
            geometry_tokens=geometry_tokens,
            pair_tokens=pair_tokens,
            auxiliary_inverse_ttc=auxiliary_inverse_ttc,
            sensor_support=sensor_support,
            diagnostics=diagnostics,
            endpoint_dense_features=endpoint_dense_features,
            transport_tokens=transport_tokens,
            transport_raw_features=transport_raw_features,
        )
