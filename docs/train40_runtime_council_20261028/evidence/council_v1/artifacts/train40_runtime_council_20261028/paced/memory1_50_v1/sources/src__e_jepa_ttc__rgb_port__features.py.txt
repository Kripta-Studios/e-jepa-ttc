"""Label-free RGB-PORT expert features with modality-specific identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .normalization import RGB_PHASE17_SHA256

RGB_PHASE17_NAMES = (
    "rgb_luminance_mean",
    "rgb_spatial_gradient_log1p",
    "rgb_a5_flow",
    "rgb_a5_margin",
    "rgb_a5_log_variance",
    "rgb_c2f_flow",
    "rgb_c2f_margin",
    "rgb_c2f_log_variance",
    "rgb_a5_phase",
    "rgb_c2f_phase",
    "rgb_pair_phase",
    "rgb_pair_minus_a5",
    "rgb_pair_minus_c2f",
    "rgb_c2f_minus_a5",
    "rgb_abs_pair_minus_a5",
    "rgb_abs_pair_minus_c2f",
    "rgb_abs_c2f_minus_a5",
)

EVENT_PHASE17_NAMES = (
    "event_count_log1p_mean",
    "event_rate_log1p_mean",
    "event_a5_flow",
    "event_a5_margin",
    "event_a5_log_variance",
    "event_c2f_flow",
    "event_c2f_margin",
    "event_c2f_log_variance",
    "event_a5_phase",
    "event_c2f_phase",
    "event_pair_phase",
    "event_pair_minus_a5",
    "event_pair_minus_c2f",
    "event_c2f_minus_a5",
    "event_abs_pair_minus_a5",
    "event_abs_pair_minus_c2f",
    "event_abs_c2f_minus_a5",
)

PAIR_FEATURE_NAMES = tuple(f"token_{index:03d}" for index in range(128)) + (
    "delta_t_seconds",
    "log_delta_t_seconds",
    "inverse_delta_t_seconds",
    "support_current",
    "support_previous_current_minimum",
)


def schema_sha256(names: Sequence[str], *, modality: str, version: int = 1) -> str:
    """Hash ordered semantics, not feature width alone."""

    value = {"names": list(names), "modality": modality, "version": version}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


EVENT_PHASE17_SHA256 = schema_sha256(EVENT_PHASE17_NAMES, modality="event", version=1)
PAIR_E_SCHEMA_SHA256 = schema_sha256(PAIR_FEATURE_NAMES, modality="event", version=1)
PAIR_R_SCHEMA_SHA256 = schema_sha256(PAIR_FEATURE_NAMES, modality="rgb", version=1)


@dataclass(frozen=True)
class ProducerObservation:
    """One frozen expert observation; targets are deliberately absent."""

    token128: Tensor
    diagnostics3: Tensor
    prediction_ttc: Tensor
    point_phase: Tensor
    support: Tensor
    known: Tensor
    anchor_us: Tensor | None = None
    available_us: Tensor | None = None

    @classmethod
    def from_output(cls, output: Mapping[str, Tensor]) -> ProducerObservation:
        required = {
            "token128",
            "prediction_ttc",
            "point_phase",
            "flow",
            "margin",
            "log_variance",
            "support",
            "known",
        }
        missing = required - set(output)
        extra = set(output) - required - {"anchor_us", "available_us"}
        if missing or extra:
            raise ValueError(
                f"producer feature output differs: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        value = cls(
            token128=output["token128"],
            diagnostics3=torch.stack(
                (output["flow"], output["margin"], output["log_variance"]), -1
            ),
            prediction_ttc=output["prediction_ttc"],
            point_phase=output["point_phase"],
            support=output["support"],
            known=output["known"],
            anchor_us=output.get("anchor_us"),
            available_us=output.get("available_us"),
        )
        value.validate()
        return value

    def validate(self) -> None:
        prefix = self.token128.shape[:-1]
        if self.token128.shape[-1:] != (128,) or self.diagnostics3.shape != (*prefix, 3):
            raise ValueError("producer token/diagnostic shapes differ")
        for value in (self.prediction_ttc, self.point_phase):
            if value.shape != prefix:
                raise ValueError("producer scalar feature shape differs")
        if (
            self.support.ndim != len(prefix) + 1
            or self.support.shape[:-1] != prefix
            or self.support.shape[-1] not in {2, 3}
        ):
            raise ValueError("producer support must preserve its T2/T3 clock")
        if self.known.shape != self.support.shape or self.known.dtype != torch.bool:
            raise ValueError("producer known mask differs from support clock")
        if not all(
            torch.isfinite(value).all()
            for value in (
                self.token128,
                self.diagnostics3,
                self.prediction_ttc,
                self.point_phase,
                self.support,
            )
        ):
            raise ValueError("producer features must be finite")
        if not ((0 <= self.support) & (self.support <= 1)).all():
            raise ValueError("producer support must be in [0,1]")
        if (self.anchor_us is None) != (self.available_us is None):
            raise ValueError("producer clocks must be supplied together")
        if self.anchor_us is not None and self.available_us is not None:
            if self.anchor_us.shape != prefix or self.available_us.shape != prefix:
                raise ValueError("producer clock shapes differ")
            if self.anchor_us.dtype != torch.int64 or self.available_us.dtype != torch.int64:
                raise TypeError("producer clocks must remain int64 microseconds")
            if (self.available_us < self.anchor_us).any():
                raise ValueError("producer availability precedes its anchor")

    def pair_input(self, delta_t_seconds: Tensor) -> Tensor:
        """Build the historical 133-D input from this producer's own support."""

        current = self.support[..., -1]
        previous_minimum = torch.minimum(self.support[..., -2], current)
        return pair133(
            self.token128,
            delta_t_seconds,
            torch.stack((current, previous_minimum), -1),
        )


def raw_rgb_statistics(rgb: Tensor) -> Tensor:
    """Return luma mean and log-gradient from raw RGB in [0,1]."""

    if rgb.ndim < 3 or rgb.shape[-3] != 3 or min(rgb.shape[-2:]) < 2:
        raise ValueError("RGB must end in [3,H>=2,W>=2]")
    if not torch.is_floating_point(rgb) or not torch.isfinite(rgb).all():
        raise TypeError("RGB feature input must be finite floating point")
    if bool((rgb < 0).any()) or bool((rgb > 1).any()):
        raise ValueError("RGB statistics require raw [0,1], not ImageNet-normalized values")
    value = rgb.float()
    luma = (
        0.2126 * value[..., 0, :, :] + 0.7152 * value[..., 1, :, :] + 0.0722 * value[..., 2, :, :]
    )
    mean = luma.mean(dim=(-2, -1))
    dx = luma.diff(dim=-1).abs().mean(dim=(-2, -1))
    dy = luma.diff(dim=-2).abs().mean(dim=(-2, -1))
    return torch.stack((mean, torch.log1p(0.5 * (dx + dy))), -1)


def event_statistics_from_channels12(inputs: Tensor) -> Tensor:
    """Mean canonical V4 count/rate channels 10/11 over their spatial maps."""

    if inputs.ndim < 3 or inputs.shape[-3] != 12:
        raise ValueError("event producer input must end in [12,H,W]")
    if not torch.is_floating_point(inputs) or not torch.isfinite(inputs).all():
        raise TypeError("event producer input must be finite floating point")
    return inputs[..., 10:12, :, :].float().mean(dim=(-2, -1))


def pair133(token128: Tensor, delta_t_seconds: Tensor, supports: Tensor) -> Tensor:
    """Build token + dt/log/inverse + current/min(previous,current) support."""

    prefix = token128.shape[:-1]
    if token128.shape[-1:] != (128,) or delta_t_seconds.shape != prefix:
        raise ValueError("PAIR token/delta shapes differ")
    if supports.shape != (*prefix, 2):
        raise ValueError("PAIR requires previous/current support")
    if not all(torch.isfinite(value).all() for value in (token128, delta_t_seconds, supports)):
        raise ValueError("PAIR inputs must be finite")
    if (delta_t_seconds <= 0).any() or ((supports < 0) | (supports > 1)).any():
        raise ValueError("PAIR requires positive delta and bounded supports")
    dt = delta_t_seconds.float()
    return torch.cat(
        (
            token128.float(),
            dt[..., None],
            torch.log(dt)[..., None],
            dt.reciprocal()[..., None],
            supports.float(),
        ),
        -1,
    )


def _phase17(base2: Tensor, a5_diag: Tensor, c2f_diag: Tensor, expert_phase: Tensor) -> Tensor:
    prefix = base2.shape[:-1]
    if (
        base2.shape[-1:] != (2,)
        or a5_diag.shape != (*prefix, 3)
        or c2f_diag.shape != (*prefix, 3)
        or expert_phase.shape != (*prefix, 3)
    ):
        raise ValueError("PHASE17 feature block shapes differ")
    if not all(torch.isfinite(value).all() for value in (base2, a5_diag, c2f_diag, expert_phase)):
        raise ValueError("unavailable observations belong in masks, not PHASE17 NaNs")
    signed = torch.stack(
        (
            expert_phase[..., 2] - expert_phase[..., 0],
            expert_phase[..., 2] - expert_phase[..., 1],
            expert_phase[..., 1] - expert_phase[..., 0],
        ),
        -1,
    )
    return torch.cat(
        (
            base2.float(),
            a5_diag.float(),
            c2f_diag.float(),
            expert_phase.float(),
            signed,
            signed.abs(),
        ),
        -1,
    )


def rgb_phase17(
    rgb: Tensor,
    a5_diagnostics3: Tensor,
    c2f_diagnostics3: Tensor,
    expert_phase: Tensor,
) -> Tensor:
    """Build R_PHASE17 from raw RGB, two frozen producers and modality-local PAIR."""

    result = _phase17(raw_rgb_statistics(rgb), a5_diagnostics3, c2f_diagnostics3, expert_phase)
    if result.shape[-1] != len(RGB_PHASE17_NAMES):
        raise AssertionError("R_PHASE17 width drifted")
    return result


def event_phase17(
    event_statistics2: Tensor,
    a5_diagnostics3: Tensor,
    c2f_diagnostics3: Tensor,
    expert_phase: Tensor,
) -> Tensor:
    """Build the matched event control from its own P-trained expert outputs."""

    result = _phase17(
        event_statistics2,
        a5_diagnostics3,
        c2f_diagnostics3,
        expert_phase,
    )
    if result.shape[-1] != len(EVENT_PHASE17_NAMES):
        raise AssertionError("event PHASE17 width drifted")
    return result


def observation_blocks(
    base_statistics2: Tensor,
    a5_output: Mapping[str, Tensor],
    c2f_output: Mapping[str, Tensor],
    pair_output: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Adapt three frozen endpoints to the cache builder's label-free blocks."""

    a5 = ProducerObservation.from_output(a5_output)
    c2f = ProducerObservation.from_output(c2f_output)
    if a5.token128.shape[:-1] != c2f.token128.shape[:-1]:
        raise ValueError("producer observation batches differ")
    pair_phase = pair_output.get("point_phase")
    if pair_phase is None or pair_phase.shape != a5.point_phase.shape:
        raise ValueError("PAIR point phase differs from producer batch")
    if base_statistics2.shape != (*a5.point_phase.shape, 2):
        raise ValueError("base observation statistics differ")
    expert_phase = torch.stack((a5.point_phase, c2f.point_phase, pair_phase), -1)
    if not torch.isfinite(expert_phase).all():
        raise ValueError("expert phase blocks must be finite")
    return {
        "base_statistics2": base_statistics2.float(),
        "a5_diagnostics3": a5.diagnostics3,
        "c2f_diagnostics3": c2f.diagnostics3,
        "expert_phase": expert_phase.float(),
    }


__all__ = [
    "EVENT_PHASE17_NAMES",
    "EVENT_PHASE17_SHA256",
    "PAIR_E_SCHEMA_SHA256",
    "PAIR_FEATURE_NAMES",
    "PAIR_R_SCHEMA_SHA256",
    "ProducerObservation",
    "RGB_PHASE17_NAMES",
    "RGB_PHASE17_SHA256",
    "event_phase17",
    "event_statistics_from_channels12",
    "observation_blocks",
    "pair133",
    "raw_rgb_statistics",
    "rgb_phase17",
    "schema_sha256",
]
