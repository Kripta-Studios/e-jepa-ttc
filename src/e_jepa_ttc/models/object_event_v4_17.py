"""Object Event TTC v4.17: signed-anchor causal temporal residual sign head.

v4.16 learned an accurate temporal classifier on train OOF but over-predicted
negative signs on unseen validation sequences. v4.17 keeps a frozen event-only
signed anchor from the three true-seed v4.8 backbones and lets the causal head
learn only an odd residual around that anchor. Magnitude remains frozen to the
median absolute v4.8 expansion.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models.object_event_v4_16 import CausalOddSignHead


@dataclass(frozen=True)
class ObjectEventV417Config:
    window_size: int = 12
    sign_hidden_dim: int = 192
    sign_bottleneck_dim: int = 96
    magnitude_floor: float = 1.0e-4
    maximum_magnitude: float = 0.25
    anchor_feature_clip: float = 6.0
    anchor_logit_strength: float = 1.0
    maximum_residual_logit: float = 3.0
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(self.window_size, self.sign_hidden_dim, self.sign_bottleneck_dim) <= 0:
            raise ValueError("v4.17 dimensions must be positive")
        if not 0.0 < self.magnitude_floor < self.maximum_magnitude:
            raise ValueError("invalid magnitude bounds")
        if min(
            self.anchor_feature_clip,
            self.anchor_logit_strength,
            self.maximum_residual_logit,
            self.epsilon,
        ) <= 0.0:
            raise ValueError("v4.17 numerical controls must be positive")


@dataclass
class ObjectEventV417Output:
    sign_logit: torch.Tensor
    negative_probability: torch.Tensor
    magnitude: torch.Tensor
    signed_expansion: torch.Tensor
    instant_sign_logits: torch.Tensor
    sign_temporal_weights: torch.Tensor
    anchor_logit: torch.Tensor
    residual_logit: torch.Tensor


class ObjectEventTTCV417(nn.Module):
    """Exact-odd anchor + bounded causal residual sign architecture.

    ``anchor_logit_windows`` follows the BCE convention used by the project:
    positive logits mean the sample is negative/receding. The last causal anchor
    is the stable default decision; the temporal head can override it only with
    bounded odd evidence. This prevents a train-split class prior from replacing
    strong event-only evidence on unseen sequences.
    """

    def __init__(
        self,
        descriptor_dim: int,
        config: ObjectEventV417Config | None = None,
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.config = config or ObjectEventV417Config()
        self.descriptor_dim = int(descriptor_dim)
        self.residual_head = CausalOddSignHead(self.descriptor_dim, self.config)  # type: ignore[arg-type]

    def _bounded(self, value: torch.Tensor) -> torch.Tensor:
        limit = float(self.config.maximum_residual_logit)
        return limit * torch.tanh(value / limit)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        anchor_magnitude: torch.Tensor,
        anchor_logit_windows: torch.Tensor,
    ) -> ObjectEventV417Output:
        if anchor_logit_windows.shape != mask.shape:
            raise ValueError("anchor_logit_windows must align with the causal mask")
        if anchor_logit_windows.dtype != windows.dtype:
            anchor_logit_windows = anchor_logit_windows.to(dtype=windows.dtype)

        raw_residual, instant_raw, weights = self.residual_head(windows, mask)
        residual = self._bounded(raw_residual)
        instant_residual = self._bounded(instant_raw)

        # causal_window_indices right-aligns the current sample at the final step.
        current_anchor = anchor_logit_windows[:, -1]
        sign_logit = current_anchor + residual
        instant = anchor_logit_windows + instant_residual
        probability = torch.sigmoid(sign_logit)
        sign = torch.where(
            sign_logit >= 0.0,
            -torch.ones_like(sign_logit),
            torch.ones_like(sign_logit),
        )
        magnitude = anchor_magnitude.clamp(
            min=self.config.magnitude_floor,
            max=self.config.maximum_magnitude,
        )
        return ObjectEventV417Output(
            sign_logit=sign_logit,
            negative_probability=probability,
            magnitude=magnitude,
            signed_expansion=sign * magnitude,
            instant_sign_logits=instant,
            sign_temporal_weights=weights,
            anchor_logit=current_anchor,
            residual_logit=residual,
        )

    @torch.no_grad()
    def oddness_error(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        anchor_magnitude: torch.Tensor,
        anchor_logit_windows: torch.Tensor,
    ) -> torch.Tensor:
        original = self(windows, mask, anchor_magnitude, anchor_logit_windows)
        reversed_features = self(
            -windows,
            mask,
            anchor_magnitude,
            -anchor_logit_windows,
        )
        return (original.sign_logit + reversed_features.sign_logit).abs()


__all__ = [
    "ObjectEventTTCV417",
    "ObjectEventV417Config",
    "ObjectEventV417Output",
]
