"""Object Event TTC v4.16: causal temporal sign + magnitude heads."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class ObjectEventV416Config:
    window_size: int = 12
    sign_hidden_dim: int = 192
    sign_bottleneck_dim: int = 96
    magnitude_hidden_dim: int = 256
    magnitude_bottleneck_dim: int = 128
    magnitude_floor: float = 1.0e-4
    maximum_magnitude: float = 0.25
    maximum_log_ratio: float = 2.0
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(
            self.window_size,
            self.sign_hidden_dim,
            self.sign_bottleneck_dim,
            self.magnitude_hidden_dim,
            self.magnitude_bottleneck_dim,
        ) <= 0:
            raise ValueError("v4.16 dimensions must be positive")
        if not 0.0 < self.magnitude_floor < self.maximum_magnitude:
            raise ValueError("invalid magnitude bounds")
        if self.maximum_log_ratio <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("v4.16 numerical controls must be positive")


@dataclass
class ObjectEventV416Output:
    sign_logit: torch.Tensor
    negative_probability: torch.Tensor
    magnitude: torch.Tensor
    signed_expansion: torch.Tensor
    instant_sign_logits: torch.Tensor
    sign_temporal_weights: torch.Tensor
    magnitude_temporal_weights: torch.Tensor
    magnitude_log_ratio: torch.Tensor


class BiasFreeOddMLP(nn.Module):
    """An MLP satisfying f(-x) == -f(x) up to floating point roundoff."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck_dim, bias=False),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, 1, bias=False),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("expected linear output")
        nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class CausalOddSignHead(nn.Module):
    """Odd per-sample evidence plus an input-independent causal recency kernel."""

    def __init__(self, descriptor_dim: int, config: ObjectEventV416Config) -> None:
        super().__init__()
        self.window_size = int(config.window_size)
        self.instant = BiasFreeOddMLP(
            descriptor_dim,
            config.sign_hidden_dim,
            config.sign_bottleneck_dim,
        )
        # Start mildly recency-biased. These weights are global parameters and do
        # not depend on the input, preserving exact oddness of the sign output.
        self.recency_logits = nn.Parameter(torch.linspace(-1.5, 0.0, self.window_size))

    def forward(
        self, windows: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if windows.ndim != 3:
            raise ValueError("windows must be [B,T,D]")
        if mask.shape != windows.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [B,T]")
        if windows.shape[1] != self.window_size:
            raise ValueError("window size mismatch")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each causal window needs at least one valid sample")
        batch, steps, dim = windows.shape
        instant = self.instant(windows.reshape(batch * steps, dim)).reshape(batch, steps)
        score = self.recency_logits[None, :].expand(batch, -1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        weights = torch.softmax(score, dim=1)
        logit = (weights * instant).sum(dim=1)
        return logit, instant, weights


class EvenMagnitudeHead(nn.Module):
    """Positive magnitude residual using only sign-even descriptor features."""

    def __init__(self, descriptor_dim: int, config: ObjectEventV416Config) -> None:
        super().__init__()
        self.config = config
        self.window_size = int(config.window_size)
        self.recency_logits = nn.Parameter(torch.linspace(-1.0, 0.0, self.window_size))
        input_dim = 2 * descriptor_dim + 1
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.magnitude_hidden_dim),
            nn.GELU(),
            nn.Linear(config.magnitude_hidden_dim, config.magnitude_bottleneck_dim),
            nn.GELU(),
            nn.Linear(config.magnitude_bottleneck_dim, 1),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("expected linear output")
        # Zero residual initially means magnitude == frozen v4.8 consensus anchor.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        anchor_magnitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if windows.ndim != 3 or mask.shape != windows.shape[:2]:
            raise ValueError("invalid magnitude window shapes")
        if anchor_magnitude.shape != windows.shape[:1]:
            raise ValueError("anchor_magnitude must be [B]")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each causal window needs at least one valid sample")
        even = windows.abs()
        score = self.recency_logits[None, :].expand(len(windows), -1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        weights = torch.softmax(score, dim=1)
        context = (weights[:, :, None] * even).sum(dim=1)
        latest = even[:, -1]
        anchor = anchor_magnitude.clamp(
            min=self.config.magnitude_floor,
            max=self.config.maximum_magnitude,
        )
        feature = torch.cat(
            (context, latest, torch.log(anchor + self.config.epsilon)[:, None]),
            dim=1,
        )
        log_ratio = self.network(feature).squeeze(-1).clamp(
            -self.config.maximum_log_ratio,
            self.config.maximum_log_ratio,
        )
        magnitude = (anchor * torch.exp(log_ratio)).clamp(
            self.config.magnitude_floor,
            self.config.maximum_magnitude,
        )
        return magnitude, log_ratio, weights


class ObjectEventTTCV416(nn.Module):
    """Temporal dual head operating on frozen multibackbone descriptors."""

    def __init__(
        self,
        descriptor_dim: int,
        config: ObjectEventV416Config | None = None,
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.config = config or ObjectEventV416Config()
        self.descriptor_dim = int(descriptor_dim)
        self.sign_head = CausalOddSignHead(self.descriptor_dim, self.config)
        self.magnitude_head = EvenMagnitudeHead(self.descriptor_dim, self.config)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        anchor_magnitude: torch.Tensor,
    ) -> ObjectEventV416Output:
        sign_logit, instant, sign_weights = self.sign_head(windows, mask)
        magnitude, log_ratio, magnitude_weights = self.magnitude_head(
            windows, mask, anchor_magnitude
        )
        probability = torch.sigmoid(sign_logit)
        sign = torch.where(sign_logit >= 0.0, -torch.ones_like(sign_logit), torch.ones_like(sign_logit))
        signed = sign * magnitude
        return ObjectEventV416Output(
            sign_logit=sign_logit,
            negative_probability=probability,
            magnitude=magnitude,
            signed_expansion=signed,
            instant_sign_logits=instant,
            sign_temporal_weights=sign_weights,
            magnitude_temporal_weights=magnitude_weights,
            magnitude_log_ratio=log_ratio,
        )

    @torch.no_grad()
    def symmetry_errors(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        anchor_magnitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original = self(windows, mask, anchor_magnitude)
        reversed_sign = self(-windows, mask, anchor_magnitude)
        sign_error = (original.sign_logit + reversed_sign.sign_logit).abs()
        magnitude_error = (original.magnitude - reversed_sign.magnitude).abs()
        return sign_error, magnitude_error


__all__ = [
    "BiasFreeOddMLP",
    "CausalOddSignHead",
    "EvenMagnitudeHead",
    "ObjectEventTTCV416",
    "ObjectEventV416Config",
    "ObjectEventV416Output",
]
