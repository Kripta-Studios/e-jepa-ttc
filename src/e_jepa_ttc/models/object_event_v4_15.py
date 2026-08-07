"""Object Event TTC v4.15: shared odd multibackbone direction probe.

V4.14 showed that independently trained v4.12 probes recover receding motion,
but their probability scales are not reproducible across seeds.  V4.15 removes
that calibration mismatch by extracting exactly odd descriptors from all frozen
v4.8 backbones and training one shared odd sign head.  The final TTC expansion
is a sign projection of the frozen v4.10 magnitude; it never blends opposite
signed values and therefore cannot create new near-zero cancellation artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_12 import ObjectEventTTCV412, ObjectEventV412Config


@dataclass(frozen=True)
class ObjectEventV415Config:
    hidden_dim: int = 192
    bottleneck_dim: int = 96
    dropout: float = 0.0
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.bottleneck_dim) <= 0:
            raise ValueError("v4.15 hidden dimensions must be positive")
        if self.dropout != 0.0:
            raise ValueError("exact odd symmetry requires dropout=0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


class OddConsensusHead(nn.Module):
    """Bias-free odd MLP: h(-x) == -h(x) up to roundoff."""

    def __init__(self, descriptor_dim: int, config: ObjectEventV415Config) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(descriptor_dim, elementwise_affine=False),
            nn.Linear(descriptor_dim, config.hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.bottleneck_dim, bias=False),
            nn.Tanh(),
            nn.Linear(config.bottleneck_dim, 1, bias=False),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected Linear output")
        nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.network(descriptor).squeeze(-1)


class ObjectEventTTCV415(nn.Module):
    """Three frozen v4.8 descriptors plus one shared odd directional head."""

    def __init__(
        self,
        extractors: Sequence[ObjectEventTTCV412],
        config: ObjectEventV415Config | None = None,
    ) -> None:
        super().__init__()
        if len(extractors) < 3:
            raise ValueError("v4.15 requires at least three frozen extractors")
        self.config = config or ObjectEventV415Config()
        self.extractors = nn.ModuleList(extractors)
        dims = {int(extractor.descriptor_dim) for extractor in self.extractors}
        if len(dims) != 1:
            raise ValueError("all v4.12 extractors must share descriptor_dim")
        self.single_descriptor_dim = dims.pop()
        self.descriptor_dim = 2 * self.single_descriptor_dim
        self.sign_head = OddConsensusHead(self.descriptor_dim, self.config)
        for extractor in self.extractors:
            extractor.requires_grad_(False)
            extractor.eval()

    def train(self, mode: bool = True) -> "ObjectEventTTCV415":
        super().train(mode)
        for extractor in self.extractors:
            extractor.eval()
        return self

    @torch.no_grad()
    def consensus_descriptor(self, events: torch.Tensor) -> torch.Tensor:
        descriptors: list[torch.Tensor] = []
        for extractor in self.extractors:
            descriptor, _ = extractor._descriptor_and_base(events)
            descriptor = functional.layer_norm(
                descriptor,
                (descriptor.shape[-1],),
                weight=None,
                bias=None,
                eps=self.config.epsilon,
            )
            descriptors.append(descriptor)
        stacked = torch.stack(descriptors, dim=0)
        robust_mean = stacked.mean(dim=0)
        robust_median = stacked.median(dim=0).values
        result = torch.cat((robust_mean, robust_median), dim=1)
        if result.shape[1] != self.descriptor_dim:
            raise AssertionError("v4.15 descriptor dimension mismatch")
        return result.detach()

    def paired_sign_logits_from_descriptor(
        self, descriptor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.sign_head(descriptor)
        return logits, -logits

    def paired_sign_logits(
        self, events: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        descriptor = self.consensus_descriptor(events)
        logits, reversed_logits = self.paired_sign_logits_from_descriptor(descriptor)
        return logits, reversed_logits, descriptor

    def forward(
        self,
        events: torch.Tensor,
        *,
        magnitude_expansion: torch.Tensor,
        negative_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _, descriptor = self.paired_sign_logits(events)
        probability = torch.sigmoid(logits)
        prediction = sign_magnitude_projection(
            magnitude_expansion,
            probability,
            negative_threshold=negative_threshold,
        )
        return prediction, probability, descriptor


def sign_magnitude_projection(
    baseline_expansion: torch.Tensor,
    negative_probability: torch.Tensor,
    *,
    negative_threshold: float,
) -> torch.Tensor:
    """Preserve magnitude and permit only confident positive-to-negative flips."""
    if baseline_expansion.shape != negative_probability.shape:
        raise ValueError("baseline and probability shapes must match")
    if not 0.5 <= negative_threshold < 1.0:
        raise ValueError("negative_threshold must lie in [0.5,1)")
    magnitude = baseline_expansion.abs()
    flip = (baseline_expansion >= 0.0) & (negative_probability >= negative_threshold)
    return torch.where(flip, -magnitude, baseline_expansion)


__all__ = [
    "ObjectEventTTCV415",
    "ObjectEventV415Config",
    "OddConsensusHead",
    "sign_magnitude_projection",
]
