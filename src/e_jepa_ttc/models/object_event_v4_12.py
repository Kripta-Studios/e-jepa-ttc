"""Object Event TTC v4.12: reversal-balanced directional sign probe.

V4.10 showed that magnitude/order are stable across seeds while a small set of
negative TTC windows is assigned the wrong sign by every final expert. V4.11
also showed that a router operating only on final predictions cannot recover the
missing direction robustly. V4.12 therefore probes the frozen v4.8 temporal
representation directly and learns only the binary direction (approach/recede).

The probe accepts event tensors only. Boxes, visible heights, sequence IDs and
track IDs are never forward inputs. A caller may provide a separately computed
magnitude (for example the fixed v4.10 ensemble magnitude); the probe supplies
only its sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48


@dataclass(frozen=True)
class ObjectEventV412Config:
    hidden_dim: int = 128
    bottleneck_dim: int = 64
    dropout: float = 0.0
    foreground_floor: float = 0.05
    activity_floor: float = 0.05
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.bottleneck_dim) <= 0:
            raise ValueError("v4.12 hidden dimensions must be positive")
        if self.dropout != 0.0:
            raise ValueError("v4.12.1 exact odd symmetry requires dropout=0")
        if min(self.foreground_floor, self.activity_floor, self.epsilon) <= 0.0:
            raise ValueError("v4.12 numerical floors must be positive")


@dataclass
class ObjectEventV412Output:
    base_expansion: torch.Tensor
    signed_expansion: torch.Tensor
    base_log_eta: torch.Tensor
    sign_logits: torch.Tensor
    negative_probability: torch.Tensor
    descriptor: torch.Tensor


def _weighted_mean_std(
    features: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 4 or weights.ndim != 3:
        raise ValueError("features must be [B,C,H,W] and weights [B,H,W]")
    if features.shape[0] != weights.shape[0] or features.shape[-2:] != weights.shape[-2:]:
        raise ValueError("feature and weight shapes do not align")
    mass = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
    normalised = weights[:, None] / mass[:, None]
    mean = (features * normalised).sum(dim=(-2, -1))
    variance = (
        (features - mean[:, :, None, None]).square() * normalised
    ).sum(dim=(-2, -1))
    return mean, torch.sqrt(variance.clamp_min(epsilon))


def _spatial_directional_moments(
    fields: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Foreground-centred first/radial moments for signed temporal fields."""

    if fields.ndim != 4 or weights.ndim != 3:
        raise ValueError("fields must be [B,K,H,W] and weights [B,H,W]")
    batch, _, height, width = fields.shape
    dtype = fields.dtype
    device = fields.device
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mass = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
    cx = (weights * xx).sum(dim=(-2, -1), keepdim=True) / mass
    cy = (weights * yy).sum(dim=(-2, -1), keepdim=True) / mass
    dx = xx[None] - cx
    dy = yy[None] - cy
    radius = torch.sqrt(dx.square() + dy.square() + epsilon)
    norm = weights[:, None] / mass[:, None]
    horizontal = (fields * dx[:, None] * norm).sum(dim=(-2, -1))
    vertical = (fields * dy[:, None] * norm).sum(dim=(-2, -1))
    radial = (fields * radius[:, None] * norm).sum(dim=(-2, -1))
    return torch.cat((horizontal, vertical, radial), dim=1).reshape(batch, -1)


class ObjectEventTTCV412(nn.Module):
    """Frozen v4.8 representation plus a trainable directional sign head."""

    def __init__(
        self,
        backbone: ObjectEventTTCV48,
        config: ObjectEventV412Config | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or ObjectEventV412Config()
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        refine = int(self.backbone.motion_config.motion_refine_dim)
        # Mean/std of temporal + 6 activity + 2 endpoint masks + local field + confidence.
        pooled_channels = refine + 10
        # First/radial moments for d01, d12, acceleration and local log eta.
        moment_channels = 4 * 3
        descriptor_dim = 2 * pooled_channels + moment_channels + 6
        self.descriptor_dim = descriptor_dim
        # Exact time-reversal antisymmetry is imposed structurally rather than
        # encouraged with a soft penalty. LayerNorm without affine parameters,
        # bias-free Linear layers and tanh are all odd functions, therefore
        # sign_head(-descriptor) == -sign_head(descriptor) up to roundoff.
        self.sign_head = nn.Sequential(
            nn.LayerNorm(descriptor_dim, elementwise_affine=False),
            nn.Linear(descriptor_dim, self.config.hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.bottleneck_dim, bias=False),
            nn.Tanh(),
            nn.Linear(self.config.bottleneck_dim, 1, bias=False),
        )
        final = self.sign_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected Linear sign output")
        nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)

    def train(self, mode: bool = True) -> "ObjectEventTTCV412":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _raw_descriptor_and_base(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = self.backbone
        maps, _, foreground_probabilities, activity = backbone._foreground_and_features(events)
        temporal = backbone.temporal_projection(backbone._temporal_maps(maps))
        temporal = functional.interpolate(
            temporal,
            size=(backbone.motion_config.field_size, backbone.motion_config.field_size),
            mode="bilinear",
            align_corners=False,
        )
        activity_features = backbone._activity_features(activity)
        endpoint_foreground = foreground_probabilities[:, 1:3]
        field_input = torch.cat((temporal, activity_features, endpoint_foreground), dim=1)
        field = backbone.field_head(field_input)
        activity_change = (activity[:, 2] - activity[:, 1]).abs()
        event_presence = torch.tanh(activity_change)
        local_log_eta = (
            backbone.motion_config.maximum_abs_log_eta
            * torch.tanh(field[:, 0])
            * event_presence
        )
        confidence = torch.sigmoid(field[:, 1])
        foreground_pair = torch.sqrt(
            (foreground_probabilities[:, 1] * foreground_probabilities[:, 2]).clamp_min(0.0)
        )
        activity_scale = activity_change.mean(dim=(-2, -1), keepdim=True).clamp_min(
            backbone.motion_config.weight_epsilon
        )
        activity_weight = (activity_change / activity_scale).clamp(0.0, 4.0)
        weights = (
            (self.config.foreground_floor + foreground_pair)
            * (self.config.activity_floor + activity_weight)
            * (backbone.motion_config.confidence_floor + confidence)
        )
        denominator = weights.sum(dim=(-2, -1)).clamp_min(self.config.epsilon)
        pooled_log_eta = (weights * local_log_eta).sum(dim=(-2, -1)) / denominator

        pooled_features = torch.cat(
            (
                temporal,
                activity_features,
                endpoint_foreground,
                local_log_eta[:, None],
                confidence[:, None],
            ),
            dim=1,
        )
        mean, std = _weighted_mean_std(
            pooled_features,
            weights,
            epsilon=self.config.epsilon,
        )
        signed_fields = torch.stack(
            (
                activity_features[:, 3],
                activity_features[:, 4],
                activity_features[:, 5],
                local_log_eta,
            ),
            dim=1,
        )
        moments = _spatial_directional_moments(
            signed_fields,
            weights,
            epsilon=self.config.epsilon,
        )
        global_features = torch.stack(
            (
                pooled_log_eta,
                activity_features[:, 3].mean(dim=(-2, -1)),
                activity_features[:, 4].mean(dim=(-2, -1)),
                activity_features[:, 5].mean(dim=(-2, -1)),
                foreground_pair.mean(dim=(-2, -1)),
                confidence.mean(dim=(-2, -1)),
            ),
            dim=1,
        )
        descriptor = torch.cat((mean, std, moments, global_features), dim=1)
        if descriptor.shape[1] != self.descriptor_dim:
            raise AssertionError(
                f"v4.12 descriptor mismatch: {descriptor.shape[1]} != {self.descriptor_dim}"
            )
        return descriptor.detach(), pooled_log_eta.detach()

    @torch.no_grad()
    def _descriptor_and_base(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a descriptor that is exactly odd under coarse time reversal.

        The cached tensor contains three ordered event windows.  We do not assume
        any polarity or temporal-bin layout inside the 12 channels.  Instead we
        antisymmetrise the frozen-backbone descriptor itself:

            d_odd(x) = 0.5 * (d_raw(x) - d_raw(Rx)).

        Since R(Rx)=x, d_odd(Rx)=-d_odd(x) exactly.  This fixes the v4.12
        overfit failure without inventing a channel-level reversal convention.
        """

        forward_descriptor, pooled_log_eta = self._raw_descriptor_and_base(events)
        reverse_descriptor, _ = self._raw_descriptor_and_base(events.flip(1))
        odd_descriptor = 0.5 * (forward_descriptor - reverse_descriptor)
        return odd_descriptor.detach(), pooled_log_eta.detach()

    def paired_sign_logits(
        self, events: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute original/reversed logits with exact antisymmetry."""

        descriptor, pooled_log_eta = self._descriptor_and_base(events)
        original_logits = self.sign_head(descriptor).squeeze(-1)
        reversed_logits = -original_logits
        return original_logits, reversed_logits, descriptor, pooled_log_eta

    def forward(
        self,
        events: torch.Tensor,
        *,
        magnitude_expansion: torch.Tensor | None = None,
        negative_threshold: float = 0.5,
    ) -> ObjectEventV412Output:
        if not 0.0 < negative_threshold < 1.0:
            raise ValueError("negative_threshold must lie in (0,1)")
        sign_logits, _, descriptor, pooled_log_eta = self.paired_sign_logits(events)
        probability = torch.sigmoid(sign_logits)
        maximum = self.backbone.config.max_abs_expansion
        base_expansion = (1.0 - torch.exp(pooled_log_eta)).clamp(
            -maximum * 0.999,
            maximum * 0.999,
        )
        magnitude = (
            base_expansion.abs()
            if magnitude_expansion is None
            else magnitude_expansion.to(device=base_expansion.device, dtype=base_expansion.dtype).abs()
        )
        sign = torch.where(
            probability >= negative_threshold,
            -torch.ones_like(probability),
            torch.ones_like(probability),
        )
        signed = (sign * magnitude).clamp(-maximum * 0.999, maximum * 0.999)
        return ObjectEventV412Output(
            base_expansion=base_expansion,
            signed_expansion=signed,
            base_log_eta=pooled_log_eta,
            sign_logits=sign_logits,
            negative_probability=probability,
            descriptor=descriptor,
        )


__all__ = [
    "ObjectEventTTCV412",
    "ObjectEventV412Config",
    "ObjectEventV412Output",
    "_spatial_directional_moments",
    "_weighted_mean_std",
]
