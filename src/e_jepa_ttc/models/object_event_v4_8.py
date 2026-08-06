"""Object Event TTC v4.8: foreground-conditioned dense temporal log-scale.

V4.7 proved that a high-resolution foreground mask is learnable, but deriving a
sub-pixel height ratio from the mask extent did not generalise. V4.8 keeps the
validated foreground network frozen and predicts a dense temporal log-scale
field from encoded event differences. Boxes and visible heights remain
training-only targets; ``forward`` accepts only the event tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_7 import ObjectEventTTCV47, ObjectEventV47Config


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


@dataclass(frozen=True)
class ObjectEventV48Config:
    motion_hidden_dim: int = 128
    motion_refine_dim: int = 96
    field_size: int = 64
    maximum_abs_log_eta: float = 0.30
    confidence_floor: float = 0.10
    activity_floor: float = 0.10
    weight_epsilon: float = 1.0e-5
    freeze_foreground: bool = True

    def __post_init__(self) -> None:
        if min(self.motion_hidden_dim, self.motion_refine_dim, self.field_size) <= 0:
            raise ValueError("v4.8 dimensions must be positive")
        if self.field_size < 16:
            raise ValueError("v4.8 field_size must be at least 16")
        if not 0.0 < self.maximum_abs_log_eta < 1.0:
            raise ValueError("maximum_abs_log_eta must lie in (0,1)")
        if min(self.confidence_floor, self.activity_floor, self.weight_epsilon) <= 0.0:
            raise ValueError("v4.8 numerical floors must be positive")


@dataclass
class ObjectEventV48Output:
    expansion: torch.Tensor
    reverse_expansion: torch.Tensor
    raw_score: torch.Tensor
    reverse_raw_score: torch.Tensor
    reversal_consistency_error: torch.Tensor
    pooled_log_eta: torch.Tensor
    local_log_eta: torch.Tensor
    confidence_logits: torch.Tensor
    confidence_probabilities: torch.Tensor
    aggregation_weights: torch.Tensor
    foreground_logits: torch.Tensor
    foreground_probabilities: torch.Tensor
    event_activity: torch.Tensor


class ObjectEventTTCV48(nn.Module):
    """Dense event-motion field pooled inside a frozen learned foreground."""

    def __init__(
        self,
        base_config: ObjectEventV41Config | None = None,
        foreground_config: ObjectEventV47Config | None = None,
        motion_config: ObjectEventV48Config | None = None,
    ) -> None:
        super().__init__()
        self.config = base_config or ObjectEventV41Config()
        self.foreground_config = foreground_config or ObjectEventV47Config()
        self.motion_config = motion_config or ObjectEventV48Config()
        self.foreground_model = ObjectEventTTCV47(
            self.config,
            self.foreground_config,
        )
        channels = self.config.embed_dim
        hidden = self.motion_config.motion_hidden_dim
        refine = self.motion_config.motion_refine_dim
        # First/second differences, acceleration and their magnitudes; no static levels.
        self.temporal_projection = nn.Sequential(
            nn.Conv2d(6 * channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, refine, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(refine), refine),
            nn.GELU(),
        )
        # Six activity channels plus the two endpoint foreground maps.
        self.field_head = nn.Sequential(
            nn.Conv2d(refine + 8, refine, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(refine), refine),
            nn.GELU(),
            nn.Conv2d(refine, refine // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(refine // 2), refine // 2),
            nn.GELU(),
            nn.Conv2d(refine // 2, 2, kernel_size=1),
        )
        final = self.field_head[-1]
        if not isinstance(final, nn.Conv2d):
            raise TypeError("Expected Conv2d output layer")
        nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(final.bias)
        self._set_foreground_trainable(not self.motion_config.freeze_foreground)

    def _set_foreground_trainable(self, trainable: bool) -> None:
        self.foreground_model.requires_grad_(trainable)
        if not trainable:
            self.foreground_model.eval()

    def train(self, mode: bool = True) -> "ObjectEventTTCV48":
        super().train(mode)
        if self.motion_config.freeze_foreground:
            self.foreground_model.eval()
        return self

    def load_v47_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.foreground_model.load_state_dict(state_dict, strict=True)
        self._set_foreground_trainable(not self.motion_config.freeze_foreground)

    def _foreground_and_features(
        self, events: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        foreground = self.foreground_model
        resized = foreground._resize(events)
        batch, steps, channels, height, width = resized.shape
        context = torch.no_grad() if self.motion_config.freeze_foreground else torch.enable_grad()
        with context:
            maps = foreground.geometry_encoder(
                resized.reshape(batch * steps, channels, height, width)
            )
            map_h, map_w = maps.shape[-2:]
            maps_by_time = maps.reshape(batch, steps, self.config.embed_dim, map_h, map_w)
            decoded = foreground.decoder(maps)
            decoded = functional.interpolate(
                decoded,
                size=(self.motion_config.field_size, self.motion_config.field_size),
                mode="bilinear",
                align_corners=False,
            )
            activity = foreground._normalised_activity(resized)
            activity_full = functional.interpolate(
                activity.reshape(batch * steps, 1, height, width),
                size=(self.motion_config.field_size, self.motion_config.field_size),
                mode="bilinear",
                align_corners=False,
            )
            foreground_logits = foreground.refine(torch.cat((decoded, activity_full), dim=1))
            foreground_probabilities = torch.sigmoid(
                foreground_logits / self.foreground_config.foreground_temperature
            )
        foreground_logits = foreground_logits.reshape(
            batch,
            steps,
            self.motion_config.field_size,
            self.motion_config.field_size,
        )
        foreground_probabilities = foreground_probabilities.reshape_as(foreground_logits)
        activity = activity_full.reshape_as(foreground_logits)
        return maps_by_time.detach() if self.motion_config.freeze_foreground else maps_by_time, foreground_logits, foreground_probabilities, activity

    @staticmethod
    def _temporal_maps(maps: torch.Tensor) -> torch.Tensor:
        if maps.ndim != 5 or maps.shape[1] != 3:
            raise ValueError("maps must be [B,3,C,H,W]")
        m0, m1, m2 = maps.unbind(dim=1)
        d01 = m1 - m0
        d12 = m2 - m1
        acceleration = d12 - d01
        return torch.cat(
            (d01, d12, acceleration, d01.abs(), d12.abs(), acceleration.abs()),
            dim=1,
        )

    @staticmethod
    def _activity_features(activity: torch.Tensor) -> torch.Tensor:
        if activity.ndim != 4 or activity.shape[1] != 3:
            raise ValueError("activity must be [B,3,H,W]")
        a0, a1, a2 = activity.unbind(dim=1)
        d01 = a1 - a0
        d12 = a2 - a1
        acceleration = d12 - d01
        return torch.stack((a0, a1, a2, d01, d12, acceleration), dim=1)

    def forward(self, events: torch.Tensor) -> ObjectEventV48Output:
        maps, foreground_logits, foreground_probabilities, activity = self._foreground_and_features(events)
        temporal = self.temporal_projection(self._temporal_maps(maps))
        temporal = functional.interpolate(
            temporal,
            size=(self.motion_config.field_size, self.motion_config.field_size),
            mode="bilinear",
            align_corners=False,
        )
        activity_features = self._activity_features(activity)
        endpoint_foreground = foreground_probabilities[:, 1:3]
        field = self.field_head(
            torch.cat((temporal, activity_features, endpoint_foreground), dim=1)
        )
        activity_change = (activity[:, 2] - activity[:, 1]).abs()
        event_presence = torch.tanh(activity_change)
        local_log_eta = (
            self.motion_config.maximum_abs_log_eta
            * torch.tanh(field[:, 0])
            * event_presence
        )
        confidence_logits = field[:, 1]
        confidence = torch.sigmoid(confidence_logits)
        foreground_pair = torch.sqrt(
            (foreground_probabilities[:, 1] * foreground_probabilities[:, 2]).clamp_min(0.0)
        )
        activity_scale = activity_change.mean(dim=(-2, -1), keepdim=True).clamp_min(
            self.motion_config.weight_epsilon
        )
        activity_weight = (activity_change / activity_scale).clamp(0.0, 4.0)
        weights = (
            foreground_pair
            * (self.motion_config.confidence_floor + confidence)
            * (self.motion_config.activity_floor + activity_weight)
        )
        denominator = weights.sum(dim=(-2, -1)).clamp_min(self.motion_config.weight_epsilon)
        pooled_log_eta = (weights * local_log_eta).sum(dim=(-2, -1)) / denominator
        maximum = self.config.max_abs_expansion
        expansion = (1.0 - torch.exp(pooled_log_eta)).clamp(
            -maximum * 0.999,
            maximum * 0.999,
        )
        reverse_log_eta = -pooled_log_eta
        reverse_expansion = (1.0 - torch.exp(reverse_log_eta)).clamp(
            -maximum * 0.999,
            maximum * 0.999,
        )
        raw_score = torch.atanh((expansion / maximum).clamp(-0.999, 0.999))
        reverse_raw_score = torch.atanh(
            (reverse_expansion / maximum).clamp(-0.999, 0.999)
        )
        return ObjectEventV48Output(
            expansion=expansion,
            reverse_expansion=reverse_expansion,
            raw_score=raw_score,
            reverse_raw_score=reverse_raw_score,
            reversal_consistency_error=(pooled_log_eta + reverse_log_eta).abs(),
            pooled_log_eta=pooled_log_eta,
            local_log_eta=local_log_eta,
            confidence_logits=confidence_logits,
            confidence_probabilities=confidence,
            aggregation_weights=weights,
            foreground_logits=foreground_logits,
            foreground_probabilities=foreground_probabilities,
            event_activity=activity,
        )


__all__ = ["ObjectEventTTCV48", "ObjectEventV48Config", "ObjectEventV48Output"]
