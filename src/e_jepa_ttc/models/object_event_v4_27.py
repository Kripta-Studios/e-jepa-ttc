"""Object Event TTC v4.27: event-native vertical scale-correlation LHR head."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48, _groups


@dataclass(frozen=True)
class ObjectEventV427Config:
    correlation_dim: int = 48
    log_scale_min: float = -0.22
    log_scale_max: float = 0.22
    scale_bins: int = 45
    correlation_temperature: float = 0.06
    foreground_floor: float = 0.05
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.correlation_dim <= 0 or self.scale_bins < 9 or self.scale_bins % 2 == 0:
            raise ValueError("v4.27 requires positive correlation_dim and an odd scale_bins >= 9")
        if not self.log_scale_min < 0.0 < self.log_scale_max:
            raise ValueError("v4.27 scale range must straddle zero")
        if min(self.correlation_temperature, self.foreground_floor, self.epsilon) <= 0.0:
            raise ValueError("v4.27 numerical controls must be positive")


@dataclass
class ObjectEventV427Output:
    expansion: torch.Tensor
    raw_score: torch.Tensor
    predicted_log_eta: torch.Tensor
    scale_logits: torch.Tensor
    scale_probabilities: torch.Tensor
    scale_entropy: torch.Tensor
    foreground_previous: torch.Tensor
    foreground_current: torch.Tensor


class ObjectEventTTCV427(nn.Module):
    """Estimate visible-height log-ratio by differentiable scale matching.

    The wrapped v4.8 backbone supplies event-only geometry features and learned
    foreground maps. Boxes/heights are never forward inputs. A small projection
    is learned, but TTC itself is constrained to come from a soft correlation
    search over physically meaningful vertical scale hypotheses.
    """

    def __init__(self, backbone: ObjectEventTTCV48, config: ObjectEventV427Config | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or ObjectEventV427Config()
        channels = int(backbone.config.embed_dim)
        hidden = self.config.correlation_dim
        self.feature_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
        )
        self.register_buffer(
            "log_scale_candidates",
            torch.linspace(self.config.log_scale_min, self.config.log_scale_max, self.config.scale_bins),
            persistent=True,
        )

    @staticmethod
    def _vertical_centroid(mask: torch.Tensor, epsilon: float) -> torch.Tensor:
        if mask.ndim != 3:
            raise ValueError("foreground mask must be [B,H,W]")
        h = mask.shape[-2]
        y = torch.linspace(-1.0, 1.0, h, device=mask.device, dtype=mask.dtype)
        marginal = mask.sum(dim=-1)
        return (marginal * y[None]).sum(dim=-1) / marginal.sum(dim=-1).clamp_min(epsilon)

    @staticmethod
    def _profile(features: torch.Tensor, mask: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask[:, None].clamp_min(0.0)
        profile = (features * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(epsilon)
        profile = functional.normalize(profile, dim=1, eps=epsilon)
        vertical_weight = mask.mean(dim=-1)
        return profile, vertical_weight

    @staticmethod
    def _stable_overlap(
        warped_weight: torch.Tensor,
        current_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Stable foreground conjunction for the correlation score.

        The original v4.27 used ``sqrt(warped * current)``. Because
        ``grid_sample(..., padding_mode="zeros")`` produces exact zeros for
        out-of-frame hypotheses, that expression has an infinite derivative at
        zero and can poison the first optimizer step with non-finite gradients.
        The product keeps the same support semantics (zero overlap remains zero)
        while having a bounded derivative everywhere.
        """
        return (warped_weight * current_weight).clamp_min(0.0)

    def _warp_profile(
        self,
        source: torch.Tensor,
        source_weight: torch.Tensor,
        source_center: torch.Tensor,
        target_center: torch.Tensor,
        log_eta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height = source.shape
        y_target = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
        ratio = torch.exp(log_eta).to(dtype=source.dtype)
        y_source = source_center[:, None] + ratio[:, None] * (y_target[None] - target_center[:, None])
        x = torch.zeros_like(y_source)
        grid = torch.stack((x, y_source), dim=-1).unsqueeze(2)
        warped = functional.grid_sample(
            source.unsqueeze(-1), grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(-1)
        warped_weight = functional.grid_sample(
            source_weight[:, None, :, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1).squeeze(-1)
        return warped, warped_weight

    def forward(self, events: torch.Tensor) -> ObjectEventV427Output:
        maps, _, foreground, _ = self.backbone._foreground_and_features(events)
        if maps.shape[1] != 3:
            raise ValueError("v4.27 expects exactly three event steps")
        previous = self.feature_projection(maps[:, 1])
        current = self.feature_projection(maps[:, 2])
        size = previous.shape[-2:]
        fg = functional.interpolate(foreground[:, 1:3], size=size, mode="bilinear", align_corners=False)
        fg_previous = fg[:, 0].clamp_min(self.config.foreground_floor)
        fg_current = fg[:, 1].clamp_min(self.config.foreground_floor)
        prev_profile, prev_weight = self._profile(previous, fg_previous, self.config.epsilon)
        curr_profile, curr_weight = self._profile(current, fg_current, self.config.epsilon)
        prev_center = self._vertical_centroid(fg_previous, self.config.epsilon)
        curr_center = self._vertical_centroid(fg_current, self.config.epsilon)

        scores: list[torch.Tensor] = []
        for candidate in self.log_scale_candidates:
            log_eta = candidate.expand(events.shape[0])
            warped, warped_weight = self._warp_profile(
                prev_profile, prev_weight, prev_center, curr_center, log_eta
            )
            local_similarity = (warped * curr_profile).sum(dim=1)
            overlap = self._stable_overlap(warped_weight, curr_weight)
            score = (local_similarity * overlap).sum(dim=-1) / overlap.sum(dim=-1).clamp_min(
                self.config.epsilon
            )
            scores.append(score)
        logits = torch.stack(scores, dim=1) / self.config.correlation_temperature
        probabilities = torch.softmax(logits, dim=1)
        candidates = self.log_scale_candidates.to(dtype=probabilities.dtype)
        predicted_log_eta = (probabilities * candidates[None]).sum(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(self.config.epsilon).log()).sum(dim=1)
        entropy = entropy / torch.log(
            torch.tensor(float(self.config.scale_bins), device=entropy.device, dtype=entropy.dtype)
        )
        maximum = float(self.backbone.config.max_abs_expansion)
        expansion = (1.0 - torch.exp(predicted_log_eta)).clamp(-maximum * 0.999, maximum * 0.999)
        raw_score = torch.atanh((expansion / maximum).clamp(-0.999, 0.999))
        return ObjectEventV427Output(
            expansion=expansion,
            raw_score=raw_score,
            predicted_log_eta=predicted_log_eta,
            scale_logits=logits,
            scale_probabilities=probabilities,
            scale_entropy=entropy,
            foreground_previous=fg_previous,
            foreground_current=fg_current,
        )


__all__ = ["ObjectEventTTCV427", "ObjectEventV427Config", "ObjectEventV427Output"]
