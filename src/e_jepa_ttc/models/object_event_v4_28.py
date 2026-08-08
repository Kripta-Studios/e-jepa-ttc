"""Object Event TTC v4.28: posterior-supervised profile or 2-D scale correlation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48, _groups
from e_jepa_ttc.models.object_event_v4_27 import ObjectEventTTCV427, ObjectEventV427Config


@dataclass(frozen=True)
class ObjectEventV428Config:
    matcher: str = "profile"
    correlation_dim: int = 48
    log_scale_min: float = -0.22
    log_scale_max: float = 0.22
    scale_bins: int = 45
    correlation_temperature: float = 0.04
    foreground_floor: float = 0.05
    activity_floor: float = 0.05
    rotation_degrees: tuple[float, ...] = (0.0,)
    pyramid_factors: tuple[int, ...] = (1,)
    batch_size: int = 8
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.matcher not in {"profile", "spatial_rotation"}:
            raise ValueError("v4.28 matcher must be profile or spatial_rotation")
        if self.correlation_dim <= 0 or self.scale_bins < 9 or self.scale_bins % 2 == 0:
            raise ValueError("v4.28 requires positive correlation_dim and odd scale_bins >= 9")
        if not self.log_scale_min < 0.0 < self.log_scale_max:
            raise ValueError("v4.28 scale range must straddle zero")
        if min(self.correlation_temperature, self.foreground_floor, self.activity_floor, self.epsilon) <= 0.0:
            raise ValueError("v4.28 numerical controls must be positive")
        if not self.rotation_degrees:
            raise ValueError("v4.28 requires at least one rotation candidate")
        if not self.pyramid_factors or any(int(x) <= 0 for x in self.pyramid_factors):
            raise ValueError("v4.28 pyramid factors must be positive")
        if self.batch_size <= 0:
            raise ValueError("v4.28 batch_size must be positive")


@dataclass
class ObjectEventV428Output:
    expansion: torch.Tensor
    raw_score: torch.Tensor
    predicted_log_eta: torch.Tensor
    scale_logits: torch.Tensor
    scale_probabilities: torch.Tensor
    scale_entropy: torch.Tensor
    log_scale_candidates: torch.Tensor
    expected_rotation_degrees: torch.Tensor
    rotation_entropy: torch.Tensor


def _normalised_entropy(probabilities: torch.Tensor, epsilon: float) -> torch.Tensor:
    entropy = -(probabilities * probabilities.clamp_min(epsilon).log()).sum(dim=-1)
    denominator = math.log(float(probabilities.shape[-1])) if probabilities.shape[-1] > 1 else 1.0
    return entropy / denominator


class ObjectEventTTCV428(nn.Module):
    """Two fixed v4.28 matcher families sharing the same v4.8 event backbone."""

    def __init__(self, backbone: ObjectEventTTCV48, config: ObjectEventV428Config) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "log_scale_candidates",
            torch.linspace(config.log_scale_min, config.log_scale_max, config.scale_bins),
            persistent=True,
        )
        rotations = torch.tensor(config.rotation_degrees, dtype=torch.float32) * (math.pi / 180.0)
        self.register_buffer("rotation_candidates_radians", rotations, persistent=True)
        if config.matcher == "profile":
            self.profile_matcher = ObjectEventTTCV427(
                backbone,
                ObjectEventV427Config(
                    correlation_dim=config.correlation_dim,
                    log_scale_min=config.log_scale_min,
                    log_scale_max=config.log_scale_max,
                    scale_bins=config.scale_bins,
                    correlation_temperature=config.correlation_temperature,
                    foreground_floor=config.foreground_floor,
                    epsilon=config.epsilon,
                ),
            )
            self.backbone = None
            self.feature_projection = None
        else:
            self.profile_matcher = None
            self.backbone = backbone
            in_channels = int(backbone.config.embed_dim) + int(backbone.motion_config.motion_refine_dim)
            hidden = int(config.correlation_dim)
            self.feature_projection = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.GroupNorm(_groups(hidden), hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_groups(hidden), hidden),
            )

    def head_parameters(self) -> list[nn.Parameter]:
        if self.profile_matcher is not None:
            return list(self.profile_matcher.feature_projection.parameters())
        assert self.feature_projection is not None
        return list(self.feature_projection.parameters())

    @staticmethod
    def _centroid_2d(weight: torch.Tensor, epsilon: float) -> torch.Tensor:
        if weight.ndim != 3:
            raise ValueError("weight must be [B,H,W]")
        batch, height, width = weight.shape
        yy = torch.linspace(-1.0, 1.0, height, device=weight.device, dtype=weight.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=weight.device, dtype=weight.dtype)
        total = weight.sum(dim=(-2, -1)).clamp_min(epsilon)
        cy = (weight.sum(dim=-1) * yy[None]).sum(dim=-1) / total
        cx = (weight.sum(dim=-2) * xx[None]).sum(dim=-1) / total
        return torch.stack((cx, cy), dim=-1)

    @staticmethod
    def _event_weight(
        foreground: torch.Tensor,
        activity: torch.Tensor,
        *,
        foreground_floor: float,
        activity_floor: float,
        epsilon: float,
    ) -> torch.Tensor:
        foreground = foreground.clamp(0.0, 1.0)
        scale = activity.mean(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
        activity = (activity / scale).clamp(0.0, 4.0)
        return (foreground_floor + foreground) * (activity_floor + activity)

    @staticmethod
    def _candidate_source_grid(
        *,
        height: int,
        width: int,
        previous_center: torch.Tensor,
        current_center: torch.Tensor,
        log_scales: torch.Tensor,
        rotations: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return [B,K,H,W,2] target->source grids for similarity transforms.

        Scale follows the v4.27 convention eta=h_prev/h_curr. Translation is
        supplied by event-derived centroids. Rotation is a nuisance variable and
        is marginalized rather than interpreted as TTC.
        """
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        target = torch.stack((xx, yy), dim=-1)[None, None]
        centered = target - current_center[:, None, None, None, :]
        scales = torch.exp(log_scales.to(device=device, dtype=dtype))
        rotations = rotations.to(device=device, dtype=dtype)
        scale_grid, rotation_grid = torch.meshgrid(scales, rotations, indexing="ij")
        scale_flat = scale_grid.reshape(-1)
        rotation_flat = rotation_grid.reshape(-1)
        cos = torch.cos(rotation_flat)[None, :, None, None]
        sin = torch.sin(rotation_flat)[None, :, None, None]
        x = centered[..., 0]
        y = centered[..., 1]
        # Inverse nuisance rotation maps target coordinates back to source.
        xr = cos * x + sin * y
        yr = -sin * x + cos * y
        source_offset = torch.stack((xr, yr), dim=-1) * scale_flat[None, :, None, None, None]
        return previous_center[:, None, None, None, :] + source_offset

    def _spatial_level_scores(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        previous_weight: torch.Tensor,
        current_weight: torch.Tensor,
        previous_center: torch.Tensor,
        current_center: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = previous.shape
        grid = self._candidate_source_grid(
            height=height,
            width=width,
            previous_center=previous_center,
            current_center=current_center,
            log_scales=self.log_scale_candidates,
            rotations=self.rotation_candidates_radians,
            dtype=previous.dtype,
            device=previous.device,
        )
        candidates = grid.shape[1]
        source = previous[:, None].expand(batch, candidates, channels, height, width).reshape(
            batch * candidates, channels, height, width
        )
        source_weight = previous_weight[:, None, None].expand(
            batch, candidates, 1, height, width
        ).reshape(batch * candidates, 1, height, width)
        flat_grid = grid.reshape(batch * candidates, height, width, 2)
        warped = functional.grid_sample(
            source, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        warped_weight = functional.grid_sample(
            source_weight, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)
        warped = functional.normalize(warped, dim=1, eps=self.config.epsilon).reshape(
            batch, candidates, channels, height, width
        )
        current = functional.normalize(current, dim=1, eps=self.config.epsilon)
        similarity = (warped * current[:, None]).sum(dim=2)
        overlap = (warped_weight.reshape(batch, candidates, height, width) * current_weight[:, None]).clamp_min(0.0)
        numerator = (similarity * overlap).sum(dim=(-2, -1))
        denominator = overlap.sum(dim=(-2, -1)).clamp_min(self.config.epsilon)
        return numerator / denominator

    def _spatial_scores(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        previous_weight: torch.Tensor,
        current_weight: torch.Tensor,
    ) -> torch.Tensor:
        previous_center = self._centroid_2d(previous_weight, self.config.epsilon)
        current_center = self._centroid_2d(current_weight, self.config.epsilon)
        scores: list[torch.Tensor] = []
        for factor in self.config.pyramid_factors:
            factor = int(factor)
            if factor == 1:
                prev_level, curr_level = previous, current
                prev_weight, curr_weight = previous_weight, current_weight
            else:
                prev_level = functional.avg_pool2d(previous, kernel_size=factor, stride=factor)
                curr_level = functional.avg_pool2d(current, kernel_size=factor, stride=factor)
                prev_weight = functional.avg_pool2d(previous_weight[:, None], factor, factor).squeeze(1)
                curr_weight = functional.avg_pool2d(current_weight[:, None], factor, factor).squeeze(1)
            scores.append(
                self._spatial_level_scores(
                    prev_level,
                    curr_level,
                    prev_weight,
                    curr_weight,
                    previous_center,
                    current_center,
                )
            )
        return torch.stack(scores, dim=0).mean(dim=0)

    def _spatial_forward(self, events: torch.Tensor) -> ObjectEventV428Output:
        assert self.feature_projection is not None and self.backbone is not None
        maps, _, foreground, activity = self.backbone._foreground_and_features(events)
        if maps.ndim != 5 or maps.shape[1] != 3:
            raise ValueError("v4.28 expects exactly three event steps")
        temporal = self.backbone.temporal_projection(self.backbone._temporal_maps(maps))
        temporal = functional.interpolate(temporal, size=maps.shape[-2:], mode="bilinear", align_corners=False)
        previous = self.feature_projection(torch.cat((maps[:, 1], temporal), dim=1))
        current = self.feature_projection(torch.cat((maps[:, 2], temporal), dim=1))
        foreground_pair = functional.interpolate(
            foreground[:, 1:3], size=maps.shape[-2:], mode="bilinear", align_corners=False
        )
        activity_pair = functional.interpolate(
            activity[:, 1:3], size=maps.shape[-2:], mode="bilinear", align_corners=False
        )
        previous_weight = self._event_weight(
            foreground_pair[:, 0], activity_pair[:, 0],
            foreground_floor=self.config.foreground_floor,
            activity_floor=self.config.activity_floor,
            epsilon=self.config.epsilon,
        )
        current_weight = self._event_weight(
            foreground_pair[:, 1], activity_pair[:, 1],
            foreground_floor=self.config.foreground_floor,
            activity_floor=self.config.activity_floor,
            epsilon=self.config.epsilon,
        )
        joint_scores = self._spatial_scores(previous, current, previous_weight, current_weight)
        scale_count = self.log_scale_candidates.numel()
        rotation_count = self.rotation_candidates_radians.numel()
        joint_logits = joint_scores.reshape(events.shape[0], scale_count, rotation_count) / self.config.correlation_temperature
        # Rotation is a nuisance variable. Marginalize it in log-space before
        # estimating the TTC-relevant scale posterior.
        scale_logits = torch.logsumexp(joint_logits, dim=-1) - math.log(float(rotation_count))
        scale_probabilities = torch.softmax(scale_logits, dim=-1)
        candidates = self.log_scale_candidates.to(dtype=scale_probabilities.dtype)
        predicted_log_eta = (scale_probabilities * candidates[None]).sum(dim=-1)
        rotation_joint = torch.softmax(joint_logits.reshape(events.shape[0], -1), dim=-1).reshape_as(joint_logits)
        rotation_probabilities = rotation_joint.sum(dim=1)
        rotation_degrees = self.rotation_candidates_radians.to(rotation_probabilities.dtype) * (180.0 / math.pi)
        expected_rotation = (rotation_probabilities * rotation_degrees[None]).sum(dim=-1)
        scale_entropy = _normalised_entropy(scale_probabilities, self.config.epsilon)
        rotation_entropy = _normalised_entropy(rotation_probabilities, self.config.epsilon)
        maximum = float(self.backbone.config.max_abs_expansion)
        expansion = (1.0 - torch.exp(predicted_log_eta)).clamp(-maximum * 0.999, maximum * 0.999)
        raw_score = torch.atanh((expansion / maximum).clamp(-0.999, 0.999))
        return ObjectEventV428Output(
            expansion=expansion,
            raw_score=raw_score,
            predicted_log_eta=predicted_log_eta,
            scale_logits=scale_logits,
            scale_probabilities=scale_probabilities,
            scale_entropy=scale_entropy,
            log_scale_candidates=self.log_scale_candidates,
            expected_rotation_degrees=expected_rotation,
            rotation_entropy=rotation_entropy,
        )

    def forward(self, events: torch.Tensor) -> ObjectEventV428Output:
        if self.profile_matcher is None:
            return self._spatial_forward(events)
        output = self.profile_matcher(events)
        zeros = torch.zeros_like(output.predicted_log_eta)
        return ObjectEventV428Output(
            expansion=output.expansion,
            raw_score=output.raw_score,
            predicted_log_eta=output.predicted_log_eta,
            scale_logits=output.scale_logits,
            scale_probabilities=output.scale_probabilities,
            scale_entropy=output.scale_entropy,
            log_scale_candidates=self.log_scale_candidates,
            expected_rotation_degrees=zeros,
            rotation_entropy=zeros,
        )


__all__ = ["ObjectEventTTCV428", "ObjectEventV428Config", "ObjectEventV428Output"]
