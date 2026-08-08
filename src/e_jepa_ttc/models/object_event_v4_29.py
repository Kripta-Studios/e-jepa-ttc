"""Event-only local-affine correspondence model for Object Event TTC v4.29.

The affine convention is ``Y_previous = X_current @ A.T + t`` in normalized
feature-map coordinates.  The module deliberately accepts only ``[B,3,C,H,W]``
event tensors; labels, boxes and identifiers live exclusively in training code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48, _groups


@dataclass(frozen=True)
class ObjectEventV429Config:
    correlation_dim: int = 48
    adjacent_radius: int = 4
    direct_radius: int = 7
    temperature: float = 0.07
    ridge: float = 1.0e-3
    huber_delta: float = 0.08
    foreground_floor: float = 0.05
    activity_floor: float = 0.05
    support_threshold: float = 0.02
    min_effective_mass: float = 4.0
    max_condition_number: float = 100.0
    min_determinant: float = 0.05
    epsilon: float = 1.0e-6
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.correlation_dim <= 0 or self.adjacent_radius < 1 or self.direct_radius < 1:
            raise ValueError("v4.29 correlation dimensions and radii must be positive")
        if (
            min(
                self.temperature,
                self.ridge,
                self.huber_delta,
                self.foreground_floor,
                self.activity_floor,
                self.support_threshold,
                self.epsilon,
            )
            <= 0.0
        ):
            raise ValueError("v4.29 numerical controls must be positive")
        if self.min_effective_mass <= 0.0 or self.max_condition_number <= 1.0:
            raise ValueError("v4.29 effective mass and condition limits are invalid")
        if self.min_determinant <= 0.0 or self.batch_size <= 0:
            raise ValueError("v4.29 determinant floor and batch size must be positive")


@dataclass
class LocalAffineFit:
    matrix: torch.Tensor
    translation: torch.Tensor
    determinant: torch.Tensor
    condition_number: torch.Tensor
    effective_weight_mass: torch.Tensor
    residual: torch.Tensor
    valid: torch.Tensor


@dataclass
class LocalCorrelation:
    displacement: torch.Tensor
    entropy: torch.Tensor
    confidence: torch.Tensor
    boundary_probability: torch.Tensor
    weight: torch.Tensor


@dataclass
class ObjectEventV429Output:
    expansion: torch.Tensor
    predicted_log_eta_vertical: torch.Tensor
    predicted_log_eta_horizontal: torch.Tensor
    predicted_log_eta_area: torch.Tensor
    affine_01: LocalAffineFit
    affine_12: LocalAffineFit
    affine_02: LocalAffineFit
    composition_matrix_error: torch.Tensor
    composition_translation_error: torch.Tensor
    correlation_entropy: torch.Tensor
    correlation_confidence: torch.Tensor
    boundary_probability: torch.Tensor
    rotation_radians: torch.Tensor
    singular_value_anisotropy: torch.Tensor
    translation_magnitude: torch.Tensor
    validity_penalty: torch.Tensor


def normalized_coordinate_grid(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return feature-pixel *centres* as ``[H,W,2]`` in ``[-1,1]``.

    Feature/ROI coordinates use the edge convention: centre ``i`` is
    ``(i + .5) * 2 / size - 1``. Consequently a one-cell displacement is
    ``2 / size`` rather than ``2 / (size - 1)``.
    """
    if height < 2 or width < 2:
        raise ValueError("local affine estimation requires a feature map at least 2x2")
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0,
        (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0,
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1)


def compose_affines(
    first: LocalAffineFit, second: LocalAffineFit
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose current->middle ``second`` then middle->previous ``first``."""
    return first.matrix @ second.matrix, (first.matrix @ second.translation[..., None]).squeeze(
        -1
    ) + first.translation


class ObjectEventTTCV429(nn.Module):
    """Shared-projection, bounded local cosine-correspondence affine estimator."""

    def __init__(
        self, backbone: ObjectEventTTCV48, config: ObjectEventV429Config | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or ObjectEventV429Config()
        incoming = int(backbone.config.embed_dim) + int(backbone.motion_config.motion_refine_dim)
        hidden = self.config.correlation_dim
        self.local_projection = nn.Sequential(
            nn.Conv2d(incoming, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
        )

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.local_projection.parameters())

    def _weights(self, foreground: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        # Normalize activity per sample so static background cannot win by area.
        normalized_activity = activity / activity.mean(dim=(-2, -1), keepdim=True).clamp_min(
            self.config.epsilon
        )
        normalized_activity = normalized_activity.clamp(0.0, 4.0)
        foreground = foreground.clamp(0.0, 1.0)
        raw = (self.config.foreground_floor + foreground) * (
            self.config.activity_floor + normalized_activity
        )
        support = (activity > self.config.support_threshold).to(raw.dtype)
        # Retain a finite floor only for active pixels; inactive background has zero support.
        return raw * support

    def _local_correlation(
        self, previous: torch.Tensor, current: torch.Tensor, weight: torch.Tensor, radius: int
    ) -> LocalCorrelation:
        batch, channels, height, width = previous.shape
        kside = 2 * radius + 1
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=previous.device),
            torch.arange(-radius, radius + 1, device=previous.device),
            indexing="ij",
        )
        offsets = torch.stack((offsets_x.reshape(-1), offsets_y.reshape(-1)), dim=-1)
        patches = functional.unfold(previous, kernel_size=kside, padding=radius).reshape(
            batch, channels, kside * kside, height, width
        )
        curr = functional.normalize(current, dim=1, eps=self.config.epsilon)[:, :, None]
        prev = functional.normalize(patches, dim=1, eps=self.config.epsilon)
        logits = (curr * prev).sum(dim=1) / self.config.temperature
        yy, xx = torch.meshgrid(
            torch.arange(height, device=previous.device),
            torch.arange(width, device=previous.device),
            indexing="ij",
        )
        source_x = xx[None, None] + offsets[:, 0, None, None]
        source_y = yy[None, None] + offsets[:, 1, None, None]
        inside = (source_x >= 0) & (source_x < width) & (source_y >= 0) & (source_y < height)
        logits = logits.masked_fill(~inside, float("-inf"))
        probability = torch.softmax(logits, dim=1)
        dx = offsets[:, 0].to(current.dtype) * (2.0 / float(width))
        dy = offsets[:, 1].to(current.dtype) * (2.0 / float(height))
        displacement = torch.stack(
            (
                (probability * dx[None, :, None, None]).sum(1),
                (probability * dy[None, :, None, None]).sum(1),
            ),
            dim=-1,
        )
        entropy_numerator = -(probability * probability.clamp_min(self.config.epsilon).log()).sum(1)
        valid_count = inside.squeeze(0).sum(dim=0).to(current.dtype)
        entropy_denominator = valid_count.clamp_min(2.0).log()
        # Normalise by candidates actually available at each position.  A corner
        # with four candidates is therefore not spuriously more confident.
        entropy = entropy_numerator / entropy_denominator[None]
        boundary = (offsets.abs().amax(dim=1) == radius).to(probability.dtype)
        boundary_probability = (probability * boundary[None, :, None, None]).sum(1).clamp(0.0, 1.0)
        confidence = probability.max(dim=1).values * (1.0 - entropy)
        return LocalCorrelation(
            displacement, entropy, confidence, boundary_probability, weight * confidence
        )

    def _fit_affine(self, correlation: LocalCorrelation) -> tuple[LocalAffineFit, torch.Tensor]:
        weight = correlation.weight
        batch, height, width = weight.shape
        coords = (
            normalized_coordinate_grid(height, width, device=weight.device, dtype=weight.dtype)
            .reshape(1, -1, 2)
            .expand(batch, -1, -1)
        )
        target = coords + correlation.displacement.reshape(batch, -1, 2)
        input_finite = torch.isfinite(weight).all(dim=(-2, -1)) & torch.isfinite(target).all(
            dim=(-2, -1)
        )
        flat_weight = weight.reshape(batch, -1)
        w = torch.where(
            torch.isfinite(flat_weight), flat_weight, torch.zeros_like(flat_weight)
        ).clamp_min(0.0)
        target = torch.where(torch.isfinite(target), target, coords)
        design = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)
        ridge = torch.eye(3, device=weight.device, dtype=weight.dtype)[None] * self.config.ridge
        target_beta = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], device=weight.device, dtype=weight.dtype
        )[None]

        def solve(local_w: torch.Tensor) -> torch.Tensor:
            normal = torch.einsum("bn,bni,bnj->bij", local_w, design, design) + ridge
            rhs = (
                torch.einsum("bn,bni,bnj->bij", local_w, design, target)
                + self.config.ridge * target_beta
            )
            return torch.linalg.solve(normal, rhs)

        beta0 = solve(w)
        residual0 = torch.linalg.vector_norm(
            torch.einsum("bni,bij->bnj", design, beta0) - target, dim=-1
        )
        irls = torch.where(
            residual0.detach() <= self.config.huber_delta,
            torch.ones_like(residual0),
            self.config.huber_delta / residual0.detach().clamp_min(self.config.epsilon),
        )
        robust_w = w * irls
        beta = solve(robust_w)
        matrix = beta[:, :2, :].transpose(-1, -2)
        translation = beta[:, 2, :]
        predicted = (
            torch.einsum("bni,bij->bnj", coords, matrix.transpose(-1, -2)) + translation[:, None]
        )
        robust_mass = robust_w.sum(-1)
        residual = (robust_w * torch.linalg.vector_norm(predicted - target, dim=-1)).sum(
            -1
        ) / robust_mass.clamp_min(self.config.epsilon)
        normal = torch.einsum("bn,bni,bnj->bij", robust_w, design, design) + ridge
        condition = torch.linalg.cond(normal)
        determinant = torch.linalg.det(matrix)
        finite = (
            input_finite
            & torch.isfinite(matrix).all(dim=(-2, -1))
            & torch.isfinite(translation).all(dim=-1)
            & torch.isfinite(condition)
            & torch.isfinite(residual)
            & torch.isfinite(determinant)
        )
        valid = (
            finite
            & (robust_mass >= self.config.min_effective_mass)
            & (condition <= self.config.max_condition_number)
            & (determinant > self.config.min_determinant)
        )
        validity_penalty = (
            functional.relu(self.config.min_determinant - determinant) / self.config.min_determinant
            + functional.relu(condition - self.config.max_condition_number)
            / self.config.max_condition_number
            + functional.relu(self.config.min_effective_mass - robust_mass)
            / self.config.min_effective_mass
        )
        return LocalAffineFit(
            matrix, translation, determinant, condition, robust_mass, residual, valid
        ), validity_penalty

    def forward(self, events: torch.Tensor) -> ObjectEventV429Output:
        if events.ndim != 5 or events.shape[1] != 3:
            raise ValueError("v4.29 forward accepts only events [B,3,C,H,W]")
        maps, _, foreground, activity = self.backbone._foreground_and_features(events)
        temporal = self.backbone.temporal_projection(self.backbone._temporal_maps(maps))
        temporal = functional.interpolate(
            temporal, size=maps.shape[-2:], mode="bilinear", align_corners=False
        )
        projected = self.local_projection(
            torch.cat(
                (
                    maps.reshape(-1, *maps.shape[2:]),
                    temporal[:, None].expand(-1, 3, -1, -1, -1).reshape(-1, *temporal.shape[1:]),
                ),
                dim=1,
            )
        ).reshape(events.shape[0], 3, -1, *maps.shape[-2:])
        fg = functional.interpolate(
            foreground, size=maps.shape[-2:], mode="bilinear", align_corners=False
        )
        act = functional.interpolate(
            activity, size=maps.shape[-2:], mode="bilinear", align_corners=False
        )
        # Current step supplies support, so the same event-only weighting is used for all pairs.
        c01 = self._local_correlation(
            projected[:, 0],
            projected[:, 1],
            self._weights(fg[:, 1], act[:, 1]),
            self.config.adjacent_radius,
        )
        c12 = self._local_correlation(
            projected[:, 1],
            projected[:, 2],
            self._weights(fg[:, 2], act[:, 2]),
            self.config.adjacent_radius,
        )
        c02 = self._local_correlation(
            projected[:, 0],
            projected[:, 2],
            self._weights(fg[:, 2], act[:, 2]),
            self.config.direct_radius,
        )
        (a01, penalty01), (a12, penalty12), (a02, penalty02) = (
            self._fit_affine(c01),
            self._fit_affine(c12),
            self._fit_affine(c02),
        )
        composed_a, composed_t = compose_affines(a01, a12)
        valid = a12.valid
        ex = torch.tensor([1.0, 0.0], device=events.device, dtype=events.dtype)
        ey = torch.tensor([0.0, 1.0], device=events.device, dtype=events.dtype)
        horizontal = (
            torch.linalg.vector_norm(a12.matrix @ ex, dim=-1).clamp_min(self.config.epsilon).log()
        )
        vertical = (
            torch.linalg.vector_norm(a12.matrix @ ey, dim=-1).clamp_min(self.config.epsilon).log()
        )
        area = 0.5 * a12.determinant.clamp_min(self.config.epsilon).log()
        nan = torch.full_like(vertical, float("nan"))
        horizontal, vertical, area = (
            torch.where(valid, horizontal, nan),
            torch.where(valid, vertical, nan),
            torch.where(valid, area, nan),
        )
        expansion = torch.where(valid, 1.0 - torch.exp(vertical), nan)
        singular = torch.linalg.svdvals(a12.matrix)
        rotation = torch.atan2(
            a12.matrix[:, 1, 0] - a12.matrix[:, 0, 1], a12.matrix[:, 0, 0] + a12.matrix[:, 1, 1]
        )
        return ObjectEventV429Output(
            expansion,
            vertical,
            horizontal,
            area,
            a01,
            a12,
            a02,
            torch.linalg.matrix_norm(a02.matrix - composed_a, ord="fro", dim=(-2, -1)),
            torch.linalg.vector_norm(a02.translation - composed_t, dim=-1),
            c12.entropy.mean((-2, -1)),
            c12.confidence.mean((-2, -1)),
            c12.boundary_probability.mean((-2, -1)),
            rotation,
            (singular[:, 0] / singular[:, 1].clamp_min(self.config.epsilon)).log(),
            torch.linalg.vector_norm(a12.translation, dim=-1),
            penalty01 + penalty12 + penalty02,
        )


__all__ = [
    "LocalAffineFit",
    "LocalCorrelation",
    "ObjectEventTTCV429",
    "ObjectEventV429Config",
    "ObjectEventV429Output",
    "compose_affines",
    "normalized_coordinate_grid",
]
