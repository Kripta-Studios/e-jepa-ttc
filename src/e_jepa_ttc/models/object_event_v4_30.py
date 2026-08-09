"""Finite event-only multiscale similarity correspondence for Object Event TTC v4.30.

The only forward argument is an event voxel tensor ``[B,3,C,H,W]``.  Labels,
boxes, IDs, and metadata are deliberately absent.  For non-zero raw event mass
the Cholesky-ridge solver produces a finite similarity estimate and covariance;
an exactly-zero event triplet is represented explicitly as ``UNKNOWN``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48, _groups


@dataclass(frozen=True)
class ObjectEventV430Config:
    arm: str = "stable_multiscale_similarity"
    correlation_dim: int = 48
    scales: tuple[int, ...] = (1, 2, 4)
    temperature: float = 0.07
    ridge: float = 0.01
    huber_delta: float = 0.08
    huber_passes: int = 3
    tile_size: int = 4
    whitening_shrinkage: float = 0.10
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.arm not in {
            "stable_multiscale_similarity",
            "stable_multiscale_similarity_normal_flow",
        }:
            raise ValueError("v4.30 has exactly the two locked arms")
        if tuple(self.scales) != (1, 2, 4):
            raise ValueError("v4.30 fixes local correlation scales to (1, 2, 4)")
        if self.correlation_dim <= 0 or self.tile_size != 4 or self.huber_passes != 3:
            raise ValueError("v4.30 fixes positive dimensions, 4x4 tiles, and three IRLS passes")
        if min(self.temperature, self.ridge, self.huber_delta, self.epsilon) <= 0.0:
            raise ValueError("v4.30 numerical controls must be positive")
        if not 0.0 <= self.whitening_shrinkage <= 1.0:
            raise ValueError("whitening_shrinkage must be in [0, 1]")


@dataclass
class LocalPosterior:
    """A local displacement posterior, retaining mean and covariance per position."""

    probabilities: torch.Tensor  # [B,K,H,W]
    offsets: torch.Tensor  # [K,2] in feature pixels
    mean: torch.Tensor  # [B,H,W,2] in feature pixels
    covariance: torch.Tensor  # [B,H,W,2,2] in feature pixels squared
    entropy: torch.Tensor  # [B,H,W]
    confidence: torch.Tensor  # [B,H,W]
    boundary_probability: torch.Tensor  # [B,H,W]
    weight: torch.Tensor  # [B,H,W]


@dataclass
class SimilarityFit:
    """Centred four-parameter similarity displacement fit and posterior."""

    kappa: torch.Tensor
    omega: torch.Tensor
    translation: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor  # [B,4,4]
    sigma2: torch.Tensor
    residual: torch.Tensor
    effective_mass: torch.Tensor
    design_rms: torch.Tensor
    matrix: torch.Tensor


@dataclass
class ObjectEventV430Output:
    expansion: torch.Tensor
    log_eta: torch.Tensor
    posterior_variance: torch.Tensor
    unknown: torch.Tensor
    fit_01: SimilarityFit
    fit_12: SimilarityFit
    fit_02: SimilarityFit
    cycle_matrix_error: torch.Tensor
    cycle_translation_error: torch.Tensor
    correlation_entropy: torch.Tensor
    correlation_confidence: torch.Tensor
    boundary_probability: torch.Tensor
    rotation_radians: torch.Tensor
    translation_magnitude: torch.Tensor
    normal_flow_residual: torch.Tensor
    foreground_map_t1: torch.Tensor
    foreground_map_t2: torch.Tensor
    support_map_t2: torch.Tensor
    posteriors_01: dict[int, LocalPosterior]
    posteriors_12: dict[int, LocalPosterior]


def feature_coordinate_grid(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return feature-pixel centres as ``[H,W,2]`` in feature-pixel units."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1)


def shrinkage_whiten(
    features: torch.Tensor, shrinkage: float = 0.10, epsilon: float = 1e-6
) -> torch.Tensor:
    """Center and shrinkage-whiten dense maps independently per sample.

    This is used for locked teacher feature maps before posterior consensus, not
    for labels or metadata.  The eigensystem is regularized by a diagonal
    shrinkage target and is finite for constant maps.
    """
    if features.ndim != 4:
        raise ValueError("dense features must be [B,C,H,W]")
    batch, channels, height, width = features.shape
    flat = features.reshape(batch, channels, height * width)
    centered = flat - flat.mean(dim=-1, keepdim=True)
    cov = centered @ centered.transpose(-1, -2) / float(max(height * width, 1))
    diagonal = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    target = torch.eye(channels, device=features.device, dtype=features.dtype)[None]
    scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(epsilon)
    regularized = (
        (1.0 - shrinkage) * cov + shrinkage * diagonal + epsilon * scale[..., None] * target
    )
    values, vectors = torch.linalg.eigh(regularized)
    inverse_sqrt = (
        vectors @ torch.diag_embed(values.clamp_min(epsilon).rsqrt()) @ vectors.transpose(-1, -2)
    )
    return (inverse_sqrt @ centered).reshape_as(features)


def geometric_mean_posteriors(
    posteriors: Iterable[torch.Tensor], epsilon: float = 1e-6
) -> torch.Tensor:
    """Consensus ``exp(mean(log P_teacher))`` normalized over candidates."""
    values = list(posteriors)
    if not values:
        raise ValueError("at least one teacher posterior is required")
    stacked = torch.stack(values, dim=0).clamp_min(epsilon)
    log_consensus = stacked.log().mean(dim=0)
    return torch.softmax(log_consensus, dim=1)


class ObjectEventTTCV430(nn.Module):
    """Event-only student using fixed-scale local correspondence and similarity WLS."""

    def __init__(
        self, backbone: ObjectEventTTCV48, config: ObjectEventV430Config | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or ObjectEventV430Config()
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

    def _activity(self, events: torch.Tensor, *, output_size: tuple[int, int]) -> torch.Tensor:
        """Locked raw-voxel activity from channels 0:10; never v4.7 normalization."""
        if events.shape[2] < 10:
            raise ValueError("v4.30 requires raw voxel channels 0:10 for activity")
        raw = events[:, :, 0:10].abs().sum(dim=2)
        positive = raw > 0
        denominator = (raw.square() * positive).sum(dim=(-2, -1), keepdim=True) / positive.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1)
        activity = (raw / denominator.sqrt().clamp_min(self.config.epsilon)).clamp(0.0, 4.0)
        return functional.interpolate(
            activity, size=output_size, mode="bilinear", align_corners=False
        )

    def _tile_normalize(self, weight: torch.Tensor) -> torch.Tensor:
        """Give every non-empty fixed 4x4 tile unit capped mass without a threshold."""
        batch, height, width = weight.shape
        pad_h = (-height) % self.config.tile_size
        pad_w = (-width) % self.config.tile_size
        padded = functional.pad(weight, (0, pad_w, 0, pad_h))
        h4, w4 = padded.shape[-2:]
        tiles = padded.reshape(batch, h4 // 4, 4, w4 // 4, 4).permute(0, 1, 3, 2, 4)
        mass = tiles.sum(dim=(-2, -1), keepdim=True)
        normalized = torch.where(mass > 0, tiles / mass.clamp_min(self.config.epsilon), tiles)
        # A point cannot carry more than its tile's fixed budget.
        normalized = normalized.clamp_max(1.0)
        return normalized.permute(0, 1, 3, 2, 4).reshape(batch, h4, w4)[..., :height, :width]

    def _weights(
        self,
        foreground: torch.Tensor,
        activity: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        confidence = torch.ones_like(activity) if confidence is None else confidence
        return self._tile_normalize(
            (0.05 + foreground.clamp(0.0, 1.0)) * activity * confidence.clamp_min(0.0)
        )

    def _posterior(
        self, previous: torch.Tensor, current: torch.Tensor, base_weight: torch.Tensor, radius: int
    ) -> LocalPosterior:
        batch, channels, height, width = previous.shape
        side = 2 * radius + 1
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=previous.device),
            torch.arange(-radius, radius + 1, device=previous.device),
            indexing="ij",
        )
        offsets = torch.stack((offsets_x.reshape(-1), offsets_y.reshape(-1)), dim=-1).to(
            previous.dtype
        )
        patches = functional.unfold(previous, kernel_size=side, padding=radius).reshape(
            batch, channels, side * side, height, width
        )
        logits = (
            functional.normalize(current, dim=1, eps=self.config.epsilon)[:, :, None]
            * functional.normalize(patches, dim=1, eps=self.config.epsilon)
        ).sum(dim=1) / self.config.temperature
        yy, xx = torch.meshgrid(
            torch.arange(height, device=previous.device),
            torch.arange(width, device=previous.device),
            indexing="ij",
        )
        source_x = xx[None, None] + offsets[:, 0, None, None]
        source_y = yy[None, None] + offsets[:, 1, None, None]
        inside = (source_x >= 0) & (source_x < width) & (source_y >= 0) & (source_y < height)
        probabilities = torch.softmax(logits.masked_fill(~inside, float("-inf")), dim=1)
        mean = torch.einsum("bkhw,ki->bhwi", probabilities, offsets)
        second = torch.einsum("bkhw,ki,kj->bhwij", probabilities, offsets, offsets)
        covariance = second - mean[..., :, None] * mean[..., None, :]
        entropy = -(probabilities * probabilities.clamp_min(self.config.epsilon).log()).sum(dim=1)
        count = inside.squeeze(0).sum(dim=0).to(previous.dtype).clamp_min(2.0)
        entropy = entropy / count.log()[None]
        boundary = (offsets.abs().amax(dim=1) == radius).to(previous.dtype)
        boundary_probability = (probabilities * boundary[None, :, None, None]).sum(dim=1)
        confidence = probabilities.max(dim=1).values * (1.0 - entropy).clamp_min(0.0)
        # ``base_weight`` already is (0.05 + foreground) * raw activity with
        # fixed tile balancing.  Confidence is the only posterior-derived
        # factor permitted by the locked support rule.
        weight = self._tile_normalize(base_weight * confidence.clamp_min(0.0))
        return LocalPosterior(
            probabilities,
            offsets,
            mean,
            covariance,
            entropy,
            confidence,
            boundary_probability,
            weight,
        )

    def _multiscale(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        foreground: torch.Tensor,
        activity: torch.Tensor,
    ) -> dict[int, LocalPosterior]:
        result: dict[int, LocalPosterior] = {}
        for scale in self.config.scales:
            if scale == 1:
                prev, curr, fg, act = previous, current, foreground, activity
            else:
                prev = functional.avg_pool2d(previous, scale, scale)
                curr = functional.avg_pool2d(current, scale, scale)
                fg = functional.avg_pool2d(foreground[:, None], scale, scale).squeeze(1)
                act = functional.avg_pool2d(activity[:, None], scale, scale).squeeze(1)
            # The radius equals the locked scale; displacement is restored to base feature pixels.
            posterior = self._posterior(prev, curr, self._weights(fg, act), radius=scale)
            posterior.mean = posterior.mean * float(scale)
            posterior.covariance = posterior.covariance * float(scale * scale)
            result[int(scale)] = posterior
        return result

    def locked_teacher_consensus(
        self,
        teacher_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
        foreground: torch.Tensor,
        activity: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Build the fixed three-teacher posterior consensus for distillation.

        Callers must provide all locked EMA geometry feature pairs for seeds
        7, 13, and 23.  This method intentionally does not select a teacher.
        """
        all_posteriors: list[dict[int, LocalPosterior]] = []
        for previous, current in teacher_pairs:
            all_posteriors.append(
                self._multiscale(
                    shrinkage_whiten(
                        previous, self.config.whitening_shrinkage, self.config.epsilon
                    ),
                    shrinkage_whiten(current, self.config.whitening_shrinkage, self.config.epsilon),
                    foreground,
                    activity,
                )
            )
        if len(all_posteriors) != 3:
            raise ValueError("v4.30 consensus requires exactly three locked teacher seeds")
        return {
            scale: geometric_mean_posteriors(
                [teacher[scale].probabilities for teacher in all_posteriors], self.config.epsilon
            )
            for scale in self.config.scales
        }

    def _fit_similarity(self, posteriors: dict[int, LocalPosterior]) -> SimilarityFit:
        """Fit one physical similarity field across all base-pixel scales.

        Every posterior mean and position is expressed in base feature pixels.
        The centre is therefore a single support-weighted physical centre across
        scales, rather than an average of unrelated coarse-grid centres.
        """
        first = posteriors[1]
        batch = first.weight.shape[0]
        positions: list[torch.Tensor] = []
        displacements: list[torch.Tensor] = []
        supports: list[torch.Tensor] = []
        for scale, item in posteriors.items():
            height, width = item.weight.shape[-2:]
            position = feature_coordinate_grid(
                height, width, device=item.weight.device, dtype=item.weight.dtype
            ) * float(scale)
            position = position.reshape(1, -1, 2).expand(batch, -1, -1)
            flat_weight = item.weight.reshape(batch, -1)
            positions.append(position)
            displacements.append(item.mean.reshape(batch, -1, 2))
            supports.append(flat_weight)

        position = torch.cat(positions, dim=1)
        displacement = torch.cat(displacements, dim=1)
        support = torch.cat(supports, dim=1).clamp_min(0.0)
        support_center = (position * support[..., None]).sum(dim=1) / support.sum(
            dim=1, keepdim=True
        ).clamp_min(self.config.epsilon)
        q = position - support_center[:, None]
        # dx = k*x - w*y + tx; dy = w*x + k*y + ty.
        zeros = torch.zeros_like(q[..., 0])
        ones = torch.ones_like(q[..., 0])
        rows = (
            torch.stack((q[..., 0], -q[..., 1], ones, zeros), dim=-1),
            torch.stack((q[..., 1], q[..., 0], zeros, ones), dim=-1),
        )
        values = (displacement[..., 0], displacement[..., 1])
        weights = (support, support)
        design = torch.cat(rows, dim=1)
        target = torch.cat(values, dim=1)
        weight = torch.cat(weights, dim=1).clamp_min(0.0)
        rms = (
            (
                (weight[..., None] * design.square()).sum(dim=1)
                / weight.sum(dim=1, keepdim=True).clamp_min(self.config.epsilon)
            )
            .sqrt()
            .clamp_min(self.config.epsilon)
        )
        normalized = design / rms[:, None]
        identity = torch.eye(4, device=design.device, dtype=design.dtype)[None]
        local_weight = weight
        beta = torch.zeros(batch, 4, device=design.device, dtype=design.dtype)
        normal = identity.expand(batch, -1, -1)
        for _ in range(3):
            normal = (
                torch.einsum("bn,bni,bnj->bij", local_weight, normalized, normalized)
                + self.config.ridge * identity
            )
            rhs = torch.einsum("bn,bni,bn->bi", local_weight, normalized, target)
            chol = torch.linalg.cholesky(normal)
            beta = torch.cholesky_solve(rhs[..., None], chol).squeeze(-1) / rms
            residual = (design * beta[:, None]).sum(dim=-1) - target
            robust = torch.where(
                residual.detach().abs() <= self.config.huber_delta,
                torch.ones_like(residual),
                self.config.huber_delta / residual.detach().abs().clamp_min(self.config.epsilon),
            )
            local_weight = weight * robust
        residual = (design * beta[:, None]).sum(dim=-1) - target
        sigma2 = (local_weight * residual.square()).sum(dim=1) / (
            local_weight.sum(dim=1) - 4.0
        ).clamp_min(1.0)
        covariance_normalized = torch.cholesky_inverse(torch.linalg.cholesky(normal))
        covariance = (
            sigma2[:, None, None] * covariance_normalized / (rms[:, :, None] * rms[:, None, :])
        )
        kappa, omega, tx, ty = beta.unbind(dim=-1)
        matrix = torch.stack(
            (torch.stack((1.0 + kappa, -omega), dim=-1), torch.stack((omega, 1.0 + kappa), dim=-1)),
            dim=-2,
        )
        return SimilarityFit(
            kappa,
            omega,
            torch.stack((tx, ty), dim=-1),
            support_center,
            covariance,
            sigma2,
            (local_weight * residual.abs()).sum(dim=1)
            / local_weight.sum(dim=1).clamp_min(self.config.epsilon),
            local_weight.sum(dim=1),
            rms,
            matrix,
        )

    def _normal_flow(
        self, events: torch.Tensor, posteriors: dict[int, LocalPosterior]
    ) -> torch.Tensor:
        """Voxel-native normal-flow residual; used only by the arm-B loss."""
        bins_per_polarity = 5
        raw = events[:, :, : 2 * bins_per_polarity].abs()
        centers = torch.linspace(
            0.0, 1.0, bins_per_polarity, device=events.device, dtype=events.dtype
        )
        surfaces = []
        for polarity in range(2):
            voxel = raw[:, :, polarity * bins_per_polarity : (polarity + 1) * bins_per_polarity]
            surfaces.append(
                (voxel * centers[None, None, :, None, None]).sum(dim=2)
                / voxel.sum(dim=2).clamp_min(self.config.epsilon)
            )
        surface = torch.stack(surfaces, dim=2).mean(dim=2)
        s1, s2 = surface[:, 1], surface[:, 2]
        residuals: list[torch.Tensor] = []
        normalizers: list[torch.Tensor] = []
        for scale in self.config.scales:
            posterior = posteriors[int(scale)]
            coarse_s1 = functional.interpolate(
                s1[:, None], size=posterior.weight.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
            coarse_s2 = functional.interpolate(
                s2[:, None], size=posterior.weight.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
            grad_y, grad_x = torch.gradient(0.5 * (coarse_s1 + coarse_s2), dim=(-2, -1))
            # Means are stored in base feature pixels; the scale-s gradient grid is coarse.
            displacement = posterior.mean / float(scale)
            residual = (
                grad_x * displacement[..., 0]
                + grad_y * displacement[..., 1]
                + (coarse_s1 - coarse_s2)
            )
            weighted = posterior.weight * (grad_x.square() + grad_y.square())
            residuals.append((weighted * residual.square()).sum(dim=(-2, -1)))
            normalizers.append(weighted.sum(dim=(-2, -1)))
        return torch.stack(residuals).sum(dim=0) / torch.stack(normalizers).sum(dim=0).clamp_min(
            self.config.epsilon
        )

    def forward(self, events: torch.Tensor) -> ObjectEventV430Output:
        if events.ndim != 5 or events.shape[1] != 3:
            raise ValueError("v4.30 forward accepts only events [B,3,C,H,W]")
        if not bool(torch.isfinite(events).all()):
            raise FloatingPointError("v4.30 event input is nonfinite; refusing to fabricate a fit")
        raw_mass = events[:, :, :10].abs().sum(dim=(1, 2, 3, 4))
        unknown = raw_mass == 0
        maps, _, foreground, _ = self.backbone._foreground_and_features(events)
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
        foreground = functional.interpolate(
            foreground, size=maps.shape[-2:], mode="bilinear", align_corners=False
        )
        activity = self._activity(events, output_size=(int(maps.shape[-2]), int(maps.shape[-1])))
        p01 = self._multiscale(projected[:, 0], projected[:, 1], foreground[:, 1], activity[:, 1])
        p12 = self._multiscale(projected[:, 1], projected[:, 2], foreground[:, 2], activity[:, 2])
        p02 = self._multiscale(projected[:, 0], projected[:, 2], foreground[:, 2], activity[:, 2])
        fit01, fit12, fit02 = (
            self._fit_similarity(p01),
            self._fit_similarity(p12),
            self._fit_similarity(p02),
        )
        radius = ((1.0 + fit12.kappa).square() + fit12.omega.square()).sqrt()
        log_eta = radius.clamp_min(self.config.epsilon).log()
        expansion = 1.0 - log_eta.exp()
        a_comp = fit01.matrix @ fit12.matrix

        def intercept(fit: SimilarityFit) -> torch.Tensor:
            return fit.center + fit.translation - (fit.matrix @ fit.center[..., None]).squeeze(-1)

        b01, b12, b02 = intercept(fit01), intercept(fit12), intercept(fit02)
        b_comp = (fit01.matrix @ b12[..., None]).squeeze(-1) + b01
        r2 = (1.0 + fit12.kappa).square() + fit12.omega.square()
        gradient = torch.stack(
            (
                (1.0 + fit12.kappa) / r2.clamp_min(self.config.epsilon),
                fit12.omega / r2.clamp_min(self.config.epsilon),
                torch.zeros_like(fit12.kappa),
                torch.zeros_like(fit12.kappa),
            ),
            dim=-1,
        )
        variance = torch.einsum("bi,bij,bj->b", gradient, fit12.covariance, gradient).clamp_min(0.0)
        normal_flow = (
            self._normal_flow(events, p12)
            if self.config.arm.endswith("normal_flow")
            else torch.zeros_like(expansion)
        )
        # Exactly-zero raw mass has no event-derived estimate. Non-zero estimates
        # remain finite through ridge Cholesky; no row is dropped or replaced.
        return ObjectEventV430Output(
            expansion=torch.where(unknown, torch.full_like(expansion, float("nan")), expansion),
            log_eta=torch.where(unknown, torch.full_like(log_eta, float("nan")), log_eta),
            posterior_variance=torch.where(
                unknown, torch.full_like(variance, float("nan")), variance
            ),
            unknown=unknown,
            fit_01=fit01,
            fit_12=fit12,
            fit_02=fit02,
            cycle_matrix_error=torch.linalg.matrix_norm(
                fit02.matrix - a_comp, ord="fro", dim=(-2, -1)
            ),
            cycle_translation_error=torch.linalg.vector_norm(b02 - b_comp, dim=-1),
            correlation_entropy=p12[1].entropy.mean(dim=(-2, -1)),
            correlation_confidence=p12[1].confidence.mean(dim=(-2, -1)),
            boundary_probability=p12[1].boundary_probability.mean(dim=(-2, -1)),
            rotation_radians=torch.atan2(fit12.omega, 1.0 + fit12.kappa),
            translation_magnitude=torch.linalg.vector_norm(fit12.translation, dim=-1),
            normal_flow_residual=normal_flow,
            foreground_map_t1=foreground[:, 1],
            foreground_map_t2=foreground[:, 2],
            support_map_t2=p12[1].weight,
            posteriors_01=p01,
            posteriors_12=p12,
        )


__all__ = [
    "LocalPosterior",
    "ObjectEventTTCV430",
    "ObjectEventV430Config",
    "ObjectEventV430Output",
    "SimilarityFit",
    "feature_coordinate_grid",
    "geometric_mean_posteriors",
    "shrinkage_whiten",
]
