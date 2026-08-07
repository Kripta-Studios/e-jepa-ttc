"""Low-dimensional radial/divergence physics bottleneck for Object Event TTC v4.18."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


FEATURE_NAMES = (
    "foreground_half_log_mass",
    "foreground_log_rms_radius",
    "foreground_radial_transport",
    "foreground_radial_energy_transport",
    "foreground_quarter_log_cov_det",
    "activity_half_log_mass",
    "activity_log_rms_radius",
    "activity_radial_transport",
    "activity_radial_energy_transport",
    "activity_quarter_log_cov_det",
)


@dataclass(frozen=True)
class ObjectEventV418Config:
    epsilon: float = 1.0e-6
    feature_clip: float = 6.0
    minimum_feature_scale: float = 1.0e-3

    def __post_init__(self) -> None:
        if min(self.epsilon, self.feature_clip, self.minimum_feature_scale) <= 0.0:
            raise ValueError("v4.18 numerical controls must be positive")


def _distribution_geometry(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return five exact endpoint-odd scale/divergence features.

    Input tensors are non-negative spatial distributions [B,H,W]. Midpoint
    coordinates are defined from first+second and are therefore unchanged under
    endpoint reversal. Signed differences/log-ratios consequently negate exactly
    (up to floating point error). A symmetric centroid-shift attenuation prevents
    strong lateral translation from masquerading as radial scale change.
    """
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError("first and second must be aligned [B,H,W]")
    first = first.float().clamp_min(0.0)
    second = second.float().clamp_min(0.0)
    batch, height, width = first.shape
    device, dtype = first.device, first.dtype

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.expand(batch, -1, -1)
    yy = yy.expand(batch, -1, -1)

    m1 = first.sum(dim=(-2, -1)).clamp_min(epsilon)
    m2 = second.sum(dim=(-2, -1)).clamp_min(epsilon)
    n1 = first / m1[:, None, None]
    n2 = second / m2[:, None, None]

    c1x = (n1 * xx).sum(dim=(-2, -1))
    c1y = (n1 * yy).sum(dim=(-2, -1))
    c2x = (n2 * xx).sum(dim=(-2, -1))
    c2y = (n2 * yy).sum(dim=(-2, -1))

    midpoint = first + second
    mm = midpoint.sum(dim=(-2, -1)).clamp_min(epsilon)
    mn = midpoint / mm[:, None, None]
    cmx = (mn * xx).sum(dim=(-2, -1))
    cmy = (mn * yy).sum(dim=(-2, -1))
    dxm = xx - cmx[:, None, None]
    dym = yy - cmy[:, None, None]
    r2_mid = dxm.square() + dym.square()
    r_mid = torch.sqrt(r2_mid + epsilon)

    er1 = (n1 * r_mid).sum(dim=(-2, -1)).clamp_min(epsilon)
    er2 = (n2 * r_mid).sum(dim=(-2, -1)).clamp_min(epsilon)
    er21 = (n1 * r2_mid).sum(dim=(-2, -1)).clamp_min(epsilon)
    er22 = (n2 * r2_mid).sum(dim=(-2, -1)).clamp_min(epsilon)

    # Covariances around each distribution's own centroid.
    dx1 = xx - c1x[:, None, None]
    dy1 = yy - c1y[:, None, None]
    dx2 = xx - c2x[:, None, None]
    dy2 = yy - c2y[:, None, None]
    v1x = (n1 * dx1.square()).sum(dim=(-2, -1)).clamp_min(epsilon)
    v1y = (n1 * dy1.square()).sum(dim=(-2, -1)).clamp_min(epsilon)
    c1xy = (n1 * dx1 * dy1).sum(dim=(-2, -1))
    v2x = (n2 * dx2.square()).sum(dim=(-2, -1)).clamp_min(epsilon)
    v2y = (n2 * dy2.square()).sum(dim=(-2, -1)).clamp_min(epsilon)
    c2xy = (n2 * dx2 * dy2).sum(dim=(-2, -1))
    det1 = (v1x * v1y - c1xy.square()).clamp_min(epsilon)
    det2 = (v2x * v2y - c2xy.square()).clamp_min(epsilon)

    mean_radius = 0.5 * (torch.sqrt(er21) + torch.sqrt(er22)).clamp_min(epsilon)
    centroid_shift = torch.sqrt((c2x - c1x).square() + (c2y - c1y).square() + epsilon)
    reliability = 1.0 / (1.0 + centroid_shift / mean_radius)

    features = torch.stack(
        (
            0.5 * torch.log(m2 / m1),
            0.5 * torch.log(er22 / er21),
            (er2 - er1) / (0.5 * (er1 + er2) + epsilon),
            (er22 - er21) / (0.5 * (er21 + er22) + epsilon),
            0.25 * torch.log(det2 / det1),
        ),
        dim=1,
    )
    return features * reliability[:, None]


def radial_physics_features(
    foreground_probabilities: torch.Tensor,
    event_activity: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return 10 endpoint-odd physical features from one frozen v4.8 backbone."""
    if foreground_probabilities.ndim != 4 or foreground_probabilities.shape[1] != 3:
        raise ValueError("foreground_probabilities must be [B,3,H,W]")
    if event_activity.shape != foreground_probabilities.shape:
        raise ValueError("event_activity must align with foreground probabilities")
    foreground = _distribution_geometry(
        foreground_probabilities[:, 1],
        foreground_probabilities[:, 2],
        epsilon=epsilon,
    )
    activity = _distribution_geometry(
        event_activity[:, 1].clamp_min(0.0),
        event_activity[:, 2].clamp_min(0.0),
        epsilon=epsilon,
    )
    return torch.cat((foreground, activity), dim=1)


def robust_seed_consensus(per_seed: torch.Tensor, *, epsilon: float = 1.0e-6) -> torch.Tensor:
    """Robust exact-odd consensus [S,B,F] -> [B,F].

    Median is odd under global sign reversal. Seed MAD is even, so multiplying
    by inverse disagreement keeps the result odd.
    """
    if per_seed.ndim != 3 or per_seed.shape[0] < 2:
        raise ValueError("per_seed must be [S,B,F] with at least two seeds")
    median = per_seed.median(dim=0).values
    mad = (per_seed - median.unsqueeze(0)).abs().median(dim=0).values
    reliability = 1.0 / (1.0 + mad / (median.abs() + 0.05 + epsilon))
    return median * reliability


def feature_scales(
    train_features: torch.Tensor,
    *,
    minimum_scale: float,
) -> torch.Tensor:
    if train_features.ndim != 2:
        raise ValueError("train_features must be [N,F]")
    return train_features.abs().median(dim=0).values.clamp_min(minimum_scale)


def normalise_physics_features(
    features: torch.Tensor,
    scales: torch.Tensor,
    *,
    clip: float,
) -> torch.Tensor:
    if features.ndim != 2 or scales.ndim != 1 or features.shape[1] != len(scales):
        raise ValueError("feature/scales shape mismatch")
    return (features / scales[None]).clamp(-clip, clip)


class MonotoneOddPhysicsHead(nn.Module):
    """Zero-bias monotone odd sign head.

    Features are oriented so positive means radial expansion/approach. The
    project's negative-class logit therefore uses a minus sign. Non-negative
    weights prevent a sequence-specific fit from learning that geometric
    contraction means approach.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.raw_weights = nn.Parameter(torch.zeros(feature_dim))

    def positive_weights(self) -> torch.Tensor:
        return functional.softplus(self.raw_weights) + 1.0e-6

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.positive_weights()
        score = (features * weights[None]).sum(dim=1) / weights.sum()
        return -score  # positive logit = negative/receding class

    @torch.no_grad()
    def oddness_error(self, features: torch.Tensor) -> torch.Tensor:
        return (self(features) + self(-features)).abs()


__all__ = [
    "FEATURE_NAMES",
    "MonotoneOddPhysicsHead",
    "ObjectEventV418Config",
    "feature_scales",
    "normalise_physics_features",
    "radial_physics_features",
    "robust_seed_consensus",
]
