from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_18 import (
    MonotoneOddPhysicsHead,
    feature_scales,
    normalise_physics_features,
    radial_physics_features,
    robust_seed_consensus,
)


def _blob(size: int, radius: float, cx: float = 0.0, cy: float = 0.0) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, size)
    x = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius**2))


def _sequence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (a + b)
    return torch.stack((middle, a, b), dim=0).unsqueeze(0)


def test_radial_features_are_endpoint_odd() -> None:
    small = _blob(32, 0.20)
    large = _blob(32, 0.32)
    forward = _sequence(small, large)
    reverse = _sequence(large, small)
    f = radial_physics_features(forward, forward)
    r = radial_physics_features(reverse, reverse)
    assert torch.allclose(f, -r, atol=2.0e-5, rtol=2.0e-5)


def test_growth_has_positive_physics_features() -> None:
    small = _blob(32, 0.18)
    large = _blob(32, 0.30)
    seq = _sequence(small, large)
    f = radial_physics_features(seq, seq)
    # The robust scale/radial proxies should agree on outward growth.
    assert float(f[0, 0]) > 0.0
    assert float(f[0, 1]) > 0.0
    assert float(f[0, 2]) > 0.0
    assert float(f[0, 3]) > 0.0
    assert float(f[0, 4]) > 0.0


def test_lateral_translation_is_attenuated() -> None:
    first = _blob(32, 0.24, cx=-0.25)
    second = _blob(32, 0.24, cx=0.25)
    seq = _sequence(first, second)
    f = radial_physics_features(seq, seq)
    assert float(f.abs().max()) < 0.25


def test_seed_consensus_preserves_oddness() -> None:
    x = torch.randn(3, 5, 10)
    forward = robust_seed_consensus(x)
    reverse = robust_seed_consensus(-x)
    assert torch.allclose(forward, -reverse, atol=1.0e-6, rtol=1.0e-6)


def test_train_scaling_does_not_center_or_break_oddness() -> None:
    x = torch.randn(20, 10)
    scales = feature_scales(x, minimum_scale=1.0e-3)
    forward = normalise_physics_features(x, scales, clip=6.0)
    reverse = normalise_physics_features(-x, scales, clip=6.0)
    assert torch.allclose(forward, -reverse)


def test_monotone_head_is_exactly_odd_and_growth_means_positive_class() -> None:
    model = MonotoneOddPhysicsHead(10)
    growth = torch.ones(4, 10)
    shrink = -growth
    growth_logit = model(growth)
    shrink_logit = model(shrink)
    assert torch.all(growth_logit < 0.0)  # negative-class logit: growth -> approach
    assert torch.all(shrink_logit > 0.0)
    assert torch.allclose(growth_logit, -shrink_logit)
    assert float(model.oddness_error(growth).max()) < 1.0e-7
