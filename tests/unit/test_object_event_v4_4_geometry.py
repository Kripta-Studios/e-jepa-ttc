from __future__ import annotations

import torch

from e_jepa_ttc.object_event_v4_4 import GEOMETRY_FEATURE_NAMES, event_geometry_features


def _gaussian(radius: float, center_x: float = 0.0) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, 64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.exp(-((xx - center_x).square() + yy.square()) / (2.0 * radius**2))


def _events(radii: tuple[float, float, float], centers: tuple[float, float, float] = (0, 0, 0)) -> torch.Tensor:
    events = torch.zeros(1, 3, 12, 64, 64)
    for step, (radius, center) in enumerate(zip(radii, centers, strict=True)):
        events[0, step, 0] = _gaussian(radius, center)
    return events


def test_geometry_proxy_has_correct_looming_sign() -> None:
    expanding = event_geometry_features(_events((0.16, 0.22, 0.31)))
    contracting = event_geometry_features(_events((0.31, 0.22, 0.16)))
    proxy = GEOMETRY_FEATURE_NAMES.index("geometry_proxy")
    assert expanding.shape == (1, len(GEOMETRY_FEATURE_NAMES))
    assert float(expanding[0, proxy]) > 0.05
    assert float(contracting[0, proxy]) < -0.05


def test_translation_does_not_masquerade_as_radial_expansion() -> None:
    translated = event_geometry_features(
        _events((0.22, 0.22, 0.22), centers=(-0.25, 0.0, 0.25))
    )
    proxy = GEOMETRY_FEATURE_NAMES.index("geometry_proxy")
    motion = GEOMETRY_FEATURE_NAMES.index("centroid_motion_02")
    assert abs(float(translated[0, proxy])) < 1.0e-3
    assert float(translated[0, motion]) > 0.4
