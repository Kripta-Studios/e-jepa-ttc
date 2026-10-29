from __future__ import annotations

import numpy as np

from e_jepa_ttc.geometry.cmax import (
    image_of_warped_events,
    maximize_radial_event_contrast,
    warp_radial_events,
)


def _synthetic_looming_events(
    *,
    inverse_ttc_per_s: float,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    anchors = rng.uniform((20.0, 15.0), (76.0, 57.0), size=(50, 2))
    endpoint = np.repeat(anchors, 10, axis=0)
    times = np.tile(np.linspace(-0.2, 0.0, 10), anchors.shape[0])
    center = np.asarray((48.0, 36.0))
    # Earlier object positions contract relative to the endpoint.
    points = center + np.exp(inverse_ttc_per_s * times)[:, None] * (endpoint - center)
    anchor_polarity = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=anchors.shape[0])
    polarity = np.repeat(anchor_polarity, 10)
    return points, times, polarity


def test_radial_warp_maps_synthetic_events_to_endpoint() -> None:
    points, times, _ = _synthetic_looming_events(inverse_ttc_per_s=0.5)
    warped = warp_radial_events(
        points,
        times,
        inverse_ttc_per_s=0.5,
        center_xy=(48.0, 36.0),
    )
    center = np.asarray((48.0, 36.0))
    expected = center + np.exp(-0.5 * times)[:, None] * (points - center)
    assert np.allclose(warped, expected)


def test_radial_warp_compensates_causal_center_translation() -> None:
    points, times, _ = _synthetic_looming_events(inverse_ttc_per_s=0.5)
    velocity = np.asarray((20.0, -10.0))
    moving_centers = np.asarray((48.0, 36.0)) + times[:, None] * velocity
    translated_points = points + times[:, None] * velocity
    warped = warp_radial_events(
        translated_points,
        times,
        inverse_ttc_per_s=0.5,
        center_xy=(48.0, 36.0),
        event_centers_xy=moving_centers,
    )
    stationary_warp = warp_radial_events(
        points,
        times,
        inverse_ttc_per_s=0.5,
        center_xy=(48.0, 36.0),
    )
    assert np.allclose(warped, stationary_warp)


def test_bilinear_iwe_preserves_event_mass_inside_image() -> None:
    points = np.asarray(((10.25, 11.75), (20.5, 21.5)))
    polarity = np.asarray((1, -1))
    image, survival = image_of_warped_events(points, polarity, image_shape=(40, 50))
    assert np.isclose(survival, 1.0)
    assert np.isclose(image.sum(), 2.0)
    assert np.isclose(image[1].sum(), 1.0)
    assert np.isclose(image[0].sum(), 1.0)


def test_cmax_recovers_synthetic_inverse_ttc() -> None:
    points, times, polarity = _synthetic_looming_events(inverse_ttc_per_s=0.5)
    result = maximize_radial_event_contrast(
        points,
        times,
        polarity,
        image_shape=(72, 96),
        center_xy=(48.0, 36.0),
        coarse_steps=49,
        minimum_relative_contrast_gain=0.001,
    )
    assert result.valid
    assert np.isclose(result.inverse_ttc_per_s, 0.5, atol=0.08)
    assert np.isclose(result.ttc_seconds, 2.0, atol=0.35)


def test_cmax_rejects_contraction_as_positive_collision() -> None:
    points, times, polarity = _synthetic_looming_events(inverse_ttc_per_s=-0.5)
    result = maximize_radial_event_contrast(
        points,
        times,
        polarity,
        image_shape=(72, 96),
        center_xy=(48.0, 36.0),
        coarse_steps=49,
        minimum_relative_contrast_gain=0.001,
    )
    assert not result.valid
    assert result.reason == "non_approaching_best_warp"
