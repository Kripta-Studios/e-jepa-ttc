"""Raw geometry, aperture ambiguity and fixed six-indicator contracts."""

import numpy as np
import pytest

from operational.release_research.normal_quality import (
    PHYSICAL_FIELDS,
    fit_surface,
    make_surface,
    physical_vector,
)


def test_surface_keeps_last_per_polarity_without_future_or_crop_leakage():
    events = np.array([[2, 3, 10, 0], [2, 3, 12, 1], [2, 3, 14, 0], [6, 3, 15, 0]])
    roi = np.array([1.0, 1.0, 5.0, 5.0])
    surface, origin, count = make_surface(events, roi, (8, 8), 10, 16)
    assert origin == (1, 1) and count == 3
    assert surface[0, 2, 1] == 14 and surface[1, 2, 1] == 12
    with pytest.raises(ValueError, match="causal"):
        make_surface(events, roi, (8, 8), 10, 15)
    with pytest.raises(ValueError, match="Sensor"):
        make_surface(events, roi, (6, 8), 10, 16)


def test_empty_surface_retains_missingness_not_certainty():
    roi = np.array([0.0, 0.0, 64.0, 64.0])
    surface, origin, count = make_surface(np.empty((0, 4), np.int64), roi, (64, 64), 0, 10000)
    result = fit_surface(surface, origin, roi, roi, 10000)
    assert count == 0 and result["accepted_planes"] == 0 and result["all"] is None
    assert physical_vector(result, result) == [None] * len(PHYSICAL_FIELDS)


def test_many_good_planes_do_not_resolve_aperture_ambiguity():
    yy, xx = np.mgrid[:64, :64]
    ticks = 10000 - 9000 + xx * 100
    events = np.column_stack((xx.ravel(), yy.ravel(), ticks.ravel(), np.zeros(xx.size, int)))
    events = events[np.argsort(events[:, 2], kind="stable")]
    roi = np.array([0.0, 0.0, 64.0, 64.0])
    surface, origin, _ = make_surface(events, roi, (64, 64), 0, 10000)
    result = fit_surface(surface, origin, roi, np.array([8.0, 8.0, 56.0, 56.0]), 10000)
    assert result["accepted_planes"] > 100
    assert not result["all"]["estimable"] and result["all"]["q"] is None
    assert result["all"]["null_leakage"] > 0.5
    assert 0 < result["independent_patch_fraction"] < 1


def test_temporal_innovation_uses_actual_endpoint_time():
    fit = {"q": 0.5, "null_leakage": 0.0, "sensitivity": 0.1, "normalized_residual": 0.01}
    first = {"all": fit, "endpoint_us": 100000}
    second = {
        "all": {**fit, "q": 0.5 / (1 - 0.2 * 0.5)},
        "endpoint_us": 300000,
        "foreground_background_disagreement": 0.2,
        "independent_patch_fraction": 0.4,
    }
    assert physical_vector(first, second)[3] == pytest.approx(0.0)
