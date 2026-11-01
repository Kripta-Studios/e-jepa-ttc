"""A second motion must not silently change foreground truth or sensor clocks."""

import numpy as np
import pytest

from operational.phase_event_expansion.background import background_fixture
from operational.phase_event_expansion.synthetic import make_fixture


def test_background_preserves_exact_foreground_and_is_reproducible():
    original = make_fixture(61, "expansion")
    fixture, foreground = background_fixture(61, 0.5)
    np.testing.assert_array_equal(fixture.events[foreground], original.events)
    assert fixture.q == original.q
    np.testing.assert_array_equal(fixture.translation, original.translation)
    repeat, mask = background_fixture(61, 0.5)
    np.testing.assert_array_equal(fixture.events, repeat.events)
    np.testing.assert_array_equal(foreground, mask)
    assert len(fixture.events) == 8192
    assert np.all(np.diff(fixture.events[:, 2]) >= 0)
    assert fixture.events[:, 2].max() <= fixture.query_us
    assert np.isin(fixture.events[:, 3], [-1, 1]).all()
    assert np.all(fixture.events[:, :2] >= fixture.roi[:2])
    assert np.all(fixture.events[:, :2] <= fixture.roi[2:])


def test_zero_background_is_original_and_invalid_fraction_rejected():
    fixture, foreground = background_fixture(73, 0)
    np.testing.assert_array_equal(fixture.events, make_fixture(73, "expansion").events)
    assert foreground.all()
    for fraction in (-0.1, 1, float("nan")):
        with pytest.raises(ValueError):
            background_fixture(73, fraction)
