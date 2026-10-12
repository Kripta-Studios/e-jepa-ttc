"""Directional representation has a precise invariance and no target inputs."""

import numpy as np
import pytest

from operational.representation_audit.directional import directions


def test_directional_invariance_to_common_token_levels():
    rng = np.random.default_rng(18)
    raw = rng.normal(size=(7, 896))
    other = raw.copy()
    shared = rng.normal(size=(7, 1, 128))
    other[:, :384] = (raw[:, :384].reshape(7, 3, 128) + shared).reshape(7, 384)
    transport_shift = rng.normal(size=(7, 256))
    other[:, 384:640] += transport_shift
    other[:, 640:] += transport_shift
    np.testing.assert_allclose(directions(raw), directions(other), atol=1e-14, rtol=1e-14)
    assert directions(raw).shape == (7, 512)


def test_directional_changes_under_time_reverse():
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(7, 896))
    other = raw.copy()
    other[:, :384] = raw[:, :384].reshape(7, 3, 128)[:, ::-1].reshape(7, 384)
    other[:, 384:640] = raw[:, 640:].reshape(7, 2, 128)[:, ::-1].reshape(7, 256)
    other[:, 640:] = raw[:, 384:640].reshape(7, 2, 128)[:, ::-1].reshape(7, 256)
    expected = -directions(raw).reshape(7, 2, 2, 128)[:, :, ::-1].reshape(7, 512)
    np.testing.assert_array_equal(directions(other), expected)


def test_bad_shape_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        directions(np.ones((7, 895)))
    with pytest.raises(ValueError):
        directions(np.full((7, 896), np.nan))
