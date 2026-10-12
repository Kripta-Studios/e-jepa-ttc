"""The deployment aggregation operates on signed phase, retaining native support."""

import numpy as np

from operational.multimodal_research.phase_ensemble import combine


def test_identical_members_preserve_signed_predictions_and_cap() -> None:
    values = np.array([-60.0, -2.0, 0.2, 3.0, 60.0])
    np.testing.assert_allclose(combine(np.tile(values, (3, 1))), values, atol=1e-12)


def test_phase_average_is_not_ttc_average() -> None:
    values = np.array([[1.0], [2.0], [4.0]])
    expected = 0.1 / (1 - np.cbrt((1 - 0.1) * (1 - 0.05) * (1 - 0.025)))
    np.testing.assert_allclose(combine(values), [expected], atol=1e-12)
    assert abs(combine(values)[0] - values.mean()) > 0.1


def test_opposite_phases_follow_native_zero_cap_without_positive_selection() -> None:
    phases = np.array([[-0.01], [0.004], [0.006]])
    values = 0.1 / -np.expm1(-phases)
    np.testing.assert_allclose(np.abs(combine(values)), [60.0], atol=1e-12)
