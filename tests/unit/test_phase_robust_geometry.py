import numpy as np

from operational.phase_robust.geometry import expansion_features


def looming(times, collision=2.0):
    size = 100 / (collision - times)
    return np.column_stack((300 - size / 2, 200 - size / 2, 300 + size / 2, 200 + size / 2))


def test_inverse_scale_fit_is_current_time_and_scale_clock_invariant():
    times = np.array([0.0, 0.07, 0.16, 0.23, 0.4])
    boxes = looming(times)
    first = expansion_features(boxes, (times * 1e6).astype(np.int64), 400000)
    shifted = expansion_features(boxes * 2, (times * 1e6).astype(np.int64) + 1234567, 1634567)
    assert first["valid"]
    np.testing.assert_allclose(first["ttc"], 1.6, atol=1e-9)
    np.testing.assert_allclose(shifted["q"], first["q"], atol=1e-9)


def test_future_boxes_are_excluded_and_missing_support_is_not_fabricated():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    boxes = looming(times)
    a = expansion_features(boxes, (times * 1e6).astype(np.int64), 200000)
    boxes[-1] = [0, 0, 1000, 1000]
    b = expansion_features(boxes, (times * 1e6).astype(np.int64), 200000)
    assert a == b
    assert not expansion_features(boxes[:1], np.array([0]), 0)["valid"]


def test_stationary_and_receding_boxes_have_explicit_physical_semantics():
    times = np.arange(7) * 0.1
    stationary = np.tile([0.0, 0.0, 50.0, 50.0], (7, 1))
    a = expansion_features(stationary, (times * 1e6).astype(np.int64), 600000)
    assert not a["valid"]
    receding = looming(-times)
    b = expansion_features(receding, (times * 1e6).astype(np.int64), 600000)
    assert b["valid"] and b["ttc"] < 0
    np.testing.assert_allclose(b["ttc"], -2.6, atol=1e-9)


def test_isolated_box_outlier_keeps_robust_motion_and_exposes_residual():
    times = np.arange(7) * 0.1
    boxes = looming(times)
    clean = expansion_features(boxes, (times * 1e6).astype(np.int64), 600000)
    boxes[3] = [0, 0, 20, 20]
    noisy = expansion_features(boxes, (times * 1e6).astype(np.int64), 600000)
    np.testing.assert_allclose(noisy["ttc"], clean["ttc"], atol=1e-9)
    assert noisy["residual"] > clean["residual"]
    assert not expansion_features(boxes, (times * 1e6).astype(np.int64), 1000000)["valid"]


def test_acceleration_changes_short_long_quality_without_future_data():
    times = np.arange(14) * 0.05
    distance = 2 - times - times**2
    size = 100 / distance
    boxes = np.column_stack((-size / 2, -size / 2, size / 2, size / 2))
    result = expansion_features(boxes, np.rint(times * 1e6).astype(np.int64), 650000)
    assert result["valid"] and result["ttc"] > 0
    assert result["quality"][13] > 0.05
