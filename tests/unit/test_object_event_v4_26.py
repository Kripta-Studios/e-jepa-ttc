import numpy as np
import pandas as pd

from e_jepa_ttc.training.object_event_v4_26 import (
    apply_residual_calibration,
    fit_residual_calibration,
    nonnegative_ridge_residual,
    predict_anchored_residual,
    residual_design_matrix,
    track_metrics,
)


def test_residual_calibration_orients_to_residual():
    score = np.array([-2.0, -1.0, 1.0, 2.0])
    residual = -0.25 * score
    calibration = fit_residual_calibration(score, residual)
    assert calibration.orientation == -1.0
    np.testing.assert_allclose(
        apply_residual_calibration(score, calibration), residual, atol=1.0e-8
    )


def test_residual_design_has_no_learned_anchor_column():
    div = np.array([1.0, 2.0])
    vert = np.array([3.0, 4.0])
    x, names = residual_design_matrix(div, vert, ["divergence", "vertical"])
    assert names == ("divergence", "vertical")
    np.testing.assert_allclose(x, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_nonnegative_residual_ridge_recovers_positive_weights():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = x @ np.array([0.3, 0.2])
    coefficients = nonnegative_ridge_residual(x, y, ridge=0.0)
    np.testing.assert_allclose(coefficients, [0.3, 0.2], atol=1.0e-8)


def test_nonnegative_residual_ridge_rejects_wrong_sign_feature():
    x = np.arange(1.0, 5.0)[:, None]
    y = -0.5 * x[:, 0]
    coefficients = nonnegative_ridge_residual(x, y, ridge=0.0)
    np.testing.assert_allclose(coefficients, [0.0], atol=1.0e-12)


def test_anchor_is_fixed_exactly_one():
    anchor = np.array([0.1, -0.2])
    x = np.array([[1.0], [2.0]])
    pred = predict_anchored_residual(anchor, x, np.array([0.25]))
    np.testing.assert_allclose(pred, anchor + np.array([0.25, 0.5]))


def test_track_metrics_detect_worst_negative_track():
    frame = pd.DataFrame({
        "sequence_id": ["s"] * 16,
        "track_id": ["good"] * 8 + ["bad"] * 8,
        "target_expansion": np.array([-1.0] * 8 + [-1.0] * 8),
    })
    prediction = np.array([-1.0] * 8 + [1.0] * 8)
    metrics, per_track = track_metrics(
        frame,
        prediction,
        minimum_track_samples=4,
        minimum_negative_track_samples=4,
    )
    assert len(per_track) == 2
    assert metrics["negative_track_count"] == 2
    assert metrics["minimum_negative_track_accuracy"] == 0.0
    assert metrics["negative_track_macro_accuracy"] == 0.5
