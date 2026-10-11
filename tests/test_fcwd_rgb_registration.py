"""Geometry, rejection and causal-history regressions for estimated FCWD RGB."""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from operational.fcwd_rgb_registration.geometry import (  # noqa: E402
    camera_parameters,
    estimate_shift,
    project_box,
    project_points,
)
from operational.rgb_port_revision.transfer_inputs import sensor_triplets  # noqa: E402


def calibration() -> dict:
    values = {
        "IntrinsicMatrix": [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [641.0, 361.0, 1.0]],
        "RadialDistortion": [[0.0, 0.0]],
        "TangentialDistortion": [[0.0, 0.0]],
    }
    return {"APScameraParams": values.copy(), "DVScameraParams": values.copy()}


def test_matlab_convention_and_projection() -> None:
    calib = calibration()
    k, _ = camera_parameters(calib["APScameraParams"])
    np.testing.assert_array_equal(k[:2, 2], [640.0, 360.0])
    points = np.array([[0.0, 0.0], [600.0, 300.0], [1279.0, 719.0]])
    np.testing.assert_allclose(
        project_points(points, calib, (17.0, -8.0)), points + [17.0, -8.0], atol=1e-9
    )
    np.testing.assert_allclose(
        project_box([100.0, 120.0, 300.0, 350.0], calib, (17.0, -8.0)), [117.0, 112.0, 317.0, 342.0]
    )


def test_focal_scale_is_optical_not_resolution_ratio() -> None:
    calib = calibration()
    calib["APScameraParams"]["IntrinsicMatrix"] = [
        [2000.0, 0.0, 0.0],
        [0.0, 2000.0, 0.0],
        [961.0, 601.0, 1.0],
    ]
    mapped = project_points(np.array([[650.0, 380.0]]), calib, (0.0, 0.0))
    np.testing.assert_allclose(mapped, [[980.0, 640.0]])


@pytest.mark.parametrize("box", [[0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 10.0, float("nan")]])
def test_invalid_boxes_rejected(box: list[float]) -> None:
    with pytest.raises(ValueError):
        project_box(box, calibration(), (0.0, 0.0))


def test_textureless_query_rejected() -> None:
    empty = np.zeros((720, 1280), np.float32)
    result = estimate_shift(
        np.zeros((1200, 1920, 3), np.uint8),
        empty,
        empty,
        [500.0, 250.0, 750.0, 500.0],
        calibration(),
    )
    assert result["accepted"] is False
    assert result["reason"] == "INSUFFICIENT_EVENT_SUPPORT"


def test_known_translation_recovered_from_independent_sensor_planes() -> None:
    cv2.setNumThreads(1)
    rgb = np.full((1200, 1920, 3), 70, np.uint8)
    rng = np.random.default_rng(19)
    for _ in range(30):
        x, y = rng.integers([400, 200], [760, 550])
        radius = int(rng.integers(5, 20))
        cv2.circle(rgb, (int(x), int(y)), radius, (220, 220, 220), 2)
    gray = cv2.cvtColor(rgb[:720, :1280], cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 30, 80).astype(np.float32) / 255 * 5
    event = cv2.warpAffine(edge, np.array([[1, 0, 12], [0, 1, 8]], np.float32), (1280, 720))
    result = estimate_shift(rgb, event, event, [390.0, 190.0, 810.0, 590.0], calibration())
    assert result["accepted"], result
    np.testing.assert_allclose(result["shift_px"], [-12.0, -8.0], atol=1)
    assert result["half_window_disagreement_px"] == 0
    assert json.loads(json.dumps(result)) == result


def test_real_fc_wd_cadence_keeps_actual_frames_and_deltas() -> None:
    clock = np.arange(30, dtype=np.int64) * 50_052
    query = int(clock[20]) + 1_000
    groups = sensor_triplets(clock, query)
    assert groups
    for ids in groups:
        assert len(ids) == 3
        assert np.all(np.diff(clock[ids]) == 100_104)
        assert clock[ids[-1]] <= query
        assert query - clock[ids[0]] <= 650_000
