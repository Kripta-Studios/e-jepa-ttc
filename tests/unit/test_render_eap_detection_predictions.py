"""Protect visualization population, geometry ordering and display-only filtering."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample
from scripts.render_eap_detection_predictions import (
    clipped_segment,
    fixed_tokens,
    native_to_original_box,
    overlay,
)


def native_box(score=0.8):
    return {
        "class_name": "car",
        "translation_m": [0, 0, 10],
        "size_lwh_m": [4, 2, 1],
        "yaw_rad": 0.3,
        "score": score,
    }


def sample():
    return RawRGBSample(
        sample_token="train:f",
        sequence_id="train",
        image=np.zeros((80, 100, 3), dtype=np.uint8),
        annotations=[],
        intrinsic=np.array([[50, 0, 50], [0, 50, 40], [0, 0, 1]], dtype=float),
        event_from_ego=np.eye(4),
        exposure_start_us=1000,
        exposure_end_us=1100,
        media={},
        member_sha256="fixture",
    )


def test_fixed_selection_uses_manifest_order():
    tokens = [f"frame-{i}" for i in range(16)]
    assert fixed_tokens(tokens, 4) == ["frame-0", "frame-5", "frame-10", "frame-15"]
    assert fixed_tokens(tokens[:2], 4) == tokens[:2]
    with pytest.raises(ValueError):
        fixed_tokens(["a", "a"], 2)


def test_native_dimensions_convert_lwh_to_lhw():
    box = native_to_original_box(native_box())
    np.testing.assert_allclose(box, [0, 0, 10, 4, 1, 2, 0.3])
    bad = native_box()
    bad["size_lwh_m"] = [4, -2, 1]
    with pytest.raises(ValueError, match="positive"):
        native_to_original_box(bad)


def test_line_clipping_handles_large_coordinates_and_outside_lines():
    np.testing.assert_allclose(clipped_segment([-1e8, 20], [1e8, 20], 100, 80), [[0, 20], [99, 20]])
    assert clipped_segment([-10, -5], [110, -5], 100, 80) is None
    assert clipped_segment([20, 30], [30, 40], 100, 80) == ((20, 30), (30, 40))


def test_threshold_changes_display_only_and_preserves_original_arrays():
    frame = sample()
    predictions = [native_box(0.1), native_box(0.8)]
    truth = [native_box()]
    unchanged = copy.deepcopy(predictions)
    tile, counts = overlay(frame, predictions, truth, label="Fixture", score_threshold=0.2)
    assert counts["pred_total"] == 2
    assert counts["pred_below_display_threshold"] == 1
    assert counts["pred_drawn"] == 1
    assert counts["gt_drawn"] == 1
    assert predictions == unchanged
    assert not frame.image.any()
    assert tile.size == (100, 170)


def test_boxes_behind_camera_are_reported_without_fake_projection():
    box = native_box()
    box["translation_m"] = [0, 0, -10]
    _, counts = overlay(sample(), [box], [], label="Behind", score_threshold=0.2)
    assert counts["pred_drawn"] == 0
    assert len(counts["projection_omissions"]) == 1
    assert "behind camera" in counts["projection_omissions"][0]["reason"]


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold"):
        overlay(sample(), [], [], label="Invalid", score_threshold=float("nan"))
