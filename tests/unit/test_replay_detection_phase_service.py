from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.detection.causal_tracker import CausalMultiObjectTracker
from scripts.replay_detection_phase_service import (
    build_frame_plan,
    causal_rgb_groups,
    project_detection,
    replay_tracker_prefix,
    suppress_duplicates,
)


def prediction(score: float = 0.9, center=(0.0, 0.0, 10.0)):
    return {
        "class_name": "car",
        "score": score,
        "translation_m": list(center),
        "size_lwh_m": [4.0, 2.0, 2.0],
        "yaw_rad": 0.0,
        "velocity_mps": [1.0, -0.5],
    }


def calibration():
    return np.array([[100.0, 0, 640.0], [0, 100.0, 360.0], [0, 0, 1.0]]), np.eye(4)


def test_projection_preserves_detector_geometry_and_velocity() -> None:
    intrinsic, transform = calibration()
    item = project_detection(
        prediction(),
        token="sequence:frame",
        index=3,
        intrinsic=intrinsic,
        event_from_ego=transform,
        minimum_score=0.1,
    )
    assert item is not None
    assert item.detection_id == "sequence:frame:3"
    assert item.center_3d_m == (0.0, 0.0, 10.0)
    assert item.size_lwh_m == (4.0, 2.0, 2.0)
    assert item.velocity == (1.0, -0.5)
    assert item.velocity_frame == "released_native_components"


def test_projection_refuses_target_fields_and_filters_low_score() -> None:
    intrinsic, transform = calibration()
    assert (
        project_detection(
            prediction(0.01),
            token="s:f",
            index=0,
            intrinsic=intrinsic,
            event_from_ego=transform,
            minimum_score=0.1,
        )
        is None
    )
    bad = prediction()
    bad["ground_truth"] = 1
    with pytest.raises(ValueError, match="must not carry"):
        project_detection(
            bad,
            token="s:f",
            index=0,
            intrinsic=intrinsic,
            event_from_ego=transform,
            minimum_score=0.1,
        )


def test_nms_is_class_aware_and_does_not_mutate_source() -> None:
    intrinsic, transform = calibration()
    first = project_detection(
        prediction(0.9),
        token="s:f",
        index=0,
        intrinsic=intrinsic,
        event_from_ego=transform,
        minimum_score=0.1,
    )
    second = project_detection(
        prediction(0.8),
        token="s:f",
        index=1,
        intrinsic=intrinsic,
        event_from_ego=transform,
        minimum_score=0.1,
    )
    assert first is not None and second is not None
    source = [second, first]
    assert suppress_duplicates(source, 0.5) == [first]
    assert source == [second, first]


def test_frame_plan_and_prefix_are_strictly_causal() -> None:
    intrinsic, transform = calibration()
    tokens = ["s:a", "s:b", "s:c"]
    media = {
        token: {
            "sample_token": token,
            "sequence_id": "s",
            "split": "train",
            "rgb_exposure_start_timestamp_us": index * 100_000,
            "rgb_exposure_end_timestamp_us": index * 100_000 + 1_000,
            "K_event": intrinsic,
            "T_event_ego": transform,
        }
        for index, token in enumerate(tokens)
    }
    frames = build_frame_plan(
        {token: [prediction()] for token in tokens}, media, minimum_score=0.1, nms_iou=0.5
    )
    assert frames[-1]["filter_accounting"] == {
        "raw_predictions": 1,
        "below_score": 0,
        "projection_or_minimum_size_rejected": 0,
        "class_aware_nms_suppressed": 0,
        "retained": 1,
    }
    tracker = CausalMultiObjectTracker(confirm_hits=2)
    receipt = replay_tracker_prefix(frames, tracker)
    assert [row["sample_token"] for row in receipt] == tokens[:-1]
    assert tracker.tracks[-1].last_timestamp_us == frames[-2]["end_us"]


def test_rgb_groups_exclude_observations_outside_frozen_history() -> None:
    timestamps = {index: index * 100_000 for index in range(8)}
    assert causal_rgb_groups(timestamps, 700_000) == [
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
        (5, 6, 7),
    ]
