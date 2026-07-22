from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e_jepa_ttc.data.eap import (
    EAPEventReader,
    box_corners_ego,
    build_eap_object_windows,
    project_box_3d_to_event,
    reconstruct_eap_object_states,
)


def test_box_projection_returns_camera_geometry() -> None:
    box = np.asarray([0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0])
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    corners, projected, nearest_depth, visible_height = project_box_3d_to_event(
        box,
        intrinsic,
        np.eye(4),
        image_size=(100, 80),
    )

    assert box_corners_ego(box).shape == (8, 3)
    assert corners.shape == (8, 3)
    assert nearest_depth == 9.0
    assert projected[0] < 50.0 < projected[2]
    assert projected[1] < 40.0 < projected[3]
    assert visible_height > 20.0


def test_reconstructed_ttc_is_signed_depth_over_depth_velocity() -> None:
    tokens = ["sequence:0", "sequence:1", "sequence:2", "sequence:3"]
    event_from_ego = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    media = pd.DataFrame(
        {
            "sample_token": tokens,
            "split": ["train"] * 4,
            "sequence_id": ["sequence"] * 4,
            "rgb_shard_path": ["rgb.tar"] * 4,
            "rgb_member_path": ["rgb/image.png"] * 4,
            "events_path": ["events.h5"] * 4,
            "labels_path": ["labels.parquet"] * 4,
            "rgb_exposure_start_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
            "rgb_exposure_end_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
            "K_event": [intrinsic] * 4,
            "T_event_ego": [event_from_ego] * 4,
        }
    )
    labels = pd.DataFrame(
        {
            "sample_token": tokens,
            "sequence_id": ["sequence"] * 4,
            "track_id": ["track"] * 4,
            "category": ["car"] * 4,
            "bbox_3d_ego": [
                np.asarray([10.0 - index, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]) for index in range(4)
            ],
        }
    )

    states = reconstruct_eap_object_states(
        media,
        labels,
        derivative_radius=1,
        maximum_gap_s=1.1,
    )

    assert len(states) == 4
    assert np.allclose([state.depth_velocity_mps for state in states], -1.0)
    assert np.allclose([state.ttc_s for state in states], [9.0, 8.0, 7.0, 6.0])
    assert all(state.ttc_source.startswith("reconstructed_public") for state in states)
    windows = build_eap_object_windows(
        states,
        history_frames=2,
        horizons_ms=(1000,),
        maximum_slop_ms=1,
        maximum_history_gap_ms=1001,
        ttc_range_s=(0.0, 20.0),
    )
    assert len(windows) == 2
    assert windows[0].future[0][1].timestamp_us > windows[0].target.timestamp_us
    assert windows[0].ego_action_valid is False


def test_eap_event_reader_applies_exact_half_open_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "events.h5"
    with h5py.File(source, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.asarray([1, 2, 3, 4, 5], dtype=np.uint16))
        events.create_dataset("y", data=np.asarray([6, 7, 8, 9, 10], dtype=np.uint16))
        events.create_dataset("t", data=np.asarray([1000, 1500, 1999, 2000, 2500]))
        events.create_dataset("p", data=np.asarray([0, 1, 0, 1, 1], dtype=np.int8))
        handle.create_dataset("ms_to_idx", data=np.asarray([0, 0, 3, 5], dtype=np.uint64))

    events = EAPEventReader(source).read_window(1500, 2500)

    assert events["t"].tolist() == [1500, 1999, 2000]
    assert events["x"].tolist() == [2, 3, 4]
    assert all(array.shape == (3,) for array in events.values())
