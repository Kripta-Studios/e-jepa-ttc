from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from e_jepa_ttc.data.annotations import LabelMeasurement
from e_jepa_ttc.data.evttc_object_cache import (
    EvTTCObjectCacheConfig,
    _actions,
    _event_box,
    _EvTTCState,
    _windows,
)


def _measurement(index: int) -> LabelMeasurement:
    return LabelMeasurement(
        sequence_id="sequence",
        frame_index=index,
        timestamp_us=1_000_000 + index * 50_000,
        category="car",
        bbox_xyxy=(192.0, 120.0, 960.0, 600.0),
        bbox_area=768.0 * 480.0,
        bbox_scale=(768.0 * 480.0) ** 0.5,
        ttc_seconds=8.0 - index * 0.05,
        image_width=1920,
        image_height=1200,
    )


def test_evttc_object_windows_and_cross_sensor_scaling() -> None:
    states = [
        _EvTTCState(
            measurement=_measurement(index),
            bbox_event_xyxy=(128.0, 72.0, 640.0, 360.0),
            depth_m=20.0 - index * 0.1,
        )
        for index in range(16)
    ]
    windows = _windows(states, EvTTCObjectCacheConfig())

    assert windows
    history, future = windows[0]
    assert len(history) == 3
    assert 100 in future
    assert _event_box(_measurement(0)) == (128.0, 72.0, 640.0, 360.0)


def test_evttc_object_windows_can_match_garl_100ms_cadence() -> None:
    states = [
        _EvTTCState(
            measurement=_measurement(index),
            bbox_event_xyxy=(128.0, 72.0, 640.0, 360.0),
            depth_m=20.0 - index * 0.1,
        )
        for index in range(16)
    ]
    config = EvTTCObjectCacheConfig(
        history_stride_frames=2,
        maximum_history_gap_ms=120,
    )
    windows = _windows(states, config)

    history, _ = windows[0]
    assert [state.timestamp_us for state in history] == [
        1_000_000,
        1_100_000,
        1_200_000,
    ]


def test_evttc_action_contract_and_disjoint_horizons_are_validated() -> None:
    assert EvTTCObjectCacheConfig().action_dim == 8
    with pytest.raises(ValueError, match="must not overlap"):
        EvTTCObjectCacheConfig(prediction_horizons_ms=(50, 100))
    with pytest.raises(ValueError, match="eight physical"):
        EvTTCObjectCacheConfig(action_dim=4)


def test_evttc_yaw_rate_converts_degrees_and_unwraps(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "navigation.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        group = handle.create_group("integratedNavigation/data")
        group.create_dataset("ts", data=np.asarray([0, 1_000_000], dtype=np.int64))
        group.create_dataset("velocity", data=np.zeros((2, 3), dtype=np.float64))
        group.create_dataset(
            "attitude",
            data=np.asarray([[0.0, 0.0, 359.0], [0.0, 0.0, 1.0]]),
        )
        _write_navigation_to_event_calibration(handle, np.eye(3))
    with h5py.File(hdf5_path, "r") as handle:
        features, valid = _actions(handle, start_us=0, end_us=1_000_000)

    assert valid
    assert features[7] == pytest.approx(np.deg2rad(2.0), abs=1e-7)


def test_evttc_actions_transform_world_velocity_to_event_camera(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "navigation.hdf5"
    navigation_rfu_to_event_optical = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
        )
    )
    with h5py.File(hdf5_path, "w") as handle:
        group = handle.create_group("integratedNavigation/data")
        group.create_dataset("ts", data=np.asarray([0, 1_000_000], dtype=np.int64))
        group.create_dataset(
            "velocity",
            data=np.asarray([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        )
        group.create_dataset("attitude", data=np.zeros((2, 3), dtype=np.float64))
        _write_navigation_to_event_calibration(
            handle,
            navigation_rfu_to_event_optical,
        )
    with h5py.File(hdf5_path, "r") as handle:
        features, valid = _actions(handle, start_us=0, end_us=1_000_000)

    assert valid
    assert features[1] == pytest.approx(0.0, abs=1e-7)
    assert features[2] == pytest.approx(0.0, abs=1e-7)
    assert features[3] == pytest.approx(10.0, abs=1e-7)


def _write_navigation_to_event_calibration(
    handle: h5py.File,
    rotation: np.ndarray,
) -> None:
    navigation = handle.require_group("integratedNavigation/data/calib")
    lidar = handle.require_group("livox/lidar/calib")
    event = handle.require_group("prophesee/event_cam_left/calib")
    identity = np.eye(4)
    event_transform = identity.copy()
    event_transform[:3, :3] = rotation
    navigation.create_dataset("T_to_lidar", data=identity)
    lidar.create_dataset("T_to_left_cam", data=identity)
    event.create_dataset("T_bfs_to_prophesee", data=event_transform)
