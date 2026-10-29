from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from operational.evttc_competition.geometry import (
    Calibration,
    input_bbox_to_event_box,
    load_calibration,
    normalize_box_xyxy,
    project_event_box_to_rgb,
)


def _rotation_z(angle: float) -> np.ndarray:
    return np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def _calibration(**changes) -> Calibration:
    values = {
        "event_intrinsics": np.asarray([100.0, 100.0, 50.0, 40.0]),
        "event_distortion": np.zeros(4),
        "event_resolution": (100, 80),
        "rgb_intrinsics": np.asarray([100.0, 100.0, 50.0, 40.0]),
        "rgb_distortion": np.zeros(4),
        "rgb_resolution": (100, 80),
        "rotation_event_to_rgb": np.eye(3),
        "source_kind": "test",
        "transform_source": "test",
    }
    values.update(changes)
    return Calibration(**values)


def test_normalize_box_converts_one_based_inclusive_to_zero_based_half_open() -> None:
    assert normalize_box_xyxy((1, 1, 10, 20), (100, 80), one_based=True, inclusive_max=True) == (
        0.0,
        0.0,
        10.0,
        20.0,
    )
    assert normalize_box_xyxy((-5, 3, 120, 90), (100, 80)) == (0.0, 3.0, 100.0, 80.0)


def test_identity_rotation_preserves_event_box() -> None:
    calibration = _calibration()
    box = (10.0, 12.0, 40.0, 55.0)
    assert project_event_box_to_rgb(box, calibration) == pytest.approx(box)
    assert input_bbox_to_event_box(box, calibration, coordinate_plane="rgb") == pytest.approx(box)


def test_projection_uses_calibrated_intrinsics_not_naive_resolution_scaling() -> None:
    calibration = _calibration(
        rgb_intrinsics=np.asarray([200.0, 180.0, 120.0, 90.0]),
        rgb_resolution=(300, 200),
    )
    projected = project_event_box_to_rgb((40.0, 30.0, 60.0, 50.0), calibration)
    assert projected == pytest.approx((100.0, 72.0, 140.0, 108.0))
    assert projected != pytest.approx((120.0, 75.0, 180.0, 125.0))


def _write_road(path: Path, transform_name: str, transform: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        for side in ("right",):
            event = handle.create_group(f"prophesee/event_cam_{side}/calib")
            event["intrinsics"] = np.asarray([100.0, 101.0, 50.0, 40.0])
            event["distortion_coeffs"] = np.zeros(4)
            event["resolution"] = np.asarray([100, 80])
            event[transform_name] = transform
            rgb = handle.create_group(f"blackflys/{side}/calib")
            rgb["intrinsics"] = np.asarray([200.0, 201.0, 100.0, 80.0])
            rgb["distortion_coeffs"] = np.zeros(4)
            rgb["resolution"] = np.asarray([200, 160])


def test_road_loader_dispatches_transform_direction_from_dataset_name(tmp_path: Path) -> None:
    rotation = _rotation_z(0.1)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    explicit = tmp_path / "explicit.hdf5"
    legacy = tmp_path / "legacy.hdf5"
    _write_road(explicit, "T_bfs_to_prophesee", transform)
    _write_road(legacy, "T_to_right_bfs", transform)
    assert load_calibration(explicit).rotation_event_to_rgb == pytest.approx(rotation.T)
    loaded_legacy = load_calibration(legacy)
    assert loaded_legacy.rotation_event_to_rgb == pytest.approx(rotation.T)
    assert (
        "data-format documentation" in loaded_legacy.metadata["transform_direction_interpretation"]
    )


def _rotation_vector_z(angle: float) -> np.ndarray:
    return np.asarray([0.0, 0.0, angle])


def _write_slider(path: Path) -> None:
    rgb_to_event = _rotation_z(0.02)
    translation = np.asarray([-4.0, 1.0, 24.0])
    rgb_vectors = np.asarray(
        [_rotation_vector_z(value) for value in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6)]
    )
    rgb_translations = np.asarray(
        [[-100.0 + 25 * index, -60.0 + 13 * index, 900.0 + 40 * index] for index in range(6)]
    )
    event_vectors = rgb_vectors + np.asarray([0.0, 0.0, 0.02])
    event_translations = np.asarray(
        [translation + rgb_to_event @ value for value in rgb_translations]
    )
    # Add one unmatched event pose; consensus must retain the six valid pairs.
    event_vectors = np.vstack((event_vectors, np.asarray([1.1, -0.4, 0.3])))
    event_translations = np.vstack((event_translations, np.asarray([500.0, -700.0, 200.0])))
    with h5py.File(path, "w") as handle:
        refs = handle.create_group("#refs#")
        rgb = refs.create_group("rgb")
        event = refs.create_group("event")
        for group, image_size, k, vectors, translations in (
            (
                rgb,
                [1080.0, 1440.0],
                [[500.0, 0.0, 0.0], [0.0, 510.0, 0.0], [720.0, 540.0, 1.0]],
                rgb_vectors,
                rgb_translations,
            ),
            (
                event,
                [480.0, 640.0],
                [[200.0, 0.0, 0.0], [0.0, 201.0, 0.0], [320.0, 240.0, 1.0]],
                event_vectors,
                event_translations,
            ),
        ):
            group["ImageSize"] = np.asarray(image_size).reshape(2, 1)
            group["K"] = np.asarray(k)
            group["RadialDistortion"] = np.zeros((2, 1))
            group["TangentialDistortion"] = np.zeros((2, 1))
            group["RotationVectors"] = vectors.T
            group["TranslationVectors"] = translations.T


def test_slider_loader_uses_all_pose_candidates_and_marks_result_provisional(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibParams.mat"
    _write_slider(path)
    loaded = load_calibration(path, "slider_candidate")
    assert loaded.event_resolution == (640, 480)
    assert loaded.rgb_resolution == (1440, 1080)
    assert loaded.rotation_event_to_rgb == pytest.approx(_rotation_z(-0.02), abs=1e-7)
    assert loaded.metadata["matched_pose_count"] == 6
    assert loaded.metadata["event_pose_count"] == 7
    assert loaded.metadata["status"] == "candidate_only_no_authoritative_intercamera_extrinsic"
    assert loaded.translation_omitted is True


def test_invalid_box_and_unknown_plane_fail_closed() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        normalize_box_xyxy((4, 2, 4, 9), (10, 10))
    with pytest.raises(ValueError, match="coordinate_plane"):
        input_bbox_to_event_box((1, 1, 2, 2), _calibration(), coordinate_plane="unknown")
