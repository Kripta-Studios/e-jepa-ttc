"""Independent camera conventions and target-free decode checks."""

import math

import numpy as np
import pytest
import torch

from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample
from e_jepa_ttc.detection.monodgp_eap_adapter import (
    camera_yaw_to_ego,
    decode_eap,
    ego_yaw_to_camera,
    prepare_sample,
    resized_intrinsic,
    wrap_angle,
)


def sample():
    transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    return RawRGBSample(
        "a:0",
        "a",
        np.zeros((384, 640, 3), dtype=np.uint8),
        [{"category": "car", "bbox_3d_ego": [2, 20, 0, 4, 1.5, 2, 0]}],
        np.array([[400, 0, 320], [0, 300, 192], [0, 0, 1.0]]),
        transform,
        100,
        110,
        {},
        "fixture",
    )


def test_known_axes_depth_center_and_heading():
    prepared = prepare_sample(sample())
    target = prepared.target
    assert target["depth"].item() == 20
    assert target["boxes_3d"][0, :2].tolist() == pytest.approx([360 / 640, 0.5])
    assert target["size_3d"][0].tolist() == [1.5, 2, 4]
    assert prepared.native_ground_truth[0]["size_lwh_m"] == [4, 2, 1.5]
    heading = target["heading_bin"].item() * math.pi / 6 + target["heading_res"].item()
    assert wrap_angle(heading) == pytest.approx(-math.pi / 2 - math.atan(0.1), abs=1e-6)
    assert target["obj_region"].sum() > 0


def test_anisotropic_resize_pixel_center_mapping():
    intrinsic = np.array([[800, 0, 640], [0, 700, 360], [0, 0, 1.0]])
    resized = resized_intrinsic(intrinsic, (1280, 720), (640, 384))
    point = np.array([1.5, 0.3, 20.0])
    original = intrinsic @ point
    original = original[:2] / original[2]
    new = resized @ point
    expected = (original + 0.5) * [0.5, 384 / 720] - 0.5
    assert new[:2] / new[2] == pytest.approx(expected)
    assert resized[0, 0] == 400
    assert resized[1, 1] == pytest.approx(700 * 384 / 720)


@pytest.mark.parametrize("yaw", [-math.pi, -2.0, 0.0, 1.3, math.pi - 0.01])
def test_heading_with_tilted_camera(yaw):
    transform = sample().event_from_ego[:3, :3]
    roll = 0.13
    tilt = np.array(
        [[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1.0]]
    )
    rotation = tilt @ transform
    camera = ego_yaw_to_camera(yaw, rotation)
    assert wrap_angle(camera_yaw_to_ego(camera, rotation) - yaw) == pytest.approx(0, abs=1e-12)


def oracle_outputs(prepared):
    target = prepared.target
    heading = torch.zeros(1, 1, 24)
    index = target["heading_bin"].item()
    heading[0, 0, index] = 10
    heading[0, 0, 12 + index] = target["heading_res"].item()
    logits = torch.full((1, 1, 8), -10.0)
    logits[0, 0, 0] = 10
    return {
        "pred_logits": logits,
        "pred_boxes": target["boxes_3d"][None],
        "pred_depth": torch.tensor([[[20.0, -20.0]]]),
        "pred_3d_dim": target["size_3d"][None],
        "pred_angle": heading,
    }


def test_decode_geometric_center_without_bottom_shift_or_uncertainty_score():
    raw = sample()
    prepared = prepare_sample(raw)
    predicted, counts = decode_eap(
        oracle_outputs(prepared), 0, prepared.calibration, raw.event_from_ego, prepared.image_size
    )
    assert counts["emitted"] == 1
    assert predicted[0]["translation_m"] == pytest.approx([2, 20, 0], abs=1e-5)
    assert predicted[0]["size_lwh_m"] == [4, 2, 1.5]
    assert predicted[0]["yaw_rad"] == pytest.approx(0, abs=1e-6)
    assert 0.99 < predicted[0]["score"] <= 1
    assert "velocity_mps" not in predicted[0]


def test_physical_invalid_predictions_are_counted_and_nan_fails():
    raw = sample()
    prepared = prepare_sample(raw)
    outputs = oracle_outputs(prepared)
    outputs["pred_3d_dim"] = -outputs["pred_3d_dim"]
    predictions, counts = decode_eap(
        outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
    )
    assert predictions == []
    assert counts["invalid_size_or_depth"] == 1
    outputs["pred_logits"][0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        decode_eap(outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size)


def test_query_capacity_never_silently_truncates():
    raw = sample()
    raw.annotations.append(dict(raw.annotations[0]))
    with pytest.raises(ValueError, match="exceed"):
        prepare_sample(raw, max_objects=1)


def test_visibility_exclusion_count_and_empty_target_shapes():
    raw = sample()
    raw.annotations[0]["bbox_3d_ego"][1] = -20
    prepared = prepare_sample(raw)
    assert prepared.target["boxes_3d"].shape == (0, 6)
    assert prepared.target["depth"].shape == (0, 1)
    assert prepared.report["ignored_by_reason"] == {"center_depth_outside_profile": 1}


def test_velocity_is_opt_in_and_preserves_native_components():
    raw = sample()
    raw.annotations[0]["velocity_3d"] = np.array([3.0, -2.0, 0.0])
    plain = prepare_sample(raw)
    prepared = prepare_sample(raw, include_velocity=True)
    assert "velocity" not in plain.target
    assert "velocity_mps" not in plain.native_ground_truth[0]
    assert prepared.target["velocity"].tolist() == [[3.0, -2.0]]
    assert prepared.target["velocity_valid"].dtype == torch.bool
    assert prepared.target["velocity_valid"].tolist() == [True]
    assert prepared.native_ground_truth[0]["velocity_mps"] == [3.0, -2.0]
    for key in plain.target:
        assert torch.equal(plain.target[key], prepared.target[key])
    assert torch.equal(plain.image, prepared.image)


@pytest.mark.parametrize("velocity", [None, [math.nan, 1, 0], [1], "unknown"])
def test_invalid_velocity_is_masked_not_invented_as_valid_zero(velocity):
    raw = sample()
    raw.annotations[0]["velocity_3d"] = velocity
    prepared = prepare_sample(raw, include_velocity=True)
    assert prepared.target["velocity_valid"].tolist() == [False]
    assert "velocity_mps" not in prepared.native_ground_truth[0]
    assert prepared.report["velocity_invalid_or_missing_targets"] == 1


def test_velocity_decode_uses_learned_output_without_ground_truth():
    raw = sample()
    prepared = prepare_sample(raw)
    outputs = oracle_outputs(prepared)
    outputs["pred_velocity"] = torch.tensor([[[2.5, -1.5]]])
    predictions, _ = decode_eap(
        outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
    )
    assert predictions[0]["velocity_mps"] == [2.5, -1.5]
    outputs["pred_velocity"][0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        decode_eap(outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size)


def test_empty_velocity_targets_have_explicit_shapes():
    raw = sample()
    raw.annotations.clear()
    prepared = prepare_sample(raw, include_velocity=True)
    assert prepared.target["velocity"].shape == (0, 2)
    assert prepared.target["velocity_valid"].shape == (0,)
