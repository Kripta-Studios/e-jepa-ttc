"""Population, ignore and diagnostic fixtures for the local detection protocol v2."""

import dataclasses
import math

import numpy as np
import pytest

from e_jepa_ttc.detection.evaluation_ground_truth import build_evaluation_ground_truth
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
# eAP ego X forward/Y left/Z up to a conventional camera X right/Y down/Z forward.
EVENT_FROM_EGO = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 0, 1]]
)


def annotation(
    x=10.0,
    y=0.0,
    z=0.0,
    category="car",
    size=(4.0, 2.0, 1.5),
    yaw=0.0,
):
    length, width, height = size
    return {
        "category": category,
        "bbox_3d_ego": [x, y, z, length, height, width, yaw],
    }


def prediction(
    x=10.0,
    y=0.0,
    z=0.0,
    score=0.9,
    category="car",
    size=(4.0, 2.0, 1.5),
    yaw=0.0,
):
    return {
        "class_name": category,
        "translation_m": [x, y, z],
        "size_lwh_m": list(size),
        "yaw_rad": yaw,
        "score": score,
    }


def ground_truth(rows, protocol=None):
    return build_evaluation_ground_truth(
        rows,
        intrinsic=K,
        event_from_ego=EVENT_FROM_EGO,
        image_wh=(100, 100),
        protocol=protocol or EvaluationProtocol.default(),
    )


def evaluate(predictions, gt, protocol=None):
    return evaluate_detections_v2(
        {"frame": predictions}, {"frame": gt}, protocol=protocol or EvaluationProtocol.default()
    )


def test_protocol_roundtrip_hash_is_canonical_and_strict():
    protocol = EvaluationProtocol.default()
    restored = EvaluationProtocol.from_dict(protocol.to_dict())
    assert restored == protocol
    assert restored.sha256 == protocol.sha256
    malformed = protocol.to_dict()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        EvaluationProtocol.from_dict(malformed)
    with pytest.raises(ValueError, match="semantics"):
        dataclasses.replace(protocol, matching_rule="ego XYZ distance")
    with pytest.raises(ValueError, match="immutable"):
        EvaluationProtocol(**{**protocol.to_dict(), "classes": list(protocol.classes)})


def test_protocol_visibility_text_tracks_configured_depth_field_by_reference():
    protocol = dataclasses.replace(EvaluationProtocol.default(), center_depth_range_m=(0.2, 80.0))
    assert "center_depth_range_m" in protocol.visibility_rule
    assert protocol.sha256 != EvaluationProtocol.default().sha256


def test_perfect_unequal_dimensions_and_known_yaw():
    protocol = EvaluationProtocol.default()
    row = annotation(size=(5.0, 1.25, 2.5), yaw=0.7)
    gt = ground_truth([row], protocol)
    result = evaluate([prediction(size=(5.0, 1.25, 2.5), yaw=0.7)], gt, protocol)
    assert result["population"]["identity_check"] is True
    assert result["dev_AP101_supported_classes"] == 1.0
    errors = result["diagnostic_matches"][0]["errors"]
    assert errors == {"xy_m": 0.0, "z_m": 0.0, "dimensions_lwh_m": [0.0] * 3, "yaw_rad": 0.0}


def test_projected_center_outside_is_still_eligible_when_cuboid_intersects_image():
    # Center projects to u=-10, while the long cuboid extends into the image.
    gt = ground_truth([annotation(x=10, y=6, size=(4, 4, 1.5))])
    assert len(gt.eligible) == 1
    assert not gt.ignored


def test_near_plane_clipping_and_rectangle_intersection_keep_visible_cuboids():
    # The cuboid straddles the 0.1 m near plane. Its clipped projection covers the
    # viewport even though its lateral corners all project beyond the image borders.
    gt = ground_truth([annotation(x=0.2, size=(0.3, 2.0, 0.2))])
    assert len(gt.eligible) == 1


def test_outside_domain_is_ignored_and_symmetric_prediction_is_filtered():
    gt = ground_truth([annotation(x=10, y=30)])
    assert not gt.eligible
    assert gt.ignored[0].ignore_reason == "projected_cuboid_outside_image"
    result = evaluate([prediction(x=10, y=30)], gt)
    assert result["population"]["ignored_ground_truth_count"] == 1
    assert result["predictions"]["outside_domain_count"] == 1
    assert result["per_class"]["car"]["thresholds"]["2.0"]["FP"] == 0


def test_ignored_gt_suppresses_prediction_but_absent_gt_does_not():
    # Default maximum matching radius is 4 m, so use a custom transparent diagnostic fixture.
    protocol = dataclasses.replace(EvaluationProtocol.default(), distance_thresholds_m=(2.0, 20.0))
    # A same-class prediction inside the protocol domain but close in ego XY to the
    # depth-ignored target is ignored rather than scored as background.
    ignored = ground_truth([annotation(x=70)], protocol)
    result = evaluate([prediction(x=59.9)], ignored, protocol)
    assert result["per_class"]["car"]["thresholds"]["20.0"]["ignored_prediction_count"] == 1
    absent = ground_truth([], protocol)
    result = evaluate([prediction(x=59.9)], absent, protocol)
    assert result["per_class"]["car"]["thresholds"]["20.0"]["FP"] == 1


def test_duplicate_eligible_prediction_stays_false_positive_even_near_ignored():
    protocol = dataclasses.replace(EvaluationProtocol.default(), distance_thresholds_m=(2.0, 4.0))
    gt = ground_truth([annotation(x=59), annotation(x=61)], protocol)
    assert len(gt.eligible) == 1 and len(gt.ignored) == 1
    result = evaluate([prediction(x=59, score=1), prediction(x=59.1, score=0.9)], gt, protocol)
    metric = result["per_class"]["car"]["thresholds"]["4.0"]
    assert (metric["TP"], metric["FP"], metric["ignored_prediction_count"]) == (1, 1, 0)
    assert metric["prediction_identity_check"] is True


def test_strict_threshold_and_z_error_are_reported_separately():
    gt = ground_truth([annotation()])
    result = evaluate([prediction(x=10.5, z=2.0)], gt)
    assert result["per_class"]["car"]["thresholds"]["0.5"]["TP"] == 0
    assert result["per_class"]["car"]["thresholds"]["1.0"]["TP"] == 1
    match = result["diagnostic_matches"][0]
    assert match["errors"]["xy_m"] == 0.5
    assert match["errors"]["z_m"] == 2.0


def test_absent_classes_are_null_and_empty_frame_is_valid():
    result = evaluate([], ground_truth([]))
    assert result["dev_AP101_supported_classes"] is None
    assert result["dev_AP101_all_eight_absent_zero"] == 0
    assert result["per_class"]["bicycle"]["AP101_mean_thresholds"] is None


def test_proposal_cap_uses_score_then_original_index_deterministically():
    protocol = dataclasses.replace(EvaluationProtocol.default(), max_predictions_per_frame=1)
    gt = ground_truth([annotation()], protocol)
    result = evaluate([prediction(x=20, score=0.1), prediction(score=0.9)], gt, protocol)
    assert result["predictions"]["capped_before_domain_filter"] == 1
    assert result["predictions"]["identity_check"] is True
    assert result["dev_AP101_supported_classes"] == 1


def test_ground_truth_hash_does_not_depend_on_detector_capacity_or_predictions():
    protocol = EvaluationProtocol.default()
    first = ground_truth([annotation(), annotation(category="truck", y=2)], protocol)
    second = ground_truth([annotation(), annotation(category="truck", y=2)], protocol)
    assert first.sha256 == second.sha256
    assert (
        evaluate([prediction()], first, protocol)["population"]["ground_truth_sha256_by_token"][
            "frame"
        ]
        == first.sha256
    )


@pytest.mark.parametrize(
    "row",
    [
        prediction(score=math.nan),
        prediction(score=True),
        prediction(size=(1, 0, 1)),
        prediction(category="SUV"),
        prediction(yaw=math.inf),
    ],
)
def test_malformed_predictions_fail(row):
    with pytest.raises(ValueError):
        evaluate([row], ground_truth([]))


def test_protocol_mismatch_and_token_mismatch_fail():
    protocol = EvaluationProtocol.default()
    other = dataclasses.replace(protocol, center_depth_range_m=(0.2, 60.0))
    gt = ground_truth([], protocol)
    with pytest.raises(ValueError, match="protocol"):
        evaluate_detections_v2({"frame": []}, {"frame": gt}, protocol=other)
    with pytest.raises(ValueError, match="tokens"):
        evaluate_detections_v2({}, {"frame": gt}, protocol=protocol)
