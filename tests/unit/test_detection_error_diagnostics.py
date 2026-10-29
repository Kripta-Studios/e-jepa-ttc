from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import pytest

from e_jepa_ttc.detection.error_diagnostics import (
    analyze_query_sidecar,
    build_error_diagnostics,
    class_agnostic_correspondence,
    oracle_predictions,
)
from e_jepa_ttc.detection.evaluation_ground_truth import EvaluationBox, EvaluationGroundTruth
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2
from scripts.analyze_detection_errors import verify_pinned_file


def tp_at(result: Mapping[str, object], category: str, threshold: str) -> int:
    per_class = result["per_class"]
    assert isinstance(per_class, Mapping)
    class_result = per_class[category]
    assert isinstance(class_result, Mapping)
    thresholds = class_result["thresholds"]
    assert isinstance(thresholds, Mapping)
    threshold_result = thresholds[threshold]
    assert isinstance(threshold_result, Mapping)
    value = threshold_result["TP"]
    assert isinstance(value, int)
    return value


def frame(protocol: EvaluationProtocol, boxes: list[EvaluationBox]) -> EvaluationGroundTruth:
    return EvaluationGroundTruth(
        protocol_sha256=protocol.sha256,
        eligible=tuple(boxes),
        ignored=(),
        raw_count=len(boxes),
        ignored_by_reason=(),
        intrinsic=((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
        event_from_ego=(
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 2.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        image_wh=(100, 100),
    )


def prediction(category: str, x: float, y: float, z: float, score: float) -> dict[str, object]:
    return {
        "class_name": category,
        "translation_m": [x, y, z],
        "size_lwh_m": [4.0, 2.0, 2.0],
        "yaw_rad": 0.0,
        "score": score,
    }


def test_correspondence_is_one_to_one_and_reports_z_separately() -> None:
    protocol = EvaluationProtocol.default()
    ground_truth = {
        "s:1": frame(protocol, [EvaluationBox("car", (10.0, 0.0, 1.0), (4.0, 2.0, 2.0), 0.0, 0)])
    }
    predictions = {
        "s:1": [
            prediction("truck", 10.1, 0.0, 5.0, 0.9),
            prediction("car", 10.2, 0.0, 1.0, 0.8),
        ]
    }
    pairs = class_agnostic_correspondence(
        predictions,
        ground_truth,
        protocol=protocol,
        threshold_m=2.0,
        max_predictions_per_frame=50,
    )
    assert len(pairs) == 1
    assert pairs[0]["prediction_index"] == 0
    assert pairs[0]["ego_z_vertical_error_m"] == 4.0
    assert pairs[0]["camera_depth_error_m"] == pytest.approx(0.1)
    result = build_error_diagnostics(predictions, ground_truth, protocol=protocol)
    assert (
        result["geometry_errors_on_class_agnostic_pairs"]["ego_z_vertical_error_m"]["mean"] == 4.0
    )
    assert result["geometry_errors_on_class_agnostic_pairs"]["camera_depth_error_m"][
        "mean"
    ] == pytest.approx(0.1)


def test_class_oracle_changes_class_then_evaluator_rematches() -> None:
    protocol = EvaluationProtocol.default()
    ground_truth = {
        "s:1": frame(protocol, [EvaluationBox("car", (10.0, 0.0, 1.0), (4.0, 2.0, 2.0), 0.0, 0)])
    }
    predictions = {"s:1": [prediction("truck", 10.0, 0.0, 1.0, 0.9)]}
    before = evaluate_detections_v2(predictions, ground_truth, protocol=protocol)
    transformed = oracle_predictions(predictions, ground_truth, protocol=protocol, kind="class")
    after = evaluate_detections_v2(transformed, ground_truth, protocol=protocol)
    assert tp_at(before, "car", "2.0") == 0
    assert tp_at(after, "car", "2.0") == 1
    assert predictions["s:1"][0]["class_name"] == "truck"


def test_duplicate_and_class_confusion_are_attributed_after_matching() -> None:
    protocol = EvaluationProtocol.default()
    ground_truth = {
        "s:1": frame(
            protocol,
            [EvaluationBox("car", (10.0, 0.0, 1.0), (4.0, 2.0, 2.0), 0.0, 0)],
        )
    }
    predictions = {
        "s:1": [
            prediction("car", 10.0, 0.0, 1.0, 0.9),
            prediction("car", 10.1, 0.0, 1.0, 0.8),
            prediction("truck", 10.0, 0.0, 1.0, 0.7),
        ]
    }
    result = build_error_diagnostics(predictions, ground_truth, protocol=protocol)
    assert result["false_positive_reasons"]["counts"] == {
        "class_confusion": 1,
        "duplicate": 1,
        "true_positive": 1,
    }


def test_ignored_object_suppresses_only_one_prediction() -> None:
    protocol = EvaluationProtocol.default()
    ignored = EvaluationBox("car", (10.0, 0.0, 1.0), (4.0, 2.0, 2.0), 0.0, 0, "outside")
    base = frame(protocol, [])
    ground_truth = {
        "s:1": EvaluationGroundTruth(
            protocol_sha256=base.protocol_sha256,
            eligible=(),
            ignored=(ignored,),
            raw_count=1,
            ignored_by_reason=(("outside", 1),),
            intrinsic=base.intrinsic,
            event_from_ego=base.event_from_ego,
            image_wh=base.image_wh,
        )
    }
    predictions = {
        "s:1": [
            prediction("car", 10.0, 0.0, 1.0, 0.9),
            prediction("car", 10.1, 0.0, 1.0, 0.8),
        ]
    }
    result = build_error_diagnostics(predictions, ground_truth, protocol=protocol)
    assert result["false_positive_reasons"]["counts"] == {
        "ignored_object_duplicate_fp": 1,
        "ignored_object_suppressed": 1,
    }


def test_pinned_label_tampering_is_rejected(tmp_path) -> None:
    relative = "data/train/seq/labels.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    original = b"pinned-label-bytes"
    path.write_bytes(original)
    manifest = {
        "files": {
            relative: {
                "sha256": hashlib.sha256(original).hexdigest(),
                "size_bytes": len(original),
            }
        }
    }
    assert verify_pinned_file(tmp_path, relative, manifest) == hashlib.sha256(original).hexdigest()
    path.write_bytes(b"tampered-label!!")
    with pytest.raises(ValueError, match="differs from pinned manifest"):
        verify_pinned_file(tmp_path, relative, manifest)


def test_query_sidecar_maps_decoder_and_runs_fixed_k_control() -> None:
    protocol = EvaluationProtocol.default()
    ground_truth = {
        "s:1": frame(
            protocol,
            [EvaluationBox("car", (10.0, 0.0, 1.0), (4.0, 2.0, 2.0), 0.0, 0)],
        )
    }
    score = 0.9
    logit = math.log(score / (1 - score))
    predictions = {"s:1": [prediction("car", 10.0, 0.0, 1.0, score)]}
    sidecar = {
        "frames": [
            {
                "sample_token": "s:1",
                "tensors": {
                    "pred_logits": {"values": [[logit, -9, -9, -9, -9, -9, -9, -9]]},
                    "pred_depth": {"values": [[10.0, 0.5]]},
                },
            }
        ]
    }
    result = analyze_query_sidecar(sidecar, predictions, ground_truth, protocol=protocol)
    assert result["matched_query_count"] == 1
    assert result["classification_quality"]["target_top1_count"] == 1
    assert result["fixed_k_query_class_control"]["dev_AP101_supported_classes"] == 1.0


def test_center_oracle_uses_only_bounded_correspondences_and_rematches() -> None:
    protocol = EvaluationProtocol.default()
    ground_truth = {
        "s:1": frame(protocol, [EvaluationBox("car", (10.0, 0.0, 1.0), (4.0, 2.0, 2.0), 0.0, 0)])
    }
    predictions = {
        "s:1": [
            prediction("car", 11.9, 0.0, 8.0, 0.9),
            prediction("car", 50.0, 0.0, 1.0, 0.8),
        ]
    }
    transformed = oracle_predictions(predictions, ground_truth, protocol=protocol, kind="center")
    assert transformed["s:1"][0]["translation_m"] == [10.0, 0.0, 1.0]
    assert transformed["s:1"][1]["translation_m"] == [50.0, 0.0, 1.0]
    result = evaluate_detections_v2(transformed, ground_truth, protocol=protocol)
    assert tp_at(result, "car", "0.5") == 1


def test_invalid_oracle_is_rejected() -> None:
    protocol = EvaluationProtocol.default()
    try:
        oracle_predictions({}, {}, protocol=protocol, kind="unknown")
    except ValueError as error:
        assert "Unknown oracle" in str(error)
    else:
        raise AssertionError("invalid oracle should fail")
