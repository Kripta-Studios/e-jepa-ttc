"""Tests for the bounded campaign constant-control scorer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from e_jepa_ttc.detection.evaluation_controls import constant_predictions
from e_jepa_ttc.detection.evaluation_ground_truth import build_evaluation_ground_truth
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_detection_campaign_controls.py"
SPEC = importlib.util.spec_from_file_location("campaign_controls", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
EVENT_FROM_EGO = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 0, 1]]
)


def annotation(category: str, x: float, y: float = 0.0) -> dict[str, object]:
    return {
        "category": category,
        "bbox_3d_ego": [x, y, 0.0, 4.0, 1.5, 2.0, 0.0],
    }


def ground_truth(*rows: dict[str, object]):
    return build_evaluation_ground_truth(
        rows,
        intrinsic=K,
        event_from_ego=EVENT_FROM_EGO,
        image_wh=(100, 100),
        protocol=EvaluationProtocol.default(),
    )


def test_compact_streaming_constant_matches_full_v2_global_metrics():
    protocol = EvaluationProtocol.default()
    truth = {
        "a": ground_truth(annotation("car", 10.0), annotation("bicycle", 12.0)),
        "b": ground_truth(
            annotation("car", 11.0),
            annotation("truck", 20.0),
            annotation("bicycle", 10.0, 7.0),
        ),
        # Strict upper depth boundary: both GT and equal prior are outside domain.
        "boundary": ground_truth(annotation("motorcycle", 60.0)),
        "c": ground_truth(),
    }
    priors = [
        {
            "class_name": "car",
            "translation_m": [10.0, 0.0, 0.0],
            "size_lwh_m": [4.0, 2.0, 1.5],
            "yaw_rad": 0.0,
            "score": 0.75,
        },
        {
            "class_name": "bicycle",
            "translation_m": [12.0, 3.1, 0.0],
            "size_lwh_m": [4.0, 2.0, 1.5],
            "yaw_rad": 0.0,
            "score": 0.25,
        },
        {
            "class_name": "motorcycle",
            "translation_m": [60.0, 0.0, 0.0],
            "size_lwh_m": [4.0, 2.0, 1.5],
            "yaw_rad": 0.0,
            "score": 0.1,
        },
    ]
    full: Any = evaluate_detections_v2(
        constant_predictions(sorted(truth), priors), truth, protocol=protocol
    )
    compact: Any = MODULE.score_constant_streaming(
        sorted(truth.items()), priors, protocol=protocol
    )
    assert compact["sample_count"] == full["sample_count"]
    assert compact["population"] == full["population"]
    assert compact["predictions"]["input_count"] == full["predictions"]["input_count"]
    assert compact["predictions"]["scored_count"] == full["predictions"]["scored_count"]
    assert compact["class_agnostic_recall"] == full["class_agnostic_recall"]
    assert compact["dev_AP101_supported_classes"] == full["dev_AP101_supported_classes"]
    for category in protocol.classes:
        assert (
            compact["per_class"][category]["AP101_mean_thresholds"]
            == full["per_class"][category]["AP101_mean_thresholds"]
        )
        for threshold in protocol.distance_thresholds_m:
            key = str(threshold)
            compact_threshold = compact["per_class"][category]["thresholds"][key]
            full_threshold = full["per_class"][category]["thresholds"][key]
            for field in ("AP101", "TP", "FP", "FN", "ignored_prediction_count"):
                assert compact_threshold[field] == full_threshold[field]


def test_streaming_requires_sorted_unique_tokens():
    protocol = EvaluationProtocol.default()
    prior = [
        {
            "class_name": "car",
            "translation_m": [10.0, 0.0, 0.0],
            "size_lwh_m": [4.0, 2.0, 1.5],
            "yaw_rad": 0.0,
            "score": 1.0,
        }
    ]
    frame = ground_truth(annotation("car", 10.0))
    with pytest.raises(ValueError, match="strictly sorted"):
        MODULE.score_constant_streaming(
            [("b", frame), ("a", frame)], prior, protocol=protocol
        )


def test_streaming_rejects_duplicate_class_priors():
    protocol = EvaluationProtocol.default()
    prior = {
        "class_name": "car",
        "translation_m": [10.0, 0.0, 0.0],
        "size_lwh_m": [4.0, 2.0, 1.5],
        "yaw_rad": 0.0,
        "score": 1.0,
    }
    with pytest.raises(ValueError, match="one prior per class"):
        MODULE.score_constant_streaming(
            [("a", ground_truth(annotation("car", 10.0)))],
            [prior, dict(prior)],
            protocol=protocol,
        )


def test_future_comparison_gate_requires_frozen_completed_predictions(tmp_path: Path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"status": "RUNNING", "frozen": False}), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="actual completed summary"):
        MODULE.require_frozen_completed_predictions(path)
