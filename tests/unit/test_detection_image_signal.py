"""Analytic tests of training-only priors and a fixed frame derangement."""

from __future__ import annotations

import copy
import json
import math

import pytest

from e_jepa_ttc.detection.development_metrics import evaluate_detections
from scripts.check_detection_image_signal import (
    analyze_results,
    derange_predictions,
    fit_constant_prior,
    load_results,
)


def box(x=0.0, category="car", yaw=0.0, score=0.9):
    return {
        "class_name": category,
        "translation_m": [x, 0.0, 0.0],
        "size_lwh_m": [4.0, 2.0, 1.0],
        "yaw_rad": yaw,
        "score": score,
        "velocity_mps": [x, 0.0],
    }


def results(predictions, truth):
    return {
        "predictions": predictions,
        "ground_truth": truth,
        "metrics": evaluate_detections(predictions, truth),
    }


def test_prior_medians_and_score_use_objects_and_frames_respectively():
    truth = {"t1": [box(0), box(20)], "t2": [box(category="truck")], "t3": []}
    priors = {row["class_name"]: row for row in fit_constant_prior(truth)}
    assert set(priors) == {"car", "truck"}
    assert priors["car"]["translation_m"] == [10.0, 0.0, 0.0]
    assert priors["car"]["score"] == 1 / 3
    assert priors["car"]["train_object_count"] == 2
    assert priors["car"]["train_frames_with_class"] == 1


def test_circular_mean_does_not_average_angles_across_wrap_to_zero():
    priors = fit_constant_prior({"t1": [box(yaw=math.pi - 0.1), box(yaw=-math.pi + 0.1)]})
    assert abs(priors[0]["yaw_rad"]) == pytest.approx(math.pi)
    assert priors[0]["yaw_undefined_fallback_zero"] is False


def test_undefined_circular_mean_is_explicitly_flagged():
    priors = fit_constant_prior({"t1": [box(yaw=0), box(yaw=math.pi)]})
    assert priors[0]["yaw_rad"] == 0
    assert priors[0]["yaw_undefined_fallback_zero"] is True


def test_selection_labels_never_change_fitted_priors():
    train = results({"t": [box(10)]}, {"t": [box(10)]})
    predictions = {"s1": [box(10)], "s2": [box(20)]}
    first = results(predictions, {"s1": [box(10)], "s2": [box(20)]})
    second = results(predictions, {"s1": [box(1000, "bus")], "s2": []})
    a = analyze_results(train, first)
    b = analyze_results(train, second)
    assert a["train_fitted_priors"] == b["train_fitted_priors"]
    assert (
        a["frame_derangement"]["assignment_sha256"] == b["frame_derangement"]["assignment_sha256"]
    )
    assert a["constant_prior"]["metrics"] != b["constant_prior"]["metrics"]


def test_derangement_is_deterministic_bijective_and_preserves_complete_predictions():
    predictions = {f"s{i}": [box(i, score=0.1 * i)] * i for i in range(1, 8)}
    original = copy.deepcopy(predictions)
    permuted, assignment, attempts = derange_predictions(predictions)
    again, assignment_again, attempts_again = derange_predictions(
        dict(reversed(list(predictions.items())))
    )
    assert permuted == again
    assert assignment == assignment_again
    assert attempts == attempts_again
    assert set(assignment) == set(assignment.values()) == set(predictions)
    assert all(target != source for target, source in assignment.items())
    assert sum(map(len, permuted.values())) == sum(map(len, predictions.values()))
    for target, source in assignment.items():
        assert permuted[target] == predictions[source]
    permuted[next(iter(permuted))].clear()
    assert predictions == original


def test_two_frame_derangement_removes_analytic_image_specific_match():
    train = results({"t": []}, {"t": [box(20)]})
    selection = results({"s1": [box(0)], "s2": [box(10)]}, {"s1": [box(0)], "s2": [box(10)]})
    report = analyze_results(train, selection)
    assert report["observed_model_metrics"]["dev_AP101_supported_classes"] == 1
    assert report["frame_derangement"]["metrics"]["dev_AP101_supported_classes"] == 0
    assert report["frame_derangement"]["fixed_points"] == 0
    assert report["dev_AP101_observed_minus_derangement"] == 1
    assert "ground_truth" not in report
    assert "predictions" not in report


def test_derangement_rejects_single_frame():
    with pytest.raises(ValueError, match="at least two"):
        derange_predictions({"s": []})


def test_result_tokens_must_be_valid_before_controls():
    train = results({"t": []}, {"t": []})
    selection = results({"s1": [], "s2": []}, {"s1": [], "s2": []})
    selection["predictions"]["extra"] = []
    with pytest.raises(ValueError, match="tokens"):
        analyze_results(train, selection)


def test_train_and_selection_must_not_overlap():
    payload = results({"s1": [], "s2": []}, {"s1": [], "s2": []})
    with pytest.raises(ValueError, match="disjoint"):
        analyze_results(payload, payload)


def test_modified_stored_metrics_are_rejected():
    train = results({"t": []}, {"t": []})
    selection = results({"s1": [], "s2": []}, {"s1": [], "s2": []})
    selection["metrics"]["dev_AP101_supported_classes"] = 1
    with pytest.raises(ValueError, match="Stored metric"):
        analyze_results(train, selection)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "results.json"
    path.write_text('{"predictions": {}, "predictions": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        load_results(path)
    path.write_text(json.dumps(results({"t": []}, {"t": []})), encoding="utf-8")
    assert load_results(path)["ground_truth"] == {"t": []}
