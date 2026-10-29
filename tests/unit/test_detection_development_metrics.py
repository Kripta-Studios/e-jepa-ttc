"""Analytic ranking and matching cases for the explicitly nonofficial scorer."""

import math

import pytest

from e_jepa_ttc.detection.development_metrics import evaluate_detections


def box(x=0.0, y=0.0, score=0.9, category="car", size=(4.0, 2.0, 1.0), yaw=0.0):
    return {
        "class_name": category,
        "translation_m": [x, y, 0.0],
        "size_lwh_m": list(size),
        "yaw_rad": yaw,
        "score": score,
    }


def test_perfect_and_absent_class_macros():
    result = evaluate_detections({"a": [box()]}, {"a": [box()]})
    assert result["dev_AP101_supported_classes"] == 1
    assert result["dev_AP101_all_eight_absent_zero"] == 1 / 8
    assert result["dev_ATE_m"] == result["dev_ASE"] == result["dev_AOE_rad"] == 0
    assert result["official_eap_score"] is False
    assert result["per_class"]["truck"]["AP101_mean_thresholds"] is None


def test_high_confidence_false_positive_reduces_ap():
    result = evaluate_detections({"a": [box(20, score=1), box(score=0.5)]}, {"a": [box()]})
    assert result["dev_AP101_supported_classes"] == 0.5


def test_duplicate_does_not_match_twice():
    result = evaluate_detections({"a": [box(), box(score=0.8)]}, {"a": [box()]})
    match = result["per_class"]["car"]["thresholds"]["2.0"]
    assert (match["TP"], match["FP"], match["FN"]) == (1, 1, 0)


def test_half_recall_precision_envelope():
    result = evaluate_detections({"a": [box()], "b": []}, {"a": [box()], "b": [box()]})
    assert result["dev_AP101_supported_classes"] == pytest.approx(51 / 101)


def test_threshold_is_strict_and_classes_do_not_cross_match():
    result = evaluate_detections({"a": [box(0.5), box(category="truck")]}, {"a": [box()]})
    assert result["per_class"]["car"]["thresholds"]["0.5"]["TP"] == 0
    assert result["per_class"]["car"]["thresholds"]["1.0"]["TP"] == 1
    assert result["per_class"]["truck"]["thresholds"]["2.0"]["FP"] == 1


def test_errors_have_known_values_and_wrapped_yaw():
    result = evaluate_detections(
        {"a": [box(0.3, 0.4, size=(8, 2, 1), yaw=math.pi - 0.1)]},
        {"a": [box(yaw=-math.pi + 0.1)]},
    )
    assert result["dev_ATE_m"] == pytest.approx(0.5)
    assert result["dev_ASE"] == pytest.approx(0.5)
    assert result["dev_AOE_rad"] == pytest.approx(0.2)


def test_matching_and_translation_error_ignore_height_coordinate():
    prediction = box()
    prediction["translation_m"][2] = 100.0
    result = evaluate_detections({"a": [prediction]}, {"a": [box()]})
    assert result["dev_AP101_supported_classes"] == 1
    assert result["dev_ATE_m"] == 0


def test_tp_errors_average_classes_equally_instead_of_pooling_objects():
    gt = {"a": [box(), box(10), box(20), box(category="truck")]}
    pred = {"a": [box(1), box(11), box(21), box(category="truck")]}
    result = evaluate_detections(pred, gt)
    assert result["per_class"]["car"]["ATE_m"] == 1
    assert result["per_class"]["truck"]["ATE_m"] == 0
    assert result["dev_ATE_m"] == 0.5


def test_empty_predictions_and_empty_ground_truth_frame():
    result = evaluate_detections({"a": [], "b": []}, {"a": [box()], "b": []})
    assert result["dev_AP101_supported_classes"] == 0
    assert result["dev_ATE_m"] is None
    result = evaluate_detections({"a": [box()]}, {"a": []})
    assert result["dev_AP101_supported_classes"] is None
    assert result["per_class"]["car"]["thresholds"]["2.0"]["FP"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_name", "SUV"),
        ("translation_m", [0, math.nan, 0]),
        ("size_lwh_m", [1, 0, 1]),
        ("score", 1.1),
        ("score", -0.1),
        ("score", True),
        ("score", math.nan),
        ("yaw_rad", math.inf),
    ],
)
def test_invalid_boxes_fail_instead_of_being_silently_dropped(field, value):
    prediction = box()
    prediction[field] = value
    with pytest.raises(ValueError):
        evaluate_detections({"a": [prediction]}, {"a": [box()]})


def test_missing_or_extra_tokens_fail():
    with pytest.raises(ValueError, match="tokens"):
        evaluate_detections({}, {"a": []})
    with pytest.raises(ValueError, match="tokens"):
        evaluate_detections({"a": [], "b": []}, {"a": []})


@pytest.mark.parametrize("thresholds", [(0,), (-1,), (1, 1), (), (math.inf,)])
def test_invalid_matching_thresholds_fail(thresholds):
    with pytest.raises(ValueError):
        evaluate_detections({"a": []}, {"a": []}, thresholds=thresholds)


def test_equal_score_ties_independent_of_mapping_iteration_order():
    gt = {"a": [box()], "b": [box()]}
    pred = {"b": [box(20)], "a": [box()]}
    first = evaluate_detections(pred, gt)
    second = evaluate_detections(dict(reversed(list(pred.items()))), gt)
    assert first == second
