"""Known native-velocity errors and matching/exclusion behavior."""

import math

import pytest

from e_jepa_ttc.detection.velocity_diagnostics import evaluate_velocity_labels


def box(x=0, score=0.9, velocity=(0, 0), category="car"):
    row = {
        "class_name": category,
        "translation_m": [x, 0, 0],
        "size_lwh_m": [4, 2, 1],
        "yaw_rad": 0,
        "score": score,
    }
    if velocity is not None:
        row["velocity_mps"] = list(velocity)
    return row


def test_known_vector_and_component_errors():
    result = evaluate_velocity_labels({"a": [box(velocity=(4, 6))]}, {"a": [box(velocity=(1, 2))]})
    assert result["valid_velocity_matches"] == 1
    assert result["component_mae_mps"] == [3, 4]
    assert result["summed_component_mae_mps"] == 7
    assert result["vector_l2_mean_mps"] == 5
    assert result["official_eap_score"] is False


def test_match_uses_confidence_not_the_best_velocity_error():
    result = evaluate_velocity_labels(
        {"a": [box(score=0.9, velocity=(2, 0)), box(score=0.8)]}, {"a": [box()]}
    )
    assert result["valid_velocity_matches"] == 1
    assert result["vector_l2_mean_mps"] == 2


def test_missing_target_velocity_consumes_box_match_without_fabricating_error():
    result = evaluate_velocity_labels({"a": [box()]}, {"a": [box(velocity=None), box(0.1)]})
    assert result["box_matches"] == 1
    assert result["ground_truth_velocity_count"] == 1
    assert result["valid_velocity_matches"] == 0
    assert result["vector_l2_mean_mps"] is None


def test_strict_distance_and_class_constraints():
    result = evaluate_velocity_labels({"a": [box(2), box(category="truck")]}, {"a": [box()]})
    assert result["box_matches"] == 0


@pytest.mark.parametrize("velocity", [None, [math.nan, 0], [0], [True, 0]])
def test_missing_or_invalid_prediction_velocity_fails(velocity):
    with pytest.raises(ValueError, match="velocity_mps"):
        evaluate_velocity_labels({"a": [box(velocity=velocity)]}, {"a": [box()]})


def test_extra_prediction_tokens_fail():
    with pytest.raises(ValueError, match="tokens"):
        evaluate_velocity_labels({"a": [], "b": []}, {"a": []})
