"""Compare velocity baselines on exactly the original confidence/XY matches."""

from __future__ import annotations

import copy

import pytest

from e_jepa_ttc.detection.development_metrics import evaluate_detections
from e_jepa_ttc.detection.velocity_diagnostics import evaluate_velocity_labels
from scripts.check_detection_velocity_signal import analyze_velocity_controls, fit_train_mean


def box(x=0.0, velocity=(0.0, 0.0), score=0.9, category="car"):
    return {
        "class_name": category,
        "translation_m": [x, 0.0, 0.0],
        "size_lwh_m": [4.0, 2.0, 1.0],
        "yaw_rad": 0.0,
        "score": score,
        "velocity_mps": list(velocity) if velocity is not None else None,
    }


def results(predictions, truth):
    metrics = evaluate_detections(predictions, truth)
    metrics["native_velocity"] = evaluate_velocity_labels(predictions, truth)
    return {"predictions": predictions, "ground_truth": truth, "metrics": metrics}


def test_known_head_zero_and_train_mean_errors():
    train = results({"t": []}, {"t": [box(velocity=(2, 4)), box(velocity=(4, 8))]})
    selection = results({"s": [box(velocity=(1, 2))]}, {"s": [box(velocity=(0, 0))]})
    report = analyze_velocity_controls(train, selection)
    assert report["train_mean_velocity_components_mps"] == [3.0, 6.0]
    assert report["train_valid_velocity_objects"] == 2
    assert report["observed_head"]["component_mae_mps"] == [1.0, 2.0]
    assert report["zero_velocity"]["component_mae_mps"] == [0.0, 0.0]
    assert report["train_mean_velocity"]["component_mae_mps"] == [3.0, 6.0]
    assert report["head_minus_zero_vector_l2_mps"] == pytest.approx(5**0.5)
    assert report["supported_matched_classes"] == ["car"]
    assert report["same_box_and_velocity_matching_counts"] is True


def test_selection_never_changes_fitted_mean():
    train = results({"t": []}, {"t": [box(velocity=(2, 4))]})
    a = results({"s": [box()]}, {"s": [box(velocity=(1, 1))]})
    b = results({"s": [box()]}, {"s": [box(velocity=(1000, 1000))]})
    first, second = analyze_velocity_controls(train, a), analyze_velocity_controls(train, b)
    assert (
        first["train_mean_velocity_components_mps"] == second["train_mean_velocity_components_mps"]
    )
    assert (
        first["observed_head"]["vector_l2_mean_mps"]
        != second["observed_head"]["vector_l2_mean_mps"]
    )


def test_missing_gt_velocity_consumes_match_without_reassignment():
    train = results({"t": []}, {"t": [box(velocity=(2, 4))]})
    selection = results(
        {"s": [box(0, velocity=(3, 4), score=1), box(0.1, score=0.5)]},
        {"s": [box(0, velocity=None), box(1, velocity=(0, 0))]},
    )
    report = analyze_velocity_controls(train, selection)
    for name in ("observed_head", "zero_velocity", "train_mean_velocity"):
        assert report[name]["box_matches"] == 2
        assert report[name]["valid_velocity_matches"] == 1
    assert report["observed_head"]["vector_l2_mean_mps"] == 0


def test_missing_predicted_velocity_is_not_repaired_by_baseline():
    train = results({"t": []}, {"t": [box()]})
    selection = results({"s": [box(100)]}, {"s": [box()]})
    selection["predictions"]["s"][0].pop("velocity_mps")
    with pytest.raises(ValueError, match="velocity_mps"):
        analyze_velocity_controls(train, selection)


def test_empty_valid_training_velocity_cannot_silently_become_zero_prior():
    with pytest.raises(ValueError, match="valid training velocities"):
        fit_train_mean({"t": [box(velocity=None)]})


def test_no_valid_matches_produce_null_errors_and_leave_inputs_unchanged():
    train = results({"t": []}, {"t": [box()]})
    selection = results({"s": [box(2, velocity=(3, 4))]}, {"s": [box()]})
    snapshot = copy.deepcopy(selection)
    report = analyze_velocity_controls(train, selection)
    assert report["observed_head"]["box_matches"] == 0
    assert report["observed_head"]["component_mae_mps"] is None
    assert report["head_minus_zero_vector_l2_mps"] is None
    assert report["supported_matched_classes"] == []
    assert selection == snapshot


def test_native_velocity_metric_mismatch_fails():
    train = results({"t": []}, {"t": [box()]})
    selection = results({"s": [box()]}, {"s": [box()]})
    selection["metrics"]["native_velocity"]["valid_velocity_matches"] = 2
    with pytest.raises(ValueError, match="Stored native_velocity"):
        analyze_velocity_controls(train, selection)


def test_native_velocity_float_roundoff_is_tolerated_but_material_change_is_not():
    train = results({"t": []}, {"t": [box()]})
    selection = results({"s": [box(velocity=(1, 1))]}, {"s": [box()]})
    selection["metrics"]["native_velocity"]["component_mae_mps"][0] += 1e-15
    analyze_velocity_controls(train, selection)
    selection["metrics"]["native_velocity"]["component_mae_mps"][0] += 1e-5
    with pytest.raises(ValueError, match="Stored native_velocity"):
        analyze_velocity_controls(train, selection)
