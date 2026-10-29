"""Controls and sequence-bootstrap tests for the fixed detection v2 population."""

import copy
import math

import numpy as np
import pytest

from e_jepa_ttc.detection.evaluation_controls import (
    constant_predictions,
    derange_predictions,
    evaluate_controls,
    fit_train_constant_prior,
    paired_sequence_bootstrap,
)
from e_jepa_ttc.detection.evaluation_ground_truth import build_evaluation_ground_truth
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
EVENT_FROM_EGO = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 0, 1]]
)


def label(category="car", x=10.0, yaw=0.0):
    return {
        "class_name": category,
        "translation_m": [x, 0.0, 0.0],
        "size_lwh_m": [4.0, 2.0, 1.5],
        "yaw_rad": yaw,
    }


def prediction(category="car", x=10.0, score=0.9):
    return {**label(category, x), "score": score}


def gt(*rows):
    annotations = [
        {
            "category": row["class_name"],
            "bbox_3d_ego": [
                *row["translation_m"],
                row["size_lwh_m"][0],
                row["size_lwh_m"][2],
                row["size_lwh_m"][1],
                row["yaw_rad"],
            ],
        }
        for row in rows
    ]
    return build_evaluation_ground_truth(
        annotations,
        intrinsic=K,
        event_from_ego=EVENT_FROM_EGO,
        image_wh=(100, 100),
        protocol=EvaluationProtocol.default(),
    )


def test_train_prior_uses_only_train_and_circular_yaw():
    protocol = EvaluationProtocol.default()
    train = {
        "train-a": [label(x=8, yaw=math.pi - 0.1)],
        "train-b": [label(x=12, yaw=-math.pi + 0.1)],
    }
    original = copy.deepcopy(train)
    prior = fit_train_constant_prior(train, protocol=protocol)
    assert train == original
    assert prior[0]["translation_m"] == [10.0, 0.0, 0.0]
    assert abs(abs(prior[0]["yaw_rad"]) - math.pi) < 1e-12
    assert prior[0]["score"] == 1.0
    assert constant_predictions(["selection"], prior)["selection"] == prior


def test_derangement_is_deterministic_bijective_and_preserves_lists():
    predictions = {
        "a": [prediction(x=8)],
        "b": [prediction(x=10), prediction(x=11)],
        "c": [prediction(x=12)],
    }
    first, assignment, attempts = derange_predictions(predictions, seed=7)
    second, second_assignment, second_attempts = derange_predictions(predictions, seed=7)
    assert (assignment, attempts) == (second_assignment, second_attempts)
    assert first == second
    assert all(target != source for target, source in assignment.items())
    assert set(assignment) == set(assignment.values())
    assert sorted(map(len, first.values())) == sorted(map(len, predictions.values()))


def test_bootstrap_keeps_sequences_paired_and_preserves_unsupported_as_null():
    protocol = EvaluationProtocol.default()
    truth = {
        "a1": gt(label(), label("bicycle")),
        "a2": gt(label()),
        "b1": gt(label()),
        "b2": gt(label()),
    }
    predictions = {token: [prediction()] for token in truth}
    control = {token: [prediction(x=20)] for token in truth}
    sequences = {"a1": "seq-a", "a2": "seq-a", "b1": "seq-b", "b2": "seq-b"}
    result = paired_sequence_bootstrap(
        {"observed": predictions, "control": control},
        truth,
        sequences,
        protocol=protocol,
        num_replicates=50,
        seed=7,
    )
    bicycle = result["arms"]["observed"]["per_class"]["bicycle"]
    assert bicycle["unsupported_replicates"] > 0
    assert None in bicycle["values"]
    # Supported replicas legitimately score zero because no bicycle is predicted;
    # unsupported replicas remain null rather than being silently converted to zero.
    assert set(bicycle["values"]) == {None, 0.0}
    assert result["observed_minus_control"]["control"]["median"] > 0
    assert result["cached_full_population"]["observed"]["supported_class_macro"] == 0.5
    assert result == paired_sequence_bootstrap(
        {"observed": predictions, "control": control},
        truth,
        sequences,
        protocol=protocol,
        num_replicates=50,
        seed=7,
    )


def test_evaluate_controls_uses_same_population_and_reports_weak_cluster_limit():
    protocol = EvaluationProtocol.default()
    truth = {"a": gt(label()), "b": gt(label())}
    result = evaluate_controls(
        {"train-a": [label(x=10)], "train-b": [label(x=12)]},
        {"a": [prediction()], "b": [prediction()]},
        truth,
        {"a": "sequence-a", "b": "sequence-b"},
        protocol=protocol,
        num_replicates=8,
    )
    assert result["protocol_sha256"] == protocol.sha256
    assert result["proposal_budget_per_frame"] == 50
    assert (
        result["observed_model_metrics"]["population"]
        == result["constant_prior"]["metrics"]["population"]
    )
    assert result["frame_derangement"]["fixed_points"] == 0
    assert "Only 2 sequence clusters" in result["paired_sequence_bootstrap"]["limitations"][0]


def test_controls_reject_leakage_and_mismatched_tokens():
    protocol = EvaluationProtocol.default()
    truth = {"shared": gt(label()), "b": gt(label())}
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_controls(
            {"shared": [label()]},
            {"shared": [prediction()], "b": [prediction()]},
            truth,
            {"shared": "s1", "b": "s2"},
            protocol=protocol,
            num_replicates=2,
        )
    with pytest.raises(ValueError, match="exact tokens"):
        paired_sequence_bootstrap(
            {"observed": {"shared": [prediction()]}},
            truth,
            {"shared": "s1", "b": "s2"},
            protocol=protocol,
            num_replicates=2,
        )
