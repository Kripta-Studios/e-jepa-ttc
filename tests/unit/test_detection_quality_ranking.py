from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from dataclasses import replace

import pytest

from e_jepa_ttc.detection.evaluation_ground_truth import (
    EvaluationBox,
    EvaluationGroundTruth,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2
from e_jepa_ttc.detection.quality_ranking import (
    FEATURE_NAMES,
    RankingRecipe,
    TrainingProvenance,
    apply_quality_ranker,
    apply_score_control,
    build_training_rows,
    fit_quality_ranker,
    monotonic_temperature_score,
    proposal_geometry_sha256,
    score_independent_pool_recall,
    token_population_sha256,
    uncertainty_control_score,
    verify_train_token_ownership,
)


def _logit(probability: float) -> float:
    return math.log(probability / (1 - probability))


def _prediction(x: float, score: float, *, z: float = 10.0) -> dict[str, object]:
    return {
        "class_name": "car",
        "translation_m": [x, 0.0, z],
        "size_lwh_m": [4.0, 1.8, 1.6],
        "yaw_rad": 0.1,
        "velocity_mps": [1.0, -2.0],
        "score": score,
    }


def _box(x: float, source: int, *, ignored: bool = False) -> EvaluationBox:
    return EvaluationBox(
        class_name="car",
        translation_m=(x, 0.0, 10.0),
        size_lwh_m=(4.0, 1.8, 1.6),
        yaw_rad=0.1,
        source_index=source,
        ignore_reason="test_ignore" if ignored else None,
    )


def _ground_truth(
    *, eligible_x: float = 0.0, ignored_x: float = 4.5
) -> dict[str, EvaluationGroundTruth]:
    protocol = EvaluationProtocol.default()
    return {
        "seq:frame": EvaluationGroundTruth(
            protocol_sha256=protocol.sha256,
            eligible=(_box(eligible_x, 10),),
            ignored=(_box(ignored_x, 20, ignored=True),),
            raw_count=2,
            ignored_by_reason=(("test_ignore", 1),),
            intrinsic=((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
            event_from_ego=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            image_wh=(100, 100),
        )
    }


def _inputs() -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    predictions = {
        "seq:frame": [
            _prediction(0.0, 0.9),
            _prediction(0.1, 0.8),
            _prediction(4.5, 0.7),
            _prediction(0.0, 0.6, z=100.0),
        ]
    }
    logits = []
    depths = []
    for index, row in enumerate(predictions["seq:frame"]):
        values = [-8.0] * 8
        score = row["score"]
        assert isinstance(score, float)
        values[0] = _logit(score)
        logits.append(values)
        depths.append([10.0 + index, -0.5 + index * 0.2])
    sidecar: dict[str, object] = {
        "schema": "monodgp_external_diagnostic_sidecar_v1",
        "checkpoint_sha256": "a" * 64,
        "frames": [
            {
                "sample_token": "seq:frame",
                "tensors": {
                    "pred_logits": {"values": logits},
                    "pred_depth": {"values": depths},
                },
            }
        ],
    }
    return predictions, sidecar


def _provenance(role: str = "train") -> TrainingProvenance:
    return TrainingProvenance(
        dataset_role=role,
        split_manifest_sha256="1" * 64,
        panel_recipe_sha256="6" * 64,
        ground_truth_receipt_sha256="7" * 64,
        predictions_sha256="2" * 64,
        sidecar_sha256="3" * 64,
        ground_truth_sha256="4" * 64,
        checkpoint_sha256="a" * 64,
        protocol_sha256=EvaluationProtocol.default().sha256,
        token_population_sha256=token_population_sha256(["seq:frame"]),
        train_ownership_sha256="5" * 64,
    )


def test_feature_taxonomy_matches_protocol_order_exactly() -> None:
    expected = tuple(f"class_logit_{name}" for name in EvaluationProtocol.default().classes)
    assert FEATURE_NAMES[:8] == expected


def test_score_independent_matching_masks_ignored_and_domain() -> None:
    predictions, sidecar = _inputs()
    rows = build_training_rows(
        predictions,
        sidecar,
        _ground_truth(),
        protocol=EvaluationProtocol.default(),
    )
    assert [(row.target_quality, row.target_kind) for row in rows] == [
        (1.0, "assigned_positive"),
        (0.0, "duplicate"),
        (None, "masked"),
        (None, "masked"),
    ]
    assert rows[2].mask_reason == "ignored_ground_truth:test_ignore"
    assert rows[3].mask_reason == (
        "prediction_domain:camera_center_depth_outside_protocol"
    )
    changed_scores = copy.deepcopy(predictions)
    for index, row in enumerate(changed_scores["seq:frame"]):
        row["score"] = predictions["seq:frame"][index]["score"]
    repeated = build_training_rows(
        changed_scores, sidecar, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    assert [row.target_quality for row in repeated] == [row.target_quality for row in rows]

    overlapping_ignore = build_training_rows(
        predictions,
        sidecar,
        _ground_truth(ignored_x=0.15),
        protocol=EvaluationProtocol.default(),
    )
    assert overlapping_ignore[1].target_kind == "duplicate"
    assert overlapping_ignore[1].target_quality == 0.0


def test_ignored_only_class_is_masked_instead_of_taught_as_background() -> None:
    predictions, sidecar = _inputs()
    original = _ground_truth()["seq:frame"]
    ignored_only = {
        "seq:frame": replace(original, eligible=(), raw_count=len(original.ignored))
    }
    rows = build_training_rows(
        predictions,
        sidecar,
        ignored_only,
        protocol=EvaluationProtocol.default(),
    )
    assert rows[2].target_quality is None
    assert rows[2].mask_reason == "ignored_ground_truth:test_ignore"


def test_train_ownership_comes_from_split_and_canonical_token_metadata() -> None:
    split = {
        "assignments": {
            "train": ["train-sequence"],
            "selection": ["selection-sequence"],
            "calibration": ["calibration-sequence"],
        }
    }
    with pytest.raises(ValueError, match="not owned"):
        verify_train_token_ownership(
            ["selection-sequence:frame"],
            split,
            {"selection-sequence:frame": "selection-sequence"},
        )
    digest = verify_train_token_ownership(
        ["train-sequence:frame"],
        split,
        {"train-sequence:frame": "train-sequence"},
    )
    assert len(digest) == 64


def test_fit_is_train_only_and_reports_absent_class_and_masks() -> None:
    predictions, sidecar = _inputs()
    rows = build_training_rows(
        predictions, sidecar, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    with pytest.raises(ValueError, match="train role"):
        fit_quality_ranker(rows, provenance=_provenance("selection"))
    bundle = fit_quality_ranker(
        rows,
        provenance=_provenance(),
        recipe=RankingRecipe(epochs=3),
    )
    training = bundle["training"]
    assert isinstance(training, dict)
    assert training["fitted_row_count"] == 2
    assert training["masked_row_count"] == 2
    support = training["positive_target_support_by_class"]
    assert isinstance(support, dict)
    assert support["car"] == 1
    assert support["bicycle"] == 0


def test_apply_preserves_fixed_pool_and_never_accepts_ground_truth() -> None:
    predictions, sidecar = _inputs()
    rows = build_training_rows(
        predictions, sidecar, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    bundle = fit_quality_ranker(
        rows, provenance=_provenance(), recipe=RankingRecipe(epochs=3)
    )
    output = apply_quality_ranker(
        predictions, sidecar, bundle, protocol=EvaluationProtocol.default()
    )
    assert "ground_truth" not in inspect.signature(apply_quality_ranker).parameters
    assert list(output) == list(predictions)
    assert len(output["seq:frame"]) == len(predictions["seq:frame"])
    immutable = {
        "class_name",
        "translation_m",
        "size_lwh_m",
        "yaw_rad",
        "velocity_mps",
    }
    for original, ranked in zip(
        predictions["seq:frame"], output["seq:frame"], strict=True
    ):
        assert all(ranked[name] == original[name] for name in immutable)
        assert ranked["classification_score"] == original["score"]
        assert ranked["original_detection_score"] == original["score"]
        assert ranked["score"] == ranked["detection_score"]

    perturbed_rows = build_training_rows(
        predictions,
        sidecar,
        _ground_truth(eligible_x=1.5),
        protocol=EvaluationProtocol.default(),
    )
    assert [row.target_quality for row in perturbed_rows] != [
        row.target_quality for row in rows
    ]
    assert apply_quality_ranker(
        predictions, sidecar, bundle, protocol=EvaluationProtocol.default()
    ) == output


def test_controls_and_missing_fields() -> None:
    values = [0.1, 0.2, 0.8, 0.9]
    transformed = [monotonic_temperature_score(value) for value in values]
    assert transformed == sorted(transformed)
    assert sorted(range(len(values)), key=values.__getitem__) == sorted(
        range(len(values)), key=transformed.__getitem__
    )
    assert 0 < uncertainty_control_score(0.8, 1.0) < 1
    predictions, sidecar = _inputs()
    baseline = apply_score_control(
        predictions,
        sidecar,
        protocol=EvaluationProtocol.default(),
        control="baseline",
    )
    temperature = apply_score_control(
        predictions,
        sidecar,
        protocol=EvaluationProtocol.default(),
        control="monotonic_temperature",
    )
    baseline_metrics = evaluate_detections_v2(
        baseline, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    temperature_metrics = evaluate_detections_v2(
        temperature, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    assert baseline_metrics["dev_AP101_supported_classes"] == temperature_metrics[
        "dev_AP101_supported_classes"
    ]
    broken = copy.deepcopy(sidecar)
    frame = broken["frames"][0]  # type: ignore[index]
    del frame["tensors"]["pred_depth"]  # type: ignore[index]
    with pytest.raises(ValueError, match="pred_logits and pred_depth"):
        build_training_rows(
            predictions, broken, _ground_truth(), protocol=EvaluationProtocol.default()
        )


def test_query_binding_supports_nonprefix_and_rejects_ambiguous_ties() -> None:
    predictions, sidecar = _inputs()
    nonprefix = copy.deepcopy(sidecar)
    tensors = nonprefix["frames"][0]["tensors"]  # type: ignore[index]
    extra_logits = [-8.0] * 8
    extra_logits[1] = _logit(0.85)
    tensors["pred_logits"]["values"].insert(1, extra_logits)  # type: ignore[index]
    tensors["pred_depth"]["values"].insert(1, [20.0, 0.0])  # type: ignore[index]
    rows = build_training_rows(
        predictions, nonprefix, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    assert len(rows) == len(predictions["seq:frame"])

    ambiguous = copy.deepcopy(sidecar)
    tied = ambiguous["frames"][0]["tensors"]["pred_logits"]["values"]  # type: ignore[index]
    tied[1] = list(tied[0])
    with pytest.raises(ValueError, match="map uniquely"):
        build_training_rows(
            predictions, ambiguous, _ground_truth(), protocol=EvaluationProtocol.default()
        )

    duplicate_frame = copy.deepcopy(sidecar)
    duplicate_frame["frames"].append(copy.deepcopy(duplicate_frame["frames"][0]))  # type: ignore[union-attr,index]
    with pytest.raises(ValueError, match="token populations differ"):
        build_training_rows(
            predictions,
            duplicate_frame,
            _ground_truth(),
            protocol=EvaluationProtocol.default(),
        )


def test_tampered_bundle_is_rejected() -> None:
    predictions, sidecar = _inputs()
    rows = build_training_rows(
        predictions, sidecar, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    bundle = fit_quality_ranker(
        rows, provenance=_provenance(), recipe=RankingRecipe(epochs=1)
    )
    bias = bundle["bias"]
    assert isinstance(bias, float)
    bundle["bias"] = bias + 1
    with pytest.raises(ValueError, match="bundle hash"):
        apply_quality_ranker(
            predictions, sidecar, bundle, protocol=EvaluationProtocol.default()
        )


def test_apply_rejects_over_cap_and_revalidates_train_provenance() -> None:
    predictions, sidecar = _inputs()
    rows = build_training_rows(
        predictions, sidecar, _ground_truth(), protocol=EvaluationProtocol.default()
    )
    bundle = fit_quality_ranker(
        rows, provenance=_provenance(), recipe=RankingRecipe(epochs=1)
    )
    oversized = {
        "seq:frame": [dict(predictions["seq:frame"][0]) for _ in range(51)]
    }
    with pytest.raises(ValueError, match="proposal cap"):
        apply_quality_ranker(
            oversized, sidecar, bundle, protocol=EvaluationProtocol.default()
        )
    with pytest.raises(ValueError, match="proposal cap"):
        apply_score_control(
            oversized,
            sidecar,
            protocol=EvaluationProtocol.default(),
            control="baseline",
        )

    changed_protocol = replace(
        EvaluationProtocol.default(), distance_thresholds_m=(1.0, 2.0, 4.0)
    )
    with pytest.raises(ValueError, match="protocol differs"):
        apply_quality_ranker(predictions, sidecar, bundle, protocol=changed_protocol)
    with pytest.raises(ValueError, match="frozen to the default"):
        apply_score_control(
            predictions, sidecar, protocol=changed_protocol, control="baseline"
        )

    invalid_role = copy.deepcopy(bundle)
    invalid_role["provenance"]["dataset_role"] = "selection"  # type: ignore[index]
    invalid_role.pop("bundle_sha256")
    encoded = json.dumps(
        invalid_role, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    invalid_role["bundle_sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="train role"):
        apply_quality_ranker(
            predictions,
            sidecar,
            invalid_role,
            protocol=EvaluationProtocol.default(),
        )


def test_greedy_v2_recall_can_change_but_pool_capacity_and_geometry_do_not() -> None:
    protocol = EvaluationProtocol.default()
    original = _ground_truth()["seq:frame"]
    ground_truth = {
        "seq:frame": replace(
            original,
            eligible=(_box(0.0, 10), _box(1.5, 11)),
            ignored=(),
            raw_count=2,
            ignored_by_reason=(),
        )
    }
    flexible_first = {
        "seq:frame": [_prediction(0.75, 0.9), _prediction(-1.0, 0.8)]
    }
    constrained_first = {
        "seq:frame": [_prediction(0.75, 0.8), _prediction(-1.0, 0.9)]
    }
    first_v2 = evaluate_detections_v2(flexible_first, ground_truth, protocol=protocol)
    second_v2 = evaluate_detections_v2(constrained_first, ground_truth, protocol=protocol)
    first_greedy = first_v2["class_agnostic_recall"]["thresholds"]["2.0"]["TP"]  # type: ignore[index]
    second_greedy = second_v2["class_agnostic_recall"]["thresholds"]["2.0"]["TP"]  # type: ignore[index]
    assert (first_greedy, second_greedy) == (1, 2)

    first_pool = score_independent_pool_recall(
        flexible_first, ground_truth, protocol=protocol
    )
    second_pool = score_independent_pool_recall(
        constrained_first, ground_truth, protocol=protocol
    )
    assert first_pool["thresholds"]["2.0"]["TP_maximum_cardinality"] == 2  # type: ignore[index]
    assert first_pool == second_pool
    assert proposal_geometry_sha256(
        flexible_first, protocol=protocol
    ) == proposal_geometry_sha256(constrained_first, protocol=protocol)
