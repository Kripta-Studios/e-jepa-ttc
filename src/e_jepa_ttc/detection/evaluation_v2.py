"""Population-aware, explicitly nonofficial detection evaluation v2."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from e_jepa_ttc.detection.evaluation_ground_truth import (
    EvaluationBox,
    EvaluationGroundTruth,
    domain_reason,
    parse_box,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol


@dataclass(frozen=True)
class _Prediction:
    token: str
    index: int
    box: EvaluationBox
    score: float


def _xy(left: EvaluationBox, right: EvaluationBox) -> float:
    return math.hypot(
        left.translation_m[0] - right.translation_m[0],
        left.translation_m[1] - right.translation_m[1],
    )


def _errors(prediction: EvaluationBox, target: EvaluationBox) -> dict[str, object]:
    return {
        "xy_m": _xy(prediction, target),
        "z_m": abs(prediction.translation_m[2] - target.translation_m[2]),
        "dimensions_lwh_m": [
            abs(left - right)
            for left, right in zip(prediction.size_lwh_m, target.size_lwh_m, strict=True)
        ],
        "yaw_rad": abs((prediction.yaw_rad - target.yaw_rad + math.pi) % (2 * math.pi) - math.pi),
    }


def _ap101(flags: Sequence[int], count: int) -> tuple[float | None, list[dict[str, float | int]]]:
    correct = 0
    curve = []
    for rank, flag in enumerate(flags, 1):
        correct += flag
        curve.append(
            {"rank": rank, "recall": correct / count if count else 0.0, "precision": correct / rank}
        )
    if count == 0:
        return None, curve
    ap = (
        sum(
            max(
                (float(point["precision"]) for point in curve if point["recall"] >= step / 100),
                default=0.0,
            )
            for step in range(101)
        )
        / 101
    )
    return ap, curve


def _match_class(
    predictions: Sequence[_Prediction],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    category: str,
    threshold: float,
) -> tuple[list[int], list[dict[str, object]], list[dict[str, object]], int]:
    assigned: set[tuple[str, int]] = set()
    ignored_assigned: set[tuple[str, int]] = set()
    flags: list[int] = []
    pairs: list[dict[str, object]] = []
    ignored_predictions: list[dict[str, object]] = []
    for prediction in predictions:
        frame = ground_truth[prediction.token]
        eligible = [
            (distance, target.source_index, target)
            for target in frame.eligible
            if target.class_name == category
            if (distance := _xy(prediction.box, target)) < threshold
        ]
        available = [row for row in eligible if (prediction.token, row[1]) not in assigned]
        if available:
            distance, source_index, target = min(available, key=lambda row: (row[0], row[1]))
            assigned.add((prediction.token, source_index))
            flags.append(1)
            pairs.append(
                {
                    "token": prediction.token,
                    "prediction_index": prediction.index,
                    "ground_truth_source_index": source_index,
                    "score": prediction.score,
                    "errors": _errors(prediction.box, target),
                }
            )
            continue
        # A duplicate close to an already assigned eligible object remains a false positive.
        if eligible:
            flags.append(0)
            continue
        ignored = [
            (distance, target.source_index, target)
            for target in frame.ignored
            if target.class_name == category
            if (distance := _xy(prediction.box, target)) < threshold
            and (prediction.token, target.source_index) not in ignored_assigned
        ]
        if ignored:
            distance, source_index, target = min(ignored, key=lambda row: (row[0], row[1]))
            ignored_assigned.add((prediction.token, source_index))
            ignored_predictions.append(
                {
                    "token": prediction.token,
                    "prediction_index": prediction.index,
                    "ignored_source_index": source_index,
                    "ignore_reason": target.ignore_reason,
                    "xy_m": distance,
                }
            )
            continue
        flags.append(0)
    return flags, pairs, ignored_predictions, len(ignored_assigned)


def _class_agnostic_recall(
    predictions: Sequence[_Prediction],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    thresholds: Sequence[float],
) -> dict[str, object]:
    count = sum(len(frame.eligible) for frame in ground_truth.values())
    results = {}
    for threshold in thresholds:
        assigned: set[tuple[str, int]] = set()
        for prediction in predictions:
            candidates = [
                (_xy(prediction.box, target), target.source_index)
                for target in ground_truth[prediction.token].eligible
                if (prediction.token, target.source_index) not in assigned
                and _xy(prediction.box, target) < threshold
            ]
            if candidates:
                _, index = min(candidates)
                assigned.add((prediction.token, index))
        results[str(threshold)] = {
            "TP": len(assigned),
            "FN": count - len(assigned),
            "recall": len(assigned) / count if count else None,
        }
    return {"eligible_ground_truth_count": count, "thresholds": results}


def evaluate_detections_v2(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    """Evaluate a fixed v2 population with symmetric domain filtering and ignored GT."""
    if not ground_truth or set(predictions) != set(ground_truth):
        raise ValueError("Prediction tokens must exactly match a nonempty ground-truth set")
    if any(not isinstance(token, str) or not token for token in ground_truth):
        raise ValueError("Tokens must be nonempty strings")
    if any(frame.protocol_sha256 != protocol.sha256 for frame in ground_truth.values()):
        raise ValueError("Ground truth was built under a different evaluation protocol")
    for frame in ground_truth.values():
        if frame.raw_count != len(frame.eligible) + len(frame.ignored):
            raise ValueError("Ground-truth denominator identity is inconsistent")
        if any(
            box.class_name not in protocol.classes or box.ignore_reason is not None
            for box in frame.eligible
        ):
            raise ValueError("Eligible ground truth has an invalid class or ignore state")
        if any(
            box.class_name not in protocol.classes or not box.ignore_reason for box in frame.ignored
        ):
            raise ValueError("Ignored ground truth has an invalid class or ignore state")
    ranked: list[_Prediction] = []
    filtered: list[dict[str, object]] = []
    capped = 0
    for token in sorted(predictions):
        frame = ground_truth[token]
        parsed = []
        for index, row in enumerate(predictions[token]):
            box, score = parse_box(
                row, source_index=index, classes=protocol.classes, prediction=True
            )
            assert score is not None
            parsed.append(_Prediction(token, index, box, score))
        parsed.sort(key=lambda item: (-item.score, item.index))
        capped += max(0, len(parsed) - protocol.max_predictions_per_frame)
        for item in parsed[: protocol.max_predictions_per_frame]:
            reason = domain_reason(
                item.box,
                intrinsic=np.asarray(frame.intrinsic),
                event_from_ego=np.asarray(frame.event_from_ego),
                image_wh=frame.image_wh,
                protocol=protocol,
            )
            if reason is None:
                ranked.append(item)
            else:
                filtered.append({"token": token, "prediction_index": item.index, "reason": reason})
    ranked.sort(key=lambda item: (-item.score, item.token, item.index))
    raw_count = sum(frame.raw_count for frame in ground_truth.values())
    eligible_count = sum(len(frame.eligible) for frame in ground_truth.values())
    ignored_count = sum(len(frame.ignored) for frame in ground_truth.values())
    reasons = Counter(
        reason
        for frame in ground_truth.values()
        for reason, count in frame.ignored_by_reason
        for _ in range(count)
    )
    per_class: dict[str, object] = {}
    supported_aps: list[float] = []
    all_pairs = []
    for category in protocol.classes:
        count = sum(
            target.class_name == category
            for frame in ground_truth.values()
            for target in frame.eligible
        )
        selected = [item for item in ranked if item.box.class_name == category]
        thresholds = {}
        class_aps = []
        diagnostic_pairs: list[dict[str, object]] = []
        for threshold in protocol.distance_thresholds_m:
            flags, pairs, ignored_predictions, _ = _match_class(
                selected, ground_truth, category=category, threshold=threshold
            )
            ap, curve = _ap101(flags, count)
            if ap is not None:
                class_aps.append(ap)
            thresholds[str(threshold)] = {
                "AP101": ap,
                "TP": sum(flags),
                "FP": len(flags) - sum(flags),
                "FN": count - sum(flags),
                "ignored_prediction_count": len(ignored_predictions),
                "ignored_predictions": ignored_predictions,
                "prediction_identity_check": len(selected) == len(flags) + len(ignored_predictions),
                "pr_curve": curve,
            }
            if threshold == protocol.diagnostic_threshold_m:
                diagnostic_pairs = pairs
        mean_ap = sum(class_aps) / len(class_aps) if class_aps else None
        if mean_ap is not None:
            supported_aps.append(mean_ap)
        all_pairs.extend({"class_name": category, **pair} for pair in diagnostic_pairs)
        class_raw_count = sum(
            target.class_name == category
            for frame in ground_truth.values()
            for target in (*frame.eligible, *frame.ignored)
        )
        class_ignored_count = sum(
            target.class_name == category
            for frame in ground_truth.values()
            for target in frame.ignored
        )
        per_class[category] = {
            "raw_ground_truth_count": class_raw_count,
            "eligible_ground_truth_count": count,
            "ignored_ground_truth_count": class_ignored_count,
            "ground_truth_identity_check": class_raw_count == count + class_ignored_count,
            "prediction_count_after_domain_filter": len(selected),
            "AP101_mean_thresholds": mean_ap,
            "thresholds": thresholds,
            "diagnostic_matches": diagnostic_pairs,
        }
    return {
        "schema": "phase_ttc_detection_evaluation_result_v2",
        "profile": protocol.profile,
        "protocol_sha256": protocol.sha256,
        "official_eap_score": False,
        "sample_count": len(ground_truth),
        "population": {
            "raw_ground_truth_count": raw_count,
            "eligible_ground_truth_count": eligible_count,
            "ignored_ground_truth_count": ignored_count,
            "ignored_by_reason": dict(sorted(reasons.items())),
            "identity_check": raw_count == eligible_count + ignored_count,
            "ground_truth_sha256_by_token": {
                token: ground_truth[token].sha256 for token in sorted(ground_truth)
            },
        },
        "predictions": {
            "input_count": sum(len(rows) for rows in predictions.values()),
            "capped_before_domain_filter": capped,
            "outside_domain_count": len(filtered),
            "outside_domain": filtered,
            "scored_count": len(ranked),
            "identity_check": sum(len(rows) for rows in predictions.values())
            == capped + len(filtered) + len(ranked),
        },
        "class_agnostic_recall": _class_agnostic_recall(
            ranked, ground_truth, protocol.distance_thresholds_m
        ),
        "supported_class_count": len(supported_aps),
        "dev_AP101_supported_classes": (
            sum(supported_aps) / len(supported_aps) if supported_aps else None
        ),
        "dev_AP101_all_eight_absent_zero": sum(supported_aps) / len(protocol.classes),
        "per_class": per_class,
        "diagnostic_matches": all_pairs,
    }
