"""Explicit development metrics; this is not the private eAP benchmark scorer.

Distance matching uses ego XY and strict thresholds. AP uses a 101-point
precision envelope without nuScenes minimum precision/recall clipping. True
positive errors average the actual matched boxes at 2 m, not recall bins.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

from e_jepa_ttc.detection.evaluation_ap_fast import ap101_value

CLASSES = (
    "car",
    "bicycle",
    "truck",
    "pedestrian",
    "motorcycle",
    "bus",
    "tricycle",
    "other_vehicle",
)
PROFILE = "eap_development_xy_ap101_envelope_v1"


@dataclass(frozen=True)
class Detection:
    """A box expressed in one common ego frame and dimension ordering."""

    category: str
    center: tuple[float, ...]
    size: tuple[float, ...]
    yaw: float
    score: float


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must contain three numbers")
    result = tuple(_finite(v, name) for v in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain three numbers")
    return result


def _box(row: Mapping[str, object], *, prediction: bool) -> Detection:
    category = row.get("class_name")
    if not isinstance(category, str) or category not in CLASSES:
        raise ValueError(f"Unknown class_name: {category!r}")
    center = _vector(row.get("translation_m"), "translation_m")
    size = _vector(row.get("size_lwh_m"), "size_lwh_m")
    if any(v <= 0 for v in size):
        raise ValueError("size_lwh_m must be strictly positive")
    yaw = _finite(row.get("yaw_rad"), "yaw_rad")
    score = _finite(row.get("score"), "score") if prediction else 1.0
    if not 0 <= score <= 1:
        raise ValueError("score must be in [0, 1]")
    return Detection(category, center, size, yaw, score)


def _xy_distance(left: Detection, right: Detection) -> float:
    return math.hypot(left.center[0] - right.center[0], left.center[1] - right.center[1])


def _tp_errors(prediction: Detection, target: Detection) -> tuple[float, float, float]:
    intersection = math.prod(min(a, b) for a, b in zip(prediction.size, target.size, strict=True))
    union = math.prod(prediction.size) + math.prod(target.size) - intersection
    return (
        _xy_distance(prediction, target),
        1 - intersection / union,
        abs((prediction.yaw - target.yaw + math.pi) % (2 * math.pi) - math.pi),
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _match(
    predictions: Sequence[tuple[str, int, Detection]],
    ground_truth: Mapping[str, Sequence[Detection]],
    *,
    category: str,
    threshold: float,
) -> tuple[list[int], list[tuple[float, float, float]]]:
    assigned: set[tuple[str, int]] = set()
    true_positives = []
    errors = []
    for token, _, prediction in predictions:
        candidates = [
            (_xy_distance(prediction, target), index, target)
            for index, target in enumerate(ground_truth[token])
            if target.category == category and (token, index) not in assigned
        ]
        nearest = min(candidates, key=lambda item: (item[0], item[1]), default=None)
        matched = nearest is not None and nearest[0] < threshold
        true_positives.append(int(matched))
        if matched and nearest is not None:
            assigned.add((token, nearest[1]))
            errors.append(_tp_errors(prediction, nearest[2]))
    return true_positives, errors


def _ap101(true_positives: Sequence[int], count: int) -> float | None:
    return ap101_value(true_positives, count)


def evaluate_detections(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    tp_threshold: float = 2.0,
) -> dict[str, object]:
    """Evaluate all supplied samples, refusing missing or extra prediction tokens.

    Empty prediction lists are valid. Absent GT classes have AP ``None`` and
    are excluded from the supported-class macro; a second macro assigns them
    zero explicitly. Confidence scores must be in [0,1]; calibration is not
    assumed or evaluated. Velocity is not scored.
    """
    if not ground_truth:
        raise ValueError("At least one ground-truth sample is required")
    if set(predictions) != set(ground_truth):
        raise ValueError("Prediction tokens must exactly match the ground-truth sample set")
    if any(not isinstance(t, str) or not t for t in ground_truth):
        raise ValueError("Tokens must be nonempty strings")
    distances = tuple(_finite(t, "threshold") for t in thresholds)
    tp_distance = _finite(tp_threshold, "tp_threshold")
    if not distances or len(set(distances)) != len(distances) or min(distances) <= 0:
        raise ValueError("Matching thresholds must be distinct and positive")
    if tp_distance <= 0:
        raise ValueError("TP threshold must be positive")
    gt = {t: [_box(r, prediction=False) for r in rows] for t, rows in ground_truth.items()}
    pred = {t: [_box(r, prediction=True) for r in rows] for t, rows in predictions.items()}
    classes = {}
    aps: list[float] = []
    macro_errors: list[list[float]] = [[], [], []]
    for category in CLASSES:
        count = sum(box.category == category for rows in gt.values() for box in rows)
        ranked = sorted(
            (
                (t, i, box)
                for t, rows in pred.items()
                for i, box in enumerate(rows)
                if box.category == category
            ),
            key=lambda item: (-item[2].score, item[0], item[1]),
        )
        threshold_results = {}
        class_aps = []
        for distance in distances:
            flags, _ = _match(ranked, gt, category=category, threshold=distance)
            ap = _ap101(flags, count)
            if ap is not None:
                class_aps.append(ap)
            threshold_results[str(distance)] = {
                "AP101": ap,
                "TP": sum(flags),
                "FP": len(flags) - sum(flags),
                "FN": count - sum(flags),
            }
        flags, errors = _match(ranked, gt, category=category, threshold=tp_distance)
        class_errors = [_mean([row[i] for row in errors]) for i in range(3)]
        for aggregate, value in zip(macro_errors, class_errors, strict=True):
            if value is not None:
                aggregate.append(value)
        ap = _mean(class_aps)
        if ap is not None:
            aps.append(ap)
        classes[category] = {
            "ground_truth_count": count,
            "prediction_count": len(ranked),
            "AP101_mean_thresholds": ap,
            "thresholds": threshold_results,
            "TP_at_error_threshold": sum(flags),
            "ATE_m": class_errors[0],
            "ASE": class_errors[1],
            "AOE_rad": class_errors[2],
        }
    return {
        "profile": PROFILE,
        "official_eap_score": False,
        "sample_count": len(gt),
        "matching": "greedy confidence-ranked, class-constrained ego XY, distance < threshold",
        "score_ties": "token lexicographic, then original prediction index",
        "ap": "101 recall points with precision envelope; no minimum recall/precision clipping",
        "tp_errors": "mean of actual matched boxes per class; macro over classes with matches",
        "velocity_evaluated": False,
        "tp_threshold_m": tp_distance,
        "supported_class_count": len(aps),
        "dev_AP101_supported_classes": _mean(aps),
        "dev_AP101_all_eight_absent_zero": sum(aps) / len(CLASSES),
        "dev_ATE_m": _mean(macro_errors[0]),
        "dev_ASE": _mean(macro_errors[1]),
        "dev_AOE_rad": _mean(macro_errors[2]),
        "per_class": classes,
    }
