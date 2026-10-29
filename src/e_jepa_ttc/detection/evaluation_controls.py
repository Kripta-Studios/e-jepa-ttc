"""Train-only and permutation controls with paired sequence-cluster bootstrap."""

from __future__ import annotations

import copy
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from e_jepa_ttc.detection.evaluation_ground_truth import (
    EvaluationBox,
    EvaluationGroundTruth,
    domain_reason,
    parse_box,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2

METHOD = "eap_v2_train_constant_prior_frame_derangement_sequence_bootstrap_v1"
YAW_RESULTANT_EPSILON = 1e-12
Box = dict[str, object]
BoxesByToken = dict[str, list[Box]]


def _finite_vector(row: Mapping[str, object], field: str, length: int) -> tuple[float, ...]:
    value = row.get(field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{field} must contain {length} numbers")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item):
            raise ValueError(f"{field} must contain finite numbers")
        result.append(float(item))
    return tuple(result)


def _training_box(row: Mapping[str, object], classes: tuple[str, ...]) -> Box:
    category = row.get("class_name")
    if not isinstance(category, str) or category not in classes:
        raise ValueError(f"Unknown training class: {category!r}")
    center = _finite_vector(row, "translation_m", 3)
    size = _finite_vector(row, "size_lwh_m", 3)
    if min(size) <= 0:
        raise ValueError("Training dimensions must be positive")
    yaw = row.get("yaw_rad")
    if isinstance(yaw, bool) or not isinstance(yaw, Real) or not math.isfinite(yaw):
        raise ValueError("Training yaw must be finite")
    return {
        "class_name": category,
        "translation_m": list(center),
        "size_lwh_m": list(size),
        "yaw_rad": float(yaw),
    }


def fit_train_constant_prior(
    train_truth: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    protocol: EvaluationProtocol,
) -> list[Box]:
    """Fit one class box using only trainer-admitted train ground truth."""
    if not train_truth or any(not isinstance(token, str) or not token for token in train_truth):
        raise ValueError("Nonempty training tokens are required")
    parsed = {
        token: [_training_box(row, protocol.classes) for row in rows]
        for token, rows in train_truth.items()
    }
    priors: list[Box] = []
    for category in protocol.classes:
        rows = [row for frame in parsed.values() for row in frame if row["class_name"] == category]
        if not rows:
            continue
        mean_sin = statistics.mean(math.sin(float(row["yaw_rad"])) for row in rows)
        mean_cos = statistics.mean(math.cos(float(row["yaw_rad"])) for row in rows)
        resultant = math.hypot(mean_sin, mean_cos)
        frames_present = sum(
            any(row["class_name"] == category for row in frame) for frame in parsed.values()
        )
        priors.append(
            {
                "class_name": category,
                "translation_m": [
                    statistics.median(float(row["translation_m"][index]) for row in rows)
                    for index in range(3)
                ],
                "size_lwh_m": [
                    statistics.median(float(row["size_lwh_m"][index]) for row in rows)
                    for index in range(3)
                ],
                "yaw_rad": (
                    0.0 if resultant <= YAW_RESULTANT_EPSILON else math.atan2(mean_sin, mean_cos)
                ),
                "score": frames_present / len(parsed),
                "train_object_count": len(rows),
                "train_frames_with_class": frames_present,
                "yaw_mean_resultant": resultant,
                "yaw_undefined_fallback_zero": resultant <= YAW_RESULTANT_EPSILON,
            }
        )
    if len(priors) > protocol.max_predictions_per_frame:
        raise ValueError("Train priors exceed the fixed protocol proposal budget")
    return priors


def constant_predictions(
    tokens: Sequence[str], priors: Sequence[Mapping[str, object]]
) -> BoxesByToken:
    """Repeat the train-fitted prior without consulting evaluation labels."""
    if len(set(tokens)) != len(tokens) or any(not token for token in tokens):
        raise ValueError("Evaluation tokens must be unique and nonempty")
    return {token: copy.deepcopy([dict(row) for row in priors]) for token in tokens}


def derange_predictions(
    predictions: Mapping[str, Sequence[Mapping[str, object]]], *, seed: int
) -> tuple[BoxesByToken, dict[str, str], int]:
    """Move complete prediction lists using the first deterministic derangement."""
    tokens = sorted(predictions)
    if len(tokens) < 2:
        raise ValueError("A derangement requires at least two frames")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed)
    for attempt in range(1, 10001):
        sources = list(tokens)
        rng.shuffle(sources)
        if all(target != source for target, source in zip(tokens, sources, strict=True)):
            assignment = dict(zip(tokens, sources, strict=True))
            return (
                {
                    target: copy.deepcopy([dict(row) for row in predictions[source]])
                    for target, source in assignment.items()
                },
                assignment,
                attempt,
            )
    raise RuntimeError("No derangement found in 10,000 draws")


@dataclass(frozen=True)
class _Flag:
    score: float
    token: str
    prediction_index: int
    true_positive: int


SequenceCache = dict[str, dict[str, dict[float, tuple[int, tuple[_Flag, ...]]]]]


def _xy(left: EvaluationBox, right: EvaluationBox) -> float:
    return math.hypot(
        left.translation_m[0] - right.translation_m[0],
        left.translation_m[1] - right.translation_m[1],
    )


def _cache_arm(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    sequence_by_token: Mapping[str, str],
    protocol: EvaluationProtocol,
) -> SequenceCache:
    cache: SequenceCache = {}
    for sequence in sorted(set(sequence_by_token.values())):
        tokens = sorted(token for token, group in sequence_by_token.items() if group == sequence)
        class_cache: dict[str, dict[float, tuple[int, tuple[_Flag, ...]]]] = {}
        for category in protocol.classes:
            count = sum(
                target.class_name == category
                for token in tokens
                for target in ground_truth[token].eligible
            )
            threshold_cache = {}
            for threshold in protocol.distance_thresholds_m:
                flags: list[_Flag] = []
                assigned: set[tuple[str, int]] = set()
                ignored_assigned: set[tuple[str, int]] = set()
                ranked: list[tuple[float, str, int, EvaluationBox]] = []
                for token in tokens:
                    frame = ground_truth[token]
                    parsed = []
                    for index, row in enumerate(predictions[token]):
                        box, score = parse_box(
                            row, source_index=index, classes=protocol.classes, prediction=True
                        )
                        assert score is not None
                        parsed.append((score, index, box))
                    parsed.sort(key=lambda item: (-item[0], item[1]))
                    for score, index, box in parsed[: protocol.max_predictions_per_frame]:
                        if box.class_name != category:
                            continue
                        if (
                            domain_reason(
                                box,
                                intrinsic=np.asarray(frame.intrinsic),
                                event_from_ego=np.asarray(frame.event_from_ego),
                                image_wh=frame.image_wh,
                                protocol=protocol,
                            )
                            is None
                        ):
                            ranked.append((score, token, index, box))
                ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
                for score, token, index, box in ranked:
                    frame = ground_truth[token]
                    eligible = [
                        (_xy(box, target), target.source_index)
                        for target in frame.eligible
                        if target.class_name == category and _xy(box, target) < threshold
                    ]
                    available = [row for row in eligible if (token, row[1]) not in assigned]
                    if available:
                        _, source_index = min(available)
                        assigned.add((token, source_index))
                        flags.append(_Flag(score, token, index, 1))
                    elif eligible:
                        flags.append(_Flag(score, token, index, 0))
                    else:
                        ignored = [
                            (_xy(box, target), target.source_index)
                            for target in frame.ignored
                            if target.class_name == category
                            and _xy(box, target) < threshold
                            and (token, target.source_index) not in ignored_assigned
                        ]
                        if ignored:
                            _, source_index = min(ignored)
                            ignored_assigned.add((token, source_index))
                        else:
                            flags.append(_Flag(score, token, index, 0))
                threshold_cache[threshold] = (count, tuple(flags))
            class_cache[category] = threshold_cache
        cache[sequence] = class_cache
    return cache


def _ap101(flags: Sequence[int], count: int) -> float | None:
    if count == 0:
        return None
    correct = 0
    curve = []
    for rank, flag in enumerate(flags, 1):
        correct += flag
        curve.append((correct / count, correct / rank))
    return (
        sum(
            max((precision for recall, precision in curve if recall >= step / 100), default=0.0)
            for step in range(101)
        )
        / 101
    )


def _bootstrap_metric(
    cache: SequenceCache,
    sampled_sequences: Sequence[str],
    protocol: EvaluationProtocol,
) -> tuple[float | None, dict[str, float | None]]:
    per_class: dict[str, float | None] = {}
    supported = []
    for category in protocol.classes:
        threshold_aps = []
        for threshold in protocol.distance_thresholds_m:
            count = 0
            combined = []
            for copy_index, sequence in enumerate(sampled_sequences):
                sequence_count, flags = cache[sequence][category][threshold]
                count += sequence_count
                combined.extend(
                    (-flag.score, copy_index, flag.token, flag.prediction_index, flag.true_positive)
                    for flag in flags
                )
            combined.sort()
            ap = _ap101([row[-1] for row in combined], count)
            if ap is not None:
                threshold_aps.append(ap)
        class_ap = statistics.mean(threshold_aps) if threshold_aps else None
        per_class[category] = class_ap
        if class_ap is not None:
            supported.append(class_ap)
    return (statistics.mean(supported) if supported else None), per_class


def _interval(values: Sequence[float | None]) -> dict[str, object]:
    finite = [value for value in values if value is not None]
    return {
        "values": list(values),
        "valid_replicates": len(finite),
        "unsupported_replicates": len(values) - len(finite),
        "median": float(np.median(finite)) if finite else None,
        "ci95": (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite
            else None
        ),
    }


def paired_sequence_bootstrap(
    arms: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    sequence_by_token: Mapping[str, str],
    *,
    protocol: EvaluationProtocol,
    num_replicates: int = 200,
    seed: int = 7,
) -> dict[str, object]:
    """Bootstrap paired whole sequences using cached within-frame match outcomes."""
    if (
        isinstance(num_replicates, bool)
        or not isinstance(num_replicates, int)
        or num_replicates < 1
    ):
        raise ValueError("num_replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if "observed" not in arms:
        raise ValueError("An observed arm is required for paired control differences")
    if set(sequence_by_token) != set(ground_truth) or any(
        set(predictions) != set(ground_truth) for predictions in arms.values()
    ):
        raise ValueError("All arms, GT and sequence assignments must share exact tokens")
    sequences = sorted(set(sequence_by_token.values()))
    if len(sequences) < 2:
        raise ValueError("Sequence bootstrap requires at least two clusters")
    caches = {
        name: _cache_arm(predictions, ground_truth, sequence_by_token, protocol)
        for name, predictions in arms.items()
    }
    rng = random.Random(seed)
    samples = [
        [rng.choice(sequences) for _ in range(len(sequences))] for _ in range(num_replicates)
    ]
    macro_values: dict[str, list[float | None]] = {name: [] for name in arms}
    class_values: dict[str, dict[str, list[float | None]]] = {
        name: {category: [] for category in protocol.classes} for name in arms
    }
    for sample in samples:
        for name, cache in caches.items():
            macro, per_class = _bootstrap_metric(cache, sample, protocol)
            macro_values[name].append(macro)
            for category, value in per_class.items():
                class_values[name][category].append(value)
    observed = "observed"
    deltas = {
        name: [
            (left - right if left is not None and right is not None else None)
            for left, right in zip(macro_values[observed], values, strict=True)
        ]
        for name, values in macro_values.items()
        if name != observed
    }
    return {
        "unit": "sequence cluster",
        "sequence_count": len(sequences),
        "sequences": sequences,
        "num_replicates": num_replicates,
        "seed": seed,
        "rng": "Python random.Random.choice with replacement",
        "paired_across_arms": True,
        "cached_full_population": {
            name: {
                "supported_class_macro": _bootstrap_metric(cache, sequences, protocol)[0],
                "per_class": _bootstrap_metric(cache, sequences, protocol)[1],
            }
            for name, cache in caches.items()
        },
        "arms": {
            name: {
                "supported_class_macro": _interval(values),
                "per_class": {
                    category: _interval(class_values[name][category])
                    for category in protocol.classes
                },
            }
            for name, values in macro_values.items()
        },
        "observed_minus_control": {name: _interval(values) for name, values in deltas.items()},
        "limitations": [
            f"Only {len(sequences)} sequence clusters are available; intervals are weak",
            "Frames are never treated as independent bootstrap units",
            "Unsupported class replicates remain null and are counted, never replaced by zero",
        ],
    }


def evaluate_controls(
    train_truth: Mapping[str, Sequence[Mapping[str, object]]],
    selection_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    selection_ground_truth: Mapping[str, EvaluationGroundTruth],
    sequence_by_token: Mapping[str, str],
    *,
    protocol: EvaluationProtocol,
    num_replicates: int = 200,
    seed: int = 7,
) -> dict[str, object]:
    """Evaluate the frozen model and two fixed negative controls under one v2 population."""
    if set(train_truth) & set(selection_ground_truth):
        raise ValueError("Train and selection tokens must be disjoint")
    if set(selection_predictions) != set(selection_ground_truth):
        raise ValueError("Selection prediction and GT tokens differ")
    priors = fit_train_constant_prior(train_truth, protocol=protocol)
    constant = constant_predictions(sorted(selection_ground_truth), priors)
    permuted, assignment, attempts = derange_predictions(selection_predictions, seed=seed)
    arms = {
        "observed": selection_predictions,
        "constant_prior": constant,
        "frame_derangement": permuted,
    }
    metrics = {
        name: evaluate_detections_v2(predictions, selection_ground_truth, protocol=protocol)
        for name, predictions in arms.items()
    }
    bootstrap = paired_sequence_bootstrap(
        arms,
        selection_ground_truth,
        sequence_by_token,
        protocol=protocol,
        num_replicates=num_replicates,
        seed=seed,
    )
    cached = bootstrap["cached_full_population"]
    assert isinstance(cached, dict)
    for name, metric in metrics.items():
        cached_arm = cached[name]
        assert isinstance(cached_arm, dict)
        expected_macro = metric["dev_AP101_supported_classes"]
        if not _same_optional_float(cached_arm["supported_class_macro"], expected_macro):
            raise RuntimeError(f"Bootstrap cache differs from v2 scorer for {name} macro")
        cached_classes = cached_arm["per_class"]
        assert isinstance(cached_classes, dict)
        for category in protocol.classes:
            expected_class = metric["per_class"][category]["AP101_mean_thresholds"]
            if not _same_optional_float(cached_classes[category], expected_class):
                raise RuntimeError(f"Bootstrap cache differs from v2 scorer for {name}/{category}")
    observed_ap = metrics["observed"]["dev_AP101_supported_classes"]
    return {
        "schema": "phase_ttc_detection_controls_v2",
        "method": METHOD,
        "protocol_sha256": protocol.sha256,
        "official_eap_score": False,
        "train_selection_tokens_disjoint": True,
        "proposal_budget_per_frame": protocol.max_predictions_per_frame,
        "constant_prior": {
            "fit_source": "trainer-admitted training ground truth only",
            "fit": (
                "component medians for center/size, circular yaw mean, class frame frequency score"
            ),
            "priors": priors,
            "boxes_per_frame": len(priors),
            "metrics": metrics["constant_prior"],
        },
        "frame_derangement": {
            "seed": seed,
            "draws_until_first_derangement": attempts,
            "fixed_points": sum(target == source for target, source in assignment.items()),
            "is_bijection": set(assignment) == set(assignment.values()),
            "predictions_before_count": sum(map(len, selection_predictions.values())),
            "predictions_after_count": sum(map(len, permuted.values())),
            "metrics": metrics["frame_derangement"],
        },
        "observed_model_metrics": metrics["observed"],
        "dev_AP101_observed_minus_constant_prior": _delta(
            observed_ap, metrics["constant_prior"]["dev_AP101_supported_classes"]
        ),
        "dev_AP101_observed_minus_derangement": _delta(
            observed_ap, metrics["frame_derangement"]["dev_AP101_supported_classes"]
        ),
        "paired_sequence_bootstrap": bootstrap,
        "limitations": [
            "The scorer and population are local v2, not the official eAP benchmark",
            "The constant prior emits one box per supported train class within the fixed Q50 cap",
            "A derangement tests frame association but does not establish causality",
            "The pilot has only two selection sequences, so cluster intervals are weak",
        ],
    }


def _delta(left: object, right: object) -> float | None:
    return float(left) - float(right) if left is not None and right is not None else None


def _same_optional_float(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
