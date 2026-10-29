"""External diagnostics over frozen detection predictions and evaluation-v2 ground truth."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from statistics import mean
from typing import Any

import numpy as np

from e_jepa_ttc.detection.evaluation_ground_truth import (
    EvaluationBox,
    EvaluationGroundTruth,
    domain_reason,
    parse_box,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    return tuple(_number(item, name) for item in value)


def _xy(left: Mapping[str, object], right: EvaluationBox) -> float:
    point = _vector(left["translation_m"], "translation_m")
    target = right.translation_m
    return math.hypot(point[0] - target[0], point[1] - target[1])


def class_agnostic_correspondence(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    protocol: EvaluationProtocol,
    threshold_m: float,
    max_predictions_per_frame: int,
) -> list[dict[str, Any]]:
    """Greedily form bounded, one-to-one prediction/eligible-GT pairs without class labels."""
    pairs: list[dict[str, Any]] = []
    for token in sorted(ground_truth):
        ranked = sorted(
            enumerate(predictions[token]),
            key=lambda item: (-_number(item[1]["score"], "score"), item[0]),
        )[:max_predictions_per_frame]
        ranked = [
            (index, row)
            for index, row in ranked
            if domain_reason(
                parse_box(
                    row,
                    source_index=index,
                    classes=protocol.classes,
                    prediction=True,
                )[0],
                intrinsic=np.asarray(ground_truth[token].intrinsic),
                event_from_ego=np.asarray(ground_truth[token].event_from_ego),
                image_wh=ground_truth[token].image_wh,
                protocol=protocol,
            )
            is None
        ]
        assigned: set[int] = set()
        for prediction_index, prediction in ranked:
            candidates = [
                (_xy(prediction, target), target.source_index, target)
                for target in ground_truth[token].eligible
                if target.source_index not in assigned and _xy(prediction, target) < threshold_m
            ]
            if not candidates:
                continue
            distance, source_index, target = min(candidates, key=lambda row: (row[0], row[1]))
            assigned.add(source_index)
            prediction_center = np.asarray(
                _vector(prediction["translation_m"], "translation_m"), dtype=np.float64
            )
            target_center = np.asarray(target.translation_m, dtype=np.float64)
            transform = np.asarray(ground_truth[token].event_from_ego, dtype=np.float64)
            prediction_camera = transform[:3, :3] @ prediction_center + transform[:3, 3]
            target_camera = transform[:3, :3] @ target_center + transform[:3, 3]
            pairs.append(
                {
                    "token": token,
                    "prediction_index": prediction_index,
                    "ground_truth_source_index": source_index,
                    "predicted_class": str(prediction["class_name"]),
                    "target_class": target.class_name,
                    "score": _number(prediction["score"], "score"),
                    "ego_xy_error_m": distance,
                    "ego_z_vertical_error_m": abs(
                        _vector(prediction["translation_m"], "translation_m")[2]
                        - target.translation_m[2]
                    ),
                    "camera_depth_error_m": abs(prediction_camera[2] - target_camera[2]),
                    "camera_xy_error_m": float(
                        np.linalg.norm(prediction_camera[:2] - target_camera[:2])
                    ),
                    "dimensions_lwh_m": [
                        abs(a - b)
                        for a, b in zip(
                            _vector(prediction["size_lwh_m"], "size_lwh_m"),
                            target.size_lwh_m,
                            strict=True,
                        )
                    ],
                    "yaw_rad": abs(
                        (_number(prediction["yaw_rad"], "yaw_rad") - target.yaw_rad + math.pi)
                        % (2 * math.pi)
                        - math.pi
                    ),
                }
            )
    return pairs


def geometric_depth_prior_diagnostics(
    pairs: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
) -> dict[str, Any]:
    """Compare camera depth with the observable fy*H/projected-height heuristic."""
    targets = {
        (token, box.source_index): box
        for token, frame in ground_truth.items()
        for box in frame.eligible
    }
    rows = []
    for pair in pairs:
        token = str(pair["token"])
        source_index = int(pair["ground_truth_source_index"])
        target = targets[(token, source_index)]
        frame = ground_truth[token]
        intrinsic = np.asarray(frame.intrinsic, dtype=np.float64)
        transform = np.asarray(frame.event_from_ego, dtype=np.float64)
        length, width, height = target.size_lwh_m
        raw_box = [*target.translation_m, length, height, width, target.yaw_rad]
        corners_ego = original_bundle_corners(raw_box)
        corners_camera = corners_ego @ transform[:3, :3].T + transform[:3, 3]
        if (corners_camera[:, 2] <= 0).any():
            continue
        projected = corners_camera @ intrinsic.T
        vertical = projected[:, 1] / projected[:, 2]
        full_height_px = float(vertical.max() - vertical.min())
        clipped_height_px = float(
            min(vertical.max(), frame.image_wh[1] - 1) - max(vertical.min(), 0)
        )
        target_camera = transform[:3, :3] @ np.asarray(target.translation_m) + transform[:3, 3]
        camera_depth = float(target_camera[2])
        row: dict[str, Any] = {
            "token": token,
            "ground_truth_source_index": source_index,
            "class_name": target.class_name,
            "camera_depth_m": camera_depth,
            "projected_full_height_px": full_height_px,
            "projected_clipped_height_px": clipped_height_px,
        }
        for profile, pixel_height in (
            ("full", full_height_px),
            ("clipped", clipped_height_px),
        ):
            if pixel_height > 0:
                prior = float(intrinsic[1, 1] * target.size_lwh_m[2] / pixel_height)
                row[f"prior_{profile}_m"] = prior
                row[f"residual_camera_depth_minus_prior_{profile}_m"] = camera_depth - prior
            else:
                row[f"prior_{profile}_m"] = None
                row[f"residual_camera_depth_minus_prior_{profile}_m"] = None
        rows.append(row)
    per_class = {}
    for category in sorted({str(row["class_name"]) for row in rows}):
        category_rows = [row for row in rows if row["class_name"] == category]
        summary: dict[str, Any] = {"count": len(category_rows)}
        for profile in ("full", "clipped"):
            key = f"residual_camera_depth_minus_prior_{profile}_m"
            values = [float(row[key]) for row in category_rows if row[key] is not None]
            summary[f"{key}_mean"] = mean(values) if values else None
            summary[f"{key}_mean_absolute"] = (
                mean(abs(value) for value in values) if values else None
            )
        per_class[category] = summary
    return {
        "definition": "observable GT geometry: camera_depth - fy*GT_height/projected_height",
        "model_internal_prior_claim": False,
        "rows": rows,
        "per_class": per_class,
    }


def oracle_predictions(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    protocol: EvaluationProtocol,
    kind: str,
) -> dict[str, list[dict[str, Any]]]:
    """Apply one declared GT diagnostic to a fixed proposal pool before full rematching."""
    valid = {"class", "center", "geometry", "ranking"}
    if kind not in valid:
        raise ValueError(f"Unknown oracle {kind!r}; expected one of {sorted(valid)}")
    result = {
        token: [copy.deepcopy(dict(row)) for row in rows] for token, rows in predictions.items()
    }
    pairs = class_agnostic_correspondence(
        predictions,
        ground_truth,
        protocol=protocol,
        threshold_m=protocol.diagnostic_threshold_m,
        max_predictions_per_frame=protocol.max_predictions_per_frame,
    )
    targets = {
        (token, box.source_index): box
        for token, frame in ground_truth.items()
        for box in frame.eligible
    }
    if kind == "ranking":
        all_scores = sorted(
            (_number(row["score"], "score") for rows in predictions.values() for row in rows),
            reverse=True,
        )
        matched = {(str(pair["token"]), int(pair["prediction_index"])) for pair in pairs}
        order = sorted(matched) + sorted(
            (token, index)
            for token, rows in predictions.items()
            for index in range(len(rows))
            if (token, index) not in matched
        )
        for score, (token, index) in zip(all_scores, order, strict=True):
            result[token][index]["score"] = score
        return result
    for pair in pairs:
        token = str(pair["token"])
        prediction = result[token][int(pair["prediction_index"])]
        target = targets[(token, int(pair["ground_truth_source_index"]))]
        if kind == "class":
            prediction["class_name"] = target.class_name
        elif kind == "center":
            prediction["translation_m"] = list(target.translation_m)
        elif kind == "geometry":
            prediction["size_lwh_m"] = list(target.size_lwh_m)
            prediction["yaw_rad"] = target.yaw_rad
    return result


def _false_positive_reasons(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    """Attribute diagnostic-threshold predictions after class-constrained matching."""
    counts: Counter[str] = Counter()
    per_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    for category in protocol.classes:
        ranked: list[tuple[float, str, int, Mapping[str, object]]] = []
        for token, rows in predictions.items():
            capped = sorted(
                enumerate(rows),
                key=lambda item: (-_number(item[1]["score"], "score"), item[0]),
            )[: protocol.max_predictions_per_frame]
            for index, row in capped:
                if row["class_name"] != category:
                    continue
                box = parse_box(
                    row,
                    source_index=index,
                    classes=protocol.classes,
                    prediction=True,
                )[0]
                frame = ground_truth[token]
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
                    ranked.append((-_number(row["score"], "score"), token, index, row))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        assigned: set[tuple[str, int]] = set()
        ignored_assigned: set[tuple[str, int]] = set()
        for _, token, _, row in ranked:
            frame = ground_truth[token]
            same = [
                (_xy(row, target), target.source_index)
                for target in frame.eligible
                if target.class_name == category
                and _xy(row, target) < protocol.diagnostic_threshold_m
            ]
            available = [item for item in same if (token, item[1]) not in assigned]
            if available:
                _, source_index = min(available)
                assigned.add((token, source_index))
                reason = "true_positive"
            elif same:
                reason = "duplicate"
            elif ignored := [
                (_xy(row, target), target.source_index)
                for target in frame.ignored
                if target.class_name == category
                and _xy(row, target) < protocol.diagnostic_threshold_m
                and (token, target.source_index) not in ignored_assigned
            ]:
                _, source_index = min(ignored)
                ignored_assigned.add((token, source_index))
                reason = "ignored_object_suppressed"
            elif any(
                _xy(row, target) < protocol.diagnostic_threshold_m
                for target in frame.ignored
                if target.class_name == category
            ):
                reason = "ignored_object_duplicate_fp"
            elif any(
                _xy(row, target) < protocol.diagnostic_threshold_m for target in frame.eligible
            ):
                reason = "class_confusion"
            else:
                reason = "background_or_localization"
            counts[reason] += 1
            per_sequence[token.split(":", 1)[0]][reason] += 1
    return {
        "threshold_m": protocol.diagnostic_threshold_m,
        "counts": dict(sorted(counts.items())),
        "per_sequence": {
            sequence: dict(sorted(values.items()))
            for sequence, values in sorted(per_sequence.items())
        },
    }


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.std() == 0 or right_array.std() == 0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def analyze_query_sidecar(
    sidecar: Mapping[str, Any],
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    """Analyze exposed query logits/depth uncertainty without altering the producer."""
    frames = sidecar.get("frames")
    if not isinstance(frames, list) or len(frames) != len(predictions):
        raise ValueError("Sidecar frame population differs from predictions")
    query_by_prediction: dict[tuple[str, int], tuple[int, list[float], list[float]]] = {}
    qxc_predictions: dict[str, list[dict[str, object]]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("Sidecar frame must be an object")
        token = frame.get("sample_token")
        tensors = frame.get("tensors")
        if (
            not isinstance(token, str)
            or token not in predictions
            or not isinstance(tensors, Mapping)
        ):
            raise ValueError("Sidecar token or tensors are invalid")
        logits_entry = tensors.get("pred_logits")
        depth_entry = tensors.get("pred_depth")
        if not isinstance(logits_entry, Mapping) or not isinstance(depth_entry, Mapping):
            raise ValueError("Sidecar lacks exposed logits or depth")
        raw_logits = logits_entry.get("values")
        raw_depth = depth_entry.get("values")
        if not isinstance(raw_logits, list) or not isinstance(raw_depth, list):
            raise ValueError("Sidecar query tensor values must be arrays")
        logits = [_vector(row, "pred_logits") for row in raw_logits]
        depths = [_vector(row, "pred_depth") for row in raw_depth]
        if len(logits) != len(depths) or any(len(row) != len(protocol.classes) for row in logits):
            raise ValueError("Sidecar query dimensions differ from detector taxonomy")
        probabilities = [tuple(1.0 / (1.0 + math.exp(-value)) for value in row) for row in logits]
        ranking = sorted(
            range(len(probabilities)),
            key=lambda query: (-max(probabilities[query]), query),
        )
        decoded = predictions[token]
        if len(decoded) != len(ranking):
            raise ValueError("Every pilot query must map to one decoded prediction")
        geometry_by_query: dict[int, Mapping[str, object]] = {}
        for prediction_index, query in enumerate(ranking):
            row = decoded[prediction_index]
            score = max(probabilities[query])
            label = probabilities[query].index(score)
            if (
                abs(_number(row["score"], "score") - score) > 1e-6
                or row["class_name"] != protocol.classes[label]
            ):
                raise ValueError("Sidecar query ordering differs from decoded predictions")
            if len(depths[query]) != 2:
                raise ValueError("pred_depth must expose depth and log variance")
            query_by_prediction[(token, prediction_index)] = (
                query,
                list(probabilities[query]),
                list(depths[query]),
            )
            geometry_by_query[query] = row
        candidates = []
        for query, class_probabilities in enumerate(probabilities):
            geometry = geometry_by_query[query]
            for class_index, score in enumerate(class_probabilities):
                candidate = copy.deepcopy(dict(geometry))
                candidate["class_name"] = protocol.classes[class_index]
                candidate["score"] = score
                candidates.append(candidate)
        qxc_predictions[token] = sorted(
            candidates, key=lambda row: -_number(row["score"], "score")
        )[: protocol.max_predictions_per_frame]
    pairs = class_agnostic_correspondence(
        predictions,
        ground_truth,
        protocol=protocol,
        threshold_m=protocol.diagnostic_threshold_m,
        max_predictions_per_frame=protocol.max_predictions_per_frame,
    )
    matched_rows = []
    for pair in pairs:
        token = str(pair["token"])
        prediction_index = int(pair["prediction_index"])
        query, query_probabilities, query_depth = query_by_prediction[(token, prediction_index)]
        target_class = str(pair["target_class"])
        target_index = protocol.classes.index(target_class)
        order = sorted(
            range(len(query_probabilities)),
            key=lambda index: (-query_probabilities[index], index),
        )
        matched_rows.append(
            {
                "token": token,
                "prediction_index": prediction_index,
                "query_index": query,
                "class_correct": pair["predicted_class"] == target_class,
                "target_class": target_class,
                "target_probability": query_probabilities[target_index],
                "target_rank": order.index(target_index) + 1,
                "classification_score": float(pair["score"]),
                "depth_m": query_depth[0],
                "depth_log_variance": query_depth[1],
                "ego_xy_error_m": float(pair["ego_xy_error_m"]),
                "ego_z_vertical_error_m": float(pair["ego_z_vertical_error_m"]),
                "camera_depth_error_m": float(pair["camera_depth_error_m"]),
            }
        )
    log_variance = [float(row["depth_log_variance"]) for row in matched_rows]
    ego_z_vertical_error = [float(row["ego_z_vertical_error_m"]) for row in matched_rows]
    ego_xy_error = [float(row["ego_xy_error_m"]) for row in matched_rows]
    camera_depth_error = [float(row["camera_depth_error_m"]) for row in matched_rows]
    class_score = [float(row["classification_score"]) for row in matched_rows]
    uncertainty_bins = []
    ordered_rows = sorted(matched_rows, key=lambda row: float(row["depth_log_variance"]))
    bin_size = max(1, math.ceil(len(ordered_rows) / 4))
    for start in range(0, len(ordered_rows), bin_size):
        rows = ordered_rows[start : start + bin_size]
        uncertainty_bins.append(
            {
                "count": len(rows),
                "depth_log_variance_mean": mean(float(row["depth_log_variance"]) for row in rows),
                "camera_depth_error_mean_m": mean(
                    float(row["camera_depth_error_m"]) for row in rows
                ),
                "ego_z_vertical_error_mean_m": mean(
                    float(row["ego_z_vertical_error_m"]) for row in rows
                ),
                "ego_xy_error_mean_m": mean(float(row["ego_xy_error_m"]) for row in rows),
            }
        )
    qxc_result = evaluate_detections_v2(qxc_predictions, ground_truth, protocol=protocol)
    return {
        "schema": "phase_ttc_query_sidecar_diagnostics_v1",
        "query_mapping": "verified decoder score/class ordering; 50 decoded queries per frame",
        "matched_query_count": len(matched_rows),
        "depth_uncertainty": {
            "semantics": "second pred_depth component trained as log variance",
            "pearson_log_variance_vs_camera_depth_error": _correlation(
                log_variance, camera_depth_error
            ),
            "pearson_log_variance_vs_ego_z_vertical_error": _correlation(
                log_variance, ego_z_vertical_error
            ),
            "pearson_log_variance_vs_ego_xy_error": _correlation(log_variance, ego_xy_error),
            "quartile_like_bins": uncertainty_bins,
        },
        "classification_quality": {
            "pearson_score_vs_camera_depth_error": _correlation(class_score, camera_depth_error),
            "target_top1_count": sum(int(row["target_rank"]) <= 1 for row in matched_rows),
            "target_top3_count": sum(int(row["target_rank"]) <= 3 for row in matched_rows),
            "target_top5_count": sum(int(row["target_rank"]) <= 5 for row in matched_rows),
            "matched_count": len(matched_rows),
        },
        "fixed_k_query_class_control": {
            "definition": "top 50 of 50 queries x 8 classes; same decoded geometry and K",
            "dev_AP101_supported_classes": qxc_result["dev_AP101_supported_classes"],
            "dev_AP101_all_eight_absent_zero": qxc_result["dev_AP101_all_eight_absent_zero"],
            "evaluation": qxc_result,
        },
        "matched_queries": matched_rows,
        "still_unavailable": [
            "decoded_box_2d_per_query",
            "geometric_depth_prior_per_query",
            "geometric_depth_residual_per_query",
            "foreground_membership_per_query",
        ],
    }


def build_error_diagnostics(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    """Return reproducible confusion, geometry and oracle diagnostics."""
    baseline = evaluate_detections_v2(predictions, ground_truth, protocol=protocol)
    pairs = class_agnostic_correspondence(
        predictions,
        ground_truth,
        protocol=protocol,
        threshold_m=protocol.diagnostic_threshold_m,
        max_predictions_per_frame=protocol.max_predictions_per_frame,
    )
    confusion = Counter((str(row["target_class"]), str(row["predicted_class"])) for row in pairs)
    by_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs:
        sequence = str(row["token"]).split(":", 1)[0]
        by_sequence[sequence]["matched_class_agnostic"] += 1
        by_sequence[sequence]["class_correct"] += int(row["target_class"] == row["predicted_class"])
    eligible_by_sequence = Counter()
    for token, frame in ground_truth.items():
        eligible_by_sequence[token.split(":", 1)[0]] += len(frame.eligible)
    geometry = {}
    for key in (
        "ego_xy_error_m",
        "ego_z_vertical_error_m",
        "camera_depth_error_m",
        "camera_xy_error_m",
        "yaw_rad",
    ):
        values = [float(row[key]) for row in pairs]
        geometry[key] = {"mean": mean(values) if values else None, "count": len(values)}
    for index, dimension in enumerate(("length", "width", "height")):
        values = [float(row["dimensions_lwh_m"][index]) for row in pairs]
        geometry[f"dimension_{dimension}_m"] = {
            "mean": mean(values) if values else None,
            "count": len(values),
        }
    quality_bins = []
    ordered = sorted(pairs, key=lambda row: float(row["score"]))
    for start in range(0, len(ordered), max(1, math.ceil(len(ordered) / 10))):
        rows = ordered[start : start + max(1, math.ceil(len(ordered) / 10))]
        quality_bins.append(
            {
                "count": len(rows),
                "score_mean": mean(float(row["score"]) for row in rows),
                "ego_xy_error_mean_m": mean(float(row["ego_xy_error_m"]) for row in rows),
                "ego_z_vertical_error_mean_m": mean(
                    float(row["ego_z_vertical_error_m"]) for row in rows
                ),
                "camera_depth_error_mean_m": mean(
                    float(row["camera_depth_error_m"]) for row in rows
                ),
            }
        )
    oracle_results = {}
    for kind in ("class", "center", "geometry", "ranking"):
        transformed = oracle_predictions(predictions, ground_truth, protocol=protocol, kind=kind)
        evaluated = evaluate_detections_v2(transformed, ground_truth, protocol=protocol)
        oracle_results[kind] = {
            "dev_AP101_supported_classes": evaluated["dev_AP101_supported_classes"],
            "dev_AP101_all_eight_absent_zero": evaluated["dev_AP101_all_eight_absent_zero"],
            "rematched_after_transformation": True,
        }
    sequence_metrics = {}
    for sequence in sorted(eligible_by_sequence):
        sequence_tokens = [token for token in ground_truth if token.split(":", 1)[0] == sequence]
        evaluated = evaluate_detections_v2(
            {token: predictions[token] for token in sequence_tokens},
            {token: ground_truth[token] for token in sequence_tokens},
            protocol=protocol,
        )
        counts = by_sequence[sequence]
        sequence_metrics[sequence] = {
            **dict(counts),
            "eligible_ground_truth": eligible_by_sequence[sequence],
            "class_agnostic_recall": (
                counts["matched_class_agnostic"] / eligible_by_sequence[sequence]
                if eligible_by_sequence[sequence]
                else None
            ),
            "dev_AP101_supported_classes": evaluated["dev_AP101_supported_classes"],
            "supported_class_count": evaluated["supported_class_count"],
        }
    return {
        "schema": "phase_ttc_detection_error_diagnostics_v1",
        "evaluation": baseline,
        "class_agnostic_correspondence": {
            "threshold_m": protocol.diagnostic_threshold_m,
            "one_to_one": True,
            "count": len(pairs),
            "pairs": pairs,
        },
        "confusion": [
            {"target_class": target, "predicted_class": predicted, "count": count}
            for (target, predicted), count in sorted(confusion.items())
        ],
        "geometry_errors_on_class_agnostic_pairs": geometry,
        "geometry_axis_semantics": {
            "ego_xy_error_m": "ground-plane center distance in ego X-forward/Y-left",
            "ego_z_vertical_error_m": "absolute vertical center error in ego Z-up",
            "camera_depth_error_m": "absolute optical-axis depth error after T_event_ego",
            "camera_xy_error_m": "camera lateral/vertical plane center error",
            "embedded_evaluation_diagnostic_z_m": "ego Z-up vertical error, not camera depth",
        },
        "geometric_depth_prior": geometric_depth_prior_diagnostics(pairs, ground_truth),
        "score_quality_bins": quality_bins,
        "false_positive_reasons": _false_positive_reasons(predictions, ground_truth, protocol),
        "per_sequence": sequence_metrics,
        "oracles": oracle_results,
        "unavailable_from_frozen_output": [
            "full_class_logits",
            "query_id",
            "box2d",
            "depth_uncertainty",
            "geometric_prior_and_residual",
        ],
    }
