"""Train-only, fixed-pool quality ranking for frozen 3D detections.

The ranker consumes only values exposed by the frozen detector at inference.
Ground truth is accepted solely by :func:`build_training_rows`; applying a
serialized bundle has no label-bearing argument.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from e_jepa_ttc.detection.evaluation_ground_truth import (
    EvaluationGroundTruth,
    domain_reason,
    parse_box,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol

SCHEMA = "phase_ttc_detection_quality_ranker_v1"
FEATURE_SCHEMA = "phase_ttc_detection_quality_features_v1"
TRAINING_ROWS_SCHEMA = "phase_ttc_detection_quality_training_rows_v1"
FEATURE_NAMES = (
    "class_logit_car",
    "class_logit_bicycle",
    "class_logit_truck",
    "class_logit_pedestrian",
    "class_logit_motorcycle",
    "class_logit_bus",
    "class_logit_tricycle",
    "class_logit_other_vehicle",
    "classification_score",
    "top1_top2_probability_margin",
    "class_probability_entropy",
    "predicted_camera_depth_m",
    "depth_log_variance",
    "log_length_m",
    "log_width_m",
    "log_height_m",
)


@dataclass(frozen=True)
class RankingRecipe:
    """Single preregistered linear-quality recipe; no hyperparameter search."""

    seed: int = 7
    epochs: int = 400
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    target_positive_weight: float = 3.0
    normalizer_std_floor: float = 1e-6


@dataclass(frozen=True)
class TrainingProvenance:
    """Content identities proving that fitting used the TRAIN role only."""

    dataset_role: str
    split_manifest_sha256: str
    panel_recipe_sha256: str
    ground_truth_receipt_sha256: str
    predictions_sha256: str
    sidecar_sha256: str
    ground_truth_sha256: str
    checkpoint_sha256: str
    protocol_sha256: str
    token_population_sha256: str
    train_ownership_sha256: str

    def validate(self) -> None:
        """Reject non-TRAIN fitting and incomplete content identities."""
        if self.dataset_role != "train":
            raise ValueError("Quality ranker fitting is restricted to the train role")
        for name, value in asdict(self).items():
            if name == "dataset_role":
                continue
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a full SHA-256 digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{name} must be hexadecimal") from error


@dataclass(frozen=True)
class QualityTrainingRow:
    """One frozen proposal with its score-independent train-only quality target."""

    token: str
    prediction_index: int
    class_name: str
    protocol_sha256: str
    features: tuple[float, ...]
    target_quality: float | None
    target_kind: str
    matched_ground_truth_source_index: int | None
    ego_xy_error_m: float | None
    mask_reason: str | None = None


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def token_population_sha256(tokens: Sequence[str]) -> str:
    """Hash a unique sorted token population."""
    if not tokens or len(tokens) != len(set(tokens)) or any(not token for token in tokens):
        raise ValueError("Tokens must be nonempty and unique")
    return _canonical_sha256(sorted(tokens))


def verify_train_token_ownership(
    tokens: Sequence[str],
    split_manifest: Mapping[str, object],
    sequence_id_by_token: Mapping[str, object],
) -> str:
    """Prove every fitted token belongs to canonical TRAIN sequence ownership."""
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Training ownership requires unique nonempty tokens")
    assignments = split_manifest.get("assignments")
    if not isinstance(assignments, Mapping) or "train" not in assignments:
        raise ValueError("Split manifest lacks assignments.train")
    owner: dict[str, str] = {}
    for role, raw_sequences in assignments.items():
        if (
            not isinstance(role, str)
            or isinstance(raw_sequences, (str, bytes))
            or not isinstance(raw_sequences, Sequence)
        ):
            raise ValueError("Split assignments must map roles to sequence arrays")
        for value in raw_sequences:
            if not isinstance(value, str) or not value:
                raise ValueError("Split sequence IDs must be nonempty strings")
            if value in owner:
                raise ValueError("Split sequence ownership overlaps roles")
            owner[value] = role
    if set(sequence_id_by_token) != set(tokens):
        raise ValueError("Canonical token metadata differs from the fitted token population")
    ownership: dict[str, str] = {}
    for token in sorted(tokens):
        sequence = sequence_id_by_token[token]
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("Canonical token metadata has an invalid sequence ID")
        if token.split(":", 1)[0] != sequence:
            raise ValueError("Token prefix differs from canonical sequence metadata")
        if owner.get(sequence) != "train":
            raise ValueError("Fitted token is not owned by split.assignments.train")
        ownership[token] = sequence
    return _canonical_sha256(ownership)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(_finite(item, name) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _sidecar_query_features(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    sidecar: Mapping[str, object],
    classes: tuple[str, ...],
) -> dict[tuple[str, int], tuple[float, ...]]:
    if sidecar.get("schema") != "monodgp_external_diagnostic_sidecar_v1":
        raise ValueError("Unexpected detector sidecar schema")
    frames = sidecar.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Detector sidecar frames are missing")
    raw_frame_tokens = [frame.get("sample_token") for frame in frames if isinstance(frame, Mapping)]
    if len(raw_frame_tokens) != len(frames) or any(
        not isinstance(token, str) or not token for token in raw_frame_tokens
    ):
        raise ValueError("Prediction and sidecar token populations differ")
    frame_tokens = [token for token in raw_frame_tokens if isinstance(token, str)]
    if (
        len(frame_tokens) != len(set(frame_tokens))
        or len(frame_tokens) != len(predictions)
        or set(predictions) != set(frame_tokens)
    ):
        raise ValueError("Prediction and sidecar token populations differ")
    result: dict[tuple[str, int], tuple[float, ...]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("Sidecar frame must be a mapping")
        token = frame.get("sample_token")
        tensors = frame.get("tensors")
        if not isinstance(token, str) or not isinstance(tensors, Mapping):
            raise ValueError("Sidecar frame lacks token or tensors")
        logits_entry = tensors.get("pred_logits")
        depth_entry = tensors.get("pred_depth")
        if not isinstance(logits_entry, Mapping) or not isinstance(depth_entry, Mapping):
            raise ValueError("Sidecar must expose pred_logits and pred_depth")
        raw_logits, raw_depth = logits_entry.get("values"), depth_entry.get("values")
        if not isinstance(raw_logits, list) or not isinstance(raw_depth, list):
            raise ValueError("Sidecar tensor values must be arrays")
        logits = [_vector(row, len(classes), "pred_logits") for row in raw_logits]
        depths = [_vector(row, 2, "pred_depth") for row in raw_depth]
        if len(logits) != len(depths) or len(logits) < len(predictions[token]):
            raise ValueError("Sidecar has fewer queries than the fixed proposal pool")
        probabilities = [tuple(_sigmoid(value) for value in row) for row in logits]
        dimensions_entry = tensors.get("pred_3d_dim")
        dimensions = None
        if isinstance(dimensions_entry, Mapping):
            raw_dimensions = dimensions_entry.get("values")
            if not isinstance(raw_dimensions, list):
                raise ValueError("Sidecar dimension values must be arrays")
            dimensions = [_vector(row, 3, "pred_3d_dim") for row in raw_dimensions]
            if len(dimensions) != len(logits):
                raise ValueError("Sidecar dimension query count differs")
        assigned_queries: set[int] = set()
        for prediction_index, prediction in enumerate(predictions[token]):
            class_name = prediction.get("class_name")
            if class_name not in classes:
                raise ValueError("Prediction class is outside the fixed taxonomy")
            score = _finite(prediction.get("score"), "score")
            length, width, height = _vector(prediction.get("size_lwh_m"), 3, "size_lwh_m")
            candidates = []
            for query, query_probabilities in enumerate(probabilities):
                query_score = max(query_probabilities)
                query_class = classes[query_probabilities.index(query_score)]
                if (
                    query not in assigned_queries
                    and query_class == class_name
                    and math.isclose(score, query_score, abs_tol=1e-7, rel_tol=0.0)
                    # The frozen decoder copies HWL to LWH without arithmetic.
                    # Exact dimensions bind close/equal scores without labels or
                    # assumptions about query order, top-k or discarded boxes.
                    and (dimensions is None or dimensions[query] == (height, width, length))
                ):
                    candidates.append(query)
            if len(candidates) != 1:
                raise ValueError(
                    "Decoded prediction does not map uniquely to one raw sidecar query"
                )
            query = candidates[0]
            assigned_queries.add(query)
            class_index = probabilities[query].index(max(probabilities[query]))
            if classes[class_index] != class_name:
                raise AssertionError("Unique query binding changed class unexpectedly")
            ordered = sorted(probabilities[query], reverse=True)
            probability_sum = sum(probabilities[query])
            normalized = [value / probability_sum for value in probabilities[query]]
            entropy = -sum(value * math.log(max(value, 1e-12)) for value in normalized)
            length, width, height = _vector(prediction.get("size_lwh_m"), 3, "size_lwh_m")
            if min(length, width, height) <= 0:
                raise ValueError("Decoded dimensions must be positive")
            features = (
                *logits[query],
                score,
                ordered[0] - ordered[1],
                entropy,
                depths[query][0],
                depths[query][1],
                math.log(length),
                math.log(width),
                math.log(height),
            )
            if len(features) != len(FEATURE_NAMES):
                raise AssertionError("Feature contract length differs from its schema")
            result[(token, prediction_index)] = features
    return result


def build_training_rows(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    sidecar: Mapping[str, object],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    protocol: EvaluationProtocol,
) -> list[QualityTrainingRow]:
    """Build TRAIN targets using class-constrained score-independent assignment."""
    if not ground_truth or set(predictions) != set(ground_truth):
        raise ValueError("Training predictions and ground truth must share one population")
    if any(frame.protocol_sha256 != protocol.sha256 for frame in ground_truth.values()):
        raise ValueError("Training ground truth uses another evaluation protocol")
    feature_map = _sidecar_query_features(predictions, sidecar, protocol.classes)
    assignments: dict[tuple[str, int], tuple[int, float]] = {}
    duplicate_keys: set[tuple[str, int]] = set()
    masked: dict[tuple[str, int], str] = {}
    maximum_threshold = max(protocol.distance_thresholds_m)
    for token in sorted(predictions):
        if len(predictions[token]) > protocol.max_predictions_per_frame:
            raise ValueError("Training pool exceeds the frozen per-frame proposal cap")
        in_domain_indices: set[int] = set()
        for index, row in enumerate(predictions[token]):
            box, _ = parse_box(row, source_index=index, classes=protocol.classes, prediction=True)
            reason = domain_reason(
                box,
                intrinsic=np.asarray(ground_truth[token].intrinsic),
                event_from_ego=np.asarray(ground_truth[token].event_from_ego),
                image_wh=ground_truth[token].image_wh,
                protocol=protocol,
            )
            if reason is None:
                in_domain_indices.add(index)
            else:
                masked[(token, index)] = f"prediction_domain:{reason}"
        for class_name in protocol.classes:
            prediction_indices = [
                index
                for index, row in enumerate(predictions[token])
                if index in in_domain_indices and row.get("class_name") == class_name
            ]
            targets = [
                target for target in ground_truth[token].eligible if target.class_name == class_name
            ]
            if not prediction_indices:
                continue
            assigned_target_indices: set[int] = set()
            costs = np.empty((len(prediction_indices), 0), dtype=np.float64)
            if targets:
                costs = np.asarray(
                    [
                        [
                            math.hypot(
                                _vector(
                                    predictions[token][index].get("translation_m"),
                                    3,
                                    "translation_m",
                                )[0]
                                - target.translation_m[0],
                                _vector(
                                    predictions[token][index].get("translation_m"),
                                    3,
                                    "translation_m",
                                )[1]
                                - target.translation_m[1],
                            )
                            for target in targets
                        ]
                        for index in prediction_indices
                    ],
                    dtype=np.float64,
                )
                bounded_costs = np.where(costs < maximum_threshold, costs, 1e6)
                prediction_rows, target_columns = linear_sum_assignment(bounded_costs)
                for prediction_row, target_column in zip(
                    prediction_rows.tolist(), target_columns.tolist(), strict=True
                ):
                    if costs[prediction_row, target_column] >= maximum_threshold:
                        continue
                    prediction_index = prediction_indices[prediction_row]
                    target = targets[target_column]
                    assignments[(token, prediction_index)] = (
                        target.source_index,
                        float(costs[prediction_row, target_column]),
                    )
                    assigned_target_indices.add(target_column)
                for prediction_row, prediction_index in enumerate(prediction_indices):
                    if (token, prediction_index) in assignments:
                        continue
                    if any(
                        costs[prediction_row, column] < maximum_threshold
                        for column in assigned_target_indices
                    ):
                        duplicate_keys.add((token, prediction_index))
            # Eligible matching has priority. Of the remaining proposals, mask
            # at most one per ignored GT, exactly preserving one-to-one ignore
            # suppression instead of teaching ignored objects as background.
            remaining = [
                index
                for index in prediction_indices
                if (token, index) not in assignments and (token, index) not in duplicate_keys
            ]
            ignored_targets = [
                target for target in ground_truth[token].ignored if target.class_name == class_name
            ]
            if remaining and ignored_targets:
                ignored_costs = np.asarray(
                    [
                        [
                            math.hypot(
                                _vector(
                                    predictions[token][index].get("translation_m"),
                                    3,
                                    "translation_m",
                                )[0]
                                - target.translation_m[0],
                                _vector(
                                    predictions[token][index].get("translation_m"),
                                    3,
                                    "translation_m",
                                )[1]
                                - target.translation_m[1],
                            )
                            for target in ignored_targets
                        ]
                        for index in remaining
                    ],
                    dtype=np.float64,
                )
                bounded_ignored_costs = np.where(
                    ignored_costs < maximum_threshold, ignored_costs, 1e6
                )
                ignored_rows, ignored_columns = linear_sum_assignment(bounded_ignored_costs)
                for row_index, column_index in zip(
                    ignored_rows.tolist(), ignored_columns.tolist(), strict=True
                ):
                    if ignored_costs[row_index, column_index] < maximum_threshold:
                        reason = ignored_targets[column_index].ignore_reason or "unspecified"
                        masked[(token, remaining[row_index])] = f"ignored_ground_truth:{reason}"
    rows: list[QualityTrainingRow] = []
    for token in sorted(predictions):
        for prediction_index, prediction in enumerate(predictions[token]):
            key = (token, prediction_index)
            assignment = assignments.get(key)
            mask_reason = masked.get(key)
            if mask_reason is not None:
                target_quality = None
                target_kind = "masked"
                source_index, error = None, None
            elif assignment is None:
                target_quality = 0.0
                target_kind = "duplicate" if key in duplicate_keys else "background"
                source_index, error = None, None
            else:
                source_index, error = assignment
                target_quality = sum(
                    error < threshold for threshold in protocol.distance_thresholds_m
                ) / len(protocol.distance_thresholds_m)
                target_kind = "assigned_positive" if target_quality > 0 else "assigned_far"
            rows.append(
                QualityTrainingRow(
                    token=token,
                    prediction_index=prediction_index,
                    class_name=str(prediction["class_name"]),
                    protocol_sha256=protocol.sha256,
                    features=feature_map[key],
                    target_quality=target_quality,
                    target_kind=target_kind,
                    matched_ground_truth_source_index=source_index,
                    ego_xy_error_m=error,
                    mask_reason=mask_reason,
                )
            )
    return rows


def fit_quality_ranker(
    rows: Sequence[QualityTrainingRow],
    *,
    provenance: TrainingProvenance,
    recipe: RankingRecipe | None = None,
) -> dict[str, object]:
    """Fit the sole preregistered linear-sigmoid candidate on TRAIN rows."""
    provenance.validate()
    recipe = RankingRecipe() if recipe is None else recipe
    if not rows:
        raise ValueError("Quality ranker requires nonempty training rows")
    if (
        token_population_sha256(sorted({row.token for row in rows}))
        != provenance.token_population_sha256
    ):
        raise ValueError("Training row population differs from its provenance")
    row_protocols = {row.protocol_sha256 for row in rows}
    if row_protocols != {provenance.protocol_sha256}:
        raise ValueError("Training row protocol differs from its provenance")
    fitted_rows = [row for row in rows if row.target_quality is not None]
    if not fitted_rows:
        raise ValueError("All quality-ranking rows are masked")
    features = torch.tensor([row.features for row in fitted_rows], dtype=torch.float64)
    fitted_targets = []
    for row in fitted_rows:
        if row.target_quality is None:
            raise AssertionError("Masked row leaked into the fitted row population")
        fitted_targets.append(row.target_quality)
    targets = torch.tensor(fitted_targets, dtype=torch.float64)
    if features.shape[1] != len(FEATURE_NAMES) or not torch.isfinite(features).all():
        raise ValueError("Training features violate the fixed finite schema")
    if not torch.all((targets >= 0) & (targets <= 1)):
        raise ValueError("Quality targets must be bounded by [0,1]")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(recipe.normalizer_std_floor)
    normalized = (features - mean) / std
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    layer = torch.nn.Linear(len(FEATURE_NAMES), 1, dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    sample_weights = 1.0 + recipe.target_positive_weight * targets
    for _ in range(recipe.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = layer(normalized).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=sample_weights
        )
        loss.backward()
        optimizer.step()
    support = Counter(
        row.class_name
        for row in fitted_rows
        if row.target_quality is not None and row.target_quality > 0
    )
    target_kinds = Counter(row.target_kind for row in rows)
    masked_by_reason_and_class = Counter(
        (row.class_name, row.mask_reason) for row in rows if row.target_quality is None
    )
    bundle: dict[str, object] = {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "model": "shared_linear_sigmoid",
        "recipe": asdict(recipe),
        "recipe_sha256": _canonical_sha256(asdict(recipe)),
        "provenance": asdict(provenance),
        "protocol_sha256": provenance.protocol_sha256,
        "normalizer": {"mean": mean.tolist(), "std": std.tolist()},
        "weights": layer.weight.detach().squeeze(0).tolist(),
        "bias": float(layer.bias.detach().item()),
        "training": {
            "row_count": len(rows),
            "fitted_row_count": len(fitted_rows),
            "masked_row_count": len(rows) - len(fitted_rows),
            "target_kind_counts": dict(sorted(target_kinds.items())),
            "masked_counts_by_class_and_reason": [
                {"class_name": key[0], "reason": key[1], "count": count}
                for key, count in sorted(masked_by_reason_and_class.items())
            ],
            "positive_target_support_by_class": {
                class_name: support.get(class_name, 0)
                for class_name in EvaluationProtocol.default().classes
            },
            "final_weighted_bce": float(loss.detach().item()),
        },
    }
    bundle["bundle_sha256"] = _canonical_sha256(bundle)
    return bundle


def _validate_bundle(
    bundle: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if bundle.get("schema") != SCHEMA or bundle.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Unexpected quality-ranker bundle schema")
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Quality-ranker feature schema differs")
    expected_hash = bundle.get("bundle_sha256")
    unhashed = dict(bundle)
    unhashed.pop("bundle_sha256", None)
    if expected_hash != _canonical_sha256(unhashed):
        raise ValueError("Quality-ranker bundle hash differs")
    provenance = bundle.get("provenance")
    expected_provenance = set(TrainingProvenance.__dataclass_fields__)
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance:
        raise ValueError("Quality-ranker provenance fields differ")
    TrainingProvenance(**{name: str(provenance[name]) for name in expected_provenance}).validate()
    if bundle.get("protocol_sha256") != provenance.get("protocol_sha256"):
        raise ValueError("Bundle protocol differs from its training provenance")
    normalizer = bundle.get("normalizer")
    if not isinstance(normalizer, Mapping):
        raise ValueError("Quality-ranker normalizer is missing")
    mean = np.asarray(normalizer.get("mean"), dtype=np.float64)
    std = np.asarray(normalizer.get("std"), dtype=np.float64)
    weights = np.asarray(bundle.get("weights"), dtype=np.float64)
    bias = _finite(bundle.get("bias"), "bias")
    expected_shape = (len(FEATURE_NAMES),)
    if (
        mean.shape != expected_shape
        or std.shape != expected_shape
        or weights.shape != expected_shape
    ):
        raise ValueError("Quality-ranker parameter shape differs")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.isfinite(weights).all():
        raise ValueError("Quality-ranker parameters must be finite")
    if np.any(std <= 0):
        raise ValueError("Quality-ranker normalization scale must be positive")
    return mean, std, weights, bias


def apply_quality_ranker(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    sidecar: Mapping[str, object],
    bundle: Mapping[str, object],
    *,
    protocol: EvaluationProtocol,
) -> dict[str, list[dict[str, object]]]:
    """Re-score a fixed proposal pool without accepting or consulting labels."""
    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping) or sidecar.get("checkpoint_sha256") != provenance.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Inference sidecar checkpoint differs from ranker training")
    if any(len(rows) > protocol.max_predictions_per_frame for rows in predictions.values()):
        raise ValueError("Inference pool exceeds the frozen per-frame proposal cap")
    mean, std, weights, bias = _validate_bundle(bundle)
    if bundle.get("protocol_sha256") != protocol.sha256:
        raise ValueError("Inference protocol differs from ranker training")
    feature_map = _sidecar_query_features(predictions, sidecar, protocol.classes)
    output: dict[str, list[dict[str, object]]] = {}
    for token in predictions:
        output[token] = []
        for prediction_index, original in enumerate(predictions[token]):
            features = np.asarray(feature_map[(token, prediction_index)], dtype=np.float64)
            logit = float(np.dot((features - mean) / std, weights) + bias)
            quality = _sigmoid(logit)
            row = dict(original)
            original_score = _finite(original.get("score"), "score")
            row.update(
                {
                    "score": quality,
                    "detection_score": quality,
                    "geometry_quality": quality,
                    "classification_score": original_score,
                    "original_detection_score": original_score,
                    "quality_ranker_bundle_sha256": bundle["bundle_sha256"],
                }
            )
            output[token].append(row)
    return output


def uncertainty_control_score(classification_score: float, depth_log_variance: float) -> float:
    """Fixed scalar control, not a calibrated probability."""
    score = min(max(_finite(classification_score, "classification_score"), 1e-9), 1 - 1e-9)
    logit = math.log(score / (1 - score)) - 0.5 * _finite(depth_log_variance, "depth_log_variance")
    return _sigmoid(logit)


def monotonic_temperature_score(score: float, *, temperature: float = 2.0) -> float:
    """Strictly monotonic control that cannot alter an AP ranking without ties."""
    value = _finite(score, "score")
    if not 0 < value < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Monotonic control requires score in (0,1) and positive temperature")
    return _sigmoid(math.log(value / (1 - value)) / temperature)


def apply_score_control(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    sidecar: Mapping[str, object],
    *,
    protocol: EvaluationProtocol,
    control: str,
) -> dict[str, list[dict[str, object]]]:
    """Apply one frozen baseline control without changing the proposal pool."""
    if control not in {"baseline", "scalar_uncertainty", "monotonic_temperature"}:
        raise ValueError("Unknown ranking control")
    if protocol.sha256 != EvaluationProtocol.default().sha256:
        raise ValueError("Ranking controls are frozen to the default evaluation protocol")
    if any(len(rows) > protocol.max_predictions_per_frame for rows in predictions.values()):
        raise ValueError("Control pool exceeds the frozen per-frame proposal cap")
    feature_map = _sidecar_query_features(predictions, sidecar, protocol.classes)
    output: dict[str, list[dict[str, object]]] = {}
    log_variance_index = FEATURE_NAMES.index("depth_log_variance")
    for token in predictions:
        output[token] = []
        for prediction_index, original in enumerate(predictions[token]):
            original_score = _finite(original.get("score"), "score")
            if control == "baseline":
                score = original_score
            elif control == "scalar_uncertainty":
                score = uncertainty_control_score(
                    original_score,
                    feature_map[(token, prediction_index)][log_variance_index],
                )
            else:
                score = monotonic_temperature_score(original_score)
            row = dict(original)
            row.update(
                {
                    "score": score,
                    "detection_score": score,
                    "classification_score": original_score,
                    "original_detection_score": original_score,
                    "ranking_control": control,
                }
            )
            output[token].append(row)
    return output


def proposal_geometry_sha256(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    protocol: EvaluationProtocol,
) -> str:
    """Hash ordered proposal geometry/class while deliberately excluding every score."""
    geometry: dict[str, list[dict[str, object]]] = {}
    for token in sorted(predictions):
        if len(predictions[token]) > protocol.max_predictions_per_frame:
            raise ValueError("Proposal pool exceeds the frozen per-frame cap")
        geometry[token] = []
        for index, row in enumerate(predictions[token]):
            box, _ = parse_box(row, source_index=index, classes=protocol.classes, prediction=True)
            value: dict[str, object] = {
                "class_name": box.class_name,
                "translation_m": list(box.translation_m),
                "size_lwh_m": list(box.size_lwh_m),
                "yaw_rad": box.yaw_rad,
            }
            if "velocity_mps" in row:
                value["velocity_mps"] = list(_vector(row["velocity_mps"], 2, "velocity_mps"))
            geometry[token].append(value)
    return _canonical_sha256(geometry)


def score_independent_pool_recall(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, EvaluationGroundTruth],
    *,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    """Maximum-cardinality class-agnostic recall of fixed in-domain geometry.

    Unlike the v2 diagnostic recall, this invariant never iterates proposals by
    detection score. It is therefore suitable for proving that a ranker retained
    the same geometric proposal capacity even when greedy v2 recall changes.
    """
    if not ground_truth or set(predictions) != set(ground_truth):
        raise ValueError("Pool recall requires identical nonempty token populations")
    if any(frame.protocol_sha256 != protocol.sha256 for frame in ground_truth.values()):
        raise ValueError("Pool recall ground truth uses another protocol")
    parsed_by_token: dict[str, list[tuple[float, float]]] = {}
    for token in sorted(predictions):
        if len(predictions[token]) > protocol.max_predictions_per_frame:
            raise ValueError("Proposal pool exceeds the frozen per-frame cap")
        parsed_by_token[token] = []
        frame = ground_truth[token]
        for index, row in enumerate(predictions[token]):
            box, _ = parse_box(row, source_index=index, classes=protocol.classes, prediction=True)
            reason = domain_reason(
                box,
                intrinsic=np.asarray(frame.intrinsic),
                event_from_ego=np.asarray(frame.event_from_ego),
                image_wh=frame.image_wh,
                protocol=protocol,
            )
            if reason is None:
                parsed_by_token[token].append((box.translation_m[0], box.translation_m[1]))

    def maximum_matches(token: str, threshold: float) -> int:
        targets = ground_truth[token].eligible
        edges: list[list[int]] = []
        for x, y in parsed_by_token[token]:
            candidates = [
                (
                    math.hypot(x - target.translation_m[0], y - target.translation_m[1]),
                    target.source_index,
                    target_index,
                )
                for target_index, target in enumerate(targets)
                if math.hypot(x - target.translation_m[0], y - target.translation_m[1]) < threshold
            ]
            edges.append([row[2] for row in sorted(candidates)])
        matched_prediction_by_target: dict[int, int] = {}

        def augment(prediction_index: int, seen: set[int]) -> bool:
            for target_index in edges[prediction_index]:
                if target_index in seen:
                    continue
                seen.add(target_index)
                previous = matched_prediction_by_target.get(target_index)
                if previous is None or augment(previous, seen):
                    matched_prediction_by_target[target_index] = prediction_index
                    return True
            return False

        for prediction_index in range(len(edges)):
            augment(prediction_index, set())
        return len(matched_prediction_by_target)

    eligible_count = sum(len(frame.eligible) for frame in ground_truth.values())
    thresholds: dict[str, object] = {}
    for threshold in protocol.distance_thresholds_m:
        matched = sum(maximum_matches(token, threshold) for token in sorted(predictions))
        thresholds[str(threshold)] = {
            "TP_maximum_cardinality": matched,
            "FN": eligible_count - matched,
            "recall": matched / eligible_count if eligible_count else None,
        }
    return {
        "schema": "phase_ttc_score_independent_pool_recall_v1",
        "proposal_geometry_sha256": proposal_geometry_sha256(predictions, protocol=protocol),
        "eligible_ground_truth_count": eligible_count,
        "thresholds": thresholds,
    }
