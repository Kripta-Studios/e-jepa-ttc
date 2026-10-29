"""Velocity errors in released label components; no TTC or official AVE claim."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from e_jepa_ttc.detection.development_metrics import CLASSES, _box, _xy_distance


def _velocity(row: Mapping[str, object], *, required: bool) -> tuple[float, float] | None:
    value = row.get("velocity_mps")
    if value is None and not required:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("velocity_mps must contain two native label components")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in value):
        raise ValueError("velocity_mps components must be finite numbers")
    return float(value[0]), float(value[1])


def evaluate_velocity_labels(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    threshold: float = 2.0,
) -> dict[str, object]:
    """Score native velocities after the same confidence/XY box association rule.

    A target without velocity may still consume a box match; it is excluded
    only from velocity errors. Missing predicted velocities are an error rather
    than an assumed zero. All averages below are over actual valid matches.
    """
    if not ground_truth or set(predictions) != set(ground_truth):
        raise ValueError("Prediction tokens must exactly match a nonempty ground-truth set")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Matching threshold must be positive and finite")
    gt = {
        token: [(_box(row, prediction=False), _velocity(row, required=False)) for row in rows]
        for token, rows in ground_truth.items()
    }
    ranked = sorted(
        (
            (token, index, _box(row, prediction=True), _velocity(row, required=True))
            for token, rows in predictions.items()
            for index, row in enumerate(rows)
        ),
        key=lambda item: (-item[2].score, item[0], item[1]),
    )
    used: set[tuple[str, int]] = set()
    errors: list[tuple[float, float, float]] = []
    by_class: dict[str, list[tuple[float, float, float]]] = {name: [] for name in CLASSES}
    for token, _, prediction, velocity in ranked:
        candidates = [
            (_xy_distance(prediction, box), index)
            for index, (box, _) in enumerate(gt[token])
            if box.category == prediction.category and (token, index) not in used
        ]
        match = min(candidates, default=None)
        if match is None or match[0] >= threshold:
            continue
        used.add((token, match[1]))
        target_velocity = gt[token][match[1]][1]
        if target_velocity is None:
            continue
        assert velocity is not None  # Every prediction was validated as required above.
        dx, dy = velocity[0] - target_velocity[0], velocity[1] - target_velocity[1]
        error = (abs(dx), abs(dy), math.hypot(dx, dy))
        errors.append(error)
        by_class[prediction.category].append(error)

    def summarize(rows: Sequence[tuple[float, float, float]]) -> dict[str, object]:
        return {
            "valid_velocity_matches": len(rows),
            "component_mae_mps": [sum(row[i] for row in rows) / len(rows) for i in (0, 1)]
            if rows
            else None,
            "summed_component_mae_mps": sum(row[0] + row[1] for row in rows) / len(rows)
            if rows
            else None,
            "vector_l2_mean_mps": sum(row[2] for row in rows) / len(rows) if rows else None,
        }

    return {
        "profile": "eap_native_velocity_label_matches_v1",
        "official_eap_score": False,
        "semantics": "Errors of the first two released velocity components; not a TTC input",
        "matching_threshold_m": threshold,
        "matching": "Confidence-ranked class/ego-XY matching; distance < threshold",
        "ground_truth_velocity_count": sum(v is not None for rows in gt.values() for _, v in rows),
        "box_matches": len(used),
        **summarize(errors),
        "per_class": {name: summarize(rows) for name, rows in by_class.items()},
    }
