"""Deterministic sequence selection for bounded public-eAP pilots."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EAP_PILOT_CATEGORIES = (
    "car",
    "truck",
    "motorcycle",
    "pedestrian",
    "bus",
    "tricycle",
    "bicycle",
    "other_vehicle",
)
EAP_PILOT_TTC_BUCKETS = (
    "approaching_0p1_2s",
    "approaching_2_4s",
    "approaching_4_8s",
    "approaching_8_20s",
    "receding_0p1_20s",
)


def eap_pilot_feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    """Encode one inventory row without using EvTTC or public-test outcomes."""

    category_counts = row["category_counts"]
    ttc_counts = row["ttc_proxy_counts"]
    projected = max(int(row["projected_state_count"]), 1)
    finite_ttc = max(
        sum(int(ttc_counts.get(name, 0)) for name in EAP_PILOT_TTC_BUCKETS),
        1,
    )
    return np.asarray(
        [
            math.log1p(int(row["event_count"])),
            math.log1p(int(row["label_count"])),
            math.log1p(int(row["track_count"])),
            math.log1p(int(row["object_window_count"])),
            *(float(category_counts.get(name, 0)) / projected for name in EAP_PILOT_CATEGORIES),
            *(float(ttc_counts.get(name, 0)) / finite_ttc for name in EAP_PILOT_TTC_BUCKETS),
        ],
        dtype=np.float64,
    )


def _standardized_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    features = np.stack([eap_pilot_feature_vector(row) for row in rows])
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (features - features.mean(axis=0)) / scale


def select_eap_pilot_sequences(
    rows: Sequence[Mapping[str, Any]],
    *,
    sequence_count: int = 12,
    validation_count: int = 3,
    anchor_sequence_ids: Sequence[str] = (),
    maximum_event_gib: float = 20.0,
) -> dict[str, list[str]]:
    """Select diverse sequences and a sequence-disjoint 9/3-style split.

    Existing, externally documented sequence attributes may be supplied as
    anchors. Remaining sequences are chosen by standardized farthest-point
    sampling. Very large event-file outliers are excluded from this bounded
    pilot but remain eligible if ``maximum_event_gib`` is raised explicitly.
    """

    if sequence_count <= 1 or not 0 < validation_count < sequence_count:
        raise ValueError("Pilot and validation counts must define non-empty splits.")
    by_id = {str(row["sequence_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("eAP inventory contains duplicate sequence IDs.")
    eligible = sorted(
        sequence_id
        for sequence_id, row in by_id.items()
        if float(row["event_file_gib"]) <= maximum_event_gib and int(row["object_window_count"]) > 0
    )
    if len(eligible) < sequence_count:
        raise ValueError("Not enough eligible eAP sequences for the requested pilot.")
    unknown_anchors = sorted(set(anchor_sequence_ids) - set(by_id))
    if unknown_anchors:
        raise ValueError(f"Unknown eAP anchor sequence IDs: {unknown_anchors}.")
    selected = [
        sequence_id for sequence_id in dict.fromkeys(anchor_sequence_ids) if sequence_id in eligible
    ][:sequence_count]
    eligible_rows = [by_id[sequence_id] for sequence_id in eligible]
    features = _standardized_features(eligible_rows)
    feature_by_id = dict(zip(eligible, features, strict=True))
    while len(selected) < sequence_count:
        candidates = [sequence_id for sequence_id in eligible if sequence_id not in selected]
        if selected:
            scores = {
                sequence_id: min(
                    float(np.linalg.norm(feature_by_id[sequence_id] - feature_by_id[chosen]))
                    for chosen in selected
                )
                for sequence_id in candidates
            }
        else:
            scores = {
                sequence_id: float(np.linalg.norm(feature_by_id[sequence_id]))
                for sequence_id in candidates
            }
        selected.append(
            min(
                candidates,
                key=lambda sequence_id: (
                    -scores[sequence_id],
                    float(by_id[sequence_id]["event_file_gib"]),
                    sequence_id,
                ),
            )
        )

    selected_features = np.stack([feature_by_id[sequence_id] for sequence_id in selected])
    center = selected_features.mean(axis=0)
    validation = [
        min(
            selected,
            key=lambda sequence_id: (
                float(np.linalg.norm(feature_by_id[sequence_id] - center)),
                sequence_id,
            ),
        )
    ]
    while len(validation) < validation_count:
        candidates = [sequence_id for sequence_id in selected if sequence_id not in validation]
        validation.append(
            min(
                candidates,
                key=lambda sequence_id: (
                    -min(
                        float(np.linalg.norm(feature_by_id[sequence_id] - feature_by_id[chosen]))
                        for chosen in validation
                    ),
                    sequence_id,
                ),
            )
        )
    validation_set = set(validation)
    return {
        "train": sorted(
            sequence_id for sequence_id in selected if sequence_id not in validation_set
        ),
        "validation": sorted(validation),
        "selected": sorted(selected),
        "excluded_large_outliers": sorted(set(by_id) - set(eligible)),
    }


__all__ = [
    "EAP_PILOT_CATEGORIES",
    "EAP_PILOT_TTC_BUCKETS",
    "eap_pilot_feature_vector",
    "select_eap_pilot_sequences",
]
