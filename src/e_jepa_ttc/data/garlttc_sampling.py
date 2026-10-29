"""Deterministic, sequence-aware cache row selection before media materialization."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def signed_ttc_bucket(value: object) -> str:
    """Return the frozen signed Garl bucket for a TTC value."""

    ttc = float(str(value))
    # Match the official release's right-closed pd.cut interval (-10, 0].
    if -10.0 < ttc <= 0.0:
        return "negative"
    if 0.0 < ttc <= 3.0:
        return "crucial"
    if 3.0 < ttc <= 6.0:
        return "small"
    if 6.0 < ttc <= 10.0:
        return "large"
    return "out_of_protocol"


def _value(row: Mapping[str, Any], *names: str, default: str = "unknown") -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name])
    return default


def sampling_stratum(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Build the observable cache stratum without using EvTTC labels."""

    sequence = _value(row, "sequence_id")
    track = _value(row, "public_track_id", "track_id")
    category = _value(row, "category", "category_name")
    ttc_bucket = signed_ttc_bucket(row.get("ttc", 0.0))
    sampling_group = _value(row, "sampling_group")
    return sequence, track, f"{category}:{ttc_bucket}", sampling_group


def _stable_seed(seed: int, role: str, sequence: str) -> int:
    digest = hashlib.sha256(f"{seed}|{role}|{sequence}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def _balanced_sequence_indices(
    frame: pd.DataFrame,
    *,
    role: str,
    seed: int,
    maximum: int | None,
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    if frame.empty:
        return [], {}, {}
    sequence_values = frame["sequence_id"].astype(str)
    sequences = sorted(sequence_values.unique().tolist())
    if maximum is not None and maximum < len(sequences):
        raise ValueError(
            f"max_samples_per_split={maximum} is smaller than the {len(sequences)} "
            f"sequences in split {role!r}; increase the cap."
        )
    candidates: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        candidates[str(row["sequence_id"])].append(int(str(index)))
    ordered: dict[str, list[int]] = {}
    for sequence in sequences:
        strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for index in candidates[sequence]:
            row = frame.loc[index].to_dict()
            _, track, category_bucket, sampling_group = sampling_stratum(row)
            strata[(track, category_bucket, sampling_group)].append(index)
        rng = np.random.default_rng(_stable_seed(seed, role, sequence))
        queues: list[list[int]] = []
        for key in sorted(strata):
            values = strata[key].copy()
            rng.shuffle(values)
            queues.append(values)
        sequence_order: list[int] = []
        while queues:
            next_queues: list[list[int]] = []
            for queue in queues:
                sequence_order.append(queue.pop(0))
                if queue:
                    next_queues.append(queue)
            queues = next_queues
        ordered[sequence] = sequence_order

    selected: list[int] = []
    pointers = {sequence: 0 for sequence in sequences}
    target = sum(len(values) for values in ordered.values()) if maximum is None else maximum
    while len(selected) < target:
        progressed = False
        for sequence in sequences:
            pointer = pointers[sequence]
            values = ordered[sequence]
            if pointer >= len(values):
                continue
            selected.append(values[pointer])
            pointers[sequence] = pointer + 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    candidate_counts: dict[str, int] = defaultdict(int)
    selected_counts: dict[str, int] = defaultdict(int)
    selected_set = set(selected)
    for index, row in frame.iterrows():
        _, track, category_bucket, sampling_group = sampling_stratum(row.to_dict())
        key = f"{role}|{row['sequence_id']}|{track}|{category_bucket}|{sampling_group}"
        candidate_counts[key] += 1
        if int(str(index)) in selected_set:
            selected_counts[key] += 1
    return selected, dict(candidate_counts), dict(selected_counts)


def select_balanced_cache_rows(
    rows: pd.DataFrame,
    sequence_roles: Mapping[str, str],
    *,
    seed: int = 7,
    max_samples_per_split: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select rows round-robin by split/sequence/track/stratum before reading media."""

    if "sequence_id" not in rows.columns:
        raise ValueError("Cache sampling requires a sequence_id column.")
    selected_indices: list[int] = []
    candidate_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for role in sorted(set(sequence_roles.values())):
        role_mask = rows["sequence_id"].astype(str).map(sequence_roles.get) == role
        role_frame = rows.loc[role_mask]
        selected, candidates, chosen = _balanced_sequence_indices(
            role_frame,
            role=role,
            seed=seed,
            maximum=max_samples_per_split,
        )
        selected_indices.extend(selected)
        candidate_counts.update(candidates)
        selected_counts.update(chosen)
    selected_frame = rows.loc[selected_indices].copy()
    selected_frame["_selection_order"] = np.arange(len(selected_frame), dtype=np.int64)
    selected_frame = selected_frame.sort_values("_selection_order", kind="stable").drop(
        columns=["_selection_order"]
    )
    report = {
        "selection_seed": seed,
        "candidate_count_by_split_sequence_track_bucket": candidate_counts,
        "selected_count_by_split_sequence_track_bucket": selected_counts,
        "candidate_count": int(len(rows)),
        "selected_count": int(len(selected_frame)),
        "discard_count": int(len(rows) - len(selected_frame)),
    }
    return selected_frame, report


__all__ = ["sampling_stratum", "select_balanced_cache_rows", "signed_ttc_bucket"]
