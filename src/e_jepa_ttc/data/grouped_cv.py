"""Deterministic grouped cross-validation for the EvTTC-32 development corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.data.types import DatasetSequence
from e_jepa_ttc.utils.io import write_structured


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_key(sequence: DatasetSequence) -> str:
    return sequence.split_group or sequence.sequence_id


def create_grouped_folds(
    sequences: list[DatasetSequence],
    *,
    folds: int = 5,
    seed: int = 7,
) -> list[dict[str, list[str]]]:
    """Assign whole groups once to validation while balancing family and speed."""

    if folds < 2:
        raise ValueError("folds must be at least two.")
    groups: dict[str, list[DatasetSequence]] = {}
    for sequence in sequences:
        groups.setdefault(_group_key(sequence), []).append(sequence)
    if len(groups) < folds:
        raise ValueError("The number of split groups must be at least the number of folds.")
    rng = np.random.default_rng(seed)
    ordered_groups = sorted(groups)
    rng.shuffle(ordered_groups)
    ordered_groups.sort(
        key=lambda group: (
            -len(groups[group]),
            groups[group][0].scenario_family or "",
            groups[group][0].speed_bucket or "",
        )
    )
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    family_counts = [Counter() for _ in range(folds)]
    speed_counts = [Counter() for _ in range(folds)]
    for group in ordered_groups:
        members = groups[group]
        family = members[0].scenario_family or "unknown"
        speed = members[0].speed_bucket or "unknown"
        selected = min(
            range(folds),
            key=lambda index: (
                family_counts[index][family],
                speed_counts[index][speed],
                sum(len(groups[item]) for item in fold_groups[index]),
                index,
            ),
        )
        fold_groups[selected].append(group)
        family_counts[selected][family] += len(members)
        speed_counts[selected][speed] += len(members)
    all_ids = {sequence.sequence_id for sequence in sequences}
    payload: list[dict[str, list[str]]] = []
    for validation_groups in fold_groups:
        validation = sorted(
            sequence.sequence_id for group in validation_groups for sequence in groups[group]
        )
        train = sorted(all_ids - set(validation))
        payload.append({"train": train, "validation": validation})
    validate_grouped_folds(sequences, payload)
    return payload


def validate_grouped_folds(
    sequences: list[DatasetSequence],
    folds: list[dict[str, list[str]]],
) -> None:
    """Prove disjointness and exactly-once validation coverage."""

    expected = {sequence.sequence_id for sequence in sequences}
    group_by_id = {sequence.sequence_id: _group_key(sequence) for sequence in sequences}
    validation_counts: Counter[str] = Counter()
    for index, fold in enumerate(folds):
        train = set(fold["train"])
        validation = set(fold["validation"])
        if train & validation:
            raise ValueError(f"Fold {index} has train/validation sequence overlap.")
        if train | validation != expected:
            raise ValueError(f"Fold {index} does not cover the complete manifest.")
        train_groups = {group_by_id[sequence_id] for sequence_id in train}
        validation_groups = {group_by_id[sequence_id] for sequence_id in validation}
        if train_groups & validation_groups:
            raise ValueError(f"Fold {index} leaks a split_group.")
        validation_counts.update(validation)
    invalid = {
        sequence_id: validation_counts[sequence_id]
        for sequence_id in expected
        if validation_counts[sequence_id] != 1
    }
    if invalid:
        raise ValueError(f"Validation coverage must be exactly once: {invalid}")


def write_grouped_cv_protocol(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    folds: int = 5,
    seed: int = 7,
    require_sequence_count: int | None = 32,
) -> dict[str, Any]:
    """Create a signed development CV protocol without opening Benchmark-10."""

    assert_no_sealed_benchmark_paths((manifest_path, output_path))
    sequences = read_manifest(manifest_path)
    if require_sequence_count is not None and len(sequences) != require_sequence_count:
        raise ValueError(
            f"Expected {require_sequence_count} EvTTC sequences, found {len(sequences)}."
        )
    fold_splits = create_grouped_folds(sequences, folds=folds, seed=seed)
    rows = []
    by_id = {sequence.sequence_id: sequence for sequence in sequences}
    for index, split in enumerate(fold_splits):
        validation = [by_id[sequence_id] for sequence_id in split["validation"]]
        rows.append(
            {
                "fold": index,
                **split,
                "validation_family_counts": dict(
                    Counter(sequence.scenario_family or "unknown" for sequence in validation)
                ),
                "validation_speed_counts": dict(
                    Counter(sequence.speed_bucket or "unknown" for sequence in validation)
                ),
            }
        )
    payload: dict[str, Any] = {
        "protocol": "evttc32_grouped_cv_v1",
        "role": "development_model_selection",
        "seed": seed,
        "fold_count": folds,
        "sequence_count": len(sequences),
        "manifest": Path(manifest_path).as_posix(),
        "manifest_sha256": _hash_file(manifest_path),
        "benchmark10_opened": False,
        "folds": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_structured(output_path, payload)
    return payload


__all__ = ["create_grouped_folds", "validate_grouped_folds", "write_grouped_cv_protocol"]
