"""Sequence-level split generation and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.data.targets import load_ttc_csv
from e_jepa_ttc.data.types import DatasetSequence
from e_jepa_ttc.utils.io import read_structured, write_structured

FINAL_CLAIM_LEVELS = frozenset({"official", "final"})
REUSED_TEST_STATUS = "reused_test_diagnostic"
LEGACY_REUSED_TEST_STATUSES = frozenset({"reused_test"})


def create_sequence_splits(
    sequences: list[DatasetSequence],
    *,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create deterministic sequence-level train/val/test splits."""

    if not sequences:
        msg = "Cannot split an empty manifest."
        raise ValueError(msg)

    ordered = sorted(sequences, key=lambda item: (item.speed_bucket or "", item.sequence_id))
    # Prefer an interpretable local EvTTC mini split when exactly the three CCRs speeds exist.
    by_speed = {sequence.speed_bucket: sequence.sequence_id for sequence in ordered}
    if {"low", "medium", "high"}.issubset(by_speed):
        return {
            "train": [by_speed["low"]],
            "validation": [by_speed["medium"]],
            "test": [by_speed["high"]],
        }

    rng = np.random.default_rng(seed)
    ids = np.array([sequence.sequence_id for sequence in ordered], dtype=object)
    rng.shuffle(ids)
    if len(ids) == 1:
        return {"train": [str(ids[0])], "validation": [], "test": []}
    if len(ids) == 2:
        return {"train": [str(ids[0])], "validation": [str(ids[1])], "test": []}

    n_train = max(1, int(round(len(ids) * 0.6)))
    n_val = max(1, int(round(len(ids) * 0.2)))
    if n_train + n_val >= len(ids):
        n_train = len(ids) - 2
        n_val = 1
    return {
        "train": [str(value) for value in ids[:n_train]],
        "validation": [str(value) for value in ids[n_train : n_train + n_val]],
        "test": [str(value) for value in ids[n_train + n_val :]],
    }


def validate_split_groups(sequences: list[DatasetSequence], splits: dict[str, list[str]]) -> None:
    """Ensure sequence IDs and split groups appear in only one split."""

    by_id = {sequence.sequence_id: sequence for sequence in sequences}
    seen_ids: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    for split_name, ids in splits.items():
        for sequence_id in ids:
            if sequence_id not in by_id:
                msg = f"Split {split_name} references unknown sequence {sequence_id}."
                raise ValueError(msg)
            if sequence_id in seen_ids:
                msg = f"Sequence {sequence_id} appears in {seen_ids[sequence_id]} and {split_name}."
                raise ValueError(msg)
            seen_ids[sequence_id] = split_name
            group = by_id[sequence_id].split_group or sequence_id
            if group in seen_groups:
                msg = f"Split group {group} appears in {seen_groups[group]} and {split_name}."
                raise ValueError(msg)
            seen_groups[group] = split_name


def split_statistics(
    sequences: list[DatasetSequence], splits: dict[str, list[str]]
) -> dict[str, Any]:
    """Compute basic target statistics per split."""

    by_id = {sequence.sequence_id: sequence for sequence in sequences}
    stats: dict[str, Any] = {}
    for split_name, ids in splits.items():
        rows = 0
        targets: list[float] = []
        for sequence_id in ids:
            sequence = by_id[sequence_id]
            ttc_csv = sequence.resolve("ttc_csv")
            if ttc_csv is None:
                continue
            table = load_ttc_csv(ttc_csv)
            rows += int(table["ttc_s"].shape[0])
            targets.extend(float(value) for value in table["ttc_s"])
        stats[split_name] = {
            "sequence_count": len(ids),
            "ttc_rows": rows,
            "ttc_min_s": min(targets) if targets else None,
            "ttc_max_s": max(targets) if targets else None,
            "ttc_mean_s": float(np.mean(targets)) if targets else None,
        }
    return stats


def write_splits(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate, validate and write split file."""

    sequences = read_manifest(manifest_path)
    splits = create_sequence_splits(sequences, seed=seed)
    validate_split_groups(sequences, splits)
    payload = {
        "version": 1,
        "seed": seed,
        "manifest": Path(manifest_path).as_posix(),
        "splits": splits,
        "statistics": split_statistics(sequences, splits),
        "notes": "Local mini split by full sequence; not a final cross-scenario protocol.",
    }
    write_structured(output_path, payload)
    return payload


def read_splits(path: str | Path) -> dict[str, list[str]]:
    """Read split mapping from a split file."""

    data = read_structured(path)
    splits = data.get("splits")
    if not isinstance(splits, dict):
        msg = f"Split file {path} does not contain a split mapping."
        raise ValueError(msg)
    return {str(name): [str(value) for value in values] for name, values in splits.items()}


def read_split_protocol(path: str | Path) -> dict[str, Any]:
    """Read claim-relevant metadata from a sequence split definition.

    Older split files did not declare a status. They remain usable for
    development, but are deliberately not treated as pristine final holdouts.
    """

    data = read_structured(path)
    status = str(data.get("status", "unspecified"))
    if status in LEGACY_REUSED_TEST_STATUSES:
        status = REUSED_TEST_STATUS
    evaluation_role = str(data.get("evaluation_role", "development"))
    raw_allowed = data.get("allowed_claim_levels")
    if raw_allowed is None:
        allowed = ("development", "diagnostic")
    elif isinstance(raw_allowed, list) and all(isinstance(value, str) for value in raw_allowed):
        allowed = tuple(raw_allowed)
    else:
        msg = f"Split file {path} has invalid allowed_claim_levels metadata."
        raise ValueError(msg)
    return {
        "path": Path(path).as_posix(),
        "protocol": data.get("protocol"),
        "status": status,
        "evaluation_role": evaluation_role,
        "allowed_claim_levels": list(allowed),
        "test_was_previously_inspected": bool(data.get("test_was_previously_inspected", False)),
    }


def assert_split_claim_allowed(path: str | Path, *, claim_level: str) -> dict[str, Any]:
    """Fail closed when a split cannot support the requested result claim."""

    claim = claim_level.strip().lower()
    if claim not in {"development", "diagnostic", "official", "final"}:
        msg = f"Unknown claim level {claim_level!r}."
        raise ValueError(msg)
    protocol = read_split_protocol(path)
    allowed = set(protocol["allowed_claim_levels"])
    if claim not in allowed:
        msg = (
            f"Split {path} has status={protocol['status']!r} and evaluation_role="
            f"{protocol['evaluation_role']!r}; it cannot produce a {claim!r} result. "
            f"Allowed claim levels: {sorted(allowed)}."
        )
        raise ValueError(msg)
    if claim in FINAL_CLAIM_LEVELS and (
        protocol["status"] == REUSED_TEST_STATUS
        or protocol["test_was_previously_inspected"]
    ):
        msg = f"Reused/inspected test split {path} cannot produce {claim!r} results."
        raise ValueError(msg)
    return {**protocol, "requested_claim_level": claim, "claim_allowed": True}
