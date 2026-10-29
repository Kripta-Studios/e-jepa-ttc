"""Sequence-disjoint development partitions; these never authorize training.

Only the public training manifest is eligible.  A split is a prospective
development artifact, not evidence that an existing TRAIN40 checkpoint is
independent of its validation or calibration sequences.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

ROLES = ("train", "selection", "calibration")


@dataclass(frozen=True)
class SequenceSplit:
    """Auditable development roles over public training sequences only."""

    schema: str
    dataset_revision: str
    media_sha256: str
    source_split: str
    seed: int
    assignments: dict[str, tuple[str, ...]]
    method: str = "sha256(seed:sequence_id); selection then calibration then train"
    status: str = "CANDIDATE_METADATA_ONLY"
    independence_scope: str = "new detector fits only; not existing TRAIN40 weights"
    grouping: str = "sequence_id; cross-sequence session linkage not verified"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable provenance and explicit role assignments."""
        return asdict(self)


def _identifiers(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError(f"{name} must contain nonempty sequence identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate sequence identifiers")
    return result


def validate_sequence_split(
    split: SequenceSplit,
    *,
    eligible_sequences: Iterable[str],
    forbidden_sequences: Iterable[str] = (),
    session_groups: Mapping[str, str] | None = None,
) -> None:
    """Reject overlap, omission, test sources and known cross-session leakage.

    ``session_groups`` must cover every eligible sequence when supplied.  An
    unknown mapping is not silently advertised as a session-disjoint split.
    """
    eligible = set(_identifiers(eligible_sequences, name="eligible_sequences"))
    if split.schema != "eap_detection_sequence_split_v1" or split.source_split != "train":
        raise ValueError("Only the public train split is eligible for development")
    if re.fullmatch(r"[0-9a-f]{40}", split.dataset_revision) is None:
        raise ValueError("dataset_revision must be a full lowercase commit SHA")
    if re.fullmatch(r"[0-9a-f]{64}", split.media_sha256) is None:
        raise ValueError("media_sha256 must identify the exact input manifest")
    if isinstance(split.seed, bool) or not isinstance(split.seed, int) or split.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if set(split.assignments) != set(ROLES):
        raise ValueError(f"Development roles must be exactly {ROLES}")
    owners: dict[str, str] = {}
    for role in ROLES:
        for sequence in _identifiers(split.assignments[role], name=role):
            if sequence in owners:
                raise ValueError(f"Sequence {sequence} overlaps {owners[sequence]} and {role}")
            owners[sequence] = role
    if set(owners) != eligible:
        raise ValueError("Assignments must cover exactly the eligible training sequences")
    if set(owners) & set(forbidden_sequences):
        raise ValueError("Forbidden/test sequences are present in the development partition")
    if session_groups is not None:
        if set(session_groups) != eligible:
            raise ValueError("Session mapping must cover exactly all eligible sequences")
        group_owners: dict[str, str] = {}
        for sequence, role in owners.items():
            group = session_groups[sequence]
            if not isinstance(group, str) or not group.strip():
                raise ValueError("Session identifiers must be nonempty strings")
            if group in group_owners and group_owners[group] != role:
                raise ValueError(f"Session {group} crosses development roles")
            group_owners[group] = role


def build_sequence_split(
    sequence_ids: Iterable[str],
    *,
    source_split: str,
    dataset_revision: str,
    media_sha256: str,
    seed: int = 7,
    train_count: int = 24,
    selection_count: int = 8,
    calibration_count: int = 8,
) -> SequenceSplit:
    """Construct a deterministic candidate without consulting labels or metrics."""
    sequences = _identifiers(sequence_ids, name="sequence_ids")
    counts = (train_count, selection_count, calibration_count)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError("Each development role needs a positive integer sequence count")
    if sum(counts) != len(sequences):
        raise ValueError("Requested role counts must sum to the manifest sequence count")
    ordered = sorted(
        sequences,
        key=lambda sequence: (hashlib.sha256(f"{seed}:{sequence}".encode()).hexdigest(), sequence),
    )
    boundary = selection_count + calibration_count
    split = SequenceSplit(
        schema="eap_detection_sequence_split_v1",
        dataset_revision=dataset_revision,
        media_sha256=media_sha256,
        source_split=source_split,
        seed=seed,
        assignments={
            "train": tuple(sorted(ordered[boundary:])),
            "selection": tuple(sorted(ordered[:selection_count])),
            "calibration": tuple(sorted(ordered[selection_count:boundary])),
        },
    )
    validate_sequence_split(split, eligible_sequences=sequences)
    return split
