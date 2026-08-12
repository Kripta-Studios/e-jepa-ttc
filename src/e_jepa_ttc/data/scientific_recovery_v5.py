"""Frozen train-only grouped-development contracts for Scientific Recovery V5."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sized
from typing import Any, Generic, TypeVar, cast

import pandas as pd
from torch.utils.data import Dataset

IDENTITY_COLUMNS = ("sequence_id", "sample_token", "track_id")
SPLIT_ALGORITHM = "sequence_row_count_greedy_sha256_tiebreak_v1"
RecordT = TypeVar("RecordT")


class SequenceIndexedView(Dataset[RecordT], Generic[RecordT]):
    """Read-only sequence-filtered view that preserves shard-local sampling."""

    def __init__(self, dataset: Dataset[RecordT], *, sequence_ids: set[str]) -> None:
        if not isinstance(dataset, Sized):
            raise TypeError("sequence-indexed base dataset must expose length")
        if not sequence_ids:
            raise ValueError("sequence-indexed view requires at least one sequence")
        self.dataset = dataset
        self.sequence_ids = frozenset(str(value) for value in sequence_ids)
        self._base_indices: list[int] = []
        identities: list[dict[str, str]] = []
        for base_index in range(len(dataset)):
            record = dataset[base_index]
            if not isinstance(record, dict):
                raise TypeError("sequence-indexed records must be mappings")
            sequence = str(record.get("sequence_id", ""))
            if sequence not in self.sequence_ids:
                continue
            identity = {column: str(record.get(column, "")) for column in IDENTITY_COLUMNS}
            if any(not value for value in identity.values()):
                raise ValueError(f"base record {base_index} lacks a complete identity")
            self._base_indices.append(base_index)
            identities.append(identity)
        if not self._base_indices:
            raise ValueError("sequence-indexed view selected no rows")
        self._identities = pd.DataFrame(identities)
        if self._identities["sample_token"].duplicated().any():
            raise ValueError("sequence-indexed view contains duplicate sample_token")

        provider = getattr(dataset, "shard_index_groups", None)
        if not callable(provider):
            raise TypeError("sequence-indexed base dataset must expose shard groups")
        base_groups = cast(tuple[tuple[int, ...], ...], provider())
        group_by_base: dict[int, int] = {}
        for group_index, group in enumerate(base_groups):
            for base_index in group:
                if base_index in group_by_base:
                    raise ValueError("base shard groups contain a duplicate index")
                group_by_base[base_index] = group_index
        if sorted(group_by_base) != list(range(len(dataset))):
            raise ValueError("base shard groups must partition the dataset")
        view_groups: list[list[int]] = [[] for _ in base_groups]
        for view_index, base_index in enumerate(self._base_indices):
            view_groups[group_by_base[base_index]].append(view_index)
        self._groups = tuple(tuple(group) for group in view_groups if group)

    def __len__(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index: int) -> RecordT:
        return self.dataset[self._base_indices[index]]

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return filtered groups whose indices partition this view."""

        return self._groups

    def identity_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the view's immutable identity table."""

        return self._identities.copy()


def _require_identities(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks identity columns: {missing}")
    identities = frame.loc[:, IDENTITY_COLUMNS].astype(str).copy()
    if identities["sample_token"].duplicated().any():
        raise ValueError(f"{source} contains duplicate sample_token")
    if (identities == "").any().any():
        raise ValueError(f"{source} contains empty identity values")
    return identities


def _values_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_cache_identities(
    metadata: pd.DataFrame,
    cache_identities: pd.DataFrame,
) -> dict[str, Any]:
    """Require exact token, sequence, and track parity with the materialized cache."""

    expected = _require_identities(metadata, source="train metadata").sort_values(
        "sample_token", kind="stable"
    )
    observed = _require_identities(cache_identities, source="cache identities").sort_values(
        "sample_token", kind="stable"
    )
    expected_tokens = expected["sample_token"].tolist()
    observed_tokens = observed["sample_token"].tolist()
    if expected_tokens != observed_tokens:
        raise ValueError("cache sample-token universe differs from train metadata")
    if expected["sequence_id"].tolist() != observed["sequence_id"].tolist():
        raise ValueError("cache sequence identity differs from train metadata")
    if expected["track_id"].tolist() != observed["track_id"].tolist():
        raise ValueError("cache track identity differs from train metadata")
    return {
        "rows": len(expected),
        "exact_sample_tokens": True,
        "exact_sequence_ids": True,
        "exact_track_ids": True,
        "sorted_sample_tokens_sha256": _values_sha256(expected_tokens),
    }


def build_train_only_grouped_dev(
    metadata: pd.DataFrame,
    *,
    folds: int = 3,
    seed: int = 20260813,
    expected_sequence_count: int = 9,
) -> dict[str, Any]:
    """Assign whole train sequences to deterministic, target-blind dev folds."""

    identities = _require_identities(metadata, source="train metadata")
    counts = Counter(identities["sequence_id"].tolist())
    sequences = sorted(counts)
    if len(sequences) != expected_sequence_count:
        raise ValueError(
            f"Expected {expected_sequence_count} train sequences, found {len(sequences)}"
        )
    if folds < 2 or len(sequences) % folds != 0:
        raise ValueError("fold count must evenly divide the train sequence universe")

    def tie_break(sequence: str) -> str:
        return hashlib.sha256(f"{seed}|{sequence}".encode()).hexdigest()

    ordered = sorted(sequences, key=lambda item: (-counts[item], tie_break(item)))
    dev_groups: list[list[str]] = [[] for _ in range(folds)]
    dev_rows = [0] * folds
    target_sequence_count = len(sequences) // folds
    for sequence in ordered:
        eligible = [
            index for index in range(folds) if len(dev_groups[index]) < target_sequence_count
        ]
        selected = min(
            eligible,
            key=lambda index: (dev_rows[index], len(dev_groups[index]), index),
        )
        dev_groups[selected].append(sequence)
        dev_rows[selected] += counts[sequence]

    fold_rows: list[dict[str, Any]] = []
    validation_counts: Counter[str] = Counter()
    universe = set(sequences)
    for index, values in enumerate(dev_groups):
        dev = sorted(values)
        train = sorted(universe - set(dev))
        if set(train) & set(dev) or set(train) | set(dev) != universe:
            raise RuntimeError(f"fold {index} is not a disjoint universe partition")
        validation_counts.update(dev)
        train_tokens = identities.loc[
            identities["sequence_id"].isin(train), "sample_token"
        ].tolist()
        dev_tokens = identities.loc[identities["sequence_id"].isin(dev), "sample_token"].tolist()
        fold_rows.append(
            {
                "fold": index,
                "train_sequence_ids": train,
                "dev_sequence_ids": dev,
                "train_rows": len(train_tokens),
                "dev_rows": len(dev_tokens),
                "train_sample_tokens_sha256": _values_sha256(train_tokens),
                "dev_sample_tokens_sha256": _values_sha256(dev_tokens),
            }
        )
    if set(validation_counts) != universe or set(validation_counts.values()) != {1}:
        raise RuntimeError("each train sequence must appear in dev exactly once")

    return {
        "protocol_version": "scientific_recovery_v5_train_only_grouped_dev_v1",
        "split_algorithm": SPLIT_ALGORITHM,
        "split_seed": seed,
        "fold_count": folds,
        "sequence_count": len(sequences),
        "sequence_ids": sequences,
        "sample_count": len(identities),
        "sorted_sample_tokens_sha256": _values_sha256(identities["sample_token"].tolist()),
        "sequence_sample_counts": dict(sorted(counts.items())),
        "folds": fold_rows,
    }


__all__ = [
    "IDENTITY_COLUMNS",
    "SPLIT_ALGORITHM",
    "SequenceIndexedView",
    "build_train_only_grouped_dev",
    "validate_cache_identities",
]
