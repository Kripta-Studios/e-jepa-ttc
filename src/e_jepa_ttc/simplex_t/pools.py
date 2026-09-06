"""Target-free, pre-score query pool and producer-family resolution."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryIdentity:
    """Only input identities participate in pool selection; no numeric labels."""

    token: str
    sequence: str
    track: str
    acquisition_group: str

    def __post_init__(self) -> None:
        if not all((self.token, self.sequence, self.track, self.acquisition_group)):
            raise ValueError("complete query identity required")


def balanced_queries(
    rows: list[QueryIdentity], *, limit: int, allowed_groups: set[str]
) -> tuple[QueryIdentity, ...]:
    """Round-robin groups then tracks, with SHA256 token order within a track.

    Group and (sequence, track) orders are lexical. Track identifiers are scoped
    to their sequence even when multiple sequences share an acquisition group.
    Callers must supply an audited query universe, never a history substitute.
    """
    if limit < 0 or len({row.token for row in rows}) != len(rows):
        raise ValueError("negative budget or duplicate token")
    grouped: dict[str, dict[tuple[str, str], list[QueryIdentity]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.acquisition_group not in allowed_groups:
            raise ValueError("query group outside approved TRAIN pool")
        grouped[row.acquisition_group][(row.sequence, row.track)].append(row)
    tracks = {}
    active = deque(sorted(grouped))
    for group in active:
        tracks[group] = deque(
            deque(
                sorted(
                    grouped[group][track],
                    key=lambda row: hashlib.sha256(row.token.encode()).digest(),
                )
            )
            for track in sorted(grouped[group])
        )
    selected = []
    while active and len(selected) < limit:
        group = active.popleft()
        track_rows = tracks[group].popleft()
        selected.append(track_rows.popleft())
        if track_rows:
            tracks[group].append(track_rows)
        if tracks[group]:
            active.append(group)
    return tuple(selected)


@dataclass(frozen=True)
class DataPoolResolution:
    """Availability is a source fact, independent of all candidate scores."""

    available: bool
    reason: str
    queries: tuple[QueryIdentity, ...]
    additional_groups: tuple[str, ...]


def expanded_pool(
    original: tuple[QueryIdentity, ...],
    candidates: list[QueryIdentity],
    *,
    original_train_groups: set[str],
    approved_expansion_groups: set[str],
) -> DataPoolResolution:
    """Keep D0 unchanged and add target-free queries up to the registered 32768."""
    if original_train_groups & approved_expansion_groups:
        raise ValueError("expansion and original acquisition groups overlap")
    original_ids = {row.token for row in original}
    if not original or len(original_ids) != len(original) or len(original) > 32768:
        raise ValueError("invalid original query cohort")
    if any(row.acquisition_group not in original_train_groups for row in original):
        raise ValueError("D0 includes a non-training group")
    if original_ids & {row.token for row in candidates}:
        raise ValueError("expansion reuses original token identity")
    additions = balanced_queries(
        candidates, limit=32768 - len(original), allowed_groups=approved_expansion_groups
    )
    groups = tuple(sorted({row.acquisition_group for row in additions}))
    if len(groups) < 6:
        return DataPoolResolution(False, "FEWER_THAN_SIX_ADDITIONAL_GROUPS", original, groups)
    return DataPoolResolution(True, "AVAILABLE", (*original, *additions), groups)


def canonical_data(folds: tuple[bool, bool, bool]) -> str:
    """D1 is canonical only when all three folds pass before scores."""
    if len(folds) != 3 or any(type(value) is not bool for value in folds):
        raise ValueError("three explicit fold availability decisions required")
    return "D1" if all(folds) else "D0"


def expansion_producer(sequence: str, ordered_families: tuple[str, str, str]) -> str:
    """Assign one existing inner family to an entire expansion sequence/history."""
    if not sequence or len(ordered_families) != 3 or len(set(ordered_families)) != 3:
        raise ValueError("sequence and three distinct ordered families required")
    if any(len(value) != 64 or set(value) - set("0123456789abcdef") for value in ordered_families):
        raise ValueError("exact lowercase family SHA256 identities required")
    index = int.from_bytes(hashlib.sha256(sequence.encode()).digest(), "big") % 3
    return ordered_families[index]


def density_size(*, d0_count: int, dense_old_available: int, diverse_available: int) -> int | None:
    """Resolve the common unique-query count before either matched-density fit."""
    if min(d0_count, dense_old_available, diverse_available) < 0 or d0_count == 0:
        raise ValueError("invalid query counts")
    count = min(32768, dense_old_available, diverse_available)
    return count if count >= 2 * d0_count else None
