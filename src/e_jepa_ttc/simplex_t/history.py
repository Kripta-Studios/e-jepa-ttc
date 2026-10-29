"""Label-free chronological context construction. No TTC/velocity fields accepted.

Rows have already passed the source/ROI/time charter. Availability is the latest
sensor/annotation dependency, not a claim about a real detector's runtime.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    token: str
    sequence: str
    track: str
    producer: str
    anchor_us: int
    available_us: int
    role: str = "TRAIN"

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, str) and v for v in (self.token, self.sequence, self.track, self.producer)
        ):
            raise ValueError("nonempty observation identity required")
        if not all(isinstance(v, (int, np.integer)) for v in (self.anchor_us, self.available_us)):
            raise ValueError("timestamps must be integer microseconds")
        if self.role not in {"TRAIN", "OLD_DEV"}:
            raise ValueError("protected or unknown observation role")


def verify_ancestry(
    training_groups: set[str], observation_groups: set[str], outer_heldout: set[str]
) -> None:
    """Every ancestor, including pretraining/scalers, supplies its full fit groups."""
    if training_groups & (observation_groups | outer_heldout):
        raise ValueError("producer ancestor has seen observed or outer-held-out groups")


def build_history(
    observations: list[Observation],
    query_indices: np.ndarray,
    length: int = 16,
    max_age_us: int = 2_000_000,
    max_gap_us: int = 250_000,
    min_spacing_us: int = 50_000,
) -> np.ndarray:
    """Right-aligned chronological indices, -1 padding, query always last.

    No cross-track/cross-producer mixing, future dependency, duplicate anchor or
    hidden interpolation. All eligibility is label-independent. Query order stays.
    """
    if length not in {1, 4, 8, 16} or max_age_us <= 0 or not 0 < min_spacing_us <= max_gap_us:
        raise ValueError("invalid frozen history settings")
    if query_indices.ndim != 1 or query_indices.dtype.kind not in "iu":
        raise ValueError("query_indices must be an integer vector")
    if len(set(query_indices.tolist())) != len(query_indices):
        raise ValueError("duplicate query index")
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    identities = set()
    for i, row in enumerate(observations):
        identity = (row.producer, row.token)
        if identity in identities:
            raise ValueError("duplicate producer/token")
        identities.add(identity)
        groups[(row.sequence, row.track, row.producer)].append(i)
    lookup = {}
    for key, ids in groups.items():
        ids.sort(key=lambda i: observations[i].anchor_us)
        times = [observations[i].anchor_us for i in ids]
        if len(set(times)) != len(times):
            raise ValueError("ambiguous duplicate anchor within a track/producer")
        lookup[key] = (ids, times)
    output = np.full((len(query_indices), length), -1, dtype=np.int64)
    for k, qi in enumerate(query_indices.tolist()):
        if qi < 0 or qi >= len(observations):
            raise ValueError("query index out of bounds")
        q = observations[qi]
        ids, times = lookup[(q.sequence, q.track, q.producer)]
        pos = bisect_right(times, q.anchor_us) - 1
        chosen = [qi]
        newest = q.anchor_us
        for j in reversed(ids[:pos]):
            row = observations[j]
            if len(chosen) == length:
                break
            if q.anchor_us - row.anchor_us > max_age_us:
                break
            if row.available_us > q.available_us:
                continue
            gap = newest - row.anchor_us
            if gap > max_gap_us:
                break
            if gap < min_spacing_us:
                continue
            chosen.append(j)
            newest = row.anchor_us
        chosen.reverse()
        output[k, -len(chosen) :] = chosen
    return output


def history_arrays(
    observations: list[Observation], features: np.ndarray, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce features, four observed timing fields in seconds, validity mask.

    Fields: anchor lag, availability lag, gap from previous retained observation,
    availability minus label anchor. Differences are integer BEFORE float casting.
    """
    if features.ndim != 2 or len(features) != len(observations) or index.ndim != 2:
        raise ValueError("feature/history shape mismatch")
    if index.dtype.kind not in "iu" or np.any(index < -1) or np.any(index >= len(observations)):
        raise ValueError("invalid context index")
    valid = index >= 0
    if np.any(~valid[:, -1]) or np.any(valid[:, :-1] & ~valid[:, 1:]):
        raise ValueError("history must be left padded and current observation valid")
    x = np.zeros((*index.shape, features.shape[1]), dtype=np.float32)
    timing = np.zeros((*index.shape, 4), dtype=np.float32)
    for b in range(len(index)):
        q = observations[int(index[b, -1])]
        previous = None
        for t, i in enumerate(index[b]):
            if i < 0:
                continue
            row = observations[int(i)]
            if (row.sequence, row.track, row.producer) != (q.sequence, q.track, q.producer):
                raise ValueError("cross-track or cross-producer context")
            if row.anchor_us > q.anchor_us or row.available_us > q.available_us:
                raise ValueError("future observation dependency")
            if previous is not None and row.anchor_us <= previous:
                raise ValueError("non-increasing history anchors")
            x[b, t] = features[i]
            timing[b, t] = [
                (q.anchor_us - row.anchor_us) / 1e6,
                (q.available_us - row.available_us) / 1e6,
                0.0 if previous is None else (row.anchor_us - previous) / 1e6,
                (row.available_us - row.anchor_us) / 1e6,
            ]
            previous = row.anchor_us
    if not np.isfinite(x).all() or not np.isfinite(timing).all():
        raise ValueError("nonfinite valid history features")
    return x, timing, valid


def feature_bytes(
    unique_rows: int, queries: int, width: int = 17, contexts: int = 16, lineages: int = 1
) -> dict[str, int]:
    if min(unique_rows, queries, width, contexts, lineages) <= 0:
        raise ValueError("positive planning dimensions required")
    return {
        "deduplicated_float32_features": unique_rows * width * 4 * lineages,
        "int32_context_index": queries * contexts * 4 * lineages,
        "naive_duplicated_float32_features": queries * contexts * width * 4 * lineages,
    }
