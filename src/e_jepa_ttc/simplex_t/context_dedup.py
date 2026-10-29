"""Content identities for the explicitly amended query-conditioned sensor cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .query_context import QueryContextInput, query_context


@dataclass(frozen=True)
class ContextSource:
    """Input-only source identity; no annotation targets or object association."""

    sequence_id: str
    raw_sha256: str
    preprocessing_sha256: str
    current: QueryContextInput


def observation_key(source: ContextSource, lag_us: int) -> str:
    """Bind producer, raw content, preprocessing, crop, timing and ROI availability.

    Query names are deliberately excluded: identical consumed inputs can share a
    cache row. ROI availability is included because gathering uses row-level time.
    """
    for digest in (source.raw_sha256, source.preprocessing_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source digest")
    observations = {row.lag_us: row for row in query_context(source.current, 16)}
    if lag_us not in observations:
        raise ValueError("unavailable or unregistered sensor lag")
    row = observations[lag_us]
    identity = {
        "schema": "simplex_t_query_context_observation_v1",
        "sequence": source.sequence_id,
        "raw": source.raw_sha256,
        "preprocessing": source.preprocessing_sha256,
        "producer": source.current.producer_family_sha256,
        "windows_us": row.windows_us,
        "square_xyxy": [float(v).hex() for v in source.current.square_xyxy],
        "anchor_us": row.anchor_us,
        "available_us": row.available_us,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def deduplicate_contexts(sources: list[ContextSource]) -> tuple[list[str], np.ndarray]:
    """Return stable unique content keys and chronological H16 indices, -1 if absent."""
    keys: list[str] = []
    positions: dict[str, int] = {}
    history = np.full((len(sources), 16), -1, dtype=np.int64)
    for i, source in enumerate(sources):
        for row in query_context(source.current, 16):
            key = observation_key(source, row.lag_us)
            if key not in positions:
                positions[key] = len(keys)
                keys.append(key)
            history[i, 15 - row.lag_us // 50_000] = positions[key]
    return keys, history
