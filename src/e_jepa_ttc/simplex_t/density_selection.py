"""Deterministic time-rank thinning, independent of all target values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def bound_selection_rows(
    binding: dict, index: dict[str, np.ndarray], index_sha256: str
) -> np.ndarray:
    """Revalidate the pinned subset against input-only parent arrays."""
    path = Path(binding["path"])
    if not path.is_absolute() or sha256(path) != binding["sha256"]:
        raise ValueError("D1 density selection reference changed")
    record = json.loads(path.read_text("utf-8"))
    if record["parent_index_sha256"] != index_sha256 or record["per_sequence_cap"] != 512:
        raise ValueError("D1 density selection parent changed")
    rows = np.asarray(record["selected_original_rows"], dtype=np.int64)
    validate_selected_rows(rows, index["sequences"], index["anchor_us"], index["tokens"])
    return rows


def compact_selected_history(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return remapped histories and their sorted original observation IDs."""
    if history.dtype != np.int64 or (history < -1).any():
        raise ValueError("invalid source observation identities")
    source_ids = np.unique(history[history >= 0])
    compact = np.full_like(history, -1)
    valid = history >= 0
    compact[valid] = np.searchsorted(source_ids, history[valid])
    return compact, source_ids


def validate_selected_rows(
    rows: np.ndarray, sequences: np.ndarray, anchor_us: np.ndarray, tokens: np.ndarray
) -> None:
    """Reject edited selections, duplicates and changes to the registered cap."""
    expected = select_time_density(sequences, anchor_us, tokens, per_sequence=512)
    if rows.dtype != np.int64 or not np.array_equal(rows, expected):
        raise ValueError("D1 selection differs from registered input-only time ranks")


def select_time_density(
    sequences: np.ndarray, anchor_us: np.ndarray, tokens: np.ndarray, *, per_sequence: int
) -> np.ndarray:
    """Keep endpoints and evenly spaced chronological ranks per sequence.

    Stable token tie-breaks make selection invariant to source row permutations.
    This thins queries, not the historical sensor dependencies of retained queries.
    """
    if (
        per_sequence < 2
        or sequences.ndim != 1
        or not (sequences.shape == anchor_us.shape == tokens.shape)
    ):
        raise ValueError("aligned input-only vectors and cap >= 2 required")
    if anchor_us.dtype.kind not in "iu" or len(np.unique(tokens)) != len(tokens):
        raise ValueError("integer timestamps and unique query identities required")
    selected = []
    for sequence in np.unique(sequences):
        rows = np.flatnonzero(sequences == sequence)
        rows = rows[np.lexsort((tokens[rows], anchor_us[rows]))]
        count = min(per_sequence, len(rows))
        ranks = (
            np.arange(count, dtype=np.int64) * (len(rows) - 1) // (count - 1)
            if count > 1
            else np.zeros(count, dtype=np.int64)
        )
        selected.extend(rows[ranks].tolist())
    return np.asarray(selected, dtype=np.int64)
