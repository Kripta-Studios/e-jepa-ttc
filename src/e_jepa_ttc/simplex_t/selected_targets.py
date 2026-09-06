"""Project only preauthorized TRAIN query labels from a pinned local Parquet."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .expansion_targets import ExpansionTargets, align_expansion_targets
from .training import state_digest


def load_selected_train_targets(
    path: Path,
    *,
    expected_sha256: str,
    tokens: np.ndarray,
    sequences: np.ndarray,
    timestamps_us: np.ndarray,
    allowed_sequences: set[str],
    pool: str,
) -> ExpansionTargets:
    """Apply query AND sequence projection before returning any target values.

    Caller verifies the authoritative role map and frozen input-only pool first.
    Parquet may scan shared row-group storage internally; only the selected rows
    and four named columns are materialized into the returned label table.
    Invalid/missing selected labels cause failure, never query deletion.
    """
    if pool not in {"DENSE_OLD", "DIVERSE_MATCHED"}:
        raise ValueError("unregistered selected TRAIN target pool")
    if (
        tokens.ndim != 1
        or not len(tokens)
        or len(np.unique(tokens)) != len(tokens)
        or sequences.shape != tokens.shape
        or timestamps_us.shape != tokens.shape
        or timestamps_us.dtype != np.int64
        or not allowed_sequences
        or any(not isinstance(v, str) or not v for v in tokens.tolist())
        or any(not isinstance(v, str) or not v for v in sequences.tolist())
        or not set(sequences) <= allowed_sequences
    ):
        raise ValueError("unauthorized or malformed selected TRAIN identity")
    if (
        len(expected_sha256) != 64
        or set(expected_sha256) - set("0123456789abcdef")
        or compute_file_hash(str(path)) != expected_sha256
    ):
        raise ValueError("selected TRAIN label file pin mismatch")
    labels = pd.read_parquet(
        path,
        columns=["sample_token", "sequence_id", "timestamp_us", "ttc"],
        filters=[
            ("sample_token", "in", tokens.tolist()),
            ("sequence_id", "in", sorted(set(sequences))),
        ],
    )
    if len(labels) != len(tokens) or set(labels.sample_token) != set(tokens):
        raise ValueError("selected target projection returned missing or foreign queries")
    result = align_expansion_targets(
        labels,
        tokens=tokens,
        sequences=sequences,
        timestamps_us=timestamps_us,
        label_sha256=expected_sha256,
    )
    return replace(
        result,
        identity_sha256=state_digest(
            {
                "namespace": "SIMPLEX_T_SELECTED_TRAIN_SUPERVISION",
                "pool": pool,
                "allowed_sequences": sorted(allowed_sequences),
                "aligned_supervision": result.identity_sha256,
            }
        ),
    )
