"""Attach pinned TRAIN targets after input-only D1 query selection is complete."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc

from .cache import training_mass
from .training import state_digest


@dataclass(frozen=True)
class ExpansionTargets:
    """Supervision in caller-specified pool order; never an eligibility mask."""

    target_ttc: np.ndarray
    target_phase: np.ndarray
    mass: np.ndarray
    identity_sha256: str


def align_expansion_targets(
    labels: pd.DataFrame,
    *,
    tokens: np.ndarray,
    sequences: np.ndarray,
    timestamps_us: np.ndarray,
    label_sha256: str,
) -> ExpansionTargets:
    """Fail on any missing/invalid target instead of filtering a frozen query pool.

    timestamps_us are the audited metadata's label reference timestamps, NOT
    guessed from shifted historical sensor windows or current ROI availability.
    Call only after role, index, family and pool identity validation.
    """
    if (
        tokens.ndim != 1
        or not len(tokens)
        or len(set(tokens)) != len(tokens)
        or sequences.shape != tokens.shape
        or timestamps_us.shape != tokens.shape
        or timestamps_us.dtype != np.int64
    ):
        raise ValueError("frozen query identity schema mismatch")
    required = {"sample_token", "sequence_id", "timestamp_us", "ttc"}
    if set(labels.columns) != required or labels.sample_token.duplicated().any():
        raise ValueError("TRAIN target table schema or token uniqueness mismatch")
    table = labels.set_index("sample_token")
    if not set(tokens) <= set(table.index):
        raise ValueError("missing target for a frozen TRAIN query")
    selected = table.loc[tokens.tolist()]
    if not np.array_equal(selected.sequence_id.to_numpy(), sequences):
        raise ValueError("TRAIN target sequence differs from frozen input identity")
    times = selected.timestamp_us.to_numpy()
    if times.dtype != np.int64 or not np.array_equal(times, timestamps_us):
        raise ValueError("TRAIN target timestamp differs from audited label reference")
    targets = selected.ttc.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(targets).all():
        raise ValueError("invalid TRAIN target; do not drop frozen queries")
    phases = phase_from_ttc(targets).astype(np.float32)
    mass = training_mass(targets, sequences)
    identity = state_digest(
        {
            "namespace": "SIMPLEX_T_PINNED_D1_TRAIN_SUPERVISION",
            "label_sha256": label_sha256,
            "tokens": tokens.tolist(),
            "sequences": sequences.tolist(),
            "timestamps_us": torch.from_numpy(timestamps_us.copy()),
            "target_phase": torch.from_numpy(phases),
            "mass": torch.from_numpy(mass),
        }
    )
    return ExpansionTargets(targets, phases, mass, identity)


def load_expansion_targets(
    path: Path,
    *,
    expected_sha256: str,
    tokens: np.ndarray,
    sequences: np.ndarray,
    timestamps_us: np.ndarray,
) -> ExpansionTargets:
    """Read only the pinned TRAIN table, with no frame/depth/velocity feature inputs."""
    if len(expected_sha256) != 64 or compute_file_hash(str(path)) != expected_sha256:
        raise ValueError("TRAIN label table differs from its authoritative pin")
    labels = pd.read_parquet(path, columns=["sample_token", "sequence_id", "timestamp_us", "ttc"])
    return align_expansion_targets(
        labels,
        tokens=tokens,
        sequences=sequences,
        timestamps_us=timestamps_us,
        label_sha256=expected_sha256,
    )
