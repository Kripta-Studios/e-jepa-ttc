"""Paired sequence-to-track bootstrap for canonical E-Clock OOF evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS, PAPER_MID_WEIGHTS

_COLUMNS = (
    "sample_token",
    "sequence_id",
    "track_id",
    "target_ttc_s",
    "scientific_mid_per_row",
)


def _normalize_side(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    missing = sorted(set(_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} bootstrap columns missing: {missing}")
    normalized = frame.loc[:, _COLUMNS].copy()
    if normalized["sample_token"].isna().any() or normalized["sample_token"].duplicated().any():
        raise ValueError(f"{label} sample tokens must be present and unique")
    normalized = normalized.sort_values("sample_token", kind="stable").reset_index(drop=True)
    for column in ("target_ttc_s", "scientific_mid_per_row"):
        numeric = np.asarray(pd.to_numeric(normalized[column], errors="coerce"), dtype=np.float64)
        normalized[column] = numeric
        if not np.isfinite(normalized[column].to_numpy()).all():
            raise ValueError(f"{label} {column} must be finite")
    return normalized


def _bucket_names(target: np.ndarray) -> np.ndarray:
    names = np.full(target.shape, "", dtype=object)
    for name, lower, upper in BUCKETS:
        names[(target > lower) & (target <= upper)] = name
    if np.any(names == ""):
        raise ValueError("bootstrap target lies outside the frozen buckets")
    return names.astype(str)


def _sequence_macro_mid(
    mid: np.ndarray,
    buckets: np.ndarray,
    replica_sequences: np.ndarray,
) -> float:
    sequence_scores: list[float] = []
    for sequence in sorted(np.unique(replica_sequences).tolist()):
        selected = replica_sequences == sequence
        weighted = 0.0
        complete = True
        for bucket, _lower, _upper in BUCKETS:
            values = mid[selected & (buckets == bucket)]
            if values.size == 0:
                complete = False
                break
            weighted += PAPER_MID_WEIGHTS[bucket] * float(np.mean(values, dtype=np.float64))
        if complete:
            sequence_scores.append(weighted)
    if not sequence_scores:
        return float("nan")
    return float(np.mean(np.asarray(sequence_scores), dtype=np.float64))


def _identity_record(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    required = {"reference_family", "path", "file_sha256", "artifact_sha256"}
    if set(value) != required:
        raise ValueError(f"{label} identity must contain exactly {sorted(required)}")
    for key in required:
        item = value[key]
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label} identity field is invalid: {key}")
    return dict(value)


def paired_hierarchical_mid_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
    reference_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bootstrap paired MiD using identical sequence and track draws.

    The seed and number of draws come only from the signed protocol. Every
    selected track contributes all its rows, preserving temporal clustering.
    """

    bootstrap = protocol.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("signed protocol lacks bootstrap configuration")
    seed = bootstrap.get("seed")
    draws = bootstrap.get("draws")
    if not isinstance(seed, int) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("signed bootstrap seed/draw count is invalid")
    if bootstrap.get("method") != "paired_hierarchical_sequence_then_track_cluster_bootstrap":
        raise ValueError("unsupported signed bootstrap method")

    left = _normalize_side(candidate, label="candidate")
    right = _normalize_side(reference, label="reference")
    left_tokens = left["sample_token"].astype(str).tolist()
    right_tokens = right["sample_token"].astype(str).tolist()
    if left_tokens != right_tokens:
        raise ValueError("paired bootstrap requires exactly the same sample tokens")
    for column in ("sequence_id", "track_id"):
        if not np.array_equal(left[column].astype(str), right[column].astype(str)):
            raise ValueError(f"paired bootstrap {column} mismatch")
    if not np.allclose(left["target_ttc_s"], right["target_ttc_s"], rtol=0.0, atol=1.0e-12):
        raise ValueError("paired bootstrap target mismatch")

    sequences = sorted(left["sequence_id"].astype(str).unique().tolist())
    if len(sequences) < 2:
        raise ValueError("hierarchical bootstrap requires at least two sequences")
    sequence_values = left["sequence_id"].astype(str).to_numpy()
    track_values = left["track_id"].astype(str).to_numpy()
    groups: dict[str, list[tuple[str, np.ndarray]]] = {}
    for sequence in sequences:
        tracks = sorted(np.unique(track_values[sequence_values == sequence]).tolist())
        if not tracks:
            raise ValueError(f"sequence has no tracks: {sequence}")
        groups[sequence] = [
            (track, np.flatnonzero((sequence_values == sequence) & (track_values == track)))
            for track in tracks
        ]

    candidate_mid = left["scientific_mid_per_row"].to_numpy(dtype=np.float64)
    reference_mid = right["scientific_mid_per_row"].to_numpy(dtype=np.float64)
    buckets = _bucket_names(left["target_ttc_s"].to_numpy(dtype=np.float64))
    rng = np.random.default_rng(seed)
    delta_values = np.empty(draws, dtype=np.float64)
    candidate_values = np.empty(draws, dtype=np.float64)
    reference_values = np.empty(draws, dtype=np.float64)
    draws_digest = hashlib.sha256()
    for draw in range(draws):
        selected_sequences = rng.integers(0, len(sequences), size=len(sequences))
        draws_digest.update(np.asarray(selected_sequences, dtype="<i8").tobytes())
        row_indices: list[np.ndarray] = []
        replica_ids: list[np.ndarray] = []
        for replica, selected_index in enumerate(selected_sequences):
            sequence = sequences[int(selected_index)]
            tracks = groups[sequence]
            selected_tracks = rng.integers(0, len(tracks), size=len(tracks))
            draws_digest.update(np.asarray(selected_tracks, dtype="<i8").tobytes())
            for track_index in selected_tracks:
                _track, rows = tracks[int(track_index)]
                row_indices.append(rows)
                replica_ids.append(
                    np.full(
                        rows.size,
                        f"{sequence}#s{replica}",
                        dtype=object,
                    )
                )
        selected_rows = np.concatenate(row_indices)
        replicas = np.concatenate(replica_ids)
        candidate_values[draw] = _sequence_macro_mid(
            candidate_mid[selected_rows], buckets[selected_rows], replicas
        )
        reference_values[draw] = _sequence_macro_mid(
            reference_mid[selected_rows], buckets[selected_rows], replicas
        )
        delta_values[draw] = candidate_values[draw] - reference_values[draw]
    finite = (
        np.isfinite(candidate_values) & np.isfinite(reference_values) & np.isfinite(delta_values)
    )
    if not finite.any():
        raise ValueError("paired bootstrap produced no finite paired draws")
    candidate_finite = candidate_values[finite]
    reference_finite = reference_values[finite]
    delta_finite = delta_values[finite]

    artifact = {
        "artifact_type": "eclock_x0_bootstrap_v2",
        "method": bootstrap["method"],
        "cluster_order": ["sequence_id", "track_id"],
        "rows_sampled_as_complete_tracks": True,
        "paired_identical_draws": True,
        "window_level_bootstrap_used": False,
        "seed": seed,
        "draws": draws,
        "draws_identity_sha256": draws_digest.hexdigest(),
        "token_count": len(left),
        "candidate_identity": _identity_record(candidate_identity, label="candidate"),
        "reference_identity": _identity_record(reference_identity, label="reference"),
        "candidate_mid": {
            "mean": float(np.mean(candidate_finite, dtype=np.float64)),
            "median": float(np.median(candidate_finite)),
        },
        "reference_mid": {
            "mean": float(np.mean(reference_finite, dtype=np.float64)),
            "median": float(np.median(reference_finite)),
        },
        "delta_candidate_minus_reference": {
            "mean": float(np.mean(delta_finite, dtype=np.float64)),
            "median": float(np.median(delta_finite)),
            "ci95_low": float(np.quantile(delta_finite, 0.025)),
            "ci95_high": float(np.quantile(delta_finite, 0.975)),
            "probability_delta_lt_zero": float(np.mean(delta_finite < 0.0)),
            "finite_draw_fraction": float(np.mean(finite)),
        },
        "protocol_sha256": str(protocol.get("artifact_sha256", "")),
    }
    return sign_artifact(artifact)


__all__ = ["paired_hierarchical_mid_bootstrap"]
