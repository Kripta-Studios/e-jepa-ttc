"""Strict Stage 63–65 metrics, paired uncertainty and gate evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BUCKETS: tuple[tuple[str, float, float, float], ...] = (
    ("positive_small", 0.0, 3.0, 0.5),
    ("positive_medium", 3.0, 6.0, 0.3),
    ("positive_large", 6.0, 10.0, 0.1),
    ("negative", -10.0, 0.0, 0.1),
)


def benchmark_phase(ttc_seconds: np.ndarray) -> np.ndarray:
    """Convert scientific TTC to the signed benchmark phase in float64."""

    values = np.asarray(ttc_seconds, dtype=np.float64)
    if not np.isfinite(values).all() or not np.all((values < 0.0) | (values > 0.1)):
        raise ValueError("TTC is outside the real benchmark-phase domain")
    return -np.log1p(-0.1 / values)


def scientific_ttc(phase: np.ndarray) -> np.ndarray:
    """Invert phase without compatibility clipping or row dropping."""

    values = np.asarray(phase, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("phase must be finite")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = 1.0 / (-np.expm1(-values) / 0.1)
    if not np.isfinite(result).all() or np.any(np.abs(result) < 0.1):
        raise ValueError("scientific TTC inversion produced an invalid row")
    if not np.allclose(benchmark_phase(result), values, rtol=0, atol=1e-7):
        raise ArithmeticError("scientific TTC phase round-trip failed")
    return result


def scientific_mid_per_row(target_ttc: np.ndarray, prediction_ttc: np.ndarray) -> np.ndarray:
    """Return unaggregated 10,000× absolute benchmark-phase error."""

    target = benchmark_phase(target_ttc)
    prediction = benchmark_phase(prediction_ttc)
    return 10_000.0 * np.abs(target - prediction)


def strict_macro_mass(target_ttc: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Return frozen global sequence/bucket mass, rejecting incomplete support."""

    target = np.asarray(target_ttc, dtype=np.float64).reshape(-1)
    sequence = np.asarray(sequences, dtype=str).reshape(-1)
    if target.shape != sequence.shape or len(target) == 0:
        raise ValueError("target/sequence shape mismatch")
    benchmark_phase(target)
    unique = np.unique(sequence)
    mass = np.zeros(len(target), dtype=np.float64)
    covered = np.zeros(len(target), dtype=bool)
    for seq in unique:
        for _, lower, upper, bucket_weight in BUCKETS:
            selected = (sequence == seq) & (target > lower) & (target <= upper)
            count = int(selected.sum())
            if count == 0:
                raise ValueError(f"sequence {seq} lacks bucket ({lower},{upper}]")
            mass[selected] = bucket_weight / (len(unique) * count)
            covered |= selected
    if not covered.all() or not np.isclose(mass.sum(), 1.0, atol=1e-12):
        raise ValueError("strict signed target universe is incomplete")
    return mass


def strict_score(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the local sequence-macro signed MiD with no missing-value omission."""

    required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s"}
    if not required <= set(frame) or len(frame) == 0:
        raise ValueError("prediction frame schema is incomplete")
    if frame["sample_token"].duplicated().any():
        raise ValueError("prediction frame contains duplicate tokens")
    for column, forbidden in (("failure", True), ("finite", False)):
        if column in frame:
            flags = frame[column]
            values = np.asarray(flags)
            if not all(isinstance(value, (bool, np.bool_)) for value in values):
                raise ValueError(f"prediction {column} flags must be explicit booleans")
            if bool(np.any(values == forbidden)):
                raise ValueError(f"prediction frame contains explicit {column} violations")
    prediction = frame["prediction_ttc_s"].to_numpy(np.float64)
    target = frame["target_ttc_s"].to_numpy(np.float64)
    if not np.isfinite(prediction).all() or np.any(np.abs(prediction) < 0.1):
        raise ValueError("prediction frame contains invalid scientific TTC")
    error = scientific_mid_per_row(target, prediction)
    mass = strict_macro_mass(target, frame["sequence_id"].astype(str).to_numpy())
    score = float(np.dot(mass, error))
    per_sequence: dict[str, Any] = {}
    for sequence in sorted(frame["sequence_id"].astype(str).unique()):
        selected = frame["sequence_id"].astype(str).eq(sequence).to_numpy()
        sequence_mass = strict_macro_mass(target[selected], np.repeat("sequence", selected.sum()))
        per_sequence[sequence] = float(np.dot(sequence_mass, error[selected]))
    return {
        "metric": "strict_sequence_macro_signed_garl_release_MiD",
        "score": score,
        "rows": len(frame),
        "tokens_unique": int(frame["sample_token"].nunique()),
        "sequences": int(frame["sequence_id"].nunique()),
        "tracks": int(frame[["sequence_id", "track_id"]].drop_duplicates().shape[0]),
        "finite_fraction": 1.0,
        "failure_rate": 0.0,
        "per_sequence": per_sequence,
    }


def align_prediction_frames(
    candidate: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort and verify paired token/target/sequence/track identity."""

    left = candidate.sort_values("sample_token").reset_index(drop=True)
    right = reference.sort_values("sample_token").reset_index(drop=True)
    identity = ["sample_token", "sequence_id", "track_id", "target_ttc_s"]
    if "outer_fold" in left or "outer_fold" in right:
        if "outer_fold" not in left or "outer_fold" not in right:
            raise ValueError("paired prediction outer_fold is missing")
        if not left["outer_fold"].equals(right["outer_fold"]):
            raise ValueError("paired prediction outer_fold mismatch")
    if len(left) != len(right) or any(not left[key].equals(right[key]) for key in identity[:3]):
        raise ValueError("paired prediction identity mismatch")
    if not np.array_equal(
        benchmark_phase(left["target_ttc_s"].to_numpy()),
        benchmark_phase(right["target_ttc_s"].to_numpy()),
    ):
        raise ValueError("paired prediction target mismatch")
    return left, right


def validate_campaign_universe(frame: pd.DataFrame, canonical: pd.DataFrame) -> None:
    """Require the complete pinned 8192-token, nine-sequence, three-fold universe."""
    required = {"sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc_s"}
    for label, table in (("canonical", canonical), ("prediction", frame)):
        if not required <= set(table) or len(table) != 8192:
            raise ValueError(f"{label} campaign universe schema/count mismatch")
        if np.asarray(table[list(required)].isna()).any() or table.sample_token.duplicated().any():
            raise ValueError(f"{label} campaign universe missing/duplicate identities")
        if table.sequence_id.nunique() != 9 or set(table.outer_fold) != {0, 1, 2}:
            raise ValueError(f"{label} campaign sequence/fold universe mismatch")
        if (table.groupby("sequence_id").outer_fold.nunique() != 1).any():
            raise ValueError(f"{label} campaign sequence crosses outer folds")
        if (table.groupby("outer_fold").sequence_id.nunique() != 3).any():
            raise ValueError(f"{label} campaign outer-dev must contain three sequences")
    align_prediction_frames(frame, canonical)


@dataclass(frozen=True)
class BootstrapResult:
    """Paired hierarchical bootstrap evidence for one candidate-reference delta."""

    point_delta: float
    bootstrap_mean: float
    bootstrap_median: float
    ci95_low: float
    ci95_high: float
    fraction_negative: float
    valid_draws: int
    attempts: int
    discarded: int
    validity_fraction: float
    draws_sha256: str


def paired_hierarchical_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    seed: int = 640065,
    valid_draws: int = 8192,
    max_attempts: int = 32768,
    resource_check: Callable[[], None] | None = None,
) -> BootstrapResult:
    """Bootstrap sequence→complete-track pairs with a shared draw for both arms."""

    left, right = align_prediction_frames(candidate, reference)
    left_score, right_score = strict_score(left)["score"], strict_score(right)["score"]
    target = left["target_ttc_s"].to_numpy(np.float64)
    delta_error = scientific_mid_per_row(
        target, left["prediction_ttc_s"].to_numpy(np.float64)
    ) - scientific_mid_per_row(target, right["prediction_ttc_s"].to_numpy(np.float64))
    sequence_values = sorted(left["sequence_id"].astype(str).unique())
    track_rows: dict[str, list[np.ndarray]] = {}
    bucket_masks: list[np.ndarray] = []
    for _, lower, upper, _ in BUCKETS:
        bucket_masks.append((target > lower) & (target <= upper))
    sequence_array = left["sequence_id"].astype(str).to_numpy()
    track_array = left["track_id"].astype(str).to_numpy()
    for sequence in sequence_values:
        tracks = sorted(np.unique(track_array[sequence_array == sequence]))
        track_rows[sequence] = [
            np.flatnonzero((sequence_array == sequence) & (track_array == track))
            for track in tracks
        ]
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    draw_digest = hashlib.sha256()
    attempts = 0
    while len(deltas) < valid_draws and attempts < max_attempts:
        if resource_check is not None and attempts % 32 == 0:
            resource_check()
        attempts += 1
        sequence_draw = rng.integers(0, len(sequence_values), size=len(sequence_values))
        occurrence_scores: list[float] = []
        encoded: list[Any] = [sequence_draw.tolist()]
        valid = True
        for sequence_index in sequence_draw:
            sequence = sequence_values[int(sequence_index)]
            rows_by_track = track_rows[sequence]
            selected_tracks = rng.integers(0, len(rows_by_track), size=len(rows_by_track))
            encoded.append(selected_tracks.tolist())
            selected_rows = np.concatenate([rows_by_track[int(index)] for index in selected_tracks])
            value = 0.0
            for bucket_index, (_, _, _, weight) in enumerate(BUCKETS):
                rows = selected_rows[bucket_masks[bucket_index][selected_rows]]
                if len(rows) == 0:
                    valid = False
                    break
                value += weight * float(delta_error[rows].mean())
            if not valid:
                break
            occurrence_scores.append(value)
        draw_digest.update(json.dumps(encoded, separators=(",", ":")).encode("utf-8"))
        if valid:
            deltas.append(float(np.mean(occurrence_scores)))
    if len(deltas) != valid_draws:
        raise RuntimeError(f"BOOTSTRAP_SUPPORT_BLOCKED: valid={len(deltas)} attempts={attempts}")
    validity = len(deltas) / attempts
    if validity < 0.99:
        raise RuntimeError(f"BOOTSTRAP_SUPPORT_BLOCKED: validity_fraction={validity}")
    values = np.asarray(deltas, dtype=np.float64)
    return BootstrapResult(
        point_delta=float(left_score - right_score),
        bootstrap_mean=float(values.mean()),
        bootstrap_median=float(np.median(values)),
        ci95_low=float(np.percentile(values, 2.5)),
        ci95_high=float(np.percentile(values, 97.5)),
        fraction_negative=float(np.mean(values < 0.0)),
        valid_draws=len(values),
        attempts=attempts,
        discarded=attempts - len(values),
        validity_fraction=validity,
        draws_sha256=draw_digest.hexdigest(),
    )


def evaluate_gate(
    result: BootstrapResult,
    *,
    point_delta_lte: float | None = None,
    point_delta_lt: float | None = None,
    ci95_high_lt: float = 0.0,
    fraction_negative_gte: float = 0.95,
) -> dict[str, Any]:
    """Evaluate a preregistered conjunction without rounding."""

    checks = {
        "point_delta_lte": point_delta_lte is None or result.point_delta <= point_delta_lte,
        "point_delta_lt": point_delta_lt is None or result.point_delta < point_delta_lt,
        "ci95_high_lt": result.ci95_high < ci95_high_lt,
        "fraction_negative_gte": result.fraction_negative >= fraction_negative_gte,
    }
    return {"passed": all(checks.values()), "checks": checks, "statistics": result.__dict__}


def next_protocol_action(
    *,
    stage63_integrity: bool,
    stage63_training_ready: bool,
    raw_available_or_supported: bool,
    stage64_seed7: str | None,
    replication: dict[int, str] | None = None,
) -> str:
    """Return the sole authorized next branch of the Stage 63–65 state machine."""

    if not stage63_integrity:
        return "STOP_INTEGRITY_BLOCKED"
    if not stage63_training_ready or not raw_available_or_supported:
        return "RUN_STAGE65"
    if stage64_seed7 is None:
        return "RUN_STAGE64_SEED7"
    if stage64_seed7 != "RAW_ALL_GATES_PASSED":
        if stage64_seed7 == "RAW_GATES_FAILED":
            return "RUN_STAGE65"
        return "STOP_RAW_TECHNICAL_OR_INTEGRITY_FAILURE"
    outcomes = replication or {}
    if set(outcomes) - {13, 23}:
        raise ValueError("replication outcomes contain an unauthorized seed")
    if any(value != "RAW_ALL_GATES_PASSED" for value in outcomes.values()):
        return "STOP_RAW_REPLICATION_NOT_CONFIRMED"
    if 13 not in outcomes or 23 not in outcomes:
        return "RUN_STAGE64_REPLICATIONS"
    if all(outcomes[seed] == "RAW_ALL_GATES_PASSED" for seed in (13, 23)):
        return "STOP_RAW_CONDITIONAL_REPLICATION_COMPLETE"
    return "STOP_RAW_REPLICATION_NOT_CONFIRMED"


__all__ = [
    "BUCKETS",
    "BootstrapResult",
    "align_prediction_frames",
    "benchmark_phase",
    "evaluate_gate",
    "next_protocol_action",
    "paired_hierarchical_bootstrap",
    "scientific_mid_per_row",
    "scientific_ttc",
    "strict_macro_mass",
    "strict_score",
]
