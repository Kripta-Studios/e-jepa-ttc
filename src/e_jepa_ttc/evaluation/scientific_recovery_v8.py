"""Fail-closed contracts and analysis primitives for Scientific Recovery V8.

This module deliberately contains no dataset access and no model execution.  Replay
and aggregation entry points use these functions to prove that comparisons are made
on the same OOF population, with the signed Garl-TTC metric and cluster-aware
inference defined before results are inspected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.evaluation.garl_ttc_protocol import sequence_macro_signed_metrics
from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleReplayControl,
    CausalScaleTTC,
    CausalScaleTTCConfig,
    CausalScaleTTCOutput,
)

OOF_V8_REQUIRED_COLUMNS: tuple[str, ...] = (
    "token_id",
    "sequence_id",
    "track_id",
    "outer_fold",
    "seed",
    "target_ttc",
    "sample_weight",
    "prediction_ttc",
    "prediction_log_variance",
    "finite",
    "failure_reason",
    "event_count",
    "event_rate",
    "support_ms",
    "model_name",
    "config_sha256",
    "checkpoint_sha256",
)

# The replay contract carries every mechanism quantity named in CODEX_HANDOFF §10.2.
# Vector-valued tensors are serialized by the replay runner as deterministic JSON
# strings (or another scalar diagnostic derived from them); this module only checks
# their column presence, leaving the storage encoding deliberately explicit to it.
REPLAY_MECHANISM_REQUIRED_COLUMNS: tuple[str, ...] = (
    "known_mask",
    "sensor_support",
    "guard_margin",
    "pair_log_height_ratio",
    "analytic_log_height_ratio",
    "residual_log_height_ratio",
    "pair_ttc",
    "pair_inverse_ttc",
    "pair_current_contribution",
    "pair_previous_contribution",
    "blend_output",
    "foreground_mass",
    "effective_mass",
    "geometry_tokens",
    "pair_tokens",
    "transport_raw",
    "transport_tokens",
    "endpoint_feature_norm",
    "occupancy",
    "occupancy_entropy",
    "motion_magnitude",
    "cycle_consistency",
)

AGGREGATE_V8_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "status",
    "git_commit",
    "protocol_sha256",
    "config_sha256",
    "seed",
    "folds",
    "row_identity_sha256",
    "target_sha256",
    "prediction_sha256",
    "checkpoint_sha256",
    "metrics",
    "per_sequence",
    "per_bucket",
    "bootstrap",
    "integrity_checks",
    "gate_decision",
    "artifact_sha256",
)

IDENTITY_COLUMNS: tuple[str, ...] = ("token_id", "sequence_id", "track_id")
TARGET_IDENTITY_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, "target_ttc")

# These names are part of the frozen aggregate contract.  A partial mapping is
# never evidence of OOF integrity: the gate must fail closed until every item has
# been produced by the replay/aggregation runner and is explicitly true.
REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS: frozenset[str] = frozenset(
    {
        "row_identity_exact",
        "fold_identity_exact",
        "target_identity_exact",
        "sample_weight_identity_exact",
        "protocol_hash_exact",
        "config_hash_exact",
        "checkpoint_hash_exact",
        "prediction_hash_exact",
        "causality_preserved",
        "future_prefix_invariance",
        "coverage_contract_verified",
    }
)


@dataclass(frozen=True)
class GeneralGateConfig:
    """Frozen V8 screen thresholds for a candidate versus its matched A5 control."""

    delta_mid_max: float = -3.0
    probability_candidate_lower_mid_min: float = 0.90
    required_finite_fraction: float = 1.0
    max_failure_rate_pct: float = 0.0
    max_coverage_drop: float = 0.01


@dataclass(frozen=True)
class MechanismRules:
    """Preregistered, conservative predicates used to classify the A5 autopsy."""

    minimum_improvement_mid: float = -3.0
    minimum_dynamic_spearman_abs: float = 0.20
    maximum_sequence_concentration: float = 0.50
    maximum_innocuous_shift_mid: float = 1.0
    maximum_shortcut_dynamic_spearman_abs: float = 0.10
    minimum_regime_complementarity: float = 0.05
    minimum_causal_regime_auroc: float = 0.60


DEFAULT_MECHANISM_RULES = MechanismRules()
DEFAULT_GENERAL_GATE_CONFIG = GeneralGateConfig()


@dataclass(frozen=True)
class FactorialReplayCell:
    """One preregistered A5 replay cell and its graph-level intervention."""

    name: str
    residual_enabled: bool
    transport_enabled: bool
    temporal_blend: str


FACTORIAL_A5_CELLS: tuple[FactorialReplayCell, ...] = (
    FactorialReplayCell("analytic_only", False, False, "current_only"),
    FactorialReplayCell("analytic_residual", True, False, "current_only"),
    FactorialReplayCell("analytic_transport", False, True, "current_only"),
    FactorialReplayCell("analytic_residual_transport", True, True, "current_only"),
    FactorialReplayCell("full", True, True, "trained"),
)


@torch.inference_mode()
def replay_factorial_a5(
    model: CausalScaleTTC,
    events: torch.Tensor,
    delta_t_s: torch.Tensor,
) -> dict[str, CausalScaleTTCOutput]:
    """Replay each frozen A5 factorial cell through the checkpoint graph.

    The model is invoked once per cell.  A cell therefore cannot be produced by
    editing a previous CSV or by copying the baseline prediction after the fact.
    """

    if not isinstance(model, CausalScaleTTC):
        raise TypeError("factorial replay requires a CausalScaleTTC checkpoint model")
    result: dict[str, CausalScaleTTCOutput] = {}
    for cell in FACTORIAL_A5_CELLS:
        result[cell.name] = model(
            events,
            delta_t_s,
            replay_control=CausalScaleReplayControl(
                residual_enabled=cell.residual_enabled,
                transport_enabled=cell.transport_enabled,
                temporal_blend=cell.temporal_blend,  # type: ignore[arg-type]
            ),
        )
    return result


def raw_mid_per_sample(
    target: Iterable[float] | np.ndarray, prediction: Iterable[float] | np.ndarray
) -> np.ndarray:
    """Return the official raw MiD term before signed-bin and sequence weighting.

    This is intentionally byte-for-byte equivalent in formula to
    :func:`scripts.analyze_v5_a8_oof_failure_modes.raw_mid_per_sample`.
    """

    truth = np.asarray(list(target), dtype=np.float64)
    estimate = np.asarray(list(prediction), dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError("target and prediction shapes differ")
    with np.errstate(divide="ignore", invalid="ignore"):
        truth_eta = 1.0 - 0.1 / truth
        estimate_eta = 1.0 - 0.1 / estimate
        return np.abs(np.log(truth_eta) - np.log(estimate_eta)) * 1.0e4


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _canonical_scalar(value: object) -> str:
    """Serialize scalar values without platform-dependent numeric formatting."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "null"
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return "nan"
        if math.isinf(numeric):
            return "inf" if numeric > 0.0 else "-inf"
        return numeric.hex()
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    return str(value)


def _frame_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    _require_columns(frame, columns, label="hash input")
    rows = sorted(
        (
            tuple(_canonical_scalar(row[column]) for column in columns)
            for _, row in frame.loc[:, list(columns)].iterrows()
        )
    )
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def row_identity_sha256(frame: pd.DataFrame) -> str:
    """Hash the order-independent V8 row identity universe."""

    return _frame_digest(frame, IDENTITY_COLUMNS)


def target_sha256(frame: pd.DataFrame) -> str:
    """Hash identity plus targets so alignment cannot hide target substitution."""

    return _frame_digest(frame, TARGET_IDENTITY_COLUMNS)


def prediction_sha256(frame: pd.DataFrame, *, prediction_column: str = "prediction_ttc") -> str:
    """Hash identity plus a prediction column, including explicit non-finite values."""

    return _frame_digest(frame, (*IDENTITY_COLUMNS, prediction_column))


def validate_oof_frame(frame: pd.DataFrame, *, label: str = "OOF V8 predictions") -> pd.DataFrame:
    """Validate the V8 OOF row schema without silently filtering failed rows."""

    _require_columns(frame, OOF_V8_REQUIRED_COLUMNS, label=label)
    result = frame.copy()
    if result.empty:
        raise ValueError(f"{label} must contain at least one row")
    for column in IDENTITY_COLUMNS + ("model_name",):
        values = result[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{label} has empty {column}")
    if result["token_id"].astype(str).duplicated().any():
        raise ValueError(f"{label} has duplicate token_id")
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{label} has duplicate token/sequence/track identities")
    for column in ("outer_fold", "seed", "event_count"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"{label} has non-integral {column}")
        if column == "event_count" and (numeric < 0).any():
            raise ValueError(f"{label} has negative event_count")
        result[column] = numeric.astype(np.int64)
    for column in ("target_ttc", "sample_weight", "event_rate", "support_ms"):
        numeric = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} has non-finite {column}")
        if column == "sample_weight" and np.any(numeric <= 0.0):
            raise ValueError(f"{label} has non-positive sample_weight")
        if column in {"event_rate", "support_ms"} and np.any(numeric < 0.0):
            raise ValueError(f"{label} has negative {column}")
        result[column] = numeric
    finite = result["finite"]
    finite_is_boolean = finite.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    if finite.isna().any() or not finite_is_boolean:
        raise ValueError(f"{label} finite must be boolean")
    finite_mask = finite.to_numpy(dtype=bool)
    prediction = pd.to_numeric(result["prediction_ttc"], errors="coerce").to_numpy(dtype=np.float64)
    log_variance = pd.to_numeric(result["prediction_log_variance"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    observed_finite = np.isfinite(prediction)
    if not np.array_equal(finite_mask, observed_finite):
        raise ValueError(f"{label} finite does not match prediction_ttc finiteness")
    if np.any(finite_mask & ~np.isfinite(log_variance)):
        raise ValueError(f"{label} has finite predictions without finite prediction_log_variance")
    reasons = result["failure_reason"].fillna("").astype(str).str.strip()
    if np.any(finite_mask & reasons.ne("")):
        raise ValueError(f"{label} records failure_reason for finite rows")
    if np.any(~finite_mask & reasons.eq("")):
        raise ValueError(f"{label} lacks failure_reason for non-finite rows")
    for column in ("config_sha256", "checkpoint_sha256"):
        if not result[column].map(_is_sha256).all():
            raise ValueError(f"{label} has invalid {column}")
    return result


def validate_replay_frame(
    frame: pd.DataFrame, *, label: str = "V8 mechanism replay"
) -> pd.DataFrame:
    """Validate a replay export with all required mechanism diagnostics present."""

    validated = validate_oof_frame(frame, label=label)
    _require_columns(validated, REPLAY_MECHANISM_REQUIRED_COLUMNS, label=label)
    return validated


def align_oof_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return exact OOF alignment, rejecting missing rows, identity drift or target drift."""

    if not frames:
        raise ValueError("at least one OOF frame is required")
    names = list(frames)
    validated = {name: validate_oof_frame(frame, label=name) for name, frame in frames.items()}
    reference_name = names[0]
    reference = validated[reference_name]
    expected_identity = row_identity_sha256(reference)
    expected_target = target_sha256(reference)
    aligned = reference.loc[:, list(TARGET_IDENTITY_COLUMNS)].copy()
    for name in names:
        current = validated[name]
        if row_identity_sha256(current) != expected_identity:
            raise ValueError(f"{name} row identities differ from {reference_name}")
        if target_sha256(current) != expected_target:
            raise ValueError(f"{name} targets differ from {reference_name}")
        current_prediction = current.loc[:, list(IDENTITY_COLUMNS) + ["prediction_ttc"]].rename(
            columns={"prediction_ttc": f"{name}_prediction_ttc"}
        )
        aligned = aligned.merge(
            current_prediction,
            on=list(IDENTITY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
    if len(aligned) != len(reference):
        raise ValueError("OOF alignment unexpectedly dropped rows")
    return aligned.sort_values(list(IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def validate_counterfactual_identity(
    reference: pd.DataFrame, counterfactual: pd.DataFrame
) -> dict[str, str]:
    """Require an intervention to retain exactly the replay identity and targets."""

    reference_validated = validate_replay_frame(reference, label="reference replay")
    counterfactual_validated = validate_replay_frame(counterfactual, label="counterfactual replay")
    reference_row_hash = row_identity_sha256(reference_validated)
    reference_target_hash = target_sha256(reference_validated)
    if row_identity_sha256(counterfactual_validated) != reference_row_hash:
        raise ValueError("counterfactual row identity differs from reference replay")
    if target_sha256(counterfactual_validated) != reference_target_hash:
        raise ValueError("counterfactual target identity differs from reference replay")
    return {
        "row_identity_sha256": reference_row_hash,
        "target_sha256": reference_target_hash,
        "reference_prediction_sha256": prediction_sha256(reference_validated),
        "counterfactual_prediction_sha256": prediction_sha256(counterfactual_validated),
    }


def _summary(frame: pd.DataFrame, *, prediction_column: str) -> dict[str, float | int]:
    target = frame["target_ttc"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    raw_mid = raw_mid_per_sample(target, prediction)
    finite = np.isfinite(prediction)
    return {
        "rows": int(len(frame)),
        "raw_mid_mean": float(np.nanmean(raw_mid)) if np.isfinite(raw_mid).any() else float("nan"),
        "mae_s": float(np.mean(np.abs(prediction[finite] - target[finite])))
        if np.any(finite)
        else float("nan"),
        "failure_count": int(np.count_nonzero(~finite)),
        "failure_rate_pct": float(np.mean(~finite) * 100.0),
    }


def _quartile_groups(values: pd.Series) -> pd.Series | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 4 or numeric.nunique(dropna=True) < 4:
        return None
    return pd.qcut(numeric, q=4, labels=("q1", "q2", "q3", "q4"), duplicates="drop")


def mechanism_cuts(
    frame: pd.DataFrame, *, prediction_column: str = "prediction_ttc"
) -> dict[str, Any]:
    """Compute prespecified mechanism cuts without using them as model features."""

    validated = validate_replay_frame(frame)
    if prediction_column not in validated.columns:
        raise ValueError(f"mechanism cuts lack prediction column {prediction_column!r}")
    result: dict[str, Any] = {}
    target = validated["target_ttc"].to_numpy(dtype=np.float64)
    cuts: dict[str, pd.Series | np.ndarray] = {
        "ttc_bucket": pd.Series(
            np.select(
                [target > 6.0, target > 3.0, target > 0.0],
                [">6", "3-6", "0-3"],
                default="negative_or_receding",
            ),
            index=validated.index,
        ),
        "sequence": validated["sequence_id"].astype(str),
        "track": validated["track_id"].astype(str),
        "guard_margin": _quartile_groups(validated["guard_margin"]),
        "confidence_or_log_variance": _quartile_groups(validated["prediction_log_variance"]),
    }
    for source, name in (
        ("event_rate", "event_density_quartile"),
        ("motion_magnitude", "motion_magnitude_quartile"),
        ("occupancy_entropy", "occupancy_entropy_quartile"),
    ):
        cuts[name] = _quartile_groups(validated[source])
    if "category" in validated.columns:
        cuts["category_analysis_only"] = validated["category"].astype(str)
    for name, labels in cuts.items():
        if labels is None:
            result[name] = {"status": "insufficient_variation"}
            continue
        labelled = validated.assign(_v8_cut=labels)
        result[name] = {
            str(label): _summary(group, prediction_column=prediction_column)
            for label, group in labelled.groupby("_v8_cut", observed=True, sort=True)
        }
    return result


def classify_mechanism(
    evidence: Mapping[str, float | bool | int], *, rules: MechanismRules = DEFAULT_MECHANISM_RULES
) -> dict[str, Any]:
    """Classify H1/H2/H3 only when its preregistered evidence is complete.

    Missing or non-finite evidence deliberately yields ``INCONCLUSIVE``.  The
    classifier records predicates rather than upgrading a result through prose.
    """

    required = (
        "a5_delta_mid_vs_reference",
        "analytic_dynamic_spearman",
        "residual_dynamic_spearman",
        "sequence_concentration",
        "innocuous_counterfactual_delta_mid",
        "regime_complementarity",
        "causal_regime_auroc",
    )
    missing = [key for key in required if key not in evidence]
    if missing:
        return {
            "decision": "INCONCLUSIVE",
            "reason": "missing_evidence",
            "missing_evidence": missing,
            "rules": asdict(rules),
        }
    values = {key: float(evidence[key]) for key in required}
    if not all(math.isfinite(value) for value in values.values()):
        return {
            "decision": "INCONCLUSIVE",
            "reason": "non_finite_evidence",
            "rules": asdict(rules),
        }
    dynamic_strength = max(
        abs(values["analytic_dynamic_spearman"]), abs(values["residual_dynamic_spearman"])
    )
    h3 = (
        values["regime_complementarity"] >= rules.minimum_regime_complementarity
        and values["causal_regime_auroc"] >= rules.minimum_causal_regime_auroc
    )
    h1 = (
        values["a5_delta_mid_vs_reference"] <= rules.minimum_improvement_mid
        and dynamic_strength >= rules.minimum_dynamic_spearman_abs
        and values["sequence_concentration"] <= rules.maximum_sequence_concentration
        and abs(values["innocuous_counterfactual_delta_mid"]) <= rules.maximum_innocuous_shift_mid
    )
    h2 = values["sequence_concentration"] > rules.maximum_sequence_concentration or (
        dynamic_strength <= rules.maximum_shortcut_dynamic_spearman_abs
        and abs(values["innocuous_counterfactual_delta_mid"]) > rules.maximum_innocuous_shift_mid
    )
    predicates = {"H1": h1, "H2": h2, "H3": h3}
    # H2 indicates a shortcut and must take precedence over a seemingly
    # separable regime result.  H3 then denotes complementary causal experts.
    decision = "H2" if h2 else "H3" if h3 else "H1" if h1 else "INCONCLUSIVE"
    return {
        "decision": decision,
        "reason": "preregistered_rules",
        "predicates": predicates,
        "evidence": values,
        "rules": asdict(rules),
    }


def hierarchical_sequence_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_prediction_column: str = "prediction_ttc",
    reference_prediction_column: str | None = None,
    resamples: int = 5000,
    seed: int = 20260814,
) -> dict[str, Any]:
    """Run deterministic sequence-to-track cluster bootstrap for signed MiD.

    Complete sequences are sampled first.  Tracks are then sampled within each
    selected sequence, and all their rows remain intact.  Replica sequence names
    preserve multiplicity when sequence-macro MiD is evaluated.
    """

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    required = ("sequence_id", "track_id", "target_ttc", candidate_prediction_column)
    if reference_prediction_column is not None:
        required = (*required, reference_prediction_column)
    _require_columns(frame, required, label="hierarchical bootstrap")
    sequences = sorted(frame["sequence_id"].astype(str).unique().tolist())
    if len(sequences) < 2:
        raise ValueError("hierarchical sequence bootstrap requires at least two sequences")
    indexed_groups: dict[str, list[np.ndarray]] = {}
    sequence_values = frame["sequence_id"].astype(str).to_numpy()
    for sequence in sequences:
        positions = np.flatnonzero(sequence_values == sequence)
        subframe = frame.iloc[positions]
        tracks = subframe.groupby("track_id", sort=True).indices
        indexed_groups[sequence] = [
            positions[np.asarray(local_rows, dtype=np.int64)] for local_rows in tracks.values()
        ]
    rng = np.random.default_rng(seed)
    candidate_values = np.empty(resamples, dtype=np.float64)
    reference_values = (
        np.empty(resamples, dtype=np.float64) if reference_prediction_column else None
    )
    target_all = frame["target_ttc"].to_numpy(dtype=np.float64)
    candidate_all = frame[candidate_prediction_column].to_numpy(dtype=np.float64)
    reference_all = (
        frame[reference_prediction_column].to_numpy(dtype=np.float64)
        if reference_prediction_column is not None
        else None
    )
    for repeat in range(resamples):
        selected_sequences = rng.integers(0, len(sequences), size=len(sequences))
        indices: list[np.ndarray] = []
        replica_sequence_ids: list[np.ndarray] = []
        for replica, selected_index in enumerate(selected_sequences):
            sequence = sequences[int(selected_index)]
            tracks = indexed_groups[sequence]
            selected_tracks = rng.integers(0, len(tracks), size=len(tracks))
            for track_index in selected_tracks:
                rows = tracks[int(track_index)]
                indices.append(rows)
                replica_sequence_ids.append(
                    np.full(len(rows), f"{sequence}#{replica}", dtype=object)
                )
        row_indices = np.concatenate(indices)
        replica_ids = np.concatenate(replica_sequence_ids)
        candidate_values[repeat] = float(
            sequence_macro_signed_metrics(
                target_all[row_indices], candidate_all[row_indices], replica_ids
            )["sequence_macro_paper_MiD_overall"]
        )
        if reference_values is not None and reference_all is not None:
            reference_values[repeat] = float(
                sequence_macro_signed_metrics(
                    target_all[row_indices], reference_all[row_indices], replica_ids
                )["sequence_macro_paper_MiD_overall"]
            )
    candidate_finite = candidate_values[np.isfinite(candidate_values)]
    if candidate_finite.size == 0:
        raise ValueError("hierarchical bootstrap produced no finite candidate MiD draws")
    result: dict[str, Any] = {
        "method": "hierarchical_sequence_then_track_cluster_bootstrap",
        "resamples": int(resamples),
        "seed": int(seed),
        "sequence_count": len(sequences),
        "candidate_mid": {
            "lower_95": float(np.quantile(candidate_finite, 0.025)),
            "median": float(np.quantile(candidate_finite, 0.5)),
            "upper_95": float(np.quantile(candidate_finite, 0.975)),
            "finite_draw_fraction": float(candidate_finite.size / resamples),
        },
    }
    if reference_values is not None:
        delta = candidate_values - reference_values
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size == 0:
            raise ValueError("hierarchical bootstrap produced no finite paired MiD draws")
        result["delta_candidate_minus_reference"] = {
            "lower_95": float(np.quantile(finite_delta, 0.025)),
            "median": float(np.quantile(finite_delta, 0.5)),
            "upper_95": float(np.quantile(finite_delta, 0.975)),
            "probability_candidate_lower_mid": float(np.mean(finite_delta < 0.0)),
            "finite_draw_fraction": float(finite_delta.size / resamples),
        }
    return result


def general_gate(
    *,
    candidate_metrics: Mapping[str, float | int],
    baseline_metrics: Mapping[str, float | int],
    bootstrap: Mapping[str, Any],
    integrity_checks: Mapping[str, bool | float | int],
    config: GeneralGateConfig = DEFAULT_GENERAL_GATE_CONFIG,
) -> dict[str, Any]:
    """Apply the frozen V8 screen gate without inferring a result from prose."""

    def metric(values: Mapping[str, float | int], *keys: str) -> float:
        for key in keys:
            if key in values:
                return float(values[key])
        raise ValueError(f"metric requires one of {keys}")

    candidate_mid = metric(candidate_metrics, "sequence_macro_MiD", "sequence_macro_mid")
    baseline_mid = metric(baseline_metrics, "sequence_macro_MiD", "sequence_macro_mid")
    candidate_finite = metric(candidate_metrics, "finite_fraction", "point_finite_fraction")
    candidate_failure = metric(candidate_metrics, "failure_rate_pct", "failure_pct")
    candidate_coverage = metric(candidate_metrics, "coverage", "known_coverage")
    baseline_coverage = metric(baseline_metrics, "coverage", "known_coverage")
    paired = bootstrap.get("delta_candidate_minus_reference", bootstrap)
    if not isinstance(paired, Mapping):
        raise ValueError("bootstrap lacks paired delta summary")
    probability = float(paired["probability_candidate_lower_mid"])
    delta = candidate_mid - baseline_mid
    missing_integrity_checks = sorted(
        REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS - set(integrity_checks)
    )
    required_integrity_checks_true = not missing_integrity_checks and all(
        integrity_checks[name] is True for name in REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS
    )
    all_reported_integrity_checks_true = bool(integrity_checks) and all(
        value is True for value in integrity_checks.values()
    )
    checks = {
        "delta_mid_at_most_minus_3": math.isfinite(delta) and delta <= config.delta_mid_max,
        "probability_candidate_lower_mid": math.isfinite(probability)
        and probability >= config.probability_candidate_lower_mid_min,
        "finite_fraction_is_1": candidate_finite == config.required_finite_fraction,
        "failure_rate_is_0": candidate_failure <= config.max_failure_rate_pct,
        "required_integrity_checks_present": not missing_integrity_checks,
        "required_integrity_checks_true": required_integrity_checks_true,
        "all_reported_integrity_checks_true": all_reported_integrity_checks_true,
        "coverage_drop_at_most_1pp": candidate_coverage
        >= baseline_coverage - config.max_coverage_drop,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "delta_candidate_minus_reference_mid": delta,
        "probability_candidate_lower_mid": probability,
        "missing_integrity_checks": missing_integrity_checks,
        "config": asdict(config),
    }


def validate_aggregate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the mandatory V8 aggregate schema and signed-hash fields."""

    missing = sorted(set(AGGREGATE_V8_REQUIRED_KEYS) - set(payload))
    if missing:
        raise ValueError(f"V8 aggregate lacks required keys: {missing}")
    for key in (
        "protocol_sha256",
        "config_sha256",
        "row_identity_sha256",
        "target_sha256",
        "prediction_sha256",
        "checkpoint_sha256",
        "artifact_sha256",
    ):
        if not _is_sha256(payload[key]):
            raise ValueError(f"V8 aggregate has invalid {key}")
    if not isinstance(payload["folds"], list) or not payload["folds"]:
        raise ValueError("V8 aggregate folds must be a non-empty list")
    for key in (
        "metrics",
        "per_sequence",
        "per_bucket",
        "bootstrap",
        "integrity_checks",
        "gate_decision",
    ):
        if not isinstance(payload[key], Mapping):
            raise ValueError(f"V8 aggregate {key} must be a mapping")
    if not isinstance(payload["seed"], int):
        raise ValueError("V8 aggregate seed must be an integer")
    if not verify_artifact_hash(dict(payload)):
        raise ValueError("V8 aggregate artifact_sha256 signature mismatch")


def aggregate_contract_hashes(frame: pd.DataFrame) -> dict[str, str]:
    """Return the three mandatory reproducibility hashes for one OOF population."""

    validated = validate_oof_frame(frame)
    return {
        "row_identity_sha256": row_identity_sha256(validated),
        "target_sha256": target_sha256(validated),
        "prediction_sha256": prediction_sha256(validated),
    }


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash JSON-compatible metadata for explicit report provenance checks."""

    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a physical replay input or checkpoint."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_causal_scale_replay_checkpoint(
    path: str | Path, *, device: torch.device
) -> CausalScaleTTC:
    """Load a real, state-dict-bearing causal-scale checkpoint fail-closed."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("causal-scale replay checkpoint must be a mapping")
    if not isinstance(payload.get("model_config"), Mapping):
        raise ValueError("causal-scale replay checkpoint lacks model_config")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("causal-scale replay checkpoint lacks model_state_dict")
    model = CausalScaleTTC(CausalScaleTTCConfig(**dict(payload["model_config"])))
    model.load_state_dict(dict(state), strict=True)
    return model.to(device).eval()


def _payload_tensor(
    payload: Mapping[str, Any], key: str, *, dtype: torch.dtype
) -> torch.Tensor:
    if key not in payload:
        raise ValueError(f"replay input lacks {key!r}")
    tensor = torch.as_tensor(payload[key], dtype=dtype)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"replay input {key!r} contains non-finite values")
    return tensor


def validate_causal_scale_replay_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable, event-only replay payload before inference.

    Payloads are materialized from the frozen cache, not from V7 prediction CSVs.
    The narrow format makes the exact input rows independently hashable and lets a
    CPU smoke test exercise the same checkpoint path as a full replay.
    """

    events = _payload_tensor(payload, "events", dtype=torch.float32)
    delta = _payload_tensor(payload, "delta_t_s", dtype=torch.float32)
    target = _payload_tensor(payload, "target_ttc", dtype=torch.float32).reshape(-1)
    weight = _payload_tensor(payload, "sample_weight", dtype=torch.float32).reshape(-1)
    if events.ndim != 5 or events.shape[1] not in {2, 3}:
        raise ValueError("replay events must have shape [N,steps={2|3},C,H,W]")
    if delta.shape != (events.shape[0], events.shape[1] - 1):
        raise ValueError("replay delta_t_s must have shape [N,steps-1]")
    if bool((delta <= 0).any()) or bool((weight <= 0).any()) or bool((target == 0).any()):
        raise ValueError("replay delta_t_s/weights must be positive and targets non-zero")
    n = int(events.shape[0])
    result: dict[str, Any] = {
        "events": events,
        "delta_t_s": delta,
        "target_ttc": target,
        "sample_weight": weight,
    }
    for key in ("token_id", "sequence_id", "track_id"):
        values = payload.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != n:
            raise ValueError(f"replay input {key!r} must have exactly N identities")
        normalized = [str(value) for value in values]
        if any(not value.strip() for value in normalized):
            raise ValueError(f"replay input {key!r} contains empty identity")
        result[key] = normalized
    if len(set(result["token_id"])) != n:
        raise ValueError("replay input has duplicate token_id")
    for key, default in (("outer_fold", 0), ("seed", 7)):
        values = payload.get(key, [default] * n)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != n:
            raise ValueError(f"replay input {key!r} must have exactly N values")
        result[key] = [int(value) for value in values]
    endpoint = payload.get("endpoint_us")
    if endpoint is not None:
        endpoints = torch.as_tensor(endpoint, dtype=torch.int64)
        if endpoints.shape != (n, events.shape[1]):
            raise ValueError("endpoint_us must have shape [N,steps]")
        if bool((endpoints[:, 1:] <= endpoints[:, :-1]).any()):
            raise ValueError(
                "endpoint_us must be strictly increasing; timestamp rollback is invalid"
            )
        result["endpoint_us"] = endpoints
    return result


def _json_tensor(value: torch.Tensor | None, index: int) -> str:
    if value is None:
        return "[]"
    return json.dumps(
        value[index].detach().float().cpu().reshape(-1).tolist(), separators=(",", ":")
    )


def _last(value: torch.Tensor, index: int) -> float:
    current = value[index]
    return float(current.reshape(-1)[-1].detach().float().cpu())


def replay_output_frame(
    output: CausalScaleTTCOutput,
    replay_input: Mapping[str, Any],
    *,
    model_name: str,
    config_sha256: str,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    """Export a model output with every §10.2 diagnostic from actual tensors."""

    values = validate_causal_scale_replay_input(replay_input)
    n = len(values["token_id"])
    if output.ttc_mean_seconds.shape != (n,):
        raise ValueError("checkpoint output batch size does not match replay input")
    events = values["events"]
    rows: list[dict[str, Any]] = []
    foreground = torch.sigmoid(output.foreground_logits)
    for index in range(n):
        known = bool(output.known_mask[index].detach().cpu())
        predicted = float(output.ttc_mean_seconds[index].detach().float().cpu())
        finite = bool(known and math.isfinite(predicted))
        event_count = int(torch.count_nonzero(events[index]).detach().cpu())
        support_ms = float(values["delta_t_s"][index].sum().detach().cpu() * 1_000.0)
        event_rate = float(event_count / max(support_ms, 1.0e-6))
        diagnostics = output.diagnostics
        transport_cycle = diagnostics.get("transport_cycle_error")
        rows.append(
            {
                "token_id": values["token_id"][index],
                "sequence_id": values["sequence_id"][index],
                "track_id": values["track_id"][index],
                "outer_fold": values["outer_fold"][index],
                "seed": values["seed"][index],
                "target_ttc": float(values["target_ttc"][index]),
                "sample_weight": float(values["sample_weight"][index]),
                "prediction_ttc": predicted if finite else float("nan"),
                "prediction_log_variance": _last(output.ttc_log_variance, index)
                if finite
                else float("nan"),
                "finite": finite,
                "failure_reason": "" if finite else "no_known_causal_support",
                "event_count": event_count,
                "event_rate": event_rate,
                "support_ms": support_ms,
                "model_name": model_name,
                "config_sha256": config_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "known_mask": known,
                "sensor_support": _last(output.sensor_support, index),
                "guard_margin": _last(output.pair_log_height_ratio.abs(), index),
                "pair_log_height_ratio": _last(output.pair_log_height_ratio, index),
                "analytic_log_height_ratio": _last(output.analytic_log_height_ratio, index),
                "residual_log_height_ratio": _last(output.residual_log_height_ratio, index),
                "pair_ttc": _last(output.pair_ttc_seconds, index),
                "pair_inverse_ttc": _last(output.pair_inverse_ttc, index),
                "pair_current_contribution": _last(output.pair_inverse_ttc, index),
                "pair_previous_contribution": float(
                    output.pair_inverse_ttc[index, -2].detach().cpu()
                )
                if output.pair_inverse_ttc.shape[1] > 1
                else 0.0,
                "blend_output": float(output.inverse_ttc_mean[index].detach().cpu()),
                "foreground_mass": float(foreground[index, -1].mean().detach().cpu()),
                "effective_mass": _last(output.diagnostics["pair_sensor_support"], index),
                "geometry_tokens": _json_tensor(output.geometry_tokens, index),
                "pair_tokens": _json_tensor(output.pair_tokens, index),
                "transport_raw": _json_tensor(output.transport_raw_features, index),
                "transport_tokens": _json_tensor(output.transport_tokens, index),
                "endpoint_feature_norm": float(
                    output.geometry_tokens[index, -1].norm().detach().cpu()
                ),
                "occupancy": float((events[index, -1].abs() > 1.0e-8).float().mean().cpu()),
                "occupancy_entropy": float(
                    -(foreground[index, -1].clamp(1e-6, 1 - 1e-6)
                    * foreground[index, -1].clamp(1e-6, 1 - 1e-6).log()
                    + (1 - foreground[index, -1].clamp(1e-6, 1 - 1e-6))
                    * (1 - foreground[index, -1].clamp(1e-6, 1 - 1e-6)).log()).mean().detach().cpu()
                ),
                "motion_magnitude": float(
                    (events[index, -1] - events[index, -2]).square().mean().sqrt().detach().cpu()
                ),
                "cycle_consistency": _last(transport_cycle, index)
                if transport_cycle is not None
                else float("nan"),
            }
        )
    return validate_replay_frame(pd.DataFrame(rows))


def assert_causal_prefix_invariance(
    model: CausalScaleTTC, events: torch.Tensor, delta_t_s: torch.Tensor
) -> None:
    """Prove that the first pair's output is invariant to an appended future endpoint."""

    if events.shape[1] != 3:
        raise ValueError("prefix invariance requires exactly three causal endpoints")
    if model.config.foreground_temporal_smoothing_mode == "symmetric_legacy":
        raise ValueError("prefix invariance rejects symmetric_legacy smoothing")
    with torch.inference_mode():
        prefix = model(events[:, :2], delta_t_s[:, :1])
        complete = model(events, delta_t_s)
    for label, before, after in (
        (
            "pair_log_height_ratio",
            prefix.pair_log_height_ratio[:, 0],
            complete.pair_log_height_ratio[:, 0],
        ),
        (
            "analytic_log_height_ratio",
            prefix.analytic_log_height_ratio[:, 0],
            complete.analytic_log_height_ratio[:, 0],
        ),
        (
            "residual_log_height_ratio",
            prefix.residual_log_height_ratio[:, 0],
            complete.residual_log_height_ratio[:, 0],
        ),
        (
            "pair_inverse_ttc",
            prefix.pair_inverse_ttc[:, 0],
            complete.pair_inverse_ttc[:, 0],
        ),
    ):
        if not torch.allclose(before, after, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError(f"future-prefix invariance failed for {label}")


__all__ = [
    "AGGREGATE_V8_REQUIRED_KEYS",
    "IDENTITY_COLUMNS",
    "OOF_V8_REQUIRED_COLUMNS",
    "REPLAY_MECHANISM_REQUIRED_COLUMNS",
    "REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS",
    "GeneralGateConfig",
    "MechanismRules",
    "FACTORIAL_A5_CELLS",
    "FactorialReplayCell",
    "aggregate_contract_hashes",
    "align_oof_frames",
    "canonical_json_sha256",
    "assert_causal_prefix_invariance",
    "classify_mechanism",
    "general_gate",
    "hierarchical_sequence_bootstrap",
    "mechanism_cuts",
    "prediction_sha256",
    "raw_mid_per_sample",
    "load_causal_scale_replay_checkpoint",
    "replay_factorial_a5",
    "replay_output_frame",
    "sha256_file",
    "row_identity_sha256",
    "target_sha256",
    "validate_aggregate_payload",
    "validate_counterfactual_identity",
    "validate_oof_frame",
    "validate_causal_scale_replay_input",
    "validate_replay_frame",
]
