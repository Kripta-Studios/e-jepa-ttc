"""Canonical, fail-closed scientific protocol checks for E-Clock X0."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import jsonschema
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from torch import nn

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS, PAPER_MID_WEIGHTS

PRODUCTION_ROW_COUNT = 8192
PRODUCTION_FOLDS = (0, 1, 2)
EXECUTABLE_ARMS = ("X0-A5-REPLAY", "X0-PAIR-U", "X0-BASE-U", "X0-DYN-U")
REFERENCE_FAMILIES = (
    "official_a5_oof",
    "official_c2f_oof",
    "nested_router_retrained_a5_constituent",
    "nested_router_retrained_c2f_constituent",
    "prospective_router_r",
)
IDENTITY_HASH_FIELDS = ("sample_token", "sequence_id", "track_id")
ROW_LEVEL_OOF_COLUMNS = (
    "sample_token",
    "sequence_id",
    "track_id",
    "outer_fold",
    "target_ttc_s",
    "target_benchmark_phase",
    "predicted_benchmark_phase",
    "predicted_inverse_ttc_raw",
    "predicted_ttc_raw",
    "predicted_ttc_clipped",
    "is_clip_saturated",
    "scientific_mid_per_row",
    "scientific_failure",
    "sample_weight",
    "arm_id",
    "seed",
    "checkpoint_sha256",
    "config_sha256",
    "protocol_sha256",
    "cache_manifest_sha256",
    "split_manifest_sha256",
)


def read_official_a5_csv(path: Path) -> pd.DataFrame:
    """Read the historical A5 CSV with its signed parser semantics.

    The frozen ``prediction_sha256`` was produced with pandas' high-precision
    parser.  Runner-produced OOF CSVs instead require ``round_trip`` parsing;
    conflating those two artifact classes changes float64 identity by one ULP.
    """

    return pd.read_csv(path, float_precision="high")


def canonical_records_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """Hash ordered typed records after stable sample-token sorting."""

    names = tuple(columns)
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"hash columns missing: {missing}")
    ordered = frame.sort_values("sample_token", kind="stable").loc[:, names]
    records = ordered.to_dict(orient="records")
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_state_sha256(module: nn.Module) -> str:
    """Hash ordered state names, shapes, dtypes and physical tensor bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def module_topology_sha256(module: nn.Module) -> str:
    """Hash learnable parameter names and shapes without their values."""

    payload = [
        {"name": name, "shape": list(parameter.shape), "requires_grad": parameter.requires_grad}
        for name, parameter in module.named_parameters()
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_signed_json(path: Path, *, schema_path: Path | None = None) -> dict[str, Any]:
    """Load one signed JSON artifact and optionally apply its closed schema."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"artifact signature mismatch: {path}")
    if schema_path is not None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(payload)
    return payload


def validate_protocol_reference_binding(
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    protocol_path: Path | None = None,
) -> None:
    """Validate the exact two-way identity used by every scientific operation."""

    if protocol.get("artifact_type") != "scientific_recovery_v9_eclock_protocol_v2":
        raise ValueError("unsupported E-Clock protocol artifact type")
    if reference.get("artifact_type") != "eclock_x0_reference_v2":
        raise ValueError("unsupported E-Clock reference artifact type")
    if not verify_artifact_hash(dict(protocol)) or not verify_artifact_hash(dict(reference)):
        raise ValueError("protocol/reference signature mismatch")
    protocol_record = reference.get("protocol")
    if not isinstance(protocol_record, Mapping):
        raise ValueError("reference lacks protocol binding")
    if protocol_record.get("artifact_sha256") != protocol.get("artifact_sha256"):
        raise ValueError("reference is bound to a different protocol artifact")
    if protocol_path is not None:
        if protocol_record.get("file_sha256") != compute_file_hash(str(protocol_path)):
            raise ValueError("reference is bound to a different protocol physical file")
        if protocol_record.get("bytes") != protocol_path.stat().st_size:
            raise ValueError("reference protocol byte count mismatch")
    registry = reference.get("reference_family_registry")
    families = reference.get("families")
    if registry != list(REFERENCE_FAMILIES) or not isinstance(families, Mapping):
        raise ValueError("reference family registry mismatch")
    if set(families) != set(REFERENCE_FAMILIES):
        raise ValueError("reference must contain exactly five families")
    for name in REFERENCE_FAMILIES:
        family = families.get(name)
        if not isinstance(family, Mapping) or family.get("reference_family") != name:
            raise ValueError("reference family key/identity mismatch")


def require_reference_family(
    reference: Mapping[str, Any], reference_family: str
) -> Mapping[str, Any]:
    """Return one exact family, rejecting aliases and family substitution."""

    if reference_family not in REFERENCE_FAMILIES:
        raise ValueError(f"unknown reference_family: {reference_family}")
    families = reference.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("reference families are missing")
    family = families.get(reference_family)
    if not isinstance(family, Mapping) or family.get("reference_family") != reference_family:
        raise ValueError("reference family identity mismatch")
    return family


def _as_finite_float64(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = np.asarray(pd.to_numeric(frame[column], errors="coerce"), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains non-finite values")
    return values


def _as_strict_bool(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if not all(isinstance(value, (bool, np.bool_)) for value in values.tolist()):
        raise ValueError(f"{column} must contain booleans")
    return values.to_numpy(dtype=bool)


def _target_phase(target_ttc_s: np.ndarray, delta_t_s: float) -> np.ndarray:
    valid = (target_ttc_s < 0.0) | (target_ttc_s > delta_t_s)
    if not valid.all():
        raise ValueError("target TTC is outside the canonical benchmark-phase domain")
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        phase = -np.log1p(-delta_t_s / target_ttc_s)
    if not np.isfinite(phase).all():
        raise ValueError("target benchmark phase is non-finite")
    return phase


def _prediction_coordinates(
    predicted_phase: np.ndarray, delta_t_s: float
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inverse = -np.expm1(-predicted_phase) / delta_t_s
        raw_ttc = np.reciprocal(inverse)
    if not np.isfinite(inverse).all() or not np.isfinite(raw_ttc).all():
        raise ValueError("predicted phase cannot be verified as finite inverse-TTC/raw TTC")
    return inverse, raw_ttc


def _bucket_names(target_ttc_s: np.ndarray) -> np.ndarray:
    result = np.full(target_ttc_s.shape, "", dtype=object)
    for name, lower, upper in BUCKETS:
        result[(target_ttc_s > lower) & (target_ttc_s <= upper)] = name
    if np.any(result == ""):
        raise ValueError("target lies outside the frozen signed TTC buckets")
    return result.astype(str)


def _require_single_string(frame: pd.DataFrame, column: str, expected: str) -> None:
    values = set(frame[column].astype(str))
    if values != {expected}:
        raise ValueError(f"{column} mismatch")


def precheck_production_oof(
    frame: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
    arm_id: str,
    config_sha256: str,
    checkpoint_sha256_by_fold: Mapping[int, str],
) -> pd.DataFrame:
    """Validate one complete OOF frame against frozen, signed canonical identity."""

    validate_protocol_reference_binding(protocol, reference)
    registry = protocol.get("executable_arm_registry")
    if arm_id not in EXECUTABLE_ARMS or registry != list(EXECUTABLE_ARMS):
        raise ValueError("arm_id is not in the closed executable registry")
    if set(frame.columns) != set(ROW_LEVEL_OOF_COLUMNS):
        missing = sorted(set(ROW_LEVEL_OOF_COLUMNS) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(ROW_LEVEL_OOF_COLUMNS))
        raise ValueError(f"row-level OOF schema mismatch; missing={missing}, extra={extra}")
    if len(frame) != int(protocol["production_row_count"]):
        raise ValueError("production OOF requires exactly 8,192 rows")
    tokens = cast(pd.Series, frame["sample_token"])
    if bool(tokens.isna().any()) or bool((tokens.astype(str).str.len() == 0).any()):
        raise ValueError("sample_token is absent")
    if bool(tokens.duplicated().any()):
        raise ValueError("sample_token is duplicated")
    if not is_integer_dtype(frame["outer_fold"].dtype):
        raise ValueError("outer_fold must have an integer dtype")
    if not is_integer_dtype(frame["seed"].dtype):
        raise ValueError("seed must have an integer dtype")
    folds = frame["outer_fold"].to_numpy(dtype=np.int64)
    if set(folds.tolist()) != set(PRODUCTION_FOLDS):
        raise ValueError("outer_fold universe must be exactly {0,1,2}")
    seeds = frame["seed"].to_numpy(dtype=np.int64)
    if set(seeds.tolist()) != {int(protocol["authorized_seed"])}:
        raise ValueError("seed mismatch")

    normalized = frame.copy()
    target_ttc = _as_finite_float64(normalized, "target_ttc_s")
    target_phase_supplied = _as_finite_float64(normalized, "target_benchmark_phase")
    predicted_phase = _as_finite_float64(normalized, "predicted_benchmark_phase")
    inverse_supplied = _as_finite_float64(normalized, "predicted_inverse_ttc_raw")
    raw_ttc_supplied = _as_finite_float64(normalized, "predicted_ttc_raw")
    clipped_supplied = _as_finite_float64(normalized, "predicted_ttc_clipped")
    mid_supplied = _as_finite_float64(normalized, "scientific_mid_per_row")
    sample_weight = _as_finite_float64(normalized, "sample_weight")
    if np.any(sample_weight <= 0.0):
        raise ValueError("sample_weight must be strictly positive")
    clip_flags = _as_strict_bool(normalized, "is_clip_saturated")
    failure_supplied = _as_strict_bool(normalized, "scientific_failure")

    metric = protocol["metric"]
    delta_t_s = float(metric["metric_delta_t_s"])
    clip_seconds = float(metric["deployment_ttc_clip_seconds"])
    minimum_abs_ttc = float(metric["minimum_abs_prediction_ttc_s"])
    target_phase = _target_phase(target_ttc, delta_t_s)
    inverse, raw_ttc = _prediction_coordinates(predicted_phase, delta_t_s)
    clipped = np.clip(raw_ttc, -clip_seconds, clip_seconds)
    saturated = np.abs(raw_ttc) > clip_seconds
    failure = ~np.isfinite(raw_ttc) | (np.abs(raw_ttc) < minimum_abs_ttc)
    scientific_mid = 1.0e4 * np.abs(target_phase - predicted_phase)
    if not np.isfinite(scientific_mid).all():
        raise ValueError("scientific MiD contains non-finite values")
    comparisons = (
        ("target_benchmark_phase", target_phase_supplied, target_phase),
        ("predicted_inverse_ttc_raw", inverse_supplied, inverse),
        ("predicted_ttc_raw", raw_ttc_supplied, raw_ttc),
        ("predicted_ttc_clipped", clipped_supplied, clipped),
        ("scientific_mid_per_row", mid_supplied, scientific_mid),
    )
    for name, supplied, recomputed in comparisons:
        if not np.allclose(supplied, recomputed, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(f"{name} disagrees with canonical float64 transformation")
    if not np.array_equal(clip_flags, saturated):
        raise ValueError("is_clip_saturated disagrees with raw TTC")
    if not np.array_equal(failure_supplied, failure):
        raise ValueError("scientific_failure disagrees with raw TTC")

    normalized["target_ttc_s"] = target_ttc
    normalized["target_benchmark_phase"] = target_phase
    normalized["predicted_benchmark_phase"] = predicted_phase
    normalized["predicted_inverse_ttc_raw"] = inverse
    normalized["predicted_ttc_raw"] = raw_ttc
    normalized["predicted_ttc_clipped"] = clipped
    normalized["is_clip_saturated"] = saturated
    normalized["scientific_mid_per_row"] = scientific_mid
    normalized["scientific_failure"] = failure
    normalized["sample_weight"] = sample_weight
    normalized["ttc_bucket"] = _bucket_names(target_ttc)

    sequences = normalized["sequence_id"].astype(str)
    expected_sequences = list(protocol["canonical_sequence_ids"])
    if sorted(sequences.unique().tolist()) != expected_sequences:
        raise ValueError("canonical sequence universe mismatch")
    sequence_to_fold = protocol["canonical_sequence_to_fold"]
    for sequence, expected_fold in sequence_to_fold.items():
        observed = set(normalized.loc[sequences == sequence, "outer_fold"].astype(int))
        if observed != {int(expected_fold)}:
            raise ValueError(f"canonical sequence-to-fold mismatch: {sequence}")
    expected_bucket_counts = protocol["canonical_bucket_counts_by_sequence"]
    for sequence in expected_sequences:
        subset = normalized.loc[sequences == sequence]
        observed = subset["ttc_bucket"].value_counts().to_dict()
        if observed != expected_bucket_counts[sequence]:
            raise ValueError(f"canonical sequence/bucket counts mismatch: {sequence}")

    hashes = protocol["canonical_hashes"]
    observed_hashes = {
        "token_identity_sha256": canonical_records_hash(
            normalized, ("sample_token", "sequence_id", "track_id")
        ),
        "target_sha256": canonical_records_hash(normalized, ("sample_token", "target_ttc_s")),
        "fold_assignment_sha256": canonical_records_hash(
            normalized, ("sample_token", "sequence_id", "outer_fold")
        ),
        "sample_weight_sha256": canonical_records_hash(
            normalized, ("sample_token", "sample_weight")
        ),
    }
    if observed_hashes != hashes:
        raise ValueError("self-consistent row hashes do not match the canonical protocol")

    _require_single_string(normalized, "arm_id", arm_id)
    _require_single_string(normalized, "config_sha256", config_sha256)
    _require_single_string(normalized, "protocol_sha256", str(protocol["artifact_sha256"]))
    _require_single_string(
        normalized,
        "cache_manifest_sha256",
        str(protocol["cache_binding"]["file_sha256"]),
    )
    _require_single_string(
        normalized,
        "split_manifest_sha256",
        str(protocol["split_binding"]["file_sha256"]),
    )
    if set(checkpoint_sha256_by_fold) != set(PRODUCTION_FOLDS):
        raise ValueError("exactly three checkpoint SHAs are required")
    for fold, expected_sha in checkpoint_sha256_by_fold.items():
        _require_single_string(
            normalized.loc[normalized["outer_fold"] == fold],
            "checkpoint_sha256",
            expected_sha,
        )
    return normalized


def production_sequence_macro_metrics(checked: pd.DataFrame) -> dict[str, Any]:
    """Aggregate canonical phase MiD directly, never through clipped TTC."""

    missing = {"sequence_id", "ttc_bucket", "scientific_mid_per_row"} - set(checked.columns)
    if missing:
        raise ValueError(f"checked OOF columns missing: {sorted(missing)}")
    per_sequence: dict[str, Any] = {}
    sequence_values: list[float] = []
    for sequence, subset in checked.groupby("sequence_id", sort=True):
        bins: dict[str, Any] = {}
        weighted = 0.0
        for bucket, _lower, _upper in BUCKETS:
            values = subset.loc[subset["ttc_bucket"] == bucket, "scientific_mid_per_row"].to_numpy(
                dtype=np.float64
            )
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"invalid sequence/bucket MiD: {sequence}/{bucket}")
            mean = float(np.mean(values, dtype=np.float64))
            bins[bucket] = {"count": int(values.size), "mid": mean}
            weighted += PAPER_MID_WEIGHTS[bucket] * mean
        if not math.isfinite(weighted):
            raise ValueError(f"non-finite sequence MiD: {sequence}")
        per_sequence[str(sequence)] = {"paper_MiD_overall": weighted, "bins": bins}
        sequence_values.append(weighted)
    overall = float(np.mean(np.asarray(sequence_values, dtype=np.float64), dtype=np.float64))
    if not math.isfinite(overall):
        raise ValueError("sequence macro MiD is non-finite")
    return {
        "protocol": "garl_signed_v1_phase_direct_float64",
        "sequence_macro_paper_MiD_overall": overall,
        "per_sequence": per_sequence,
    }


def clipping_diagnostics(checked: pd.DataFrame) -> dict[str, Any]:
    """Report deployment clipping effects without influencing scientific MiD."""

    target = checked["target_benchmark_phase"].to_numpy(dtype=np.float64)
    raw_phase = checked["predicted_benchmark_phase"].to_numpy(dtype=np.float64)
    clipped_ttc = checked["predicted_ttc_clipped"].to_numpy(dtype=np.float64)
    delta_t_s = 0.1
    clipped_phase = _target_phase(clipped_ttc, delta_t_s)
    raw = 1.0e4 * np.abs(target - raw_phase)
    diagnostic = 1.0e4 * np.abs(target - clipped_phase)
    difference = diagnostic - raw
    return {
        "clip_fraction": float(
            np.mean(
                checked["is_clip_saturated"].to_numpy(dtype=np.float64),
                dtype=np.float64,
            )
        ),
        "mid_raw_mean_per_row": float(np.mean(raw, dtype=np.float64)),
        "mid_clipped_mean_per_row_diagnostic_only": float(np.mean(diagnostic, dtype=np.float64)),
        "delta_clipped_minus_raw_diagnostic_only": float(np.mean(difference, dtype=np.float64)),
        "rows_improved_by_clipping": int(np.count_nonzero(difference < 0.0)),
        "rows_worsened_by_clipping": int(np.count_nonzero(difference > 0.0)),
        "deployment_clipping_not_used_for_scientific_metric": True,
    }


__all__ = [
    "EXECUTABLE_ARMS",
    "IDENTITY_HASH_FIELDS",
    "PRODUCTION_FOLDS",
    "PRODUCTION_ROW_COUNT",
    "REFERENCE_FAMILIES",
    "ROW_LEVEL_OOF_COLUMNS",
    "canonical_records_hash",
    "clipping_diagnostics",
    "load_signed_json",
    "module_topology_sha256",
    "precheck_production_oof",
    "production_sequence_macro_metrics",
    "require_reference_family",
    "tensor_state_sha256",
    "validate_protocol_reference_binding",
]
