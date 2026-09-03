"""Fail-closed production protocol checks for E-Clock X0."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS, sequence_macro_signed_metrics
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase

PRODUCTION_ROW_COUNT = 8192
PRODUCTION_FOLD_COUNT = 3
IDENTITY_HASH_FIELDS = ("sample_token", "sequence_id", "track_id")


def canonical_records_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """Hash ordered, typed records after stable sample-token sorting."""

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


def _bucket_name(target: np.ndarray) -> np.ndarray:
    result = np.full(target.shape, "", dtype=object)
    for name, lower, upper in BUCKETS:
        mask = (target > lower) & (target <= upper)
        result[mask] = name
    if np.any(result == ""):
        raise ValueError("target outside the frozen signed TTC buckets")
    return result.astype(str)


def precheck_production_oof(
    frame: pd.DataFrame,
    *,
    expected_hashes: Mapping[str, str],
    required_sequences: Iterable[str],
) -> pd.DataFrame:
    """Reject every integrity/finiteness defect before historical macro-MiD."""

    required = {
        "sample_token",
        "sequence_id",
        "track_id",
        "outer_fold",
        "target_ttc",
        "ttc_prediction_s",
        "sample_weight",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"production OOF columns missing: {missing}")
    if len(frame) != PRODUCTION_ROW_COUNT:
        raise ValueError(f"production OOF requires exactly {PRODUCTION_ROW_COUNT} rows")
    tokens = cast(pd.Series, frame["sample_token"])
    if bool(tokens.isna().any()) or bool((tokens.astype(str).str.len() == 0).any()):
        raise ValueError("sample_token is absent")
    if bool(tokens.duplicated().any()):
        raise ValueError("sample_token is duplicated")
    folds = np.asarray(pd.to_numeric(frame["outer_fold"], errors="coerce"), dtype=np.float64)
    if not np.isfinite(folds).all() or len(np.unique(folds)) != PRODUCTION_FOLD_COUNT:
        raise ValueError("production OOF requires exactly three finite folds")
    normalized = frame.copy()
    numeric_columns = ("target_ttc", "ttc_prediction_s", "sample_weight")
    for column in numeric_columns:
        values = np.asarray(pd.to_numeric(normalized[column], errors="coerce"), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
        normalized[column] = values
    sample_weight = np.asarray(normalized["sample_weight"], dtype=np.float64)
    target_ttc = np.asarray(normalized["target_ttc"], dtype=np.float64)
    prediction_ttc = np.asarray(normalized["ttc_prediction_s"], dtype=np.float64)
    if np.any(sample_weight <= 0.0):
        raise ValueError("sample_weight must be strictly positive")
    target_phase, target_valid = ttc_to_benchmark_phase(
        torch.from_numpy(target_ttc),
        metric_delta_t_s=0.1,
    )
    prediction_phase, prediction_valid = ttc_to_benchmark_phase(
        torch.from_numpy(prediction_ttc),
        metric_delta_t_s=0.1,
    )
    per_row_mid = 1.0e4 * (target_phase - prediction_phase).abs()
    if not bool(target_valid.all()) or not bool(prediction_valid.all()):
        raise ValueError("target or prediction is outside the benchmark-phase domain")
    if not bool(torch.isfinite(per_row_mid).all()):
        raise ValueError("per-row MiD contains non-finite values")
    normalized["mid_per_row"] = per_row_mid.numpy()
    normalized["ttc_bucket"] = _bucket_name(target_ttc)

    observed_sequences = set(normalized["sequence_id"].astype(str))
    required_sequence_set = set(str(value) for value in required_sequences)
    if observed_sequences != required_sequence_set:
        raise ValueError("required sequence universe mismatch")
    required_buckets = {name for name, _lower, _upper in BUCKETS}
    for sequence in sorted(required_sequence_set):
        subset = normalized[normalized["sequence_id"].astype(str) == sequence]
        if set(subset["ttc_bucket"].astype(str)) != required_buckets:
            raise ValueError(f"sequence {sequence!r} lacks a required TTC bucket")
        for bucket in sorted(required_buckets):
            values = np.asarray(
                subset.loc[subset["ttc_bucket"] == bucket, "mid_per_row"],
                dtype=np.float64,
            )
            if values.size == 0 or not math.isfinite(float(np.mean(values))):
                raise ValueError(f"sequence/bucket aggregate is invalid: {sequence}/{bucket}")

    hash_specs = {
        "identity_sha256": IDENTITY_HASH_FIELDS,
        "target_sha256": ("sample_token", "target_ttc"),
        "fold_sha256": ("sample_token", "outer_fold"),
        "weight_sha256": ("sample_token", "sample_weight"),
    }
    if set(expected_hashes) != set(hash_specs):
        raise ValueError("expected identity hash contract is incomplete or has unknown fields")
    for key, columns in hash_specs.items():
        observed = canonical_records_hash(normalized, columns)
        if observed != expected_hashes[key]:
            raise ValueError(f"{key} mismatch")
    return normalized


def production_sequence_macro_metrics(
    frame: pd.DataFrame,
    *,
    expected_hashes: Mapping[str, str],
    required_sequences: Iterable[str],
) -> dict[str, Any]:
    """Run historical macro-MiD only after the strict X0 precheck."""

    checked = precheck_production_oof(
        frame,
        expected_hashes=expected_hashes,
        required_sequences=required_sequences,
    )
    metrics = sequence_macro_signed_metrics(
        checked["target_ttc"].to_numpy(dtype=np.float64),
        checked["ttc_prediction_s"].to_numpy(dtype=np.float64),
        checked["sequence_id"].astype(str).to_numpy(),
    )
    overall = float(metrics["sequence_macro_paper_MiD_overall"])
    if not math.isfinite(overall):
        raise ValueError("sequence macro MiD is non-finite")
    for sequence, payload in metrics["per_sequence"].items():
        if not math.isfinite(float(payload["paper_MiD_overall"])):
            raise ValueError(f"sequence aggregate is non-finite: {sequence}")
        for bucket, bucket_payload in payload["bins"].items():
            if not math.isfinite(float(bucket_payload["mid"])):
                raise ValueError(f"bucket aggregate is non-finite: {sequence}/{bucket}")
    return metrics


__all__ = [
    "IDENTITY_HASH_FIELDS",
    "PRODUCTION_FOLD_COUNT",
    "PRODUCTION_ROW_COUNT",
    "canonical_records_hash",
    "module_topology_sha256",
    "precheck_production_oof",
    "production_sequence_macro_metrics",
    "tensor_state_sha256",
]
