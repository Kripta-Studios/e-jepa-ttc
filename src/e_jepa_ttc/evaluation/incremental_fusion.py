"""Preregistered cross-fitted complementarity and X1 evidence utilities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_bootstrap import (
    paired_hierarchical_mid_bootstrap,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    canonical_records_hash,
    production_sequence_macro_metrics,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS

DYNAMIC_SLOT_NAMES: tuple[str, ...] = (
    "translation_x",
    "translation_y",
    "divergence_x",
    "divergence_y",
    "divergence_isotropic",
    "flow_magnitude",
    "confidence_margin",
    "entropy",
    "cycle_error",
)
X05_ARMS: tuple[str, ...] = (
    "X05-A5-RAW",
    "X05-A5-CAL",
    "X05-A5-ZERO9",
    "X05-A5-DYN9",
    "X05-A5-SHUFFLE9",
    "X05-A5-DYNPRED",
    "X05-A5-PAIRPRED",
    "X05-A5-BASEPRED",
)
X1_ARMS: tuple[str, ...] = (
    "X1-A5-REPLAY",
    "X1-A5-ZERO-U",
    "X1-A5-DYN-U",
    "X1-A5-SHUFFLE-U",
)
RIDGE_LAMBDAS: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)

FEATURE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "sample_token",
    "sequence_id",
    "track_id",
    "outer_fold",
    "target_ttc_s",
    "target_benchmark_phase",
    "sample_weight",
)
FEATURE_PREDICTION_COLUMNS: tuple[str, ...] = (
    "a5_predicted_benchmark_phase",
    "base_predicted_benchmark_phase",
    "dyn_predicted_benchmark_phase",
    "pair_predicted_benchmark_phase",
)
FEATURE_COLUMNS: tuple[str, ...] = (
    *FEATURE_IDENTITY_COLUMNS,
    *FEATURE_PREDICTION_COLUMNS,
    *DYNAMIC_SLOT_NAMES,
    "transport_valid",
    "a5_checkpoint_sha256",
    "base_checkpoint_sha256",
    "dyn_checkpoint_sha256",
    "pair_checkpoint_sha256",
    "x0_protocol_sha256",
    "x0_reference_sha256",
    "cache_manifest_sha256",
    "split_manifest_sha256",
)


def atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, sign: bool = True
) -> dict[str, Any]:
    """Write canonical scientific JSON atomically, optionally self-signing it."""

    value = dict(payload)
    if sign:
        value = sign_artifact(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return value


def load_signed_artifact(path: Path, *, artifact_type: str | None = None) -> dict[str, Any]:
    """Load a self-hashed artifact and reject type/signature drift."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"artifact signature mismatch: {path}")
    if artifact_type is not None and payload.get("artifact_type") != artifact_type:
        raise ValueError(f"artifact type mismatch: {path}")
    return payload


def _bucket_names(target: np.ndarray) -> np.ndarray:
    result = np.full(target.shape, "", dtype=object)
    for name, lower, upper in BUCKETS:
        result[(target > lower) & (target <= upper)] = name
    if np.any(result == ""):
        raise ValueError("target lies outside frozen TTC buckets")
    return result.astype(str)


def validate_feature_table(
    frame: pd.DataFrame,
    *,
    x0_protocol: Mapping[str, Any],
    replay_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate the complete 8,192-row feature table before any meta-test use."""

    if tuple(frame.columns) != FEATURE_COLUMNS:
        missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(FEATURE_COLUMNS))
        raise ValueError(f"feature schema mismatch; missing={missing}, extra={extra}")
    sample_tokens = frame["sample_token"].to_numpy()
    if len(frame) != 8192 or bool(pd.isna(sample_tokens).any()):
        raise ValueError("feature table must contain exactly 8,192 identified rows")
    if frame["sample_token"].astype(str).duplicated().any():
        raise ValueError("feature table contains duplicate sample_token values")
    numeric_columns = (
        "outer_fold",
        "target_ttc_s",
        "target_benchmark_phase",
        "sample_weight",
        *FEATURE_PREDICTION_COLUMNS,
        *DYNAMIC_SLOT_NAMES,
    )
    normalized = frame.copy()
    for column in numeric_columns:
        values = np.asarray(pd.to_numeric(normalized[column], errors="coerce"), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"feature table numeric column is non-finite: {column}")
        if column == "outer_fold":
            if not np.equal(values, np.floor(values)).all():
                raise ValueError("feature table outer_fold values must be integers")
            normalized[column] = values.astype(np.int64)
        else:
            normalized[column] = values
    if set(normalized["outer_fold"].astype(int)) != {0, 1, 2}:
        raise ValueError("feature table outer fold universe mismatch")
    if set(normalized["sequence_id"].astype(str)) != set(x0_protocol["canonical_sequence_ids"]):
        raise ValueError("feature table sequence universe mismatch")
    for sequence, fold in x0_protocol["canonical_sequence_to_fold"].items():
        observed = set(
            normalized.loc[normalized["sequence_id"].astype(str) == sequence, "outer_fold"].astype(
                int
            )
        )
        if observed != {int(fold)}:
            raise ValueError(f"feature table sequence/fold mismatch: {sequence}")
    hashes = {
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
    if hashes != x0_protocol["canonical_hashes"]:
        raise ValueError("feature table canonical identity hashes mismatch")
    expected_strings = {
        "x0_protocol_sha256": str(x0_protocol["artifact_sha256"]),
        "cache_manifest_sha256": str(x0_protocol["cache_binding"]["file_sha256"]),
        "split_manifest_sha256": str(x0_protocol["split_binding"]["file_sha256"]),
    }
    for column, expected in expected_strings.items():
        if set(normalized[column].astype(str)) != {expected}:
            raise ValueError(f"feature table provenance mismatch: {column}")
    required_replay = {
        "row_count": 8192,
        "finite_fraction": 1.0,
        "failure_rate": 0.0,
        "identity_hashes_exact": True,
        "replay_matches_x0": True,
        "target_not_passed_to_extractor": True,
        "sealed_evaluation_opened": False,
    }
    for key, expected in required_replay.items():
        if replay_manifest.get(key) != expected:
            raise ValueError(f"feature replay integrity gate failed: {key}")
    strict_transport = np.asarray(
        normalized["transport_valid"].map(lambda value: isinstance(value, (bool, np.bool_))),
        dtype=bool,
    )
    if not bool(np.all(strict_transport)):
        raise ValueError("transport_valid must contain strict booleans")
    if not bool(normalized["transport_valid"].all()):
        raise ValueError("feature replay contains invalid transport rows")
    return normalized.sort_values("sample_token", kind="stable").reset_index(drop=True)


def deterministic_within_sequence_shuffle(
    slots: np.ndarray,
    sequence_ids: Sequence[str],
    *,
    seed: int,
    outer_fold: int,
    partition: str,
) -> tuple[np.ndarray, str]:
    """Permute complete nine-slot rows within sequence without target access."""

    values = np.asarray(slots, dtype=np.float64)
    sequences = np.asarray(sequence_ids, dtype=str)
    if values.ndim != 2 or values.shape[1] != 9 or values.shape[0] != sequences.size:
        raise ValueError("shuffle expects slots [N,9] aligned with N sequences")
    output = np.empty_like(values)
    digest = hashlib.sha256()
    for sequence in sorted(np.unique(sequences).tolist()):
        indices = np.flatnonzero(sequences == sequence)
        identity = f"{seed}|{outer_fold}|{partition}|{sequence}".encode()
        local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
        permutation = np.random.default_rng(local_seed).permutation(indices.size)
        output[indices] = values[indices[permutation]]
        digest.update(sequence.encode())
        digest.update(np.asarray(permutation, dtype="<i8").tobytes())
    if not np.isfinite(output).all():
        raise ValueError("shuffle produced non-finite slots")
    return output, digest.hexdigest()


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(np.mean(values, axis=0, dtype=np.float64), dtype=np.float64).reshape(-1)
    std = np.asarray(np.std(values, axis=0, dtype=np.float64), dtype=np.float64).reshape(-1)
    std = np.asarray(np.where(std > 1.0e-12, std, 1.0), dtype=np.float64)
    return mean, std


def _ridge_fit(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, float]:
    if regularization < 0.0:
        raise ValueError("ridge lambda cannot be negative")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or w.shape != y.shape:
        raise ValueError("ridge input shapes mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("ridge inputs must be finite")
    if np.any(w <= 0.0):
        raise ValueError("ridge weights must be strictly positive")
    design = np.column_stack((np.ones(x.shape[0], dtype=np.float64), x))
    root = np.sqrt(w / np.mean(w, dtype=np.float64))
    weighted_x = design * root[:, None]
    weighted_y = y * root
    gram = weighted_x.T @ weighted_x
    rhs = weighted_x.T @ weighted_y
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise ValueError("ridge fit produced non-finite coefficients")
    return coefficients[1:], float(coefficients[0])


def _ridge_predict(features: np.ndarray, coefficients: np.ndarray, intercept: float) -> np.ndarray:
    result = np.asarray(features, dtype=np.float64) @ coefficients + intercept
    if not np.isfinite(result).all():
        raise ValueError("ridge prediction is non-finite")
    return result


def scientific_prediction_frame(
    identity: pd.DataFrame,
    prediction_phase: np.ndarray,
    *,
    arm_id: str,
) -> pd.DataFrame:
    """Build the canonical bootstrap/metric frame from phase predictions."""

    phase = np.asarray(prediction_phase, dtype=np.float64)
    target_phase = identity["target_benchmark_phase"].to_numpy(dtype=np.float64)
    if phase.shape != target_phase.shape or not np.isfinite(phase).all():
        raise ValueError("prediction phase shape/finiteness mismatch")
    result = identity.loc[:, FEATURE_IDENTITY_COLUMNS].copy()
    result["predicted_benchmark_phase"] = phase
    result["scientific_mid_per_row"] = 1.0e4 * np.abs(target_phase - phase)
    result["ttc_bucket"] = _bucket_names(result["target_ttc_s"].to_numpy(dtype=np.float64))
    result["arm_id"] = arm_id
    return result


def _feature_matrix(frame: pd.DataFrame, arm_id: str, slots: np.ndarray) -> np.ndarray:
    a5 = frame["a5_predicted_benchmark_phase"].to_numpy(dtype=np.float64)[:, None]
    mapping = {
        "X05-A5-CAL": a5,
        "X05-A5-ZERO9": np.column_stack((a5, np.zeros_like(slots))),
        "X05-A5-DYN9": np.column_stack((a5, slots)),
        "X05-A5-SHUFFLE9": np.column_stack((a5, slots)),
        "X05-A5-DYNPRED": np.column_stack(
            (a5, frame["dyn_predicted_benchmark_phase"].to_numpy(dtype=np.float64))
        ),
        "X05-A5-PAIRPRED": np.column_stack(
            (a5, frame["pair_predicted_benchmark_phase"].to_numpy(dtype=np.float64))
        ),
        "X05-A5-BASEPRED": np.column_stack(
            (a5, frame["base_predicted_benchmark_phase"].to_numpy(dtype=np.float64))
        ),
    }
    if arm_id not in mapping:
        raise ValueError(f"unsupported ridge arm: {arm_id}")
    return np.asarray(mapping[arm_id], dtype=np.float64)


def _loso_select_lambda(
    train: pd.DataFrame,
    features: np.ndarray,
    *,
    lambdas: Sequence[float],
) -> tuple[float, list[dict[str, Any]]]:
    sequences = train["sequence_id"].astype(str).to_numpy()
    target = train["target_benchmark_phase"].to_numpy(dtype=np.float64)
    weights = train["sample_weight"].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for regularization in lambdas:
        predictions = np.empty_like(target)
        folds: list[dict[str, Any]] = []
        for heldout in sorted(np.unique(sequences).tolist()):
            fit_mask = sequences != heldout
            test_mask = ~fit_mask
            mean, std = _standardize_fit(features[fit_mask])
            coefficients, intercept = _ridge_fit(
                (features[fit_mask] - mean) / std,
                target[fit_mask],
                weights[fit_mask],
                float(regularization),
            )
            predictions[test_mask] = _ridge_predict(
                (features[test_mask] - mean) / std, coefficients, intercept
            )
            folds.append(
                {
                    "heldout_sequence": heldout,
                    "train_rows": int(np.sum(fit_mask)),
                    "heldout_rows": int(np.sum(test_mask)),
                    "normalization_fit_on_inner_train_only": True,
                }
            )
        checked = scientific_prediction_frame(train, predictions, arm_id="X05-LOSO")
        score = float(
            production_sequence_macro_metrics(checked)["sequence_macro_paper_MiD_overall"]
        )
        records.append(
            {
                "lambda": float(regularization),
                "loso_sequence_macro_mid": score,
                "folds": folds,
            }
        )
    best = min(records, key=lambda item: (item["loso_sequence_macro_mid"], item["lambda"]))
    return float(best["lambda"]), records


def run_x05_cross_fit(
    features: pd.DataFrame,
    *,
    x0_protocol: Mapping[str, Any],
    output_root: Path,
    shuffle_seed: int = 20260904,
    lambdas: Sequence[float] = RIDGE_LAMBDAS,
) -> dict[str, Any]:
    """Run all preregistered X0.5 arms with fold-isolated LOSO ridge selection."""

    if output_root.exists():
        raise FileExistsError(f"X0.5 output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    slots_all = features.loc[:, DYNAMIC_SLOT_NAMES].to_numpy(dtype=np.float64)
    predictions: dict[str, np.ndarray] = {
        "X05-A5-RAW": features["a5_predicted_benchmark_phase"].to_numpy(dtype=np.float64).copy()
    }
    for arm in X05_ARMS:
        predictions.setdefault(arm, np.empty(len(features), dtype=np.float64))
    fold_records: list[dict[str, Any]] = []
    for outer_fold in (0, 1, 2):
        test_mask = features["outer_fold"].to_numpy(dtype=np.int64) == outer_fold
        train_mask = ~test_mask
        train = features.loc[train_mask].reset_index(drop=True)
        test = features.loc[test_mask].reset_index(drop=True)
        train_slots = slots_all[train_mask]
        test_slots = slots_all[test_mask]
        shuffled_train, train_shuffle_sha = deterministic_within_sequence_shuffle(
            train_slots,
            train["sequence_id"].astype(str).tolist(),
            seed=shuffle_seed,
            outer_fold=outer_fold,
            partition="meta-train",
        )
        shuffled_test, test_shuffle_sha = deterministic_within_sequence_shuffle(
            test_slots,
            test["sequence_id"].astype(str).tolist(),
            seed=shuffle_seed,
            outer_fold=outer_fold,
            partition="meta-test",
        )
        arm_records: dict[str, Any] = {}
        for arm in X05_ARMS:
            if arm == "X05-A5-RAW":
                continue
            train_arm_slots = shuffled_train if arm == "X05-A5-SHUFFLE9" else train_slots
            test_arm_slots = shuffled_test if arm == "X05-A5-SHUFFLE9" else test_slots
            train_matrix = _feature_matrix(train, arm, train_arm_slots)
            test_matrix = _feature_matrix(test, arm, test_arm_slots)
            regularization, cv_records = _loso_select_lambda(train, train_matrix, lambdas=lambdas)
            mean, std = _standardize_fit(train_matrix)
            coefficients, intercept = _ridge_fit(
                (train_matrix - mean) / std,
                train["target_benchmark_phase"].to_numpy(dtype=np.float64),
                train["sample_weight"].to_numpy(dtype=np.float64),
                regularization,
            )
            fold_prediction = _ridge_predict((test_matrix - mean) / std, coefficients, intercept)
            predictions[arm][np.flatnonzero(test_mask)] = fold_prediction
            arm_records[arm] = {
                "selected_lambda": regularization,
                "lambda_selection": cv_records,
                "normalization_mean": mean.tolist(),
                "normalization_std": std.tolist(),
                "coefficients_standardized": coefficients.tolist(),
                "intercept": intercept,
                "meta_train_rows": len(train),
                "meta_test_rows": len(test),
                "meta_train_sequences": sorted(train["sequence_id"].astype(str).unique().tolist()),
                "meta_test_sequences": sorted(test["sequence_id"].astype(str).unique().tolist()),
                "outer_meta_test_used_for_fit_or_selection": False,
            }
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "train_shuffle_permutation_sha256": train_shuffle_sha,
                "test_shuffle_permutation_sha256": test_shuffle_sha,
                "shuffle_target_blind": True,
                "shuffle_within_sequence": True,
                "arms": arm_records,
            }
        )
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("X0.5 produced non-finite meta-OOF predictions")

    arm_frames: dict[str, pd.DataFrame] = {}
    arm_manifests: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, Any] = {}
    for arm in X05_ARMS:
        arm_frame = scientific_prediction_frame(features, predictions[arm], arm_id=arm)
        arm_path = output_root / "arms" / arm / "meta_oof_predictions.csv"
        arm_path.parent.mkdir(parents=True, exist_ok=True)
        arm_frame.to_csv(arm_path, index=False)
        metrics = production_sequence_macro_metrics(arm_frame)
        per_fold = {
            str(fold): float(
                production_sequence_macro_metrics(arm_frame.loc[arm_frame["outer_fold"] == fold])[
                    "sequence_macro_paper_MiD_overall"
                ]
            )
            for fold in (0, 1, 2)
        }
        manifest = atomic_write_json(
            arm_path.with_suffix(".manifest.json"),
            {
                "artifact_type": "eclock_x05_arm_oof_manifest_v1",
                "arm_id": arm,
                "row_count": len(arm_frame),
                "oof_path": str(arm_path),
                "oof_file_sha256": compute_file_hash(str(arm_path)),
                "oof_bytes": arm_path.stat().st_size,
                "sample_token_sha256": canonical_records_hash(
                    arm_frame, ("sample_token", "sequence_id", "track_id")
                ),
                "metrics": metrics,
                "per_fold_mid": per_fold,
                "finite_fraction": 1.0,
                "failure_rate": 0.0,
                "outer_meta_test_used_for_fit_or_selection": False,
            },
        )
        arm_frames[arm] = arm_frame
        arm_manifests[arm] = manifest
        aggregates[arm] = {
            "mid": metrics["sequence_macro_paper_MiD_overall"],
            "per_fold_mid": per_fold,
            "metrics": metrics,
            "manifest_sha256": manifest["artifact_sha256"],
            "oof_file_sha256": manifest["oof_file_sha256"],
        }
    fold_summary = atomic_write_json(
        output_root / "x05_meta_fold_summary.json",
        {
            "artifact_type": "eclock_x05_meta_fold_summary_v1",
            "folds": fold_records,
            "lambda_grid": [float(value) for value in lambdas],
            "ridge_precision": "float64_cpu",
            "standardization_scope": "inner-train-for-LOSO_then_full-meta-train",
            "shuffle_seed": shuffle_seed,
        },
    )
    comparison_specs = {
        "dyn9_vs_cal": ("X05-A5-DYN9", "X05-A5-CAL"),
        "dyn9_vs_shuffle": ("X05-A5-DYN9", "X05-A5-SHUFFLE9"),
        "dyn9_vs_a5_raw": ("X05-A5-DYN9", "X05-A5-RAW"),
        "zero9_vs_cal": ("X05-A5-ZERO9", "X05-A5-CAL"),
    }
    comparisons: dict[str, Any] = {}
    for name, (candidate, reference) in comparison_specs.items():
        candidate_manifest = arm_manifests[candidate]
        reference_manifest = arm_manifests[reference]
        bootstrap = paired_hierarchical_mid_bootstrap(
            arm_frames[candidate],
            arm_frames[reference],
            protocol=x0_protocol,
            candidate_identity={
                "reference_family": candidate,
                "path": candidate_manifest["oof_path"],
                "file_sha256": candidate_manifest["oof_file_sha256"],
                "artifact_sha256": candidate_manifest["artifact_sha256"],
            },
            reference_identity={
                "reference_family": reference,
                "path": reference_manifest["oof_path"],
                "file_sha256": reference_manifest["oof_file_sha256"],
                "artifact_sha256": reference_manifest["artifact_sha256"],
            },
        )
        delta = float(aggregates[candidate]["mid"] - aggregates[reference]["mid"])
        comparison = atomic_write_json(
            output_root / "comparisons" / f"x05_comparison_{name}.json",
            {
                "artifact_type": "eclock_x05_comparison_v1",
                "comparison": f"{candidate}_minus_{reference}",
                "candidate": candidate,
                "reference": reference,
                "delta_mid": delta,
                "bootstrap": bootstrap,
            },
        )
        comparisons[name] = comparison
    aggregate = atomic_write_json(
        output_root / "x05_aggregate.json",
        {
            "artifact_type": "eclock_x05_aggregate_v1",
            "row_count": 8192,
            "finite_fraction": 1.0,
            "failure_rate": 0.0,
            "arms": aggregates,
            "fold_summary_sha256": fold_summary["artifact_sha256"],
            "comparisons": {name: value["artifact_sha256"] for name, value in comparisons.items()},
        },
    )
    cal = comparisons["dyn9_vs_cal"]
    shuffle = comparisons["dyn9_vs_shuffle"]
    raw = comparisons["dyn9_vs_a5_raw"]
    cal_delta = cal["bootstrap"]["delta_candidate_minus_reference"]
    shuffle_delta = shuffle["bootstrap"]["delta_candidate_minus_reference"]
    incremental = (
        cal["delta_mid"] < 0.0
        and cal_delta["ci95_high"] < 0.0
        and cal_delta["probability_delta_lt_zero"] >= 0.90
        and shuffle["delta_mid"] < 0.0
        and shuffle_delta["ci95_high"] < 0.0
        and shuffle_delta["probability_delta_lt_zero"] >= 0.90
    )
    if not incremental:
        decision = "GLOBAL_DYNAMIC_SLOTS_REDUNDANT_WITH_A5_SCREEN"
    elif raw["delta_mid"] > -1.0:
        decision = "INCREMENTAL_SIGNAL_TOO_SMALL_FOR_X1"
    else:
        decision = "X1_AUTHORIZED"
    gate = atomic_write_json(
        output_root / "x05_gate.json",
        {
            "artifact_type": "eclock_x05_gate_v1",
            "decision": decision,
            "incremental_signal_supported": incremental,
            "x1_authorized": decision == "X1_AUTHORIZED",
            "scientific_stop": decision != "X1_AUTHORIZED",
            "integrity": {
                "row_count": 8192,
                "finite_fraction": 1.0,
                "failure_rate": 0.0,
                "identity_hashes_exact": True,
                "replay_matches_x0": True,
                "outer_meta_test_used_for_fit_or_selection": False,
                "sealed_evaluation_opened": False,
            },
            "dyn9_minus_cal": {
                "delta_mid": cal["delta_mid"],
                **cal_delta,
            },
            "dyn9_minus_shuffle": {
                "delta_mid": shuffle["delta_mid"],
                **shuffle_delta,
            },
            "dyn9_minus_a5_raw": {
                "delta_mid": raw["delta_mid"],
                **raw["bootstrap"]["delta_candidate_minus_reference"],
            },
            "aggregate_sha256": aggregate["artifact_sha256"],
        },
    )
    return gate


def evaluate_x1_gate(
    *,
    comparisons: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen seed-7 X1 gates without post-hoc alternatives."""

    required_integrity = {
        "row_count": 8192,
        "finite_fraction": 1.0,
        "failure_rate": 0.0,
        "a5_replay_exact": True,
        "zero_initialization_replay_exact": True,
        "matched_topology_init_order_budget": True,
        "a5_and_transport_frozen": True,
        "outer_dev_evaluations_per_arm_fold": 1,
        "sealed_evaluation_opened": False,
    }
    for key, expected in required_integrity.items():
        if integrity.get(key) != expected:
            return sign_artifact(
                {
                    "artifact_type": "eclock_x1_gate_v1",
                    "decision": "INVALID_X1",
                    "failed_integrity_key": key,
                    "scientific_stop": True,
                }
            )

    def passed(name: str, *, delta_limit: float, strict_delta: bool = False) -> bool:
        item = comparisons[name]
        boot = item["bootstrap"]["delta_candidate_minus_reference"]
        delta_ok = (
            item["delta_mid"] < delta_limit if strict_delta else item["delta_mid"] <= delta_limit
        )
        return bool(
            delta_ok and boot["ci95_high"] < 0.0 and boot["probability_delta_lt_zero"] >= 0.90
        )

    primary = passed("dyn_vs_zero", delta_limit=-1.0)
    shuffle = passed("dyn_vs_shuffle", delta_limit=0.0, strict_delta=True)
    practical = (
        passed("dyn_vs_a5", delta_limit=-3.0) and float(integrity["coverage_drop_pp"]) <= 1.0
    )
    if primary and shuffle and practical:
        decision = "X1_SEED7_SUPPORTED_REPLICATION_REQUIRED"
    elif primary and shuffle:
        decision = "X1_INCREMENTAL_BUT_NOT_COMPETITIVE"
    else:
        decision = "X1_SEED7_NEGATIVE"
    return sign_artifact(
        {
            "artifact_type": "eclock_x1_gate_v1",
            "decision": decision,
            "primary_passed": primary,
            "shuffle_passed": shuffle,
            "practical_utility_passed": practical,
            "replication_authorized": primary and shuffle and practical,
            "scientific_stop": not (primary and shuffle and practical),
            "integrity": dict(integrity),
        }
    )


__all__ = [
    "DYNAMIC_SLOT_NAMES",
    "FEATURE_COLUMNS",
    "FEATURE_IDENTITY_COLUMNS",
    "RIDGE_LAMBDAS",
    "X05_ARMS",
    "X1_ARMS",
    "atomic_write_json",
    "deterministic_within_sequence_shuffle",
    "evaluate_x1_gate",
    "load_signed_artifact",
    "run_x05_cross_fit",
    "scientific_prediction_frame",
    "validate_feature_table",
]
