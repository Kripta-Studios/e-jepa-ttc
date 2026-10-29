"""Fail-closed E-Clock OOF aggregation and internal reference comparison."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_bootstrap import (
    paired_hierarchical_mid_bootstrap,
)
from e_jepa_ttc.evaluation.collision_clock_config import load_x0_config
from e_jepa_ttc.evaluation.collision_clock_gates import evaluate_x0_height_gate
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    EXECUTABLE_ARMS,
    canonical_records_hash,
    clipping_diagnostics,
    load_signed_json,
    precheck_production_oof,
    production_sequence_macro_metrics,
    read_official_a5_csv,
    require_reference_family,
    validate_protocol_reference_binding,
)
from e_jepa_ttc.training.collision_clock_eap import require_frozen_checkpoint


def _read_oof_csv(path: Path) -> pd.DataFrame:
    """Read runner output without moving persisted float64 values by one ULP."""

    return pd.read_csv(path, float_precision="round_trip")


def _reference_identity(family: Mapping[str, Any]) -> dict[str, str]:
    physical = family.get("physical_references")
    artifact = family.get("artifact_reference")
    if not isinstance(physical, list) or len(physical) < 1 or not isinstance(artifact, Mapping):
        raise ValueError("reference family physical/artifact identity is incomplete")
    record = physical[0]
    return {
        "reference_family": str(family["reference_family"]),
        "path": str(record["path"]),
        "file_sha256": str(record["file_sha256"]),
        "artifact_sha256": str(artifact["artifact_sha256"]),
    }


def _phase_from_ttc(ttc: np.ndarray, delta_t_s: float) -> np.ndarray:
    valid = (ttc < 0.0) | (ttc > delta_t_s)
    if not valid.all():
        raise ValueError("reference prediction lies outside benchmark-phase domain")
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        phase = -np.log1p(-delta_t_s / ttc)
    if not np.isfinite(phase).all():
        raise ValueError("reference prediction phase is non-finite")
    return phase


def load_official_a5_reference_frame(
    *,
    source_root: Path,
    family: Mapping[str, Any],
    checked_candidate: pd.DataFrame,
    delta_t_s: float,
) -> pd.DataFrame:
    """Load and recalculate official A5 on the candidate's canonical target rows."""

    if family.get("reference_family") != "official_a5_oof":
        raise ValueError("official A5 loader refuses every other reference family")
    identity = _reference_identity(family)
    path = source_root / Path(identity["path"])
    if not path.is_file() or compute_file_hash(str(path)) != identity["file_sha256"]:
        raise ValueError("official A5 physical reference SHA mismatch")
    physical = family["physical_references"][0]
    if path.stat().st_size != int(physical["bytes"]):
        raise ValueError("official A5 physical reference byte count mismatch")
    source = read_official_a5_csv(path)
    required = {
        "sample_token",
        "sequence_id",
        "track_id",
        "target_ttc_s",
        "point_prediction_ttc_s",
        "fold",
        "seed",
    }
    if not required.issubset(source.columns):
        raise ValueError("official A5 OOF schema mismatch")
    normalized = source.loc[:, sorted(required)].rename(
        columns={"point_prediction_ttc_s": "prediction_ttc_s", "fold": "outer_fold"}
    )
    merged = checked_candidate[
        ["sample_token", "sequence_id", "track_id", "target_ttc_s", "outer_fold"]
    ].merge(
        normalized,
        on="sample_token",
        how="outer",
        suffixes=("_canonical", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if set(merged["_merge"]) != {"both"}:
        raise ValueError("official A5 token coverage differs from candidate")
    for column in ("sequence_id", "track_id", "outer_fold"):
        if not bool((merged[f"{column}_canonical"] == merged[f"{column}_source"]).all()):
            raise ValueError(f"official A5 {column} mismatch")
    target = merged["target_ttc_s_canonical"].to_numpy(dtype=np.float64)
    source_target = merged["target_ttc_s_source"].to_numpy(dtype=np.float64)
    if not np.allclose(target, source_target, rtol=0.0, atol=1.0e-12):
        raise ValueError("official A5 target mismatch")
    prediction = merged["prediction_ttc_s"].to_numpy(dtype=np.float64)
    phase = _phase_from_ttc(prediction, delta_t_s)
    target_phase = _phase_from_ttc(target, delta_t_s)
    result = pd.DataFrame(
        {
            "sample_token": merged["sample_token"].astype(str),
            "sequence_id": merged["sequence_id_canonical"].astype(str),
            "track_id": merged["track_id_canonical"].astype(str),
            "target_ttc_s": target,
            "prediction_ttc_s": prediction,
            "scientific_mid_per_row": 1.0e4 * np.abs(target_phase - phase),
        }
    ).sort_values("sample_token", kind="stable")
    if canonical_records_hash(result, ("sample_token", "prediction_ttc_s")) != family.get(
        "prediction_sha256"
    ):
        raise ValueError("official A5 prediction identity mismatch")
    return result.reset_index(drop=True)


def aggregate_verified_frame(
    candidate: pd.DataFrame,
    official_a5: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
    config_sha256: str,
    checkpoint_sha256_by_fold: Mapping[int, str],
    candidate_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Recalculate all metrics/bootstrap/gates and only then sign scientific evidence."""

    validate_protocol_reference_binding(protocol, reference)
    arm_id = str(config.get("arm_id", ""))
    if arm_id not in EXECUTABLE_ARMS:
        raise ValueError("arm_id is not in the closed scientific OOF registry")
    if config.get("execution_authorized") is not True:
        raise PermissionError("arm config is not authorized for execution")
    if config.get("seed") != 7 or config.get("folds") != [0, 1, 2]:
        raise ValueError("scientific aggregate requires seed 7 and all three folds")
    if config.get("checkpoint_policy") != "last_update_fixed_budget":
        raise ValueError("aggregate checkpoint policy mismatch")
    family = require_reference_family(reference, "official_a5_oof")
    if protocol.get("official_a5_reference_family") != "official_a5_oof":
        raise ValueError("protocol official A5 family mismatch")
    checked = precheck_production_oof(
        candidate,
        protocol=protocol,
        reference=reference,
        arm_id=arm_id,
        config_sha256=config_sha256,
        checkpoint_sha256_by_fold=checkpoint_sha256_by_fold,
    )
    required_reference_columns = {
        "sample_token",
        "sequence_id",
        "track_id",
        "target_ttc_s",
        "prediction_ttc_s",
    }
    if not required_reference_columns.issubset(official_a5.columns):
        raise ValueError("official A5 comparison schema is incomplete")
    reference_checked = official_a5.loc[:, sorted(required_reference_columns)].copy()
    reference_tokens = reference_checked["sample_token"].astype(str)
    if len(official_a5) != len(checked) or reference_tokens.nunique() != len(checked):
        raise ValueError("official A5 must cover exactly the production token universe")
    aligned_identity = checked[["sample_token", "sequence_id", "track_id", "target_ttc_s"]].merge(
        reference_checked,
        on="sample_token",
        how="outer",
        suffixes=("_candidate", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    if set(aligned_identity["_merge"]) != {"both"}:
        raise ValueError("official A5 token set differs from candidate")
    for column in ("sequence_id", "track_id"):
        if not bool(
            (
                aligned_identity[f"{column}_candidate"] == aligned_identity[f"{column}_reference"]
            ).all()
        ):
            raise ValueError(f"official A5 {column} differs from candidate")
    if not np.allclose(
        aligned_identity["target_ttc_s_candidate"],
        aligned_identity["target_ttc_s_reference"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("official A5 target differs from candidate")
    if canonical_records_hash(
        reference_checked, ("sample_token", "prediction_ttc_s")
    ) != family.get("prediction_sha256"):
        raise ValueError("official A5 supplied prediction vector is not signed")
    reference_target = reference_checked["target_ttc_s"].to_numpy(dtype=np.float64)
    reference_prediction = reference_checked["prediction_ttc_s"].to_numpy(dtype=np.float64)
    delta_t_s = float(protocol["metric"]["metric_delta_t_s"])
    reference_checked["scientific_mid_per_row"] = 1.0e4 * np.abs(
        _phase_from_ttc(reference_target, delta_t_s)
        - _phase_from_ttc(reference_prediction, delta_t_s)
    )
    candidate_metrics = production_sequence_macro_metrics(checked)
    target = reference_checked["target_ttc_s"].to_numpy(dtype=np.float64)
    bucket = np.full(target.shape, "", dtype=object)
    for name, lower, upper in (
        ("crucial", 0.0, 3.0),
        ("small", 3.0, 6.0),
        ("large", 6.0, 10.0),
        ("negative", -10.0, 0.0),
    ):
        bucket[(target > lower) & (target <= upper)] = name
    reference_checked["ttc_bucket"] = bucket.astype(str)
    reference_metrics = production_sequence_macro_metrics(reference_checked)
    reference_mid = float(reference_metrics["sequence_macro_paper_MiD_overall"])
    if not math.isclose(reference_mid, float(family["recomputed_mid"]), abs_tol=1.0e-9):
        raise ValueError("official A5 recomputed MiD disagrees with signed reference")

    reference_identity = _reference_identity(family)
    bootstrap_artifact = paired_hierarchical_mid_bootstrap(
        checked,
        reference_checked,
        protocol=protocol,
        candidate_identity=candidate_identity,
        reference_identity=reference_identity,
    )
    candidate_mid = float(candidate_metrics["sequence_macro_paper_MiD_overall"])
    delta = candidate_mid - reference_mid
    if arm_id == "X0-A5-REPLAY":
        aligned = checked[["sample_token", "predicted_ttc_clipped"]].merge(
            official_a5[["sample_token", "prediction_ttc_s"]],
            on="sample_token",
            validate="one_to_one",
        )
        replay_check = checked.copy()
        replay_check["scientific_mid_per_row"] = 1.0e4 * np.abs(
            replay_check["target_benchmark_phase"].to_numpy(dtype=np.float64)
            - _phase_from_ttc(
                replay_check["predicted_ttc_clipped"].to_numpy(dtype=np.float64),
                delta_t_s,
            )
        )
        replay_mid = float(
            production_sequence_macro_metrics(replay_check)["sequence_macro_paper_MiD_overall"]
        )
        tolerance = float(protocol["gates"]["a5_replay_mid_tolerance"])
        if not np.allclose(
            aligned["predicted_ttc_clipped"],
            aligned["prediction_ttc_s"],
            rtol=0.0,
            atol=tolerance,
        ) or not math.isclose(replay_mid, reference_mid, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("X0-A5-REPLAY does not reproduce official_a5_oof")
        gate = {
            "decision": "OFFICIAL_A5_REPLAY_REPRODUCED",
            "passed": True,
            "scientific_mid_raw": candidate_mid,
            "legacy_official_replay_mid_clipped_diagnostic_only": replay_mid,
            "deployment_clipping_not_used_for_scientific_metric": True,
        }
    elif arm_id == "X0-DYN-U":
        paired = bootstrap_artifact["delta_candidate_minus_reference"]
        gate = evaluate_x0_height_gate(
            {
                "row_count": len(checked),
                "finite_fraction": 1.0,
                "failure_rate": float(
                    np.mean(
                        checked["scientific_failure"].to_numpy(dtype=np.float64),
                        dtype=np.float64,
                    )
                ),
                "coverage_drop_pp": 0.0,
                "delta_mid_vs_official_a5_oof": delta,
                "probability_delta_below_zero": paired["probability_delta_lt_zero"],
                "paired_ci95_upper": paired["ci95_high"],
                "identity_hashes_exact": True,
                "height_interface_bypassed": True,
                "global_transport_foreground_free": True,
                "motion_feature_schema_exact": True,
                "prefix_causality_passed": True,
                "forbidden_feature_audit_passed": True,
                "reference_identity": reference_identity,
            }
        )
    else:
        gate = {"decision": "INTEGRITY_CHAIN_COMPLETE", "passed": True}
    return sign_artifact(
        {
            "artifact_type": "eclock_x0_aggregate_v2",
            "arm_id": arm_id,
            "evidence_class": "scientific_oof",
            "scientific_result": True,
            "reference_family": "official_a5_oof",
            "reference_identity": reference_identity,
            "config_sha256": config_sha256,
            "protocol_sha256": protocol["artifact_sha256"],
            "reference_sha256": reference["artifact_sha256"],
            "checkpoint_sha256_by_fold": {
                str(key): value for key, value in checkpoint_sha256_by_fold.items()
            },
            "metrics": candidate_metrics,
            "reference_metrics": reference_metrics,
            "delta_mid_vs_official_a5_oof": delta,
            "clipping_diagnostics": clipping_diagnostics(checked),
            "bootstrap": bootstrap_artifact,
            "gate_decision": gate,
            "integrity_chain_complete": True,
        }
    )


def aggregate_run(
    *,
    config_path: Path,
    protocol_path: Path,
    reference_path: Path,
    run_root: Path,
    source_root: Path,
    schema_root: Path,
) -> dict[str, Any]:
    """Load three physical folds and invoke the closed in-memory aggregate."""

    config = load_x0_config(
        config_path, schema_path=schema_root / "scientific_recovery_v9_eclock_config_v2.schema.json"
    )
    arm_id = str(config.get("arm_id", ""))
    if arm_id not in EXECUTABLE_ARMS:
        raise ValueError("unknown or forbidden scientific arm_id")
    protocol = load_signed_json(
        protocol_path,
        schema_path=schema_root / "scientific_recovery_v9_eclock_protocol_v2.schema.json",
    )
    reference = load_signed_json(
        reference_path,
        schema_path=schema_root / "scientific_recovery_v9_eclock_reference_v2.schema.json",
    )
    validate_protocol_reference_binding(protocol, reference, protocol_path=protocol_path)
    frames: list[pd.DataFrame] = []
    checkpoints: dict[int, str] = {}
    summaries: list[str] = []
    official_family = require_reference_family(reference, "official_a5_oof")
    for fold in (0, 1, 2):
        fold_root = run_root / f"fold-{fold}"
        summary_path = fold_root / "fold_summary.json"
        oof_path = fold_root / "oof_predictions.csv"
        if not summary_path.is_file() or not oof_path.is_file():
            raise ValueError(f"fold {fold} physical outputs are incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not verify_artifact_hash(summary):
            raise ValueError(f"fold {fold} summary signature mismatch")
        if (
            summary.get("artifact_type") != "eclock_x0_fold_summary_v2"
            or summary.get("arm_id") != arm_id
            or summary.get("outer_fold") != fold
            or summary.get("seed") != 7
            or summary.get("status") != "completed_after_frozen_checkpoint"
        ):
            raise ValueError(f"fold {fold} summary identity mismatch")
        checkpoint_path = Path(str(summary.get("checkpoint_path", "")))
        if not checkpoint_path.is_file():
            raise ValueError(f"fold {fold} checkpoint is missing")
        checkpoint_sha = compute_file_hash(str(checkpoint_path))
        if summary.get("external_official_a5") is True:
            records = official_family.get("official_fold_checkpoints")
            if not isinstance(records, list):
                raise ValueError("official A5 fold checkpoint registry is missing")
            expected = [record for record in records if int(record["outer_fold"]) == fold]
            if (
                arm_id != "X0-A5-REPLAY"
                or len(expected) != 1
                or expected[0]["file_sha256"] != checkpoint_sha
            ):
                raise ValueError(f"fold {fold} external checkpoint is not official A5")
        else:
            frozen = require_frozen_checkpoint(checkpoint_path)
            if summary.get("checkpoint_manifest_sha256") != frozen.get("artifact_sha256"):
                raise ValueError(f"fold {fold} checkpoint manifest identity mismatch")
        if (
            summary.get("checkpoint_file_sha256") != checkpoint_sha
            or summary.get("oof_file_sha256") != compute_file_hash(str(oof_path))
            or summary.get("oof_bytes") != oof_path.stat().st_size
        ):
            raise ValueError(f"fold {fold} physical hash mismatch")
        # These CSVs are a persistence boundary for signed float64 scientific
        # coordinates.  Pandas' default fast parser can move a decimal by one
        # ULP on a read -> write -> read cycle, breaking the canonical target
        # identity even when the in-memory value was unchanged.  The round-trip
        # parser restores the exact IEEE-754 value emitted by the runner.
        frame = _read_oof_csv(oof_path)
        if set(frame["outer_fold"].astype(int)) != {fold}:
            raise ValueError(f"fold {fold} OOF rows are mixed")
        frames.append(frame)
        checkpoints[fold] = checkpoint_sha
        summaries.append(summary["artifact_sha256"])
    candidate = pd.concat(frames, ignore_index=True)
    official = load_official_a5_reference_frame(
        source_root=source_root,
        family=official_family,
        checked_candidate=candidate,
        delta_t_s=float(protocol["metric"]["metric_delta_t_s"]),
    )
    candidate_identity = {
        "reference_family": arm_id,
        "path": str(run_root),
        "file_sha256": hashlib_for_strings(summaries),
        "artifact_sha256": hashlib_for_strings(checkpoints.values()),
    }
    return aggregate_verified_frame(
        candidate,
        official,
        config=config,
        protocol=protocol,
        reference=reference,
        config_sha256=compute_file_hash(str(config_path)),
        checkpoint_sha256_by_fold=checkpoints,
        candidate_identity=candidate_identity,
    )


def hashlib_for_strings(values: Iterable[object]) -> str:
    """Hash an ordered collection of already-verified identities."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "aggregate_run",
    "aggregate_verified_frame",
    "load_official_a5_reference_frame",
]
