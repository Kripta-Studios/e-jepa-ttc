"""Signed, fail-closed primary X0 comparison of DYN-U against matched BASE-U."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_bootstrap import paired_hierarchical_mid_bootstrap
from e_jepa_ttc.evaluation.collision_clock_config import load_x0_config, validate_matched_base_dyn
from e_jepa_ttc.evaluation.collision_clock_gates import evaluate_x0_primary_dyn_vs_base_gate
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    load_signed_json,
    precheck_production_oof,
    production_sequence_macro_metrics,
    validate_protocol_reference_binding,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS
from e_jepa_ttc.training.collision_clock_eap import require_frozen_checkpoint


def _load_arm(
    run_root: Path,
    *,
    arm_id: str,
    config_sha256: str,
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[int, str], str]:
    aggregate_path = run_root / "aggregate.json"
    if not aggregate_path.is_file():
        raise ValueError(f"{arm_id} aggregate is missing")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if (
        not isinstance(aggregate, dict)
        or not verify_artifact_hash(aggregate)
        or aggregate.get("artifact_type") != "eclock_x0_aggregate_v2"
        or aggregate.get("arm_id") != arm_id
        or aggregate.get("config_sha256") != config_sha256
        or aggregate.get("protocol_sha256") != protocol.get("artifact_sha256")
        or aggregate.get("reference_sha256") != reference.get("artifact_sha256")
        or aggregate.get("integrity_chain_complete") is not True
    ):
        raise ValueError(f"{arm_id} aggregate identity mismatch")
    frames: list[pd.DataFrame] = []
    checkpoint_hashes: dict[int, str] = {}
    commits: set[str] = set()
    for fold in (0, 1, 2):
        fold_root = run_root / f"fold-{fold}"
        summary_path = fold_root / "fold_summary.json"
        predictions_path = fold_root / "oof_predictions.csv"
        if not summary_path.is_file() or not predictions_path.is_file():
            raise ValueError(f"{arm_id} fold {fold} is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not verify_artifact_hash(summary):
            raise ValueError(f"{arm_id} fold {fold} summary signature mismatch")
        if (
            summary.get("arm_id") != arm_id
            or summary.get("outer_fold") != fold
            or summary.get("seed") != 7
            or summary.get("external_official_a5") is not False
            or summary.get("outer_dev_evaluations") != 1
        ):
            raise ValueError(f"{arm_id} fold {fold} summary identity mismatch")
        if summary.get("oof_file_sha256") != compute_file_hash(str(predictions_path)):
            raise ValueError(f"{arm_id} fold {fold} prediction SHA mismatch")
        checkpoint = Path(str(summary.get("checkpoint_path", "")))
        manifest = require_frozen_checkpoint(checkpoint)
        identity = manifest.get("scientific_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{arm_id} fold {fold} checkpoint identity is missing")
        if (
            identity.get("arm_id") != arm_id
            or identity.get("outer_fold") != fold
            or identity.get("seed") != 7
            or identity.get("config_sha256") != config_sha256
            or identity.get("protocol_sha256") != protocol.get("artifact_sha256")
        ):
            raise ValueError(f"{arm_id} fold {fold} checkpoint identity mismatch")
        commits.add(str(identity.get("git_commit_observed", "")))
        checkpoint_hashes[fold] = str(manifest["checkpoint_file_sha256"])
        frames.append(pd.read_csv(predictions_path))
    if len(commits) != 1 or "" in commits:
        raise ValueError(f"{arm_id} folds do not share one training commit")
    frame = pd.concat(frames, ignore_index=True)
    checked = precheck_production_oof(
        frame,
        protocol=protocol,
        reference=reference,
        arm_id=arm_id,
        config_sha256=config_sha256,
        checkpoint_sha256_by_fold=checkpoint_hashes,
    )
    aggregate_mid = float(aggregate["metrics"]["sequence_macro_paper_MiD_overall"])
    observed_mid = float(
        production_sequence_macro_metrics(checked)["sequence_macro_paper_MiD_overall"]
    )
    if not np.isclose(aggregate_mid, observed_mid, rtol=0, atol=1e-12):
        raise ValueError(f"{arm_id} aggregate MiD differs from physical OOF rows")
    expected_checkpoints = {str(key): value for key, value in checkpoint_hashes.items()}
    if aggregate.get("checkpoint_sha256_by_fold") != expected_checkpoints:
        raise ValueError(f"{arm_id} aggregate checkpoint set mismatch")
    return checked, checkpoint_hashes, next(iter(commits))


def _group_deltas(aligned: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        str(key): float(group["dyn_mid"].mean() - group["base_mid"].mean())
        for key, group in aligned.groupby(column, sort=True)
    }


def _checkpoint_set_sha256(checkpoints: Mapping[int, str]) -> str:
    payload = json.dumps(
        {str(key): value for key, value in sorted(checkpoints.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_dyn_vs_base(
    *,
    base_run_root: Path,
    dyn_run_root: Path,
    base_config_path: Path,
    dyn_config_path: Path,
    protocol_path: Path,
    reference_path: Path,
    schema_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recalculate and sign the preregistered primary comparison from physical CSVs."""

    config_schema = schema_root / "scientific_recovery_v9_eclock_config_v2.schema.json"
    base_config = load_x0_config(base_config_path, schema_path=config_schema)
    dyn_config = load_x0_config(dyn_config_path, schema_path=config_schema)
    validate_matched_base_dyn(base_config, dyn_config)
    protocol = load_signed_json(
        protocol_path,
        schema_path=schema_root / "scientific_recovery_v9_eclock_protocol_v2.schema.json",
    )
    reference = load_signed_json(
        reference_path,
        schema_path=schema_root / "scientific_recovery_v9_eclock_reference_v2.schema.json",
    )
    validate_protocol_reference_binding(protocol, reference, protocol_path=protocol_path)
    if protocol.get("primary_comparison") != "X0-DYN-U_vs_X0-BASE-U":
        raise ValueError("signed protocol primary comparison is not DYN-U vs BASE-U")
    base_sha = compute_file_hash(str(base_config_path))
    dyn_sha = compute_file_hash(str(dyn_config_path))
    base, base_checkpoints, base_commit = _load_arm(
        base_run_root,
        arm_id="X0-BASE-U",
        config_sha256=base_sha,
        protocol=protocol,
        reference=reference,
    )
    dyn, dyn_checkpoints, dyn_commit = _load_arm(
        dyn_run_root,
        arm_id="X0-DYN-U",
        config_sha256=dyn_sha,
        protocol=protocol,
        reference=reference,
    )
    if base_commit != dyn_commit:
        raise ValueError("BASE/DYN training commits differ")
    identity_columns = [
        "sample_token",
        "sequence_id",
        "track_id",
        "outer_fold",
        "target_ttc_s",
        "target_benchmark_phase",
        "sample_weight",
    ]
    aligned = base[identity_columns + ["scientific_mid_per_row", "scientific_failure"]].merge(
        dyn[identity_columns + ["scientific_mid_per_row", "scientific_failure"]],
        on="sample_token",
        suffixes=("_base", "_dyn"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if set(aligned["_merge"]) != {"both"}:
        raise ValueError("BASE/DYN token universes differ")
    for column in identity_columns[1:]:
        left = aligned[f"{column}_base"].to_numpy()
        right = aligned[f"{column}_dyn"].to_numpy()
        if np.issubdtype(left.dtype, np.number):
            equal = np.allclose(
                left.astype(np.float64), right.astype(np.float64), rtol=0, atol=1e-12
            )
        else:
            equal = np.array_equal(left.astype(str), right.astype(str))
        if not equal:
            raise ValueError(f"BASE/DYN {column} mismatch")
    bootstrap = paired_hierarchical_mid_bootstrap(
        dyn,
        base,
        protocol=protocol,
        candidate_identity={
            "reference_family": "X0-DYN-U",
            "path": str(dyn_run_root),
            "file_sha256": compute_file_hash(str(dyn_run_root / "aggregate.json")),
            "artifact_sha256": _checkpoint_set_sha256(dyn_checkpoints),
        },
        reference_identity={
            "reference_family": "X0-BASE-U",
            "path": str(base_run_root),
            "file_sha256": compute_file_hash(str(base_run_root / "aggregate.json")),
            "artifact_sha256": _checkpoint_set_sha256(base_checkpoints),
        },
    )
    base_metrics = production_sequence_macro_metrics(base)
    dyn_metrics = production_sequence_macro_metrics(dyn)
    base_mid = float(base_metrics["sequence_macro_paper_MiD_overall"])
    dyn_mid = float(dyn_metrics["sequence_macro_paper_MiD_overall"])
    aligned = aligned.rename(
        columns={
            "scientific_mid_per_row_base": "base_mid",
            "scientific_mid_per_row_dyn": "dyn_mid",
            "outer_fold_base": "outer_fold",
            "sequence_id_base": "sequence_id",
            "track_id_base": "track_id",
            "target_ttc_s_base": "target_ttc_s",
        }
    )
    bucket = np.full(len(aligned), "", dtype=object)
    target = aligned["target_ttc_s"].to_numpy(dtype=np.float64)
    for name, lower, upper in BUCKETS:
        bucket[(target > lower) & (target <= upper)] = name
    aligned["ttc_bucket"] = bucket.astype(str)
    aligned["sequence_track"] = (
        aligned["sequence_id"].astype(str) + "/" + aligned["track_id"].astype(str)
    )
    track_scores = cast(
        pd.DataFrame,
        aligned.groupby(["sequence_id", "track_id"], sort=True)[["base_mid", "dyn_mid"]].mean(),
    )
    oracle = base.sort_values("sample_token").copy()
    oracle["scientific_mid_per_row"] = np.minimum(
        base.sort_values("sample_token")["scientific_mid_per_row"].to_numpy(dtype=np.float64),
        dyn.sort_values("sample_token")["scientific_mid_per_row"].to_numpy(dtype=np.float64),
    )
    paired = bootstrap["delta_candidate_minus_reference"]
    base_failure = float(np.mean(base["scientific_failure"].to_numpy(dtype=np.float64)))
    dyn_failure = float(np.mean(dyn["scientific_failure"].to_numpy(dtype=np.float64)))
    row_win_rate = float(
        np.mean(
            aligned["dyn_mid"].to_numpy(dtype=np.float64)
            < aligned["base_mid"].to_numpy(dtype=np.float64)
        )
    )
    track_win_rate = float(
        np.mean(
            track_scores["dyn_mid"].to_numpy(dtype=np.float64)
            < track_scores["base_mid"].to_numpy(dtype=np.float64)
        )
    )
    error_correlation = float(
        np.corrcoef(
            aligned["base_mid"].to_numpy(dtype=np.float64),
            aligned["dyn_mid"].to_numpy(dtype=np.float64),
        )[0, 1]
    )
    gate = evaluate_x0_primary_dyn_vs_base_gate(
        {
            "row_count": len(aligned),
            "base_finite_fraction": 1.0,
            "dyn_finite_fraction": 1.0,
            "base_failure_rate": base_failure,
            "dyn_failure_rate": dyn_failure,
            "coverage_delta_pp": 0.0,
            "finite_draw_fraction": paired["finite_draw_fraction"],
            "paired_ci95_upper": paired["ci95_high"],
            "primary_comparison_signed": True,
            "identity_hashes_exact": True,
            "matched_config_contract": True,
            "paired_identical_draws": bootstrap["paired_identical_draws"],
            "incomplete_draws_disclosed": bootstrap["incomplete_draws_disclosed"],
            "official_a5_not_used_as_primary_reference": True,
        }
    )
    comparison = sign_artifact(
        {
            "artifact_type": "eclock_x0_cross_arm_comparison_v1",
            "status": "complete",
            "primary_comparison": "X0-DYN-U_vs_X0-BASE-U",
            "git_commit": base_commit,
            "seed": 7,
            "row_count": len(aligned),
            "protocol_sha256": protocol["artifact_sha256"],
            "reference_sha256": reference["artifact_sha256"],
            "base_config_sha256": base_sha,
            "dyn_config_sha256": dyn_sha,
            "base_checkpoint_sha256_by_fold": {str(k): v for k, v in base_checkpoints.items()},
            "dyn_checkpoint_sha256_by_fold": {str(k): v for k, v in dyn_checkpoints.items()},
            "base_mid": base_mid,
            "dyn_mid": dyn_mid,
            "delta_mid_dyn_minus_base": dyn_mid - base_mid,
            "base_failure_rate": base_failure,
            "dyn_failure_rate": dyn_failure,
            "coverage_delta_pp": 0.0,
            "bootstrap": bootstrap,
            "fold_deltas": _group_deltas(aligned, "outer_fold"),
            "sequence_deltas": _group_deltas(aligned, "sequence_id"),
            "bucket_deltas": _group_deltas(aligned, "ttc_bucket"),
            "track_deltas": _group_deltas(aligned, "sequence_track"),
            "row_win_rate": row_win_rate,
            "track_win_rate": track_win_rate,
            "error_correlation": error_correlation,
            "oracle_base_dyn_mid_diagnostic": float(
                production_sequence_macro_metrics(oracle)["sequence_macro_paper_MiD_overall"]
            ),
            "gate_decision": gate,
        }
    )
    gate_artifact = sign_artifact(
        {
            "artifact_type": "eclock_x0_cross_arm_gate_v1",
            "comparison_artifact_sha256": comparison["artifact_sha256"],
            "primary_comparison": comparison["primary_comparison"],
            **gate,
        }
    )
    return comparison, gate_artifact


__all__ = ["compare_dyn_vs_base"]
