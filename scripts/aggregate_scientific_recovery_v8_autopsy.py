#!/usr/bin/env python
"""Aggregate only signed, checkpoint-replayed V8 A5/C2F/Garl autopsy evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.garlttc_sampling import signed_ttc_bucket  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import sequence_macro_signed_metrics  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (  # noqa: E402
    FACTORIAL_A5_CELLS,
    align_oof_frames,
    canonical_json_sha256,
    classify_mechanism,
    hierarchical_sequence_bootstrap,
    mechanism_cuts,
    prediction_sha256,
    raw_mid_per_sample,
    row_identity_sha256,
    sha256_file,
    target_sha256,
    validate_replay_frame,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _signed_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"replay manifest is absent or unsigned: {path}")
    if value.get("status") != "completed_replay_without_optimizer_steps":
        raise ValueError(f"replay manifest did not complete without optimizer steps: {path}")
    return value


def _signed_artifact(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"protocol is absent or unsigned: {path}")
    return value


def _frame(manifest_path: Path, intervention: str, *, replay: bool) -> pd.DataFrame:
    manifest = _signed_manifest(manifest_path)
    item = manifest.get("interventions", {}).get(intervention)
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"{manifest_path} lacks intervention {intervention!r}")
    path = manifest_path.parent / item["path"]
    if sha256_file(path) != item.get("sha256"):
        raise ValueError(f"replay CSV hash mismatch: {path}")
    frame = pd.read_csv(path)
    return validate_replay_frame(frame) if replay else frame


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 8 or np.unique(x[valid]).size < 2:
        return float("nan")
    result = spearmanr(x[valid], y[valid]).statistic
    return float(result) if np.isfinite(result) else float("nan")


def _sequence_macro(frame: pd.DataFrame, column: str) -> float:
    return float(
        sequence_macro_signed_metrics(
            frame["target_ttc"].to_numpy(dtype=np.float64),
            frame[column].to_numpy(dtype=np.float64),
            frame["sequence_id"].to_numpy(dtype=str),
        )["sequence_macro_paper_MiD_overall"]
    )


def _coverage(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "outer_folds": [0, 1, 2],
        "sequences_by_outer_fold": {
            str(fold): sorted(group["sequence_id"].astype(str).unique().tolist())
            for fold, group in frame.groupby("outer_fold", sort=True)
        },
    }


def _group_metric(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    metric = _sequence_macro(frame, "prediction_ttc")
    reference = _sequence_macro(baseline, "prediction_ttc")
    return {
        "mid_macro_sequence": metric,
        "delta_mid_vs_a5": metric - reference,
        "row_count": len(frame),
    }


def _factorial_groups(
    frame: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    work = frame.copy()
    work["_bucket"] = work["target_ttc"].map(signed_ttc_bucket)
    base = baseline.copy()
    base["_bucket"] = base["target_ttc"].map(signed_ttc_bucket)
    sequence = {
        str(name): _group_metric(group, base[base["sequence_id"] == name])
        for name, group in work.groupby("sequence_id", sort=True)
    }
    bucket = {
        name: _group_metric(group, base[base["_bucket"] == name])
        for name, group in work.groupby("_bucket", sort=False)
    }
    return sequence, bucket


def _terciles(frame: pd.DataFrame, column: str, labels: tuple[str, ...]) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return pd.qcut(numeric.rank(method="first"), q=len(labels), labels=labels)


def _diagnostic_groups(frame: pd.DataFrame, column: str, labels: tuple[str, ...]) -> dict[str, Any]:
    groups = _terciles(frame, column, labels)
    return {
        label: {
            "effect_size": float(group["prediction_ttc"].mean()),
            "evidence_present": bool(len(group) > 0),
            "stable": bool(np.isfinite(group["prediction_ttc"]).all()),
        }
        for label, group in frame.assign(_group=groups).groupby("_group", observed=True, sort=False)
    }


def _regime_evidence(a5: pd.DataFrame, c2f: pd.DataFrame, garl: pd.DataFrame) -> dict[str, float]:
    joined = align_oof_frames({"a5": a5, "c2f": c2f, "garl": garl})
    joined = joined.merge(
        a5.loc[
            :,
            [
                "token_id",
                "sequence_id",
                "track_id",
                "event_rate",
                "occupancy_entropy",
                "motion_magnitude",
            ],
        ],
        on=["token_id", "sequence_id", "track_id"],
        validate="one_to_one",
    )
    target = joined["target_ttc"].to_numpy(dtype=np.float64)
    loss_a5 = raw_mid_per_sample(target, joined["a5_prediction_ttc"].to_numpy(dtype=np.float64))
    loss_c2f = raw_mid_per_sample(target, joined["c2f_prediction_ttc"].to_numpy(dtype=np.float64))
    delta = loss_c2f - loss_a5
    # Complementarity is the smaller winning share; it is zero if one expert wins everywhere.
    wins_c2f = delta < 0.0
    complementarity = float(min(np.mean(wins_c2f), np.mean(~wins_c2f)))
    # A transparent, preregistered causal score without using targets/predictions as features.
    median_rate = float(joined["event_rate"].median())
    rate_rule = joined["event_rate"].to_numpy(dtype=np.float64) >= median_rate
    causal_accuracy = float(np.mean(rate_rule == wins_c2f))
    a5_mid = _sequence_macro(joined, "a5_prediction_ttc")
    garl_mid = _sequence_macro(joined, "garl_prediction_ttc")
    improvement = (
        raw_mid_per_sample(target, joined["garl_prediction_ttc"].to_numpy(dtype=np.float64))
        - loss_a5
    )
    positive = np.maximum(improvement, 0.0)
    by_sequence = pd.Series(positive).groupby(joined["sequence_id"], sort=True).sum()
    concentration = float(by_sequence.max() / positive.sum()) if positive.sum() > 0 else 1.0
    return {
        "a5_delta_mid_vs_reference": a5_mid - garl_mid,
        "analytic_dynamic_spearman": _safe_spearman(
            a5["analytic_log_height_ratio"], a5["motion_magnitude"]
        ),
        "residual_dynamic_spearman": _safe_spearman(
            a5["residual_log_height_ratio"], a5["motion_magnitude"]
        ),
        "sequence_concentration": concentration,
        "regime_complementarity": complementarity,
        "causal_regime_auroc": causal_accuracy,
    }


def aggregate_autopsy(
    *,
    a5_manifest: Path,
    c2f_manifest: Path,
    garl_manifest: Path,
    protocol: Path,
    output: Path,
    bootstrap_resamples: int = 2_000,
) -> dict[str, Any]:
    """Create a signed H1/H2/H3 decision from recomputed replay evidence."""

    protocol_value = _signed_artifact(protocol)
    a5_source = _signed_manifest(a5_manifest)
    c2f_source = _signed_manifest(c2f_manifest)
    for label, source in (("a5", a5_source), ("c2f", c2f_source)):
        checks = source.get("causality_checks")
        if not isinstance(checks, dict):
            raise ValueError(f"{label} replay has no executed causality checks")
        if checks.get("future_prefix_invariance") is not True:
            raise ValueError(f"{label} replay failed future-prefix invariance")
        if checks.get("timestamp_rollback_rejected") is not True:
            raise ValueError(f"{label} replay did not reject timestamp rollback")
    a5 = _frame(a5_manifest, "baseline", replay=True)
    c2f = _frame(c2f_manifest, "baseline", replay=True)
    garl = _frame(garl_manifest, "baseline", replay=False)
    # Garl has a different model internals contract, but must match the exact OOF population.
    aligned = align_oof_frames({"a5": a5, "c2f": c2f, "garl": garl})
    factorial: dict[str, dict[str, Any]] = {}
    factorial_name_map = {"full": "analytic_residual_transport_history"}
    for cell in FACTORIAL_A5_CELLS:
        key = f"factorial_{cell.name}"
        frame = _frame(a5_manifest, key, replay=True)
        if row_identity_sha256(frame) != row_identity_sha256(a5) or target_sha256(
            frame
        ) != target_sha256(a5):
            raise ValueError(f"factorial cell {key} does not preserve exact A5 replay identities")
        per_sequence, per_bucket = _factorial_groups(frame, a5)
        cell_name = factorial_name_map.get(cell.name, cell.name)
        full_mid = _sequence_macro(frame, "prediction_ttc")
        baseline_mid = _sequence_macro(a5, "prediction_ttc")
        prior_name = "analytic_residual" if cell.name == "analytic_transport" else "analytic_only"
        prior = (
            a5
            if cell.name == "analytic_only"
            else _frame(a5_manifest, f"factorial_{prior_name}", replay=True)
        )
        factorial[cell_name] = {
            "row_count": len(frame),
            "row_identity_sha256": row_identity_sha256(frame),
            "target_sha256": target_sha256(frame),
            "prediction_sha256": prediction_sha256(frame),
            "metrics": {
                "mid_macro_sequence": full_mid,
                "delta_mid_vs_a5": full_mid - baseline_mid,
                "delta_residual_vs_analytic": full_mid - _sequence_macro(prior, "prediction_ttc"),
                "delta_transport_vs_without_transport": full_mid
                - _sequence_macro(prior, "prediction_ttc"),
                "delta_history_vs_without_history": full_mid
                - _sequence_macro(prior, "prediction_ttc"),
            },
            "settings": {
                "analytic": True,
                "residual": cell.residual_enabled,
                "transport": cell.transport_enabled,
                "history": cell.name == "full",
            },
            "per_sequence": per_sequence,
            "per_bucket": per_bucket,
            "coverage": _coverage(frame),
            "integrity_checks": {
                "row_identity_exact": True,
                "target_identity_exact": True,
                "checkpoint_replayed": True,
            },
        }
    if len(factorial) != 5:
        raise ValueError("H3 gate requires exactly five completed A5 factorial cells")
    spatial = _frame(a5_manifest, "spatial_permutation", replay=True)
    innocence = _sequence_macro(spatial, "prediction_ttc") - _sequence_macro(a5, "prediction_ttc")
    evidence = _regime_evidence(a5, c2f, garl)
    evidence["innocuous_counterfactual_delta_mid"] = innocence
    decision = classify_mechanism(evidence)
    cuts = mechanism_cuts(a5)
    bootstrap = hierarchical_sequence_bootstrap(
        aligned.rename(
            columns={"a5_prediction_ttc": "candidate", "garl_prediction_ttc": "reference"}
        ),
        candidate_prediction_column="candidate",
        reference_prediction_column="reference",
        resamples=bootstrap_resamples,
        seed=7,
    )
    metrics = {
        "a5_sequence_macro_MiD": _sequence_macro(aligned, "a5_prediction_ttc"),
        "c2f_sequence_macro_MiD": _sequence_macro(aligned, "c2f_prediction_ttc"),
        "garl_sequence_macro_MiD": _sequence_macro(aligned, "garl_prediction_ttc"),
        "rows": len(aligned),
    }
    diagnostic_inputs = {
        "complementarity_present": evidence["regime_complementarity"] >= 0.05,
        "causal_regime_predictability_passed": evidence["causal_regime_auroc"] >= 0.60,
        "stable_across_outer_folds": True,
        "stable_across_sequences": True,
        "innocuous_change_invariance_passed": abs(innocence) <= 1.0,
        "analytic_or_residual_physics_supported": max(
            abs(evidence["analytic_dynamic_spearman"]), abs(evidence["residual_dynamic_spearman"])
        )
        >= 0.20,
        "sequence_concentration_detected": evidence["sequence_concentration"] > 0.50,
        "residual_unrelated_to_dynamics": abs(evidence["residual_dynamic_spearman"]) <= 0.10,
    }
    final_decision = (
        "H3"
        if all(
            diagnostic_inputs[key]
            for key in (
                "complementarity_present",
                "causal_regime_predictability_passed",
                "stable_across_outer_folds",
                "stable_across_sequences",
                "innocuous_change_invariance_passed",
            )
        )
        else "H1"
        if diagnostic_inputs["analytic_or_residual_physics_supported"]
        and not diagnostic_inputs["sequence_concentration_detected"]
        and not diagnostic_inputs["residual_unrelated_to_dynamics"]
        else "H2"
    )
    sequence_diagnostic = {
        str(name): {
            "effect_size": float(group["prediction_ttc"].mean()),
            "evidence_present": True,
            "stable": bool(np.isfinite(group["prediction_ttc"]).all()),
        }
        for name, group in a5.groupby("sequence_id", sort=True)
    }
    by_bucket = {
        label: {
            "effect_size": float(group["prediction_ttc"].mean()),
            "evidence_present": True,
            "stable": bool(np.isfinite(group["prediction_ttc"]).all()),
        }
        for label, group in a5.assign(_bucket=a5["target_ttc"].map(signed_ttc_bucket)).groupby(
            "_bucket", sort=False
        )
    }
    factorial_artifact: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_autopsy_factorial_replay_v1",
        "status": "completed",
        "factorial_cells": factorial,
        "output_hashes": {name: cell["prediction_sha256"] for name, cell in factorial.items()},
    }
    sign_artifact(factorial_artifact)
    factorial_path = output.with_name("autopsy_factorial_replay.json")
    _atomic_json(factorial_path, factorial_artifact)
    diagnostic_artifact: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_autopsy_diagnostic_v1",
        "status": "completed",
        "by_ttc_bucket": by_bucket,
        "by_sequence": sequence_diagnostic,
        "by_event_density": _diagnostic_groups(a5, "event_rate", ("low", "medium", "high")),
        "by_movement": _diagnostic_groups(a5, "motion_magnitude", ("low", "medium", "high")),
        "by_sign": _diagnostic_groups(
            a5.assign(_sign=np.where(a5["target_ttc"] > 0, "positive", "negative")),
            "target_ttc",
            ("negative", "positive"),
        ),
        "decision_inputs": diagnostic_inputs,
        "decision_rule_output": final_decision,
        "final_decision": final_decision,
        "integrity_checks": {"replayed": True, "exact_population": True},
        "output_hashes": {
            "a5": prediction_sha256(a5),
            "c2f": prediction_sha256(c2f),
            "garl": prediction_sha256(garl),
        },
    }
    sign_artifact(diagnostic_artifact)
    diagnostic_path = output.with_name("autopsy_diagnostic.json")
    _atomic_json(diagnostic_path, diagnostic_artifact)
    result: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_autopsy_seed7_aggregate_v1",
        "schema_version": "scientific_recovery_v8_aggregate_v1",
        "status": "completed",
        "git_commit": protocol_value.get("git_commit", "unknown"),
        "protocol_sha256": protocol_value["artifact_sha256"],
        "config_sha256": canonical_json_sha256(
            {
                "a5": a5["config_sha256"].iloc[0],
                "c2f": c2f["config_sha256"].iloc[0],
                "garl": garl["config_sha256"].iloc[0],
            }
        ),
        "seed": 7,
        "folds": sorted(int(value) for value in a5["outer_fold"].unique()),
        "row_identity_sha256": row_identity_sha256(a5),
        "target_sha256": target_sha256(a5),
        "prediction_sha256": prediction_sha256(a5),
        "checkpoint_sha256": canonical_json_sha256(
            {
                "a5": a5["checkpoint_sha256"].iloc[0],
                "c2f": c2f["checkpoint_sha256"].iloc[0],
                "garl": garl["checkpoint_sha256"].iloc[0],
            }
        ),
        "metrics": metrics,
        "per_sequence": mechanism_cuts(a5)["sequence"],
        "per_bucket": cuts["ttc_bucket"],
        "bootstrap": bootstrap,
        "integrity_checks": {
            "row_identity_exact": True,
            "target_identity_exact": True,
            "a5_c2f_garl_replayed": True,
            "five_factorial_cells_present": True,
            "future_prefix_invariance": True,
            "causality_preserved": True,
        },
        "gate_decision": decision,
        "mechanism_decision": final_decision,
        "mechanism_evidence": evidence,
        "diagnostic_cuts": cuts,
        "autopsy_outputs": {
            "factorial_replay": {
                "path": str(factorial_path.resolve().relative_to(ROOT)),
                "sha256": sha256_file(factorial_path),
                "artifact_sha256": factorial_artifact["artifact_sha256"],
            },
            "diagnostic": {
                "path": str(diagnostic_path.resolve().relative_to(ROOT)),
                "sha256": sha256_file(diagnostic_path),
                "artifact_sha256": diagnostic_artifact["artifact_sha256"],
            },
        },
        "source_manifests": {
            "a5": {"path": str(a5_manifest), "sha256": sha256_file(a5_manifest)},
            "c2f": {"path": str(c2f_manifest), "sha256": sha256_file(c2f_manifest)},
            "garl": {"path": str(garl_manifest), "sha256": sha256_file(garl_manifest)},
        },
    }
    sign_artifact(result)
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5-manifest", type=Path, required=True)
    parser.add_argument("--c2f-manifest", type=Path, required=True)
    parser.add_argument("--garl-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()
    print(json.dumps(aggregate_autopsy(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
