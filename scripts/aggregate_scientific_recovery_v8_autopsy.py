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
    OOF_V8_REQUIRED_COLUMNS,
    REPLAY_MECHANISM_REQUIRED_COLUMNS,
    align_oof_frames,
    canonical_json_sha256,
    classify_mechanism,
    hierarchical_sequence_bootstrap,
    mechanism_cuts_from_pruned_replay,
    prediction_sha256,
    raw_mid_per_sample,
    row_identity_sha256,
    sha256_file,
    target_sha256,
    validate_oof_frame,
)
from e_jepa_ttc.scientific_provenance import (  # noqa: E402
    assert_autopsy_replay_producer_reusable,
    require_clean_scientific_worktree,
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


# Aggregation never consumes the high-dimensional JSON payloads carried by the
# replay CSVs (geometry_tokens, pair_tokens, transport tensors, ...).  Those
# columns dominate both file size and pandas object-memory use.  We still hash
# the complete CSV and verify that its header satisfies the complete replay
# schema, but only materialize the scalar columns used by the preregistered
# autopsy calculations.
_AUTOPSY_SCALAR_REPLAY_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (*OOF_V8_REQUIRED_COLUMNS,
         "guard_margin",
         "analytic_log_height_ratio",
         "residual_log_height_ratio",
         "occupancy_entropy",
         "motion_magnitude")
    )
)


_GARL_COMPARATOR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "token_id",
    "sequence_id",
    "track_id",
    "outer_fold",
    "seed",
    "target_ttc",
    "prediction_ttc",
    "prediction_log_variance",
    "event_rate",
)


def _frame(manifest_path: Path, intervention: str, *, replay: bool) -> pd.DataFrame:
    manifest = _signed_manifest(manifest_path)
    item = manifest.get("interventions", {}).get(intervention)
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"{manifest_path} lacks intervention {intervention!r}")
    path = manifest_path.parent / item["path"]
    if sha256_file(path) != item.get("sha256"):
        raise ValueError(f"replay CSV hash mismatch: {path}")

    # Header validation is deliberately performed against the full replay
    # contract before column pruning.  This prevents a narrow analysis read from
    # accepting a structurally incomplete replay artifact.
    header = pd.read_csv(path, nrows=0)
    required = set(OOF_V8_REQUIRED_COLUMNS)
    if replay:
        required.update(REPLAY_MECHANISM_REQUIRED_COLUMNS)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"replay CSV lacks required columns {missing}: {path}")

    if replay:
        usecols = list(_AUTOPSY_SCALAR_REPLAY_COLUMNS)
        if "category" in header.columns:
            usecols.append("category")
    else:
        usecols = list(OOF_V8_REQUIRED_COLUMNS)
    frame = pd.read_csv(path, usecols=usecols)
    # The OOF validator covers identity, target, prediction finiteness, weights,
    # config/checkpoint provenance and event-support scalars.  The omitted replay
    # mechanism tensors were already proven present in the signed, fully hashed
    # artifact header and are not inputs to any aggregate statistic.
    return validate_oof_frame(frame, label=f"{intervention} replay")


def _garl_frame(manifest_path: Path, intervention: str = "baseline") -> pd.DataFrame:
    """Load the frozen external Garl OOF binding without inventing replay provenance."""

    manifest = _signed_manifest(manifest_path)
    item = manifest.get("interventions", {}).get(intervention)
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"{manifest_path} lacks intervention {intervention!r}")
    path = manifest_path.parent / item["path"]
    if sha256_file(path) != item.get("sha256"):
        raise ValueError(f"Garl comparator CSV hash mismatch: {path}")

    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(_GARL_COMPARATOR_REQUIRED_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(f"Garl comparator lacks required columns {missing}: {path}")
    frame = pd.read_csv(path, usecols=list(_GARL_COMPARATOR_REQUIRED_COLUMNS))
    if frame.empty:
        raise ValueError("Garl comparator must contain at least one row")
    for column in ("token_id", "sequence_id", "track_id"):
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Garl comparator has empty {column}")
    if frame["token_id"].astype(str).duplicated().any():
        raise ValueError("Garl comparator has duplicate token_id")
    if frame.duplicated(["token_id", "sequence_id", "track_id"]).any():
        raise ValueError("Garl comparator has duplicate row identity")
    for column in ("outer_fold", "seed"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"Garl comparator has non-integral {column}")
        frame[column] = numeric.astype(np.int64)
    if set(frame["outer_fold"].unique().tolist()).difference({0, 1, 2}):
        raise ValueError("Garl comparator has outer_fold outside {0,1,2}")
    if not np.all(frame["seed"].to_numpy(dtype=np.int64) == 7):
        raise ValueError("Garl comparator is not the frozen seed-7 OOF binding")
    for column in ("target_ttc", "prediction_ttc", "prediction_log_variance", "event_rate"):
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Garl comparator has non-finite {column}")
        if column == "event_rate" and np.any(numeric < 0.0):
            raise ValueError("Garl comparator has negative event_rate")
        frame[column] = numeric
    return frame


def _align_with_external_garl(
    a5: pd.DataFrame, c2f: pd.DataFrame, garl: pd.DataFrame
) -> pd.DataFrame:
    """Align A5/C2F with a narrower frozen external Garl OOF comparator."""

    aligned = align_oof_frames({"a5": a5, "c2f": c2f})
    if len(garl) != len(a5):
        raise ValueError("Garl comparator row count differs from A5 OOF population")
    if row_identity_sha256(garl) != row_identity_sha256(a5):
        raise ValueError("Garl comparator row identities differ from A5")
    if target_sha256(garl) != target_sha256(a5):
        raise ValueError("Garl comparator targets differ from A5")

    reference_assignment = a5.loc[
        :, ["token_id", "sequence_id", "track_id", "outer_fold"]
    ].copy()
    comparator_assignment = garl.loc[
        :, ["token_id", "sequence_id", "track_id", "outer_fold"]
    ].copy()
    assignment = reference_assignment.merge(
        comparator_assignment,
        on=["token_id", "sequence_id", "track_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_a5", "_garl"),
    )
    if len(assignment) != len(a5) or not np.array_equal(
        assignment["outer_fold_a5"].to_numpy(dtype=np.int64),
        assignment["outer_fold_garl"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Garl comparator outer-fold assignments differ from A5")

    garl_prediction = garl.loc[
        :, ["token_id", "sequence_id", "track_id", "prediction_ttc"]
    ].rename(columns={"prediction_ttc": "garl_prediction_ttc"})
    aligned = aligned.merge(
        garl_prediction,
        on=["token_id", "sequence_id", "track_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(a5):
        raise ValueError("Garl comparator alignment unexpectedly dropped rows")
    return aligned


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


def _diagnostic_records(frame: pd.DataFrame, groups: pd.Series) -> dict[str, Any]:
    """Return schema-compatible, effect-based regime diagnostics."""

    result: dict[str, Any] = {}
    for label, group in frame.assign(_group=groups).groupby(
        "_group", observed=True, sort=False
    ):
        effect = pd.to_numeric(group["c2f_minus_a5_raw_mid"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(effect)
        win_fraction = float(np.mean(effect[finite] < 0.0)) if np.any(finite) else float("nan")
        result[str(label)] = {
            "effect_size": float(np.mean(effect[finite])) if np.any(finite) else 0.0,
            "evidence_present": bool(np.any(finite)),
            "stable": bool(np.any(finite) and 0.02 <= win_fraction <= 0.98),
        }
    return result


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return an exact rank AUROC, or NaN when either class is absent."""

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_positive = float(ranks[labels].sum())
    return float(
        (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def _auc_separability(labels: np.ndarray, scores: np.ndarray) -> float:
    auc = _binary_auc(labels, scores)
    return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")


def _regime_evidence(
    a5: pd.DataFrame, c2f: pd.DataFrame, garl: pd.DataFrame
) -> dict[str, Any]:
    joined = _align_with_external_garl(a5, c2f, garl)
    joined = joined.merge(
        a5.loc[
            :,
            [
                "token_id",
                "sequence_id",
                "track_id",
                "outer_fold",
                "event_rate",
                "occupancy_entropy",
                "motion_magnitude",
            ],
        ],
        on=["token_id", "sequence_id", "track_id"],
        validate="one_to_one",
        suffixes=("", "_mechanism"),
    )
    if "outer_fold_mechanism" in joined:
        if not np.array_equal(
            joined["outer_fold"].to_numpy(dtype=np.int64),
            joined["outer_fold_mechanism"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("autopsy regime evidence outer-fold identity mismatch")
    target = joined["target_ttc"].to_numpy(dtype=np.float64)
    loss_a5 = raw_mid_per_sample(target, joined["a5_prediction_ttc"].to_numpy(dtype=np.float64))
    loss_c2f = raw_mid_per_sample(target, joined["c2f_prediction_ttc"].to_numpy(dtype=np.float64))
    delta = loss_c2f - loss_a5
    wins_c2f = delta < 0.0
    complementarity = float(min(np.mean(wins_c2f), np.mean(~wins_c2f)))

    # Event rate is a frozen causal observable.  AUROC is reported as
    # separability (max(AUC, 1-AUC)) because either high or low event rate may
    # identify the C2F-favouring regime; no target/prediction value is a feature.
    event_rate = joined["event_rate"].to_numpy(dtype=np.float64)
    causal_auc = _auc_separability(wins_c2f, event_rate)
    per_fold_auc: dict[str, float] = {}
    for fold, group in joined.assign(_wins=wins_c2f).groupby("outer_fold", sort=True):
        per_fold_auc[str(int(fold))] = _auc_separability(
            group["_wins"].to_numpy(dtype=bool), group["event_rate"].to_numpy(dtype=np.float64)
        )
    stable_folds = bool(
        len(per_fold_auc) == 3
        and all(np.isfinite(value) and value >= 0.55 for value in per_fold_auc.values())
    )

    per_sequence_complementarity: dict[str, float] = {}
    for sequence, group in joined.assign(_wins=wins_c2f).groupby("sequence_id", sort=True):
        share = float(group["_wins"].mean())
        per_sequence_complementarity[str(sequence)] = min(share, 1.0 - share)
    stable_sequence_fraction = float(
        np.mean([value >= 0.02 for value in per_sequence_complementarity.values()])
    ) if per_sequence_complementarity else 0.0
    stable_sequences = bool(stable_sequence_fraction >= 0.75)

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
        "causal_regime_auroc": causal_auc,
        "causal_regime_auroc_by_outer_fold": per_fold_auc,
        "stable_across_outer_folds": stable_folds,
        "per_sequence_complementarity": per_sequence_complementarity,
        "stable_sequence_fraction": stable_sequence_fraction,
        "stable_across_sequences": stable_sequences,
    }


def aggregate_autopsy(
    *,
    a5_manifest: Path,
    c2f_manifest: Path,
    garl_manifest: Path,
    protocol: Path,
    output: Path,
    bootstrap_resamples: int = 5_000,
) -> dict[str, Any]:
    """Create a signed H1/H2/H3 decision from recomputed replay evidence."""

    protocol_value = _signed_artifact(protocol)
    sample_contract = protocol_value["sample_contract"]
    closed_evaluation = protocol_value["closed_evaluation"]
    protocol_file_sha256 = sha256_file(protocol)
    fold_defs = {
        str(item["fold"]): sorted(item["dev_sequence_ids"])
        for item in sample_contract["fold_definitions"]
    }
    exact_coverage = {
        "outer_folds": [0, 1, 2],
        "sequences_by_outer_fold": fold_defs,
        "sealed_evaluation_closed": True,
    }
    a5_source = _signed_manifest(a5_manifest)
    c2f_source = _signed_manifest(c2f_manifest)
    garl_source = _signed_manifest(garl_manifest)
    producer = require_clean_scientific_worktree()
    for label, source, path in (
        ("a5", a5_source, a5_manifest),
        ("c2f", c2f_source, c2f_manifest),
        ("garl", garl_source, garl_manifest),
    ):
        assert_autopsy_replay_producer_reusable(
            source,
            expected_commit=producer["git_commit"],
            source=f"{label} replay {path}",
        )
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
    garl = _garl_frame(garl_manifest, "baseline")
    # Garl has a different model internals contract, but must match the exact OOF population.
    aligned = _align_with_external_garl(a5, c2f, garl)
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
        factorial[cell_name] = {
            "row_count": len(frame),
            "row_identity_sha256": row_identity_sha256(frame),
            "target_sha256": target_sha256(frame),
            "prediction_sha256": prediction_sha256(frame),
            "metrics": {
                "mid_macro_sequence": full_mid,
                "delta_mid_vs_a5": full_mid - baseline_mid,
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
    cell_mid = {name: float(value["metrics"]["mid_macro_sequence"]) for name, value in factorial.items()}
    factorial_contrasts = {
        "residual_on_analytic": cell_mid["analytic_residual"] - cell_mid["analytic_only"],
        "transport_on_analytic": cell_mid["analytic_transport"] - cell_mid["analytic_only"],
        "transport_given_residual": cell_mid["analytic_residual_transport"] - cell_mid["analytic_residual"],
        "residual_given_transport": cell_mid["analytic_residual_transport"] - cell_mid["analytic_transport"],
        "history_given_analytic_residual_transport": cell_mid["analytic_residual_transport_history"] - cell_mid["analytic_residual_transport"],
    }
    for name, cell in factorial.items():
        residual_delta = (
            factorial_contrasts["residual_on_analytic"]
            if name == "analytic_residual"
            else factorial_contrasts["residual_given_transport"]
            if name in {"analytic_residual_transport", "analytic_residual_transport_history"}
            else 0.0
        )
        transport_delta = (
            factorial_contrasts["transport_on_analytic"]
            if name == "analytic_transport"
            else factorial_contrasts["transport_given_residual"]
            if name in {"analytic_residual_transport", "analytic_residual_transport_history"}
            else 0.0
        )
        history_delta = (
            factorial_contrasts["history_given_analytic_residual_transport"]
            if name == "analytic_residual_transport_history"
            else 0.0
        )
        cell["metrics"].update(
            {
                "delta_residual_vs_analytic": float(residual_delta),
                "delta_transport_vs_without_transport": float(transport_delta),
                "delta_history_vs_without_history": float(history_delta),
            }
        )
    spatial = _frame(a5_manifest, "spatial_permutation", replay=True)
    innocence = _sequence_macro(spatial, "prediction_ttc") - _sequence_macro(a5, "prediction_ttc")
    evidence = _regime_evidence(a5, c2f, garl)
    evidence["innocuous_counterfactual_delta_mid"] = innocence
    decision = classify_mechanism(evidence)
    cuts = mechanism_cuts_from_pruned_replay(a5)
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
        "stable_across_outer_folds": bool(evidence["stable_across_outer_folds"]),
        "stable_across_sequences": bool(evidence["stable_across_sequences"]),
        "innocuous_change_invariance_passed": abs(innocence) <= 1.0,
        "analytic_or_residual_physics_supported": max(
            abs(evidence["analytic_dynamic_spearman"]), abs(evidence["residual_dynamic_spearman"])
        )
        >= 0.20,
        "sequence_concentration_detected": evidence["sequence_concentration"] > 0.50,
        "residual_unrelated_to_dynamics": abs(evidence["residual_dynamic_spearman"]) <= 0.10,
    }
    final_decision = str(decision["decision"])

    joined_experts = align_oof_frames({"a5": a5, "c2f": c2f})
    joined_experts = joined_experts.merge(
        a5.loc[:, ["token_id", "sequence_id", "track_id", "event_rate", "motion_magnitude"]],
        on=["token_id", "sequence_id", "track_id"],
        validate="one_to_one",
    )
    joined_experts["loss_a5"] = raw_mid_per_sample(
        joined_experts["target_ttc"], joined_experts["a5_prediction_ttc"]
    )
    joined_experts["loss_c2f"] = raw_mid_per_sample(
        joined_experts["target_ttc"], joined_experts["c2f_prediction_ttc"]
    )
    joined_experts["c2f_minus_a5_raw_mid"] = joined_experts["loss_c2f"] - joined_experts["loss_a5"]
    sequence_diagnostic = {
        str(name): {
            "effect_size": float(group["c2f_minus_a5_raw_mid"].mean()),
            "evidence_present": bool(len(group) > 0),
            "stable": bool(
                len(group) > 0
                and 0.02
                <= float((group["c2f_minus_a5_raw_mid"] < 0.0).mean())
                <= 0.98
            ),
        }
        for name, group in joined_experts.groupby("sequence_id", sort=True)
    }
    joined_experts["_bucket"] = joined_experts["target_ttc"].map(signed_ttc_bucket)
    by_bucket = {
        str(label): {
            "effect_size": float(group["c2f_minus_a5_raw_mid"].mean()),
            "evidence_present": bool(len(group) > 0),
            "stable": bool(
                len(group) > 0
                and 0.02
                <= float((group["c2f_minus_a5_raw_mid"] < 0.0).mean())
                <= 0.98
            ),
        }
        for label, group in joined_experts.groupby("_bucket", sort=False)
    }
    factorial_artifact: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_autopsy_factorial_replay_v1",
        "status": "completed",
        "protocol_artifact_sha256": protocol_value["artifact_sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "sample_contract": sample_contract,
        "closed_evaluation": closed_evaluation,
        "factorial_cells": factorial,
        "factorial_contrasts_mid": factorial_contrasts,
        "output_hashes": {name: cell["prediction_sha256"] for name, cell in factorial.items()},
    }
    sign_artifact(factorial_artifact)
    factorial_path = output.with_name("autopsy_factorial_replay.json")
    _atomic_json(factorial_path, factorial_artifact)
    diagnostic_artifact: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_autopsy_diagnostic_v1",
        "status": "completed",
        "protocol_artifact_sha256": protocol_value["artifact_sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "sample_contract": sample_contract,
        "closed_evaluation": closed_evaluation,
        "by_ttc_bucket": by_bucket,
        "by_sequence": sequence_diagnostic,
        "by_event_density": _diagnostic_records(
            joined_experts,
            _terciles(joined_experts, "event_rate", ("low", "medium", "high")),
        ),
        "by_movement": _diagnostic_records(
            joined_experts,
            _terciles(joined_experts, "motion_magnitude", ("low", "medium", "high")),
        ),
        "by_sign": _diagnostic_records(
            joined_experts,
            pd.Series(
                np.where(joined_experts["target_ttc"] > 0, "positive", "negative"),
                index=joined_experts.index,
            ),
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
        "schema_version": protocol_value["schema_version"],
        "status": "completed",
        "stage": "autopsy",
        "arm": "autopsy",
        "candidate_id": "A_AUTOPSY",
        "git_commit": protocol_value["git_base_commit"],
        "producer_git_commit": producer["git_commit"],
        "producer_git_dirty": producer["git_dirty"],
        "protocol_sha256": protocol_value["artifact_sha256"],
        "protocol_artifact_sha256": protocol_value["artifact_sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "config_sha256": canonical_json_sha256(
            {
                "a5": a5["config_sha256"].iloc[0],
                "c2f": c2f["config_sha256"].iloc[0],
                "garl_external_binding_manifest_sha256": sha256_file(garl_manifest),
            }
        ),
        "seed": 7,
        "folds": {
            label: {
                "status": "completed",
                "sequence_ids": sequences,
                "row_count": int(sample_contract["row_count_contract"]["by_outer_fold"][label]),
            }
            for label, sequences in fold_defs.items()
        },
        "row_count": int(sample_contract["rows"]),
        "row_identity_sha256": sample_contract["row_identity_sha256"],
        "target_identity_sha256": sample_contract["target_identity_sha256"],
        "target_sha256": sample_contract["target_sha256"],
        "mid_sample_weight_sha256": sample_contract["mid_sample_weight_sha256"],
        "fold_assignment_sha256": sample_contract["fold_assignment_sha256"],
        "prediction_sha256": prediction_sha256(a5),
        "checkpoint_sha256": canonical_json_sha256(
            {
                "a5": a5["checkpoint_sha256"].iloc[0],
                "c2f": c2f["checkpoint_sha256"].iloc[0],
                "garl_frozen_oof_source_sha256": protocol_value["sources"][
                    "garl_oof_predictions"
                ]["sha256"],
            }
        ),
        "metrics": metrics,
        "per_sequence": cuts["sequence"],
        "per_bucket": cuts["ttc_bucket"],
        "bootstrap": bootstrap,
        "integrity_checks": {
            "row_identity_exact": True,
            "target_identity_exact": True,
            "a5_c2f_replayed_garl_frozen_oof_bound": True,
            "garl_comparator_kind": "frozen_external_oof_binding",
            "five_factorial_cells_present": True,
            "future_prefix_invariance": True,
            "causality_preserved": True,
            "producer_git_commit_matches_head": (
                producer["git_commit"] == a5_source.get("git_commit")
                and producer["git_commit"] == c2f_source.get("git_commit")
                and producer["git_commit"] == garl_source.get("git_commit")
            ),
        },
        "gate_decision": decision,
        "mechanism_decision": final_decision,
        "mechanism_evidence": evidence,
        "diagnostic_cuts": cuts,
        "coverage": exact_coverage,
        "sample_contract": sample_contract,
        "closed_evaluation": closed_evaluation,
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
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    args = parser.parse_args()
    try:
        result = aggregate_autopsy(**vars(args))
    except BaseException as error:
        print(
            f"V8 autopsy aggregation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
