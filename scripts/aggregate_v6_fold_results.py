#!/usr/bin/env python
"""Aggregate clean V6.1 and A5 fold-local OOF results against V5 and Garl."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from scripts.aggregate_v5_fold_results import (  # noqa: E402
    _git_revision,
    _metrics,
    _read_predictions,
    _sha256,
    _summary,
    _token_sha,
    _validate_summary,
    align_fold_predictions,
)
from scripts.paired_grouped_bootstrap import run as run_pair  # noqa: E402

ARMS = ("a5_causal", "v6_1", "a8_0", "a6", "garl")
PAIRINGS = (
    ("v6_1", "a8_0"),
    ("v6_1", "garl"),
    ("a5_causal", "v6_1"),
    ("a5_causal", "a8_0"),
    ("a5_causal", "a6"),
    ("a5_causal", "garl"),
)
RUN_NAMES = {
    "a5_causal": "scientific_recovery_v6_a5_causal_grouped_fold{fold}_seed7",
    "v6_1": "scientific_recovery_v6_1_dual_transport_r2_fold{fold}_seed7",
    "a8_0": "scientific_recovery_v5_a8_0_fold_chain_fold{fold}_seed7",
    "a6": "scientific_recovery_v5_a6_fold_chain_fold{fold}_seed7",
    "garl": "scientific_recovery_v5_garl_fold_chain_fold{fold}_seed7",
}


def _signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"invalid signed artifact: {path}")
    return payload


def _validate_frozen_inputs(
    grouped_protocol: dict[str, Any],
    v6_manifest: dict[str, Any],
    v5_aggregate: dict[str, Any],
) -> None:
    if grouped_protocol.get("status") != "frozen_before_a8_results":
        raise ValueError("the grouped protocol was not frozen before A8")
    if v6_manifest.get("status") != "frozen_before_v6_training":
        raise ValueError("V6 run configs were not frozen before training")
    contracts = v6_manifest.get("contracts", {})
    required = {
        "a5_causal_is_diagnostic_geometry_unconstrained": True,
        "folds_unchanged_from_v5": True,
        "private_test_opened": False,
        "public_validation_opened": False,
        "v6_1_single_change_transport_radius_1_to_2": True,
    }
    if any(contracts.get(key) is not expected for key, expected in required.items()):
        raise ValueError("V6 manifest contracts do not match the frozen experiment")
    if v5_aggregate.get("status") != "completed_development_gate_evaluation":
        raise ValueError("V5 aggregate is incomplete")
    if v5_aggregate.get("contracts", {}).get("private_test_opened") is not False:
        raise ValueError("V5 aggregate opened private/test")


def _validate_v6_summary(
    payload: dict[str, Any],
    *,
    arm: str,
    fold: int,
    manifest: dict[str, Any],
) -> None:
    key = "a5_causal" if arm == "a5_causal" else "v6_1_r2"
    config_record = manifest["configs"][f"{key}_fold{fold}"]
    if payload.get("config", {}).get("sha256") != config_record["sha256"]:
        raise ValueError(f"{arm} fold {fold} config differs from the frozen manifest")
    contract = payload.get("decision_contract", {})
    if contract.get("public_validation_used_for_selection") is not False:
        raise ValueError(f"{arm} fold {fold} used public validation")
    if contract.get("private_test_remains_closed") is not True:
        raise ValueError(f"{arm} fold {fold} does not attest a closed private test")
    if arm == "a5_causal":
        if contract.get("diagnostic_only_until_geometry_is_reassessed") is not True:
            raise ValueError("A5 causal is missing its diagnostic-only contract")
        if contract.get("geometry_preservation_required") is not False:
            raise ValueError("A5 causal incorrectly claims required geometry preservation")
    else:
        change = contract.get("representation_change", {})
        dual = contract.get("dual_stream_contract", {})
        if int(change.get("transport_radius", -1)) != 2:
            raise ValueError(f"V6.1 fold {fold} is not radius 2")
        if dual.get("geometry_must_equal_parent_by_construction") is not True:
            raise ValueError(f"V6.1 fold {fold} lacks the geometry-preservation contract")


def _geometry_summary(summaries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    keys = ("delta_log_height_vs_bbox", "delta_log_height_vs_physical")
    result: dict[str, Any] = {}
    for key in keys:
        folds = [
            summary["dev_metrics"]["geometry_diagnostics"][key]["macro_by_sequence"]
            for summary in summaries.values()
        ]
        result[key] = {
            metric: float(np.mean([float(fold[metric]) for fold in folds]))
            for metric in (
                "mae",
                "pearson",
                "prediction_target_std_ratio",
                "sign_accuracy",
                "slope",
            )
        }
    return result


def aggregate(
    *,
    run_root: Path,
    grouped_protocol_path: Path,
    v6_manifest_path: Path,
    v5_aggregate_path: Path,
    historical_a5_pair_path: Path,
    cluster_metadata: Path,
    audit_dir: Path,
    output_dir: Path,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Build the exact OOF aggregate and paired sequence+track comparisons."""

    grouped_protocol = _signed(grouped_protocol_path)
    v6_manifest = _signed(v6_manifest_path)
    v5_aggregate = _signed(v5_aggregate_path)
    historical_a5_pair = _signed(historical_a5_pair_path)
    _validate_frozen_inputs(grouped_protocol, v6_manifest, v5_aggregate)
    fold_contracts = {int(item["fold"]): item for item in grouped_protocol["folds"]}
    frames: dict[str, list[pd.DataFrame]] = {arm: [] for arm in ARMS}
    summaries: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    sources: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(3):
        current: dict[str, pd.DataFrame] = {}
        for arm in ARMS:
            run_dir = run_root / RUN_NAMES[arm].format(fold=fold)
            prediction = run_dir / (
                "dev_predictions.parquet" if arm == "garl" else "dev_predictions.csv"
            )
            summary_path = run_dir / "summary.json"
            current[arm] = _read_predictions(prediction)
            summaries[arm][fold] = _summary(summary_path)
            _validate_summary(summaries[arm][fold], arm=arm, fold=fold, prediction=prediction)
            if arm in {"a5_causal", "v6_1"}:
                _validate_v6_summary(summaries[arm][fold], arm=arm, fold=fold, manifest=v6_manifest)
            sources[f"{arm}_fold{fold}"] = {
                "predictions": {"path": str(prediction.resolve()), "sha256": _sha256(prediction)},
                "summary": {
                    "path": str(summary_path.resolve()),
                    "sha256": _sha256(summary_path),
                    "artifact_sha256": summaries[arm][fold]["artifact_sha256"],
                },
            }
        current = align_fold_predictions(current)
        contract = fold_contracts[fold]
        for arm, frame in current.items():
            if len(frame) != int(contract["dev_rows"]):
                raise ValueError(f"{arm} fold {fold} row count differs from protocol")
            if _token_sha(frame["sample_token"]) != contract["dev_sample_tokens_sha256"]:
                raise ValueError(f"{arm} fold {fold} tokens differ from protocol")
            frames[arm].append(frame.assign(fold=fold))

    models: dict[str, Any] = {}
    oof_paths: dict[str, Path] = {}
    for arm, parts in frames.items():
        oof = pd.concat(parts, ignore_index=True)
        if (
            len(oof) != int(grouped_protocol["sample_count"])
            or oof["sample_token"].duplicated().any()
        ):
            raise ValueError(f"{arm} OOF population is not an exact partition")
        if _token_sha(oof["sample_token"]) != grouped_protocol["sorted_sample_tokens_sha256"]:
            raise ValueError(f"{arm} OOF token universe differs from protocol")
        path = output_dir / f"{arm}_outer_dev_predictions.csv"
        oof.drop(columns="fold").to_csv(path, index=False)
        oof_paths[arm] = path
        fold_metrics = [_metrics(part) for part in parts]
        mids = np.asarray([item["sequence_macro_MiD"] for item in fold_metrics])
        models[arm] = {
            "folds": {str(index): value for index, value in enumerate(fold_metrics)},
            "fold_MiD_mean": float(mids.mean()),
            "fold_MiD_sample_std": float(mids.std(ddof=1)),
            "worst_fold_MiD": float(mids.max()),
            "outer_dev_9_sequence": _metrics(oof),
            "parameter_count": int(
                summaries[arm][0].get(
                    "parameter_count",
                    summaries[arm][0].get("resources", {}).get("parameter_count", 0),
                )
            ),
        }
    for arm in ("a5_causal", "v6_1", "a8_0"):
        models[arm]["geometry_diagnostics"] = _geometry_summary(summaries[arm])
    models["a5_causal"]["claim_scope"] = "diagnostic_geometry_unconstrained"
    models["v6_1"]["geometry_claim_scope"] = "exact_frozen_fold_parent"

    paired: dict[str, Any] = {}
    for first, second in PAIRINGS:
        key = f"{first}_vs_{second}"
        path = output_dir / f"paired_{key}_outer_dev.json"
        result = run_pair(
            oof_paths[first],
            oof_paths[second],
            path,
            first_label=first,
            second_label=second,
            fold=None,
            resamples=resamples,
            seed=seed,
            cluster_metadata=cluster_metadata,
            protocol=grouped_protocol_path,
        )
        paired[key] = {
            "path": str(path.resolve()),
            "artifact_sha256": result["artifact_sha256"],
            "delta_first_minus_second": result["delta_first_minus_second"],
            "bootstrap": result["bootstrap"],
        }

    geometry_paths = [audit_dir / f"geometry_v6_1_fold{fold}.json" for fold in range(3)]
    for path in geometry_paths:
        audit = _signed(path)
        if audit.get("status") != "completed_exact_primary_geometry" or audit.get("arm") != "v6_1":
            raise ValueError(f"V6.1 geometry audit failed: {path}")
    for name in ("prefix_causality_v6_1.json", "prefix_causality_a5_causal.json"):
        audit = _signed(audit_dir / name)
        if audit.get("status") != "PASS":
            raise ValueError(f"prefix causality audit failed: {name}")

    v6_mid = models["v6_1"]["outer_dev_9_sequence"]["sequence_macro_MiD"]
    a8_mid = models["a8_0"]["outer_dev_9_sequence"]["sequence_macro_MiD"]
    a5_mid = models["a5_causal"]["outer_dev_9_sequence"]["sequence_macro_MiD"]
    garl_mid = models["garl"]["outer_dev_9_sequence"]["sequence_macro_MiD"]
    gate_checks = {
        "improves_a8_aggregate": v6_mid < a8_mid,
        "first_stage_MiD_le_175": v6_mid <= 175.0,
        "strong_target_MiD_le_160": v6_mid <= 160.0,
        "geometry_exact_parent": True,
        "model_prefix_causality": True,
        "coverage_not_materially_worse": models["v6_1"]["outer_dev_9_sequence"]["coverage"]
        >= models["a8_0"]["outer_dev_9_sequence"]["coverage"] - 0.01,
        "public_validation_used_for_selection": False,
        "private_test_opened": False,
    }
    required_gate = (
        "improves_a8_aggregate",
        "first_stage_MiD_le_175",
        "geometry_exact_parent",
        "model_prefix_causality",
        "coverage_not_materially_worse",
    )
    clean_ttc_order = sorted(
        (arm for arm in ARMS),
        key=lambda arm: models[arm]["outer_dev_9_sequence"]["sequence_macro_MiD"],
    )
    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v6_fold_chain_aggregate_v1",
        "status": "completed_development_gate_evaluation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "aggregation_revision": _git_revision(),
        "models": models,
        "paired_outer_dev": paired,
        "clean_ttc_ranking": clean_ttc_order,
        "v6_1_gate": {
            "checks": gate_checks,
            "decision": "PASS" if all(gate_checks[key] for key in required_gate) else "FAIL",
        },
        "a5_assessment": {
            "clean_fold_local_ttc_best_ejepa": a5_mid < min(v6_mid, a8_mid),
            "clean_fold_local_ttc_beats_garl": a5_mid < garl_mid,
            "promotion_eligible": False,
            "reason": "geometry_unconstrained_diagnostic_comparator",
            "historical_result_is_separate_population_and_protocol": True,
            "historical_paired_artifact": {
                "path": str(historical_a5_pair_path.resolve()),
                "sha256": _sha256(historical_a5_pair_path),
                "artifact_sha256": historical_a5_pair["artifact_sha256"],
                "claim_scope": historical_a5_pair.get("claim_scope", "diagnostic_only"),
            },
        },
        "protocol": {
            "grouped_dev": {
                "path": str(grouped_protocol_path.resolve()),
                "sha256": _sha256(grouped_protocol_path),
                "artifact_sha256": grouped_protocol["artifact_sha256"],
            },
            "v6_runs": {
                "path": str(v6_manifest_path.resolve()),
                "sha256": _sha256(v6_manifest_path),
                "artifact_sha256": v6_manifest["artifact_sha256"],
            },
        },
        "contracts": {
            "outer_dev_is_development_not_test": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "strict_end_to_end_streaming_causality_claimed": False,
            "garl_preprocessing_identical": False,
            "garl_evaluation_and_oracle_roi_privilege_matched": True,
            "sota_claim_authorized": False,
        },
        "sources": sources,
    }
    if not all(
        math.isfinite(float(models[arm]["outer_dev_9_sequence"]["sequence_macro_MiD"]))
        for arm in ARMS
    ):
        raise ValueError("an arm has non-finite aggregate MiD")
    sign_artifact(report)
    output = output_dir / "aggregate.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts/runs")
    parser.add_argument("--grouped-protocol", type=Path, required=True)
    parser.add_argument("--v6-manifest", type=Path, required=True)
    parser.add_argument("--v5-aggregate", type=Path, required=True)
    parser.add_argument("--historical-a5-pair", type=Path, required=True)
    parser.add_argument("--cluster-metadata", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    try:
        report = aggregate(
            run_root=args.run_root.resolve(strict=True),
            grouped_protocol_path=args.grouped_protocol.resolve(strict=True),
            v6_manifest_path=args.v6_manifest.resolve(strict=True),
            v5_aggregate_path=args.v5_aggregate.resolve(strict=True),
            historical_a5_pair_path=args.historical_a5_pair.resolve(strict=True),
            cluster_metadata=args.cluster_metadata.resolve(strict=True),
            audit_dir=args.audit_dir.resolve(strict=True),
            output_dir=args.output_dir.resolve(),
            resamples=args.resamples,
            seed=args.seed,
        )
    except Exception as error:
        parser.exit(2, f"V6 fold aggregation failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "aggregate.json"),
                "gate": report["v6_1_gate"],
                "a5": report["a5_assessment"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
