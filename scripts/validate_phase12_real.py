"""Validate the non-synthetic evidence contract for PLAN.md Phases 9 and 12.

The validator is deliberately read-only.  It never trains, opens a dataset, or
writes an audit artifact.  Its purpose is to fail closed when a fixture smoke,
an old diagnostic run, or an incomplete freeze is accidentally presented as
final real-data evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_LOW_LABEL_FRACTIONS = frozenset({0.01, 0.05, 0.10, 0.25, 1.0})
REQUIRED_SEEDS = frozenset({7, 13, 23})
REAL_DATASET_SCOPES = frozenset({"real", "eap", "garl", "evttc"})


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    """Hash a signed JSON object after removing its signature field."""

    unsigned = dict(payload)
    unsigned.pop(field, None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, [f"missing file: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid JSON {path}: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"JSON root must be an object: {path}"]
    return payload, []


def _signed_real_artifact(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    payload, errors = _read_json(path)
    if payload is None:
        return None, [f"{label}: {error}" for error in errors]
    signature = payload.get("artifact_sha256")
    if not isinstance(signature, str) or signature != _canonical_hash(payload, "artifact_sha256"):
        errors.append(f"{label}: artifact_sha256 missing or invalid")
    if payload.get("metrics_are_not_real_dataset_results") is True:
        errors.append(f"{label}: explicitly marked as non-real dataset evidence")
    evidence_type = str(payload.get("evidence_type", "")).lower()
    if any(token in evidence_type for token in ("synthetic", "fixture", "smoke")):
        errors.append(f"{label}: evidence_type={evidence_type!r} is diagnostic only")
    scope = payload.get("scope")
    scope_mapping = scope if isinstance(scope, dict) else {}
    dataset_scope = str(
        payload.get(
            "dataset_scope",
            payload.get("data_scope", scope_mapping.get("dataset_scope", "")),
        )
    ).lower()
    real_marker = bool(
        payload.get("real_dataset") is True
        or payload.get("real_validation_samples") is True
        or scope_mapping.get("real_dataset") is True
        or dataset_scope in REAL_DATASET_SCOPES
    )
    if not real_marker:
        errors.append(f"{label}: no explicit real-dataset marker")
    return payload, errors


def _check_freeze(path: Path) -> list[str]:
    payload, errors = _read_json(path)
    if payload is None:
        return [f"freeze: {error}" for error in errors]
    if payload.get("artifact_type") != "evttc_final_freeze_manifest_v1":
        errors.append("freeze: wrong artifact_type")
    if payload.get("dirty_worktree") is not False:
        errors.append("freeze: dirty_worktree must be false")
    if payload.get("evttc_used_for_selection") is not False:
        errors.append("freeze: evttc_used_for_selection must be false")
    if payload.get("predictions_frozen_before_benchmark") is not True:
        errors.append("freeze: predictions_frozen_before_benchmark must be true")
    if not isinstance(payload.get("checkpoints"), list) or not payload["checkpoints"]:
        errors.append("freeze: at least one checkpoint is required")
    resources = payload.get("frozen_resources")
    required_resources = {"config", "preprocessing", "protocol", "selection_audit"}
    if not isinstance(resources, dict) or not required_resources.issubset(resources):
        errors.append("freeze: config, preprocessing, protocol and selection_audit must be hashed")
    signature = payload.get("freeze_manifest_sha256")
    if not isinstance(signature, str) or signature != _canonical_hash(
        payload, "freeze_manifest_sha256"
    ):
        errors.append("freeze: freeze_manifest_sha256 missing or invalid")
    return errors


def _check_robustness(path: Path) -> list[str]:
    payload, errors = _signed_real_artifact(path, "robustness")
    if payload is None:
        return errors
    models = payload.get("comparison_models", payload.get("models"))
    if isinstance(models, dict):
        model_names = {str(name).lower() for name in models}
    elif isinstance(models, list):
        model_names = {str(name).lower() for name in models}
    else:
        model_names = set()
    required_names = {"official", "scratch", "jepa"}
    if not required_names.issubset(model_names):
        errors.append("robustness: official, scratch and jepa comparisons are required")
    if not payload.get("corruptions") and not payload.get("perturbations"):
        errors.append("robustness: corruption/intensity matrix is missing")
    return errors


def _check_calibration(path: Path) -> list[str]:
    payload, errors = _signed_real_artifact(path, "calibration")
    if payload is None:
        return errors
    calibration_split = str(payload.get("calibration_split", ""))
    evaluation_split = str(payload.get("evaluation_split", ""))
    if not calibration_split or not evaluation_split or calibration_split == evaluation_split:
        errors.append(
            "calibration: calibration and evaluation splits must be explicit and disjoint"
        )
    if int(payload.get("calibration_count", 0)) <= 0:
        errors.append("calibration: calibration_count must be positive")
    return errors


def _check_low_label(path: Path) -> list[str]:
    payload, errors = _signed_real_artifact(path, "low_label")
    if payload is None:
        return errors
    fractions = {round(float(value), 2) for value in payload.get("fractions", [])}
    if fractions != REQUIRED_LOW_LABEL_FRACTIONS:
        errors.append("low_label: fractions must be exactly 1/5/10/25/100 percent")
    seeds = {int(value) for value in payload.get("seeds", [])}
    if seeds != REQUIRED_SEEDS:
        errors.append("low_label: seeds must be exactly 7, 13 and 23")
    for field in ("sequence_grouped", "same_ids_across_methods", "training_completed"):
        if payload.get(field) is not True:
            errors.append(f"low_label: {field}=true is required")
    return errors


def _check_onnx_equivalence(path: Path) -> list[str]:
    payload, errors = _signed_real_artifact(path, "onnx_equivalence")
    if payload is None:
        return errors
    if payload.get("real_validation_samples") is not True:
        errors.append("onnx_equivalence: real validation samples are required")
    if payload.get("final_test_opened") is True:
        errors.append("onnx_equivalence: final test must not have been opened")
    if payload.get("status") not in (None, "passed"):
        errors.append("onnx_equivalence: status is not passed")
    return errors


def _check_onnx_benchmark(path: Path) -> list[str]:
    payload, errors = _signed_real_artifact(path, "onnx_benchmark")
    if payload is None:
        return errors
    if payload.get("batch_size") != 1:
        errors.append("onnx_benchmark: batch_size must be 1")
    if payload.get("warmup_iterations") != 50:
        errors.append("onnx_benchmark: warmup_iterations must be 50")
    if payload.get("measured_iterations") != 500:
        errors.append("onnx_benchmark: measured_iterations must be 500")
    if not all(key in payload for key in ("p50_ms", "p90_ms", "p95_ms")):
        errors.append("onnx_benchmark: p50, p90 and p95 are required")
    if not payload.get("stages"):
        errors.append("onnx_benchmark: separated pipeline stages are required")
    if "parameter_count" not in payload or "flops" not in payload:
        errors.append("onnx_benchmark: parameter_count and flops are required")
    return errors


def _check_report(path: Path) -> list[str]:
    payload, errors = _read_json(path)
    if payload is None:
        return [f"report: {error}" for error in errors]
    if not payload.get("artifacts"):
        errors.append("report: regenerable artifact list is empty")
    if not payload.get("source_artifacts") and not payload.get("artifact_root"):
        errors.append("report: source_artifacts or artifact_root is required")
    return errors


def _check_document(path: Path, label: str) -> list[str]:
    if not path.is_file():
        return [f"{label}: missing file: {path}"]
    if not path.read_text(encoding="utf-8").strip():
        return [f"{label}: document is empty"]
    return []


def audit(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "phase9_freeze": _check_freeze(args.freeze),
        "robustness_real": _check_robustness(args.robustness),
        "calibration_real": _check_calibration(args.calibration),
        "low_label_real": _check_low_label(args.low_label),
        "onnx_equivalence_real": _check_onnx_equivalence(args.onnx_equivalence),
        "onnx_benchmark_real": _check_onnx_benchmark(args.onnx_benchmark),
        "report_regenerable": _check_report(args.report),
        "model_card": _check_document(args.model_card, "model_card"),
        "dataset_card": _check_document(args.dataset_card, "dataset_card"),
        "limitations": _check_document(args.limitations, "limitations"),
    }
    return {
        "artifact_type": "phase9_phase12_real_gate_audit_v1",
        "status": "passed" if all(not errors for errors in checks.values()) else "failed",
        "claim_allowed": all(not errors for errors in checks.values()),
        "gates": {
            name: {"passed": not errors, "errors": errors} for name, errors in checks.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--low-label", type=Path, required=True)
    parser.add_argument("--onnx-equivalence", type=Path, required=True)
    parser.add_argument("--onnx-benchmark", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--limitations", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["claim_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
