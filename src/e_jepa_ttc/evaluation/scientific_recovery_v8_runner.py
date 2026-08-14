"""Fail-closed orchestration helpers for the Scientific Recovery V8 protocol.

The V8 runner intentionally owns *control flow*, not experiment mathematics.  It
checks the frozen protocol before every stage, records a signed state transition,
and delegates work to the narrow scripts that own each scientific operation.  This
keeps a partially completed screen resumable without turning an old, unsigned
result into evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash

ROOT = Path(__file__).resolve().parents[3]
STAGES: tuple[str, ...] = (
    "preflight",
    "autopsy",
    "router",
    "temporal",
    "adaptive",
    "jepa",
    "multiseed_replication",
    "robustness",
    "export",
    "package",
    "screen",
    "all",
)
SCREEN_STAGES: tuple[str, ...] = (
    "preflight",
    "autopsy",
    "router",
    "temporal",
    "adaptive",
    "jepa",
)
ALL_STAGES: tuple[str, ...] = (
    *SCREEN_STAGES,
    "multiseed_replication",
    "robustness",
    "export",
    "package",
)
_SEALED_MARKERS = ("public_validation", "private_test", "evttc_test", "codabench")
_CLOSED_FLAGS = {
    "public_validation_used_for_selection",
    "private_test_opened",
    "evttc_test_opened",
    "codabench_opened",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_OUTER_FOLDS = frozenset({0, 1, 2})
_CAUSAL_ROUTER_FEATURES = frozenset(
    {"event_count", "event_rate", "flow", "temporal_support", "support_ms"}
)
_C1_OPENING_ARTIFACT_TYPE = "scientific_recovery_v8_c1_opening_decision_v1"
_C1_PLAN_ARTIFACT_TYPE = "scientific_recovery_v8_preregistered_analysis_plan_v1"
_C1_EVIDENCE_ARTIFACT_TYPE = "scientific_recovery_v8_regime_evidence_v1"
_C1_CAUSALITY_ARTIFACT_TYPE = "scientific_recovery_v8_causal_invariance_v1"


class V8IntegrityError(RuntimeError):
    """Raised when an input, result, or stage transition fails the V8 contract."""


@dataclass(frozen=True)
class FrozenV8Inputs:
    """Verified V8 protocol plus the manifest generated from it."""

    protocol_path: Path
    manifest_path: Path
    protocol: Mapping[str, Any]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class RunResumeState:
    """The only three permissible transitions for one training run directory."""

    status: str
    resume: bool
    detail: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> Path:
    """Resolve a repository-local path and reject paths escaping the checkout."""

    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise V8IntegrityError(f"V8 path escapes repository: {path}") from error
    return resolved


def _read_signed(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise V8IntegrityError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise V8IntegrityError(f"invalid JSON for {label}: {path}") from error
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise V8IntegrityError(f"invalid signed {label}: {path}")
    return payload


def _assert_closed(value: object, *, label: str) -> None:
    """Reject any source claiming or naming a sealed evaluation split."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in _CLOSED_FLAGS and nested is not False:
                raise V8IntegrityError(f"sealed-evaluation flag is not false in {label}: {key}")
            _assert_closed(nested, label=label)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _assert_closed(nested, label=label)
        return
    if isinstance(value, str):
        normalized = value.lower().replace("\\", "/")
        if any(marker in normalized for marker in _SEALED_MARKERS):
            raise V8IntegrityError(f"sealed-evaluation path rejected in {label}: {value}")


def _assert_finite(value: object, *, label: str) -> None:
    """Reject NaN/Infinity in signed result metadata rather than accepting it."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite(nested, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_finite(nested, label=f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise V8IntegrityError(f"non-finite value in {label}")


def _status_is_complete(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status", "")).lower()
    return status in {"completed", "complete", "passed", "success", "completed_seed7_oof_gate"}


def verify_frozen_inputs(protocol_path: Path, manifest_path: Path) -> FrozenV8Inputs:
    """Validate protocol, manifest, frozen configs, hashes, and split closure."""

    protocol_path = _repo_path(protocol_path)
    manifest_path = _repo_path(manifest_path)
    protocol = _read_signed(protocol_path, label="V8 protocol")
    manifest = _read_signed(manifest_path, label="V8 frozen manifest")
    if protocol.get("artifact_type") != "scientific_recovery_v8_temporal_protocol_v1":
        raise V8IntegrityError("unexpected V8 protocol artifact type")
    if manifest.get("artifact_type") != "scientific_recovery_v8_frozen_config_manifest_v1":
        raise V8IntegrityError("unexpected V8 frozen manifest artifact type")
    _assert_closed(protocol.get("closed_evaluation", {}), label="protocol")
    _assert_closed(manifest.get("closed_evaluation", {}), label="frozen manifest")
    declared_protocol = manifest.get("protocol")
    if not isinstance(declared_protocol, Mapping):
        raise V8IntegrityError("frozen manifest lacks protocol source")
    if declared_protocol.get("artifact_sha256") != protocol.get("artifact_sha256"):
        raise V8IntegrityError("frozen manifest protocol signature differs from protocol")
    if declared_protocol.get("sha256") != _sha256_file(protocol_path):
        raise V8IntegrityError("frozen manifest protocol file hash differs from protocol")
    sample = protocol.get("sample_contract")
    integrity = manifest.get("integrity")
    if not isinstance(sample, Mapping) or not isinstance(integrity, Mapping):
        raise V8IntegrityError("V8 protocol/manifest lacks sample integrity contract")
    if integrity != sample:
        for key, value in sample.items():
            if integrity.get(key) != value:
                raise V8IntegrityError(f"frozen manifest differs from protocol for {key}")
        raise V8IntegrityError("frozen manifest sample integrity differs from protocol")
    configs = manifest.get("enabled_seed7_configs")
    if not isinstance(configs, Mapping) or not configs:
        raise V8IntegrityError("frozen manifest has no enabled seed-7 configurations")
    for name, entry in configs.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise V8IntegrityError(f"invalid frozen configuration entry: {name}")
        config_path = _repo_path(ROOT / str(entry["path"]))
        if not config_path.is_file() or entry.get("sha256") != _sha256_file(config_path):
            raise V8IntegrityError(f"frozen configuration hash mismatch: {name}")
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise V8IntegrityError(f"invalid frozen configuration YAML: {name}") from error
        _assert_closed(config, label=f"frozen config {name}")
    templates = manifest.get("conditional_templates", {})
    if not isinstance(templates, Mapping):
        raise V8IntegrityError("frozen manifest conditional_templates must be a mapping")
    for template_name, template in templates.items():
        if not isinstance(template, Mapping):
            raise V8IntegrityError(f"invalid conditional template: {template_name}")
        fold_configs = template.get("fold_configs", [])
        if not isinstance(fold_configs, list):
            raise V8IntegrityError(f"conditional template lacks fold_configs: {template_name}")
        for index, entry in enumerate(fold_configs):
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                raise V8IntegrityError(f"invalid conditional config {template_name}[{index}]")
            config_path = _repo_path(ROOT / str(entry["path"]))
            if not config_path.is_file() or entry.get("sha256") != _sha256_file(config_path):
                raise V8IntegrityError(f"conditional config hash mismatch: {template_name}[{index}]")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, Mapping):
                raise V8IntegrityError(f"conditional config is not a mapping: {template_name}[{index}]")
            _assert_closed(config, label=f"conditional config {template_name}[{index}]")
    models = manifest.get("model_configs")
    if not isinstance(models, Mapping) or not models:
        raise V8IntegrityError("frozen manifest has no model configurations")
    for name, entry in models.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise V8IntegrityError(f"invalid frozen model configuration entry: {name}")
        model_path = _repo_path(ROOT / str(entry["path"]))
        if not model_path.is_file() or entry.get("sha256") != _sha256_file(model_path):
            raise V8IntegrityError(f"frozen model configuration hash mismatch: {name}")
    plans = manifest.get("c1_analysis_plans")
    if not isinstance(plans, Mapping) or plans != protocol.get("c1_analysis_plans"):
        raise V8IntegrityError("frozen C1 analysis plans differ from the protocol")
    expected_contract = protocol.get("sample_contract")
    for route, entry in plans.items():
        if not isinstance(route, str) or not isinstance(entry, Mapping):
            raise V8IntegrityError("frozen C1 analysis plan entry is invalid")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
            raise V8IntegrityError("frozen C1 analysis plan path must be relative")
        plan_path = _repo_path(ROOT / raw_path)
        if not plan_path.is_file() or entry.get("sha256") != _sha256_file(plan_path):
            raise V8IntegrityError(f"frozen C1 analysis plan hash mismatch: {route}")
        plan = _read_signed(plan_path, label=f"frozen C1 analysis plan {route}")
        if (
            plan.get("artifact_type") != _C1_PLAN_ARTIFACT_TYPE
            or entry.get("artifact_sha256") != plan.get("artifact_sha256")
            or plan.get("plan_id") != route
            or plan.get("sample_contract") != expected_contract
        ):
            raise V8IntegrityError(f"frozen C1 analysis plan contract mismatch: {route}")
        _assert_closed(plan.get("closed_evaluation", {}), label=f"frozen C1 analysis plan {route}")
    return FrozenV8Inputs(protocol_path, manifest_path, protocol, manifest)


def stage_summary_path(results_root: Path, stage: str) -> Path:
    """Return the signed stage-summary location, independent of training runs."""

    return results_root / "stages" / stage / "summary.json"


def stage_state_path(results_root: Path, stage: str) -> Path:
    """Return the durable signed state location for a stage."""

    return results_root / "stages" / stage / "state.json"


def write_stage_state(
    *,
    results_root: Path,
    stage: str,
    status: str,
    frozen: FrozenV8Inputs,
    detail: str,
    command: Sequence[str] | None = None,
    candidate: str | None = None,
) -> dict[str, Any]:
    """Atomically publish signed stage state and its signed completion summary."""

    if stage not in STAGES or stage in {"screen", "all"}:
        raise ValueError(f"not a concrete V8 stage: {stage}")
    now = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_stage_state_v1",
        "schema_version": "scientific_recovery_v8_temporal_v1",
        "stage": stage,
        "status": status,
        "detail": detail,
        "created_at_utc": now,
        "protocol_sha256": frozen.protocol["artifact_sha256"],
        "protocol_file_sha256": _sha256_file(frozen.protocol_path),
        "frozen_manifest_sha256": frozen.manifest["artifact_sha256"],
        "frozen_manifest_file_sha256": _sha256_file(frozen.manifest_path),
        "closed_evaluation": dict(frozen.protocol["closed_evaluation"]),
    }
    if command is not None:
        payload["command"] = list(command)
    if candidate is not None:
        key = "multiseed_replication_candidate" if stage == "multiseed_replication" else "candidate"
        payload[key] = candidate
    sign_artifact(payload)
    for target in (stage_state_path(results_root, stage), stage_summary_path(results_root, stage)):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)
    return payload


def stage_is_complete(*, results_root: Path, stage: str, frozen: FrozenV8Inputs) -> bool:
    """A stage is reusable only when its signed completion summary matches inputs."""

    path = stage_summary_path(results_root, stage)
    if not path.is_file():
        return False
    try:
        payload = _read_signed(path, label=f"V8 {stage} stage summary")
        _assert_finite(payload, label=f"V8 {stage} stage summary")
        _assert_closed(payload.get("closed_evaluation", {}), label=f"V8 {stage} stage summary")
    except V8IntegrityError:
        return False
    return bool(
        _status_is_complete(payload)
        and payload.get("stage") == stage
        and payload.get("protocol_sha256") == frozen.protocol["artifact_sha256"]
        and payload.get("frozen_manifest_sha256") == frozen.manifest["artifact_sha256"]
    )


def assess_run_resume_state(run_dir: Path) -> RunResumeState:
    """Return completed/resume/new, preserving a corrupt checkpoint untouched.

    A corrupt ``last.pt`` produces a separate signed ``failed_integrity.json``.
    It is deliberately never deleted or replaced; recovery then requires explicit
    human action rather than silently losing potentially useful forensic evidence.
    """

    run_dir = _repo_path(run_dir)
    summary = run_dir / "summary.json"
    if summary.is_file():
        payload = _read_signed(summary, label=f"run summary {run_dir.name}")
        _assert_finite(payload, label=f"run summary {run_dir.name}")
        _assert_closed(payload.get("closed_evaluation", {}), label=f"run summary {run_dir.name}")
        if not _status_is_complete(payload):
            raise V8IntegrityError(f"signed run summary is not complete: {run_dir}")
        return RunResumeState("completed", False, "signed completed summary")
    last_checkpoint = run_dir / "state" / "last.pt"
    if not last_checkpoint.exists():
        return RunResumeState("new", False, "no completed summary or last checkpoint")
    try:
        import torch

        torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # torch emits implementation-specific error types.
        failure = {
            "artifact_type": "scientific_recovery_v8_run_failure_v1",
            "status": "failed_integrity",
            "run_dir": str(run_dir.relative_to(ROOT)),
            "checkpoint": str(last_checkpoint.relative_to(ROOT)),
            "detail": f"corrupt state/last.pt: {type(error).__name__}: {error}",
        }
        sign_artifact(failure)
        failure_path = run_dir / "failed_integrity.json"
        failure_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        return RunResumeState("failed_integrity", False, failure["detail"])
    return RunResumeState("resume", True, "valid state/last.pt without completed summary")


def _script_command(stage: str, *, device: str, candidate: str | None) -> list[str]:
    """Return the frozen delegation command for a concrete stage."""

    python = ["uv", "run", "--no-sync", "python"]
    if stage == "preflight":
        return [*python, "scripts/smoke_scientific_recovery_v8.py", "--device", "cpu"]
    scripts: dict[str, list[str]] = {
        "autopsy": ["scripts/replay_scientific_recovery_v8_mechanisms.py", "--device", device],
        "router": ["scripts/run_scientific_recovery_v8_nested_router.py", "--device", device],
        "temporal": ["scripts/run_scientific_recovery_v8_temporal.py", "--device", device],
        "adaptive": ["scripts/run_scientific_recovery_v8_adaptive.py", "--device", device],
        "jepa": ["scripts/run_scientific_recovery_v8_jepa_attribution.py", "--device", device],
        "robustness": ["scripts/run_scientific_recovery_v8_robustness.py", "--device", device],
        "export": ["scripts/export_scientific_recovery_v8_onnx.py", "--device", device],
        "package": ["scripts/package_scientific_recovery_v8_evidence.py"],
    }
    if stage == "multiseed_replication":
        if not candidate:
            raise V8IntegrityError(
                "multiseed_replication stage requires an explicit frozen candidate id"
            )
        return [
            *python,
            "scripts/run_scientific_recovery_v8_multiseed_replication.py",
            "--device",
            device,
            "--candidate",
            candidate,
        ]
    if stage not in scripts:
        raise ValueError(f"no V8 command registered for stage {stage!r}")
    return [*python, *scripts[stage]]


def _assert_script_exists(command: Sequence[str]) -> None:
    script = next((Path(value) for value in command if str(value).startswith("scripts/")), None)
    if script is None:
        return
    path = ROOT / script
    if not path.is_file():
        raise V8IntegrityError(
            f"required V8 stage implementation is absent: {script}; run the owning phase first"
        )


def _signed_result_artifacts(results_root: Path) -> Iterable[tuple[Path, Mapping[str, Any]]]:
    if not results_root.exists():
        return ()
    results: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(results_root.rglob("*.json")):
        try:
            payload = _read_signed(path, label="V8 result artifact")
        except V8IntegrityError:
            continue
        results.append((path, payload))
    return tuple(results)


def _gate_passed(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("gate_decision", payload.get("gates", {}))
    if not isinstance(gate, Mapping):
        return False
    return bool(gate.get("passed") is True or gate.get("screen_passed") is True)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _assert_evidence_binding(payload: Mapping[str, Any], *, frozen: FrozenV8Inputs) -> None:
    """Bind a signed C1 artifact to the exact frozen protocol and sample contract."""

    expected_contract = frozen.protocol.get("sample_contract")
    if not isinstance(expected_contract, Mapping):
        raise V8IntegrityError("frozen V8 protocol lacks a sample contract")
    if payload.get("protocol_artifact_sha256") != frozen.protocol.get("artifact_sha256"):
        raise V8IntegrityError("C1 evidence protocol artifact signature mismatch")
    if payload.get("protocol_file_sha256") != _sha256_file(frozen.protocol_path):
        raise V8IntegrityError("C1 evidence protocol file signature mismatch")
    if payload.get("sample_contract") != expected_contract:
        raise V8IntegrityError("C1 evidence sample/fold contract differs from frozen protocol")
    _assert_closed(payload.get("closed_evaluation", {}), label="C1 evidence")


def _resolve_c1_evidence_ref(
    *,
    ref: object,
    opening_path: Path,
    frozen: FrozenV8Inputs,
    label: str,
    artifact_type: str,
) -> Mapping[str, Any]:
    """Load one path-bound C1 evidence artifact and reject aggregate reuse."""

    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise V8IntegrityError(f"C1 {label} reference lacks a relative path")
    if not _is_sha256(ref.get("sha256")):
        raise V8IntegrityError(f"C1 {label} reference lacks a SHA-256")
    raw_path = Path(str(ref["path"]))
    if raw_path.is_absolute():
        raise V8IntegrityError(f"C1 {label} reference path must be relative")
    candidate = _repo_path(ROOT / raw_path)
    evidence_root = (ROOT / "artifacts" / "scientific_recovery_v8").resolve()
    try:
        candidate.relative_to(evidence_root)
    except ValueError as error:
        raise V8IntegrityError(f"C1 {label} path is outside V8 artifacts") from error
    if candidate == opening_path.resolve():
        raise V8IntegrityError(f"C1 {label} cannot self-reference an opening decision")
    if not candidate.is_file():
        raise V8IntegrityError(f"missing C1 {label} artifact: {candidate}")
    if _sha256_file(candidate) != ref["sha256"]:
        raise V8IntegrityError(f"C1 {label} file checksum mismatch")
    payload = _read_signed(candidate, label=f"C1 {label} artifact")
    if payload.get("artifact_type") != artifact_type:
        raise V8IntegrityError(f"C1 {label} uses an invalid artifact type")
    _assert_evidence_binding(payload, frozen=frozen)
    return payload


def _resolve_frozen_analysis_plan(
    *, ref: object, route: str, frozen: FrozenV8Inputs
) -> Mapping[str, Any]:
    """Resolve an opening route to exactly one plan signed during the freeze."""

    plans = frozen.manifest.get("c1_analysis_plans")
    protocol_plans = frozen.protocol.get("c1_analysis_plans")
    if not isinstance(plans, Mapping) or plans != protocol_plans:
        raise V8IntegrityError("frozen C1 analysis plans do not match the protocol")
    expected = plans.get(route)
    if not isinstance(expected, Mapping) or ref != expected:
        raise V8IntegrityError("C1 opening does not reference its exact frozen analysis plan")
    raw_path = expected.get("path")
    if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
        raise V8IntegrityError("frozen C1 analysis plan path must be relative")
    path = _repo_path(ROOT / raw_path)
    if not path.is_file() or expected.get("sha256") != _sha256_file(path):
        raise V8IntegrityError("frozen C1 analysis plan file checksum mismatch")
    plan = _read_signed(path, label="frozen C1 analysis plan")
    if plan.get("artifact_type") != _C1_PLAN_ARTIFACT_TYPE:
        raise V8IntegrityError("frozen C1 analysis plan uses an invalid artifact type")
    if expected.get("artifact_sha256") != plan.get("artifact_sha256"):
        raise V8IntegrityError("frozen C1 analysis plan artifact signature mismatch")
    if plan.get("plan_id") != route or plan.get("sample_contract") != frozen.protocol.get(
        "sample_contract"
    ):
        raise V8IntegrityError("frozen C1 analysis plan contract differs from the protocol")
    _assert_closed(plan.get("closed_evaluation", {}), label="frozen C1 analysis plan")
    return plan


def _resolve_source_aggregate(
    *,
    ref: object,
    opening_path: Path,
    frozen: FrozenV8Inputs,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load the route-specific seed-7 aggregate that supplies C1 gate provenance."""

    contract = plan.get("source_aggregate_contract")
    if not isinstance(contract, Mapping):
        raise V8IntegrityError("frozen C1 analysis plan lacks a source aggregate contract")
    expected_type = contract.get("artifact_type")
    if not isinstance(expected_type, str):
        raise V8IntegrityError("frozen C1 source aggregate type is invalid")
    aggregate = _resolve_c1_evidence_ref(
        ref=ref,
        opening_path=opening_path,
        frozen=frozen,
        label="source aggregate",
        artifact_type=expected_type,
    )
    for key in ("stage", "arm", "candidate_id", "seed"):
        if aggregate.get(key) != contract.get(key):
            raise V8IntegrityError(f"C1 source aggregate differs from frozen plan for {key}")
    if aggregate.get("status") != "completed":
        raise V8IntegrityError("C1 source aggregate is not completed")
    _assert_finite(aggregate, label="C1 source aggregate")
    if not _exact_coverage(aggregate, frozen=frozen):
        raise V8IntegrityError("C1 source aggregate lacks exact outer-fold coverage")
    for key in (
        "row_identity_sha256",
        "target_identity_sha256",
        "target_sha256",
        "mid_sample_weight_sha256",
        "fold_assignment_sha256",
    ):
        if aggregate["sample_contract"].get(key) != frozen.protocol["sample_contract"].get(key):
            raise V8IntegrityError(f"C1 source aggregate integrity mismatch for {key}")
    if contract.get("aggregate_schema") is not None:
        _validate_primary_aggregate(aggregate=aggregate, contract=contract, frozen=frozen)
    if contract.get("primary_ttc_gate_required") is True:
        _require_primary_ttc_gate(aggregate)
    if contract.get("required_outputs") is not None:
        _resolve_autopsy_outputs(
            aggregate=aggregate,
            contract=contract,
            opening_path=opening_path,
            frozen=frozen,
        )
    return aggregate


def _validate_primary_aggregate(
    *, aggregate: Mapping[str, Any], contract: Mapping[str, Any], frozen: FrozenV8Inputs
) -> None:
    """Validate a router/EXP6 aggregate against its frozen full-evaluation schema."""

    schema = contract.get("aggregate_schema")
    if not isinstance(schema, list) or not all(isinstance(field, str) for field in schema):
        raise V8IntegrityError("frozen source aggregate schema is invalid")
    missing = [field for field in schema if field not in aggregate]
    if missing:
        raise V8IntegrityError(f"C1 source aggregate lacks schema fields: {missing}")
    if aggregate.get("schema_version") != frozen.protocol.get("schema_version"):
        raise V8IntegrityError("C1 source aggregate schema version mismatch")
    if aggregate.get("protocol_sha256") != frozen.protocol.get("artifact_sha256"):
        raise V8IntegrityError("C1 source aggregate protocol signature mismatch")
    if aggregate.get("git_commit") != frozen.protocol.get("git_base_commit"):
        raise V8IntegrityError("C1 source aggregate git commit differs from frozen base")
    if aggregate.get("row_count") != contract.get("row_count"):
        raise V8IntegrityError("C1 source aggregate row count differs from frozen contract")
    sample_contract = frozen.protocol.get("sample_contract")
    if not isinstance(sample_contract, Mapping):
        raise V8IntegrityError("frozen V8 protocol lacks sample contract")
    for key in ("row_identity_sha256", "target_sha256"):
        if aggregate.get(key) != sample_contract.get(key):
            raise V8IntegrityError(f"C1 source aggregate {key} differs from frozen contract")
    expected_configs = contract.get("config_sha256_by_fold")
    if (
        not isinstance(expected_configs, Mapping)
        or aggregate.get("config_sha256") != expected_configs
    ):
        raise V8IntegrityError("C1 source aggregate config hashes differ from frozen route")
    expected_coverage = _expected_fold_sequences(frozen)
    row_counts = _frozen_row_count_contract(frozen)
    folds = aggregate.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != set(expected_coverage):
        raise V8IntegrityError("C1 source aggregate folds are incomplete")
    predictions = _fold_hash_mapping(aggregate.get("prediction_sha256"), label="prediction")
    checkpoints = _fold_hash_mapping(aggregate.get("checkpoint_sha256"), label="checkpoint")
    if set(predictions) != set(expected_coverage) or set(checkpoints) != set(expected_coverage):
        raise V8IntegrityError("C1 source aggregate prediction/checkpoint folds are incomplete")
    for fold, sequences in expected_coverage.items():
        entry = folds.get(fold)
        if not isinstance(entry, Mapping):
            raise V8IntegrityError(f"C1 source aggregate fold {fold} is invalid")
        if entry.get("status") != "completed" or entry.get("sequence_ids") != sequences:
            raise V8IntegrityError(f"C1 source aggregate fold {fold} is incomplete")
        if (
            entry.get("prediction_sha256") != predictions[fold]
            or entry.get("checkpoint_sha256") != checkpoints[fold]
        ):
            raise V8IntegrityError(f"C1 source aggregate fold {fold} hash binding mismatch")
        if entry.get("row_count") != row_counts["by_outer_fold"].get(fold):
            raise V8IntegrityError(f"C1 source aggregate fold {fold} row count is invalid")
    _require_primary_metrics(aggregate.get("metrics"), contract=contract)
    _require_bootstrap_metrics(aggregate.get("bootstrap"), contract=contract)
    per_sequence = aggregate.get("per_sequence")
    if not isinstance(per_sequence, Mapping) or set(per_sequence) != {
        sequence for values in expected_coverage.values() for sequence in values
    }:
        raise V8IntegrityError("C1 source aggregate per-sequence coverage is incomplete")
    _require_group_metrics(
        per_sequence,
        expected_counts=row_counts["by_sequence"],
        contract=contract,
        label="C1 source per-sequence metrics",
    )
    buckets = aggregate.get("per_bucket")
    expected_buckets = contract.get("required_bucket_ids")
    if not isinstance(buckets, Mapping) or not isinstance(expected_buckets, list):
        raise V8IntegrityError("C1 source aggregate bucket contract is invalid")
    if set(buckets) != set(expected_buckets):
        raise V8IntegrityError("C1 source aggregate per-bucket coverage is incomplete")
    _require_group_metrics(
        buckets,
        expected_counts=row_counts["by_bucket"],
        contract=contract,
        label="C1 source per-bucket metrics",
    )
    _require_all_true(aggregate.get("integrity_checks"), label="C1 source integrity checks")


def _frozen_row_count_contract(frozen: FrozenV8Inputs) -> Mapping[str, Mapping[str, int]]:
    sample_contract = frozen.protocol.get("sample_contract")
    if not isinstance(sample_contract, Mapping):
        raise V8IntegrityError("frozen V8 protocol lacks sample contract")
    contract = sample_contract.get("row_count_contract")
    if not isinstance(contract, Mapping) or contract.get("total") != 8192:
        raise V8IntegrityError("frozen V8 row-count contract is invalid")
    required = ("by_outer_fold", "by_sequence", "by_bucket")
    if not all(isinstance(contract.get(key), Mapping) for key in required):
        raise V8IntegrityError("frozen V8 row-count contract is incomplete")
    return contract  # Type narrowed above; mappings are checked by all().


def _require_primary_metrics(value: object, *, contract: Mapping[str, Any]) -> None:
    keys = contract.get("primary_metric_keys")
    if not isinstance(value, Mapping) or not isinstance(keys, list) or set(value) != set(keys):
        raise V8IntegrityError("C1 source primary metrics schema is incomplete")
    for key in keys:
        item = value.get(key)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise V8IntegrityError(f"C1 source primary metric is invalid: {key}")


def _require_bootstrap_metrics(value: object, *, contract: Mapping[str, Any]) -> None:
    keys = contract.get("bootstrap_keys")
    if not isinstance(value, Mapping) or not isinstance(keys, list) or set(value) != set(keys):
        raise V8IntegrityError("C1 source bootstrap schema is incomplete")
    for key in keys:
        item = value.get(key)
        if key == "resamples":
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise V8IntegrityError("C1 source bootstrap resamples is invalid")
        elif (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise V8IntegrityError(f"C1 source bootstrap metric is invalid: {key}")


def _require_group_metrics(
    value: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
    contract: Mapping[str, Any],
    label: str,
) -> None:
    keys = contract.get("per_group_metric_keys")
    if not isinstance(keys, list) or set(value) != set(expected_counts):
        raise V8IntegrityError(f"{label} coverage is incomplete")
    for group, record in value.items():
        if not isinstance(record, Mapping) or set(record) != set(keys):
            raise V8IntegrityError(f"{label} schema is invalid for {group}")
        if record.get("row_count") != expected_counts[group]:
            raise V8IntegrityError(f"{label} row count is invalid for {group}")
        for key in ("mid_macro_sequence", "delta_mid_vs_a5"):
            item = record.get(key)
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise V8IntegrityError(f"{label} metric is invalid for {group}.{key}")


def _fold_hash_mapping(value: object, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"0", "1", "2"}:
        raise V8IntegrityError(f"C1 source aggregate {label} hashes are incomplete")
    if not all(
        isinstance(fold, str) and _is_sha256(hash_value) for fold, hash_value in value.items()
    ):
        raise V8IntegrityError(f"C1 source aggregate {label} hashes are invalid")
    return value


def _require_finite_mapping(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise V8IntegrityError(f"{label} is missing")
    _assert_finite(value, label=label)


def _require_all_true(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise V8IntegrityError(f"{label} is missing")

    def all_true(nested: object) -> bool:
        if isinstance(nested, Mapping):
            return bool(nested) and all(all_true(item) for item in nested.values())
        return nested is True

    if not all_true(value):
        raise V8IntegrityError(f"{label} must contain only passing checks")


def _require_primary_ttc_gate(aggregate: Mapping[str, Any]) -> None:
    """Recompute the frozen primary TTC gate from finite aggregate components."""

    metrics = aggregate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise V8IntegrityError("C1 source aggregate lacks primary TTC gate metrics")
    bootstrap = aggregate.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise V8IntegrityError("C1 source aggregate lacks bootstrap gate metrics")
    required = {
        "delta_mid_vs_a5": lambda value: value <= -3.0,
        "finite_fraction": lambda value: value == 1.0,
        "failure_rate": lambda value: value == 0.0,
        "coverage_drop_max_pp": lambda value: value <= 1.0,
    }
    values: dict[str, float] = {}
    for key in required:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise V8IntegrityError(f"C1 primary TTC gate metric is missing or non-numeric: {key}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise V8IntegrityError(f"C1 primary TTC gate metric is non-finite: {key}")
        values[key] = numeric
    probability = bootstrap.get("probability_delta_lt_zero")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise V8IntegrityError("C1 bootstrap probability is missing or non-numeric")
    if not math.isfinite(float(probability)):
        raise V8IntegrityError("C1 bootstrap probability is non-finite")
    passed = (
        all(predicate(values[key]) for key, predicate in required.items())
        and float(probability) >= 0.90
    )
    decision = aggregate.get("gate_decision")
    if not isinstance(decision, Mapping) or decision.get("passed") is not passed:
        raise V8IntegrityError(
            "C1 source aggregate gate decision disagrees with primary TTC metrics"
        )
    if not passed:
        raise V8IntegrityError("C1 source aggregate does not pass the primary TTC gate")


def _resolve_autopsy_outputs(
    *,
    aggregate: Mapping[str, Any],
    contract: Mapping[str, Any],
    opening_path: Path,
    frozen: FrozenV8Inputs,
) -> None:
    """Require the H3 aggregate to bind its factorial replay and diagnostics."""

    required_outputs = contract.get("required_outputs")
    if not isinstance(required_outputs, Mapping):
        raise V8IntegrityError("frozen H3 plan has invalid required outputs")
    refs = aggregate.get("autopsy_outputs")
    if not isinstance(refs, Mapping) or set(refs) != set(required_outputs):
        raise V8IntegrityError("H3 source aggregate lacks exact required output references")
    for name, artifact_type in required_outputs.items():
        if not isinstance(name, str) or not isinstance(artifact_type, str):
            raise V8IntegrityError("frozen H3 output contract is invalid")
        ref = refs.get(name)
        output = _resolve_c1_evidence_ref(
            ref=ref,
            opening_path=opening_path,
            frozen=frozen,
            label=f"H3 {name}",
            artifact_type=artifact_type,
        )
        if output.get("status") != "completed":
            raise V8IntegrityError(f"H3 {name} output is not completed")
        if not isinstance(ref, Mapping) or ref.get("artifact_sha256") != output.get(
            "artifact_sha256"
        ):
            raise V8IntegrityError(f"H3 {name} output artifact signature mismatch")
        _assert_finite(output, label=f"H3 {name} output")
        if name == "factorial_replay":
            _validate_factorial_replay(output=output, contract=contract, frozen=frozen)
        elif name == "diagnostic":
            _validate_autopsy_diagnostic(output=output, contract=contract, frozen=frozen)
            if output.get("final_decision") != aggregate.get("mechanism_decision"):
                raise V8IntegrityError("H3 source decision differs from diagnostic final decision")


def _validate_factorial_replay(
    *, output: Mapping[str, Any], contract: Mapping[str, Any], frozen: FrozenV8Inputs
) -> None:
    schema = contract.get("factorial_replay_schema")
    if not isinstance(schema, Mapping):
        raise V8IntegrityError("frozen H3 factorial replay schema is invalid")
    combinations = schema.get("combinations")
    fields = schema.get("required_cell_fields")
    definitions = schema.get("combination_definitions")
    metric_keys = schema.get("metric_keys")
    if (
        not isinstance(combinations, list)
        or not isinstance(fields, list)
        or not isinstance(definitions, Mapping)
        or not isinstance(metric_keys, list)
    ):
        raise V8IntegrityError("frozen H3 factorial replay requirements are invalid")
    cells = output.get("factorial_cells")
    if not isinstance(cells, Mapping) or set(cells) != set(combinations):
        raise V8IntegrityError("H3 factorial replay lacks exact frozen combinations")
    sample_contract = frozen.protocol.get("sample_contract")
    if not isinstance(sample_contract, Mapping):
        raise V8IntegrityError("frozen V8 protocol lacks sample contract")
    row_counts = _frozen_row_count_contract(frozen)
    for combination in combinations:
        cell = cells.get(combination)
        if not isinstance(cell, Mapping) or any(field not in cell for field in fields):
            raise V8IntegrityError(f"H3 factorial cell is incomplete: {combination}")
        if cell.get("row_count") != schema.get("row_count"):
            raise V8IntegrityError(f"H3 factorial cell row count differs: {combination}")
        for key in ("row_identity_sha256", "target_sha256"):
            if cell.get(key) != sample_contract.get(key):
                raise V8IntegrityError(f"H3 factorial cell integrity mismatch: {combination}.{key}")
        if not _is_sha256(cell.get("prediction_sha256")):
            raise V8IntegrityError(f"H3 factorial cell prediction hash is invalid: {combination}")
        if cell.get("settings") != definitions.get(combination):
            raise V8IntegrityError(f"H3 factorial cell settings differ: {combination}")
        if not _exact_coverage(cell, frozen=frozen):
            raise V8IntegrityError(f"H3 factorial cell coverage is incomplete: {combination}")
        _require_exact_numeric_metrics(
            cell.get("metrics"), keys=metric_keys, label=f"H3 factorial metrics {combination}"
        )
        _require_factorial_group_metrics(
            cell.get("per_sequence"),
            expected_counts=row_counts["by_sequence"],
            label=f"H3 factorial per-sequence {combination}",
        )
        _require_factorial_group_metrics(
            cell.get("per_bucket"),
            expected_counts=row_counts["by_bucket"],
            label=f"H3 factorial per-bucket {combination}",
        )
        _require_all_true(cell.get("integrity_checks"), label=f"H3 factorial checks {combination}")
    _require_hash_mapping(output.get("output_hashes"), label="H3 factorial output hashes")


def _require_exact_numeric_metrics(value: object, *, keys: list[object], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise V8IntegrityError(f"{label} schema is incomplete")
    for key in keys:
        item = value.get(key)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise V8IntegrityError(f"{label} value is invalid: {key}")


def _require_factorial_group_metrics(
    value: object, *, expected_counts: Mapping[str, int], label: str
) -> None:
    required = ["mid_macro_sequence", "delta_mid_vs_a5", "row_count"]
    if not isinstance(value, Mapping) or set(value) != set(expected_counts):
        raise V8IntegrityError(f"{label} coverage is incomplete")
    for group, record in value.items():
        if not isinstance(record, Mapping) or set(record) != set(required):
            raise V8IntegrityError(f"{label} schema is invalid for {group}")
        if record.get("row_count") != expected_counts[group]:
            raise V8IntegrityError(f"{label} row count is invalid for {group}")
        for key in required[:2]:
            item = record.get(key)
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise V8IntegrityError(f"{label} metric is invalid for {group}.{key}")


def _validate_autopsy_diagnostic(
    *, output: Mapping[str, Any], contract: Mapping[str, Any], frozen: FrozenV8Inputs
) -> None:
    schema = contract.get("diagnostic_schema")
    if not isinstance(schema, Mapping):
        raise V8IntegrityError("frozen H3 diagnostic schema is invalid")
    fields = schema.get("required_fields")
    dimensions = schema.get("dimensions")
    record_schema = schema.get("dimension_record_schema")
    rule = schema.get("decision_rule")
    if (
        not isinstance(fields, list)
        or not isinstance(dimensions, Mapping)
        or not isinstance(record_schema, Mapping)
        or not isinstance(rule, Mapping)
    ):
        raise V8IntegrityError("frozen H3 diagnostic requirements are invalid")
    if any(field not in output for field in fields):
        raise V8IntegrityError("H3 diagnostic lacks required decision inputs")
    expected_sequences = {
        sequence for values in _expected_fold_sequences(frozen).values() for sequence in values
    }
    by_sequence = output.get("by_sequence")
    if not isinstance(by_sequence, Mapping) or set(by_sequence) != expected_sequences:
        raise V8IntegrityError("H3 diagnostic sequence coverage is incomplete")
    _require_diagnostic_records(
        by_sequence, schema=record_schema, label="H3 diagnostic per-sequence metrics"
    )
    for name, expected in dimensions.items():
        values = output.get(name)
        if (
            not isinstance(expected, list)
            or not isinstance(values, Mapping)
            or set(values) != set(expected)
        ):
            raise V8IntegrityError(f"H3 diagnostic dimension is incomplete: {name}")
        _require_diagnostic_records(values, schema=record_schema, label=f"H3 diagnostic {name}")
    decision = _recompute_h3_decision(output.get("decision_inputs"), rule=rule)
    if output.get("final_decision") != decision or output.get("decision_rule_output") != decision:
        raise V8IntegrityError("H3 diagnostic decision does not match its frozen rule output")
    _require_all_true(output.get("integrity_checks"), label="H3 diagnostic checks")
    _require_hash_mapping(output.get("output_hashes"), label="H3 diagnostic output hashes")


def _require_diagnostic_records(
    value: Mapping[str, Any], *, schema: Mapping[str, Any], label: str
) -> None:
    numeric = schema.get("numeric")
    booleans = schema.get("booleans")
    required = (
        set(numeric) | set(booleans)
        if isinstance(numeric, list) and isinstance(booleans, list)
        else None
    )
    if required is None:
        raise V8IntegrityError("frozen H3 diagnostic record schema is invalid")
    for group, record in value.items():
        if not isinstance(record, Mapping) or set(record) != required:
            raise V8IntegrityError(f"{label} record schema is invalid for {group}")
        for key in numeric:
            item = record.get(key)
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise V8IntegrityError(f"{label} numeric value is invalid for {group}.{key}")
        for key in booleans:
            if record.get(key) is not True and record.get(key) is not False:
                raise V8IntegrityError(f"{label} boolean is invalid for {group}.{key}")


def _recompute_h3_decision(inputs: object, *, rule: Mapping[str, Any]) -> str:
    expected = rule.get("inputs")
    h3_inputs = rule.get("h3_all_true")
    h1_true = rule.get("h1_all_true")
    h1_false = rule.get("h1_all_false")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(expected, list)
        or not isinstance(h3_inputs, list)
        or not isinstance(h1_true, list)
        or not isinstance(h1_false, list)
        or set(inputs) != set(expected)
        or any(inputs.get(key) is not True and inputs.get(key) is not False for key in expected)
    ):
        raise V8IntegrityError("H3 diagnostic decision inputs are invalid")
    if all(inputs[key] is True for key in h3_inputs):
        return "H3"
    if all(inputs[key] is True for key in h1_true) and all(
        inputs[key] is False for key in h1_false
    ):
        return "H1"
    if rule.get("otherwise") != "H2":
        raise V8IntegrityError("frozen H3 diagnostic fallback rule is invalid")
    return "H2"


def _require_hash_mapping(value: object, *, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or not value
        or not all(_is_sha256(item) for item in value.values())
    ):
        raise V8IntegrityError(f"{label} are missing or invalid")


def _expected_fold_sequences(frozen: FrozenV8Inputs) -> dict[str, list[str]]:
    sample_contract = frozen.protocol.get("sample_contract")
    if not isinstance(sample_contract, Mapping):
        raise V8IntegrityError("frozen V8 protocol lacks sample contract")
    definitions = sample_contract.get("fold_definitions")
    if not isinstance(definitions, list):
        raise V8IntegrityError("frozen V8 protocol lacks fold definitions")
    expected: dict[str, list[str]] = {}
    for definition in definitions:
        if not isinstance(definition, Mapping):
            raise V8IntegrityError("frozen V8 fold definition is invalid")
        fold = definition.get("fold")
        sequences = definition.get("dev_sequence_ids")
        if not isinstance(fold, int) or not isinstance(sequences, list):
            raise V8IntegrityError("frozen V8 fold definition is incomplete")
        expected[str(fold)] = sorted(str(sequence) for sequence in sequences)
    if {int(fold) for fold in expected} != _REQUIRED_OUTER_FOLDS:
        raise V8IntegrityError("frozen V8 protocol does not contain the three outer folds")
    return expected


def _exact_coverage(evidence: Mapping[str, Any], *, frozen: FrozenV8Inputs) -> bool:
    coverage = evidence.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("outer_folds") != [0, 1, 2]:
        return False
    sequences = coverage.get("sequences_by_outer_fold")
    if not isinstance(sequences, Mapping):
        return False
    expected = _expected_fold_sequences(frozen)
    if set(sequences) != set(expected) or not all(isinstance(fold, str) for fold in sequences):
        return False
    if not all(isinstance(values, list) for values in sequences.values()):
        return False
    actual = {fold: sorted(values) for fold, values in sequences.items()}
    return actual == expected


def _stable_by_fold_and_sequence(
    value: object, *, evidence: Mapping[str, Any], frozen: FrozenV8Inputs
) -> bool:
    if not isinstance(value, Mapping):
        return False
    stable_folds = value.get("stable_by_outer_fold")
    stable_sequences = value.get("stable_by_sequence")
    if not isinstance(stable_folds, Mapping) or not isinstance(stable_sequences, Mapping):
        return False
    if not all(
        stable_folds.get(str(fold), stable_folds.get(fold)) is True
        for fold in _REQUIRED_OUTER_FOLDS
    ):
        return False
    if not _exact_coverage(evidence, frozen=frozen):
        return False
    expected = _expected_fold_sequences(frozen)
    sequence_ids = [sequence for values in expected.values() for sequence in values]
    return set(stable_sequences) == set(sequence_ids) and all(
        stable_sequences.get(sequence) is True for sequence in sequence_ids
    )


def _router_evidence_opens_c1(evidence: Mapping[str, Any], *, frozen: FrozenV8Inputs) -> bool:
    dependence = evidence.get("stable_temporal_density_feature_dependence")
    if not isinstance(dependence, Mapping):
        return False
    features = dependence.get("features")
    if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
        return False
    normalized = {feature.lower() for feature in features}
    if not normalized & _CAUSAL_ROUTER_FEATURES:
        return False
    return _stable_by_fold_and_sequence(dependence, evidence=evidence, frozen=frozen)


def _autopsy_evidence_opens_c1(plan: Mapping[str, Any]) -> bool:
    """The H3 decision is authoritative only in the completed source aggregate."""

    return plan.get("status") == "frozen_before_v8_training"


def _exp6_evidence_opens_c1(
    plan: Mapping[str, Any], evidence: Mapping[str, Any], *, frozen: FrozenV8Inputs
) -> bool:
    heterogeneity = evidence.get("exp6_stable_regime_heterogeneity")
    return bool(
        plan.get("status") == "frozen_before_v8_training"
        and isinstance(heterogeneity, Mapping)
        and _stable_by_fold_and_sequence(heterogeneity, evidence=evidence, frozen=frozen)
    )


def assert_adaptive_gate(*, results_root: Path, frozen: FrozenV8Inputs) -> None:
    """Open C1 only from signed, preregistered causal regime evidence."""

    for opening_path, opening in _signed_result_artifacts(results_root):
        if opening.get("artifact_type") != _C1_OPENING_ARTIFACT_TYPE:
            continue
        try:
            _assert_evidence_binding(opening, frozen=frozen)
            refs = opening.get("evidence_refs")
            if not isinstance(refs, Mapping):
                raise V8IntegrityError("C1 opening decision lacks evidence references")
            route = opening.get("opening_route")
            if not isinstance(route, str):
                raise V8IntegrityError("C1 opening decision lacks a route")
            plan = _resolve_frozen_analysis_plan(
                ref=refs.get("analysis_plan"), route=route, frozen=frozen
            )
            source_aggregate = _resolve_source_aggregate(
                ref=refs.get("source_aggregate"),
                opening_path=opening_path,
                frozen=frozen,
                plan=plan,
            )
            evidence = _resolve_c1_evidence_ref(
                ref=refs.get("regime_evidence"),
                opening_path=opening_path,
                frozen=frozen,
                label="regime evidence",
                artifact_type=_C1_EVIDENCE_ARTIFACT_TYPE,
            )
            invariance = _resolve_c1_evidence_ref(
                ref=refs.get("causal_invariance"),
                opening_path=opening_path,
                frozen=frozen,
                label="causal invariance",
                artifact_type=_C1_CAUSALITY_ARTIFACT_TYPE,
            )
            if invariance.get("passed") is not True or not _exact_coverage(evidence, frozen=frozen):
                raise V8IntegrityError("C1 evidence lacks exact causal coverage")
            arm = str(opening.get("arm", "")).lower()
            if (
                route == "autopsy_h3"
                and arm == "autopsy"
                and source_aggregate.get("mechanism_decision") == "H3"
                and _autopsy_evidence_opens_c1(plan)
            ):
                return
            if route == "exp6_regime" and arm == "exp6_3" and _gate_passed(source_aggregate):
                if _exp6_evidence_opens_c1(plan, evidence, frozen=frozen):
                    return
            if route == "router_regime" and arm == "router" and _gate_passed(source_aggregate):
                if _router_evidence_opens_c1(evidence, frozen=frozen):
                    return
        except V8IntegrityError:
            continue
    raise V8IntegrityError("C1/adaptive is closed: no signed mechanism opening gate was found")


def assert_multiseed_replication_candidate(results_root: Path, candidate: str) -> None:
    """Require one signed seed-7 aggregate to nominate the fixed replication candidate."""

    matches = []
    for path, payload in _signed_result_artifacts(results_root):
        if int(payload.get("seed", -1)) != 7:
            continue
        candidate_id = str(
            payload.get("candidate_id", payload.get("arm", payload.get("model_name", "")))
        )
        gate = payload.get("gate_decision", payload.get("gates", {}))
        nominated = bool(payload.get("multiseed_replication_candidate") is True)
        if isinstance(gate, Mapping):
            nominated = nominated or bool(gate.get("multiseed_replication_candidate") is True)
        if candidate_id == candidate and nominated:
            _assert_finite(payload, label=f"multiseed replication aggregate {path}")
            _assert_closed(
                payload.get("closed_evaluation", {}),
                label=f"multiseed replication aggregate {path}",
            )
            matches.append(path)
    if len(matches) != 1:
        raise V8IntegrityError(
            "multiseed replication requires exactly one signed seed-7 aggregate with "
            f"multiseed_replication_candidate=true for {candidate!r}; found {len(matches)}"
        )


CommandRunner = Callable[[Sequence[str]], int]


def _subprocess_runner(command: Sequence[str]) -> int:
    return subprocess.run(list(command), cwd=ROOT, check=False).returncode


def run_stage(
    *,
    stage: str,
    device: str,
    protocol_path: Path,
    manifest_path: Path,
    results_root: Path,
    candidate: str | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Validate and execute one concrete V8 stage exactly once when complete."""

    if stage not in STAGES or stage in {"screen", "all"}:
        raise ValueError(f"run_stage expects a concrete V8 stage, got {stage!r}")
    frozen = verify_frozen_inputs(protocol_path, manifest_path)
    results_root = _repo_path(results_root)
    if stage_is_complete(results_root=results_root, stage=stage, frozen=frozen):
        return write_stage_state(
            results_root=results_root,
            stage=stage,
            status="completed",
            frozen=frozen,
            detail="reused signed completed stage summary",
            candidate=candidate,
        )
    if stage == "adaptive":
        assert_adaptive_gate(results_root=results_root, frozen=frozen)
    if stage == "multiseed_replication":
        if candidate is None:
            raise V8IntegrityError("multiseed_replication stage requires --candidate")
        assert_multiseed_replication_candidate(results_root, candidate)
    command = _script_command(stage, device=device, candidate=candidate)
    _assert_script_exists(command)
    write_stage_state(
        results_root=results_root,
        stage=stage,
        status="running",
        frozen=frozen,
        detail="validated inputs; delegated command started",
        command=command,
        candidate=candidate,
    )
    exit_code = (command_runner or _subprocess_runner)(command)
    if exit_code != 0:
        return write_stage_state(
            results_root=results_root,
            stage=stage,
            status="failed",
            frozen=frozen,
            detail=f"delegated command returned exit code {exit_code}",
            command=command,
            candidate=candidate,
        )
    return write_stage_state(
        results_root=results_root,
        stage=stage,
        status="completed",
        frozen=frozen,
        detail="delegated command completed; downstream artifacts must be independently signed",
        command=command,
        candidate=candidate,
    )


def run_requested_stages(
    *,
    stage: str,
    device: str,
    protocol_path: Path,
    manifest_path: Path,
    results_root: Path,
    candidate: str | None = None,
    command_runner: CommandRunner | None = None,
) -> list[dict[str, Any]]:
    """Expand ``screen``/``all`` and execute stages in the preregistered order."""

    if stage not in STAGES:
        raise ValueError(f"unsupported V8 stage {stage!r}; choose from {', '.join(STAGES)}")
    selected = SCREEN_STAGES if stage == "screen" else ALL_STAGES if stage == "all" else (stage,)
    outputs = []
    for item in selected:
        result = run_stage(
            stage=item,
            device=device,
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            results_root=results_root,
            candidate=candidate,
            command_runner=command_runner,
        )
        outputs.append(result)
        if result["status"] != "completed":
            break
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the Python runner invoked by the PowerShell wrapper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "artifacts/scientific_recovery_v8"
    )
    parser.add_argument("--candidate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run V8 stages with shell-safe nonzero failures."""

    args = parse_args(argv)
    try:
        outputs = run_requested_stages(
            stage=args.stage,
            device=args.device,
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            results_root=args.results_root,
            candidate=args.candidate,
        )
    except (OSError, ValueError, V8IntegrityError) as error:
        print(f"V8 runner failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"stages": [(item["stage"], item["status"]) for item in outputs]}))
    return 0 if outputs and outputs[-1]["status"] == "completed" else 2


__all__ = [
    "ALL_STAGES",
    "SCREEN_STAGES",
    "STAGES",
    "FrozenV8Inputs",
    "RunResumeState",
    "V8IntegrityError",
    "assess_run_resume_state",
    "assert_adaptive_gate",
    "assert_multiseed_replication_candidate",
    "run_requested_stages",
    "run_stage",
    "stage_is_complete",
    "stage_state_path",
    "stage_summary_path",
    "verify_frozen_inputs",
    "write_stage_state",
]


if __name__ == "__main__":
    raise SystemExit(main())
