"""Immutable contract for the additive RGB-PORT host pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import (
    atomic_write_bytes,
    atomic_write_json,
    read_bytes_shared,
    read_json_shared,
    sha256_file,
)
from operational.rgb_port_acceleration.contracts import validate_acceleration_freeze
from operational.rgb_port_revision.migration import source_matches

SCHEMA = "rgb_port_pipeline_freeze_v2"
EVENT_FIT_IDS = ("E_A5_MATCHED", "E_C2F_MATCHED")
CANARY = {
    "scheduled_updates": 300,
    "subwindow_updates": 100,
    "subwindow_count": 3,
    "baseline_window_updates": 1000,
    "decision_boundary": "DURABLE_FULL_CHECKPOINT",
    "measurement_clock": "SCIENCE_STARTED_TO_DURABLE_CHECKPOINT_EXCLUDING_RESTORE_PREWARM",
    "baseline_updates_per_minute": 93.51292713076926,
    "minimum_speedup_factor_exclusive": 1.05,
    "minimum_updates_per_minute_exclusive": 98.18857348730772,
    "minimum_subwindow_rate_inclusive": 88.83728077423079,
    "enable_rule": (
        "median_3x100_rate_and_total300_rate_strictly_above_threshold;"
        "each_window_at_least_95pct_baseline;all_integrity_gates_pass"
    ),
    "integrity_gates": [
        "byte_parity",
        "schedule_order_unchanged",
        "zero_recovery_or_resource_failures",
        "finite_outputs",
    ],
    "foreground_wait_ratio": "DIAGNOSTIC_ONLY_NO_THRESHOLD",
    "overlap_fraction": "1-source_wait_seconds/source_prepare_seconds_DIAGNOSTIC_ONLY",
    "rollback_rule": "otherwise_continue_from_durable_canary_boundary_with_V1_runtime",
    "downstream_scope": "PASS_enables_A5_and_future_C2F;FAIL_keeps_V1_for_both",
    "forbidden_effects": ["new_scientific_arm", "extra_optimizer_update", "schedule_change"],
}


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "identity_sha256"}


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _executed_paths(repository: Path) -> list[Path]:
    paths = [
        repository / "operational/rgb_port_pipeline_v2/prefetch.py",
        repository / "operational/rgb_port_pipeline_v2/producer.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pipeline executed source is incomplete: {missing}")
    return paths


def _orchestration_paths(repository: Path) -> list[Path]:
    paths = [
        repository / "operational/rgb_port_pipeline_v2/__init__.py",
        repository / "operational/rgb_port_pipeline_v2/contracts.py",
        repository / "operational/rgb_port_pipeline_v2/queue.py",
        repository / "operational/rgb_port_pipeline_v2/receipts.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pipeline orchestration source is incomplete: {missing}")
    return paths


def _validate_admission(
    path: Path,
    *,
    kind: str,
    acceleration: Mapping[str, Any],
    acceleration_file_sha256: str,
    source_sha256: Mapping[str, str],
    orchestration_source_sha256: Mapping[str, str],
    a5_origin: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json_shared(path)
    scope = value.get("event_fit_ids", value.get("fit_ids"))
    if kind == "cpu":
        evidence_valid = (
            value.get("batch_byte_parity") is True
            and value.get("schedule_order_unchanged") is True
            and value.get("rng_cursor_unchanged") is True
            and value.get("deferred_oserror_passed") is True
        )
    else:
        paused_a5 = value.get("paused_a5_origin")
        evidence_valid = (
            value.get("finite_outputs") is True
            and value.get("recovery_failures") == 0
            and value.get("resource_failures") == 0
            and value.get("checkpoint_integrity") is True
            and isinstance(paused_a5, dict)
            and a5_origin.get("mode") == "PAUSED_FULL_CHECKPOINT"
            and paused_a5.get("fit_id") == EVENT_FIT_IDS[0]
            and isinstance(paused_a5.get("completed_updates"), int)
            and paused_a5.get("completed_updates", 0) > 0
            and isinstance(paused_a5.get("checkpoint_sha256"), str)
            and isinstance(paused_a5.get("identity_sha256"), str)
            and paused_a5.get("completed_updates") == a5_origin.get("completed_updates")
            and paused_a5.get("checkpoint_sha256") == a5_origin.get("checkpoint_sha256")
            and paused_a5.get("identity_sha256") == a5_origin.get("identity_sha256")
        )
    if (
        value.get("status") != "PASSED"
        or value.get("optimizer_updates") != 0
        or set(scope or ()) != set(EVENT_FIT_IDS)
        or value.get("admission_kind", value.get("kind")) != kind
        or value.get("original_acceleration_freeze_sha256") != acceleration_file_sha256
        or value.get("original_acceleration_freeze_identity_sha256")
        != acceleration["identity_sha256"]
        or value.get("source_sha256") != dict(source_sha256)
        or value.get("orchestration_source_sha256") != dict(orchestration_source_sha256)
        or not evidence_valid
    ):
        raise ValueError(f"Pipeline {kind} admission is outside the authorized zero-update scope")
    return value


def _seal_origin(run: Path, fit_id: str) -> dict[str, Any]:
    fit = run / "fits" / fit_id
    pointer_path = fit / "CHECKPOINT_POINTER.json"
    if not pointer_path.is_file():
        return {"fit_id": fit_id, "mode": "FRESH_UNSTARTED", "completed_updates": 0}
    pointer = read_json_shared(pointer_path)
    receipt = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
    update = int(pointer.get("completed_updates", -1))
    checkpoint = fit / "checkpoint_versions" / str(pointer.get("version", ""))
    if (
        update <= 0
        or receipt.get("completed_updates") != update
        or receipt.get("identity_sha256") != pointer.get("identity_sha256")
        or receipt.get("checkpoint_sha256") != pointer.get("checkpoint_sha256")
        or receipt.get("status") not in {"PAUSED_RESOURCE", "COMPLETE", "COMPLETED"}
        or not checkpoint.is_file()
        or sha256_file(checkpoint) != pointer.get("checkpoint_sha256")
    ):
        raise RuntimeError(f"Pipeline origin is not a sealed full checkpoint: {fit_id}")
    target = run / "pipeline_v2/origins" / f"{fit_id}_{update:06d}.pt"
    if target.exists() and sha256_file(target) != pointer["checkpoint_sha256"]:
        raise RuntimeError(f"Immutable pipeline origin changed: {fit_id}")
    if not target.exists():
        atomic_write_bytes(target, read_bytes_shared(checkpoint))
    if sha256_file(target) != pointer["checkpoint_sha256"]:
        raise RuntimeError(f"Pipeline origin copy was not exact: {fit_id}")
    return {
        "fit_id": fit_id,
        "mode": "PAUSED_FULL_CHECKPOINT",
        "completed_updates": update,
        "checkpoint_path": str(target.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(target),
        "identity_sha256": pointer["identity_sha256"],
    }


def build_pipeline_freeze(
    run: Path,
    cpu_admission: Path,
    gpu_admission: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Seal the additive pipeline against the current paused event checkpoint."""
    run = run.resolve(strict=True)
    repository = _repository()
    source_freeze = run / "SOURCE_FREEZE.json"
    acceleration_path = run / "ACCELERATION_FREEZE.json"
    baseline_path = run / "current_bottleneck_20261009/V1_RATE_BASELINE.json"
    acceleration = validate_acceleration_freeze(acceleration_path)
    baseline = read_json_shared(baseline_path)
    source_sha256 = {str(path.resolve()): sha256_file(path) for path in _executed_paths(repository)}
    orchestration_source_sha256 = {
        str(path.resolve()): sha256_file(path) for path in _orchestration_paths(repository)
    }
    origins = {fit_id: _seal_origin(run, fit_id) for fit_id in EVENT_FIT_IDS}
    if (
        baseline.get("schema") != "rgb_port_pipeline_performance_baseline_v1"
        or baseline.get("updates") != CANARY["baseline_window_updates"]
        or baseline.get("updates_per_minute") != CANARY["baseline_updates_per_minute"]
        or baseline.get("acceleration_freeze_file_sha256") != sha256_file(acceleration_path)
    ):
        raise ValueError("V1 performance baseline differs from the preregistered canary")
    admissions: dict[str, dict[str, Any]] = {}
    for kind, source in (("cpu", cpu_admission), ("gpu", gpu_admission)):
        source = source.resolve(strict=True)
        value = _validate_admission(
            source,
            kind=kind,
            acceleration=acceleration,
            acceleration_file_sha256=sha256_file(acceleration_path),
            source_sha256=source_sha256,
            orchestration_source_sha256=orchestration_source_sha256,
            a5_origin=origins[EVENT_FIT_IDS[0]],
        )
        target = run / "pipeline_v2/admissions" / f"{kind.upper()}_ADMISSION.json"
        if target.exists() and read_json_shared(target) != value:
            raise RuntimeError(f"Immutable pipeline {kind} admission changed")
        if not target.exists():
            atomic_write_json(target, value)
        admissions[kind] = {"path": str(target.resolve(strict=True)), "sha256": sha256_file(target)}
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "event_fit_ids": list(EVENT_FIT_IDS),
        "scope": "HOST_PREFETCH_EVENT_ONLY",
        "scientific_identity_unchanged": True,
        "original_source_freeze_path": str(source_freeze.resolve(strict=True)),
        "original_source_freeze_sha256": sha256_file(source_freeze),
        "original_acceleration_freeze_path": str(acceleration_path.resolve(strict=True)),
        "original_acceleration_freeze_sha256": sha256_file(acceleration_path),
        "original_acceleration_freeze_identity_sha256": acceleration["identity_sha256"],
        "admissions": admissions,
        "performance_baseline": {
            "path": str(baseline_path.resolve(strict=True)),
            "sha256": sha256_file(baseline_path),
        },
        "origins": origins,
        "canary": CANARY,
        "source_sha256": source_sha256,
        "orchestration_source_sha256": orchestration_source_sha256,
    }
    payload["identity_sha256"] = canonical_sha256(_canonical_payload(payload))
    target = output.resolve() if output is not None else run / "PIPELINE_FREEZE.json"
    if target.exists() and read_json_shared(target) != payload:
        raise RuntimeError("Pipeline freeze changed")
    if not target.exists():
        atomic_write_json(target, payload)
    return payload


def validate_pipeline_freeze(path: Path) -> dict[str, Any]:
    """Fail closed on any changed source, admission, parent freeze, or origin."""
    value = read_json_shared(path)
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != "FROZEN"
        or value.get("event_fit_ids") != list(EVENT_FIT_IDS)
        or value.get("scope") != "HOST_PREFETCH_EVENT_ONLY"
        or value.get("canary") != CANARY
        or value.get("identity_sha256") != canonical_sha256(_canonical_payload(value))
    ):
        raise ValueError("Pipeline freeze contract differs")
    for field in ("source_sha256", "orchestration_source_sha256"):
        entries = value.get(field)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Pipeline freeze lacks {field}")
        for source, digest in entries.items():
            if not source_matches(Path(source), digest, path.parent):
                raise RuntimeError(f"Frozen pipeline source changed: {source}")
    for prefix in ("original_source_freeze", "original_acceleration_freeze"):
        if sha256_file(Path(value[f"{prefix}_path"])) != value[f"{prefix}_sha256"]:
            raise RuntimeError(f"Frozen pipeline parent changed: {prefix}")
    acceleration = validate_acceleration_freeze(Path(value["original_acceleration_freeze_path"]))
    if acceleration["identity_sha256"] != value["original_acceleration_freeze_identity_sha256"]:
        raise RuntimeError("Acceleration canonical identity changed")
    admissions = value.get("admissions", {})
    if set(admissions) != {"cpu", "gpu"}:
        raise ValueError("Pipeline freeze lacks its exact admissions")
    for kind, item in admissions.items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("Pipeline admission changed")
        _validate_admission(
            Path(item["path"]),
            kind=kind,
            acceleration=acceleration,
            acceleration_file_sha256=value["original_acceleration_freeze_sha256"],
            source_sha256=value["source_sha256"],
            orchestration_source_sha256=value["orchestration_source_sha256"],
            a5_origin=value["origins"][EVENT_FIT_IDS[0]],
        )
    baseline = value.get("performance_baseline", {})
    if sha256_file(Path(baseline["path"])) != baseline.get("sha256"):
        raise RuntimeError("Pipeline performance baseline changed")
    for fit_id, origin in value.get("origins", {}).items():
        if fit_id not in EVENT_FIT_IDS or origin.get("fit_id") != fit_id:
            raise ValueError("Pipeline origin scope changed")
        if origin.get("mode") == "PAUSED_FULL_CHECKPOINT" and sha256_file(
            Path(origin["checkpoint_path"])
        ) != origin.get("checkpoint_sha256"):
            raise RuntimeError(f"Immutable pipeline origin changed: {fit_id}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--cpu-admission", type=Path, required=True)
    build.add_argument("--gpu-admission", type=Path, required=True)
    build.add_argument("--output", type=Path)
    check = sub.add_parser("validate")
    check.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    value = (
        build_pipeline_freeze(args.run, args.cpu_admission, args.gpu_admission, args.output)
        if args.action == "build"
        else validate_pipeline_freeze(args.freeze)
    )
    print(value["identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
