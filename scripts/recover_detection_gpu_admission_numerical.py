"""Prepare and verify a numerical-only recovery of an interrupted GPU admission.

The recovery never upgrades the interrupted aggregate admission: performance and
memory remain INCONCLUSIVE.  GPU execution is intentionally left to the reviewed
serial admission command emitted by ``prepare``.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection_gpu_admission import _graphics_baseline, _run_arm, resource_admission

from e_jepa_ttc.detection.campaign_training import canonical_sha256, sha256_file
from e_jepa_ttc.detection.gpu_admission_harness import (
    bind_plan,
    load_plan,
    validate_checkpoint_lineage,
    validate_initial_state,
    verify_numerical_metrics,
)


def _write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite recovery evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "last_write_time_utc": datetime.fromtimestamp(
            resolved.stat().st_mtime, tz=UTC
        ).isoformat(),
    }


def prepare(original_plan_path: Path, output_dir: Path, optimized_output: Path) -> dict[str, Any]:
    """Seal completed controls and emit a fresh optimized-only execution plan."""
    original = load_plan(original_plan_path)
    if optimized_output.exists():
        raise FileExistsError(f"Fresh optimized output already exists: {optimized_output}")
    for name in ("old_A", "old_B"):
        for field in ("metrics_path", "summary_path", "evaluation_path"):
            Path(original["arms"][name][field]).resolve(strict=True)

    recovered = copy.deepcopy(original)
    recovered.pop("plan_sha256")
    recovered["graphics_baseline"] = _graphics_baseline()
    arm = recovered["arms"]["optimized"]
    old_output = str(Path(arm["output"]).resolve())
    fresh_output = str(optimized_output.resolve())
    arm["command"] = [fresh_output if value == old_output else value for value in arm["command"]]
    arm["output"] = fresh_output
    arm["metrics_path"] = str(optimized_output.resolve() / "metrics.jsonl")
    arm["summary_path"] = str(optimized_output.resolve() / "summary.json")
    arm["evaluation_path"] = str(
        optimized_output.resolve() / "evaluation" / "stop_diagnostic_update_001040.json"
    )
    recovered = bind_plan(recovered)
    validate_checkpoint_lineage(recovered)
    initial = validate_initial_state(recovered)

    old_evidence = {
        name: {
            field: _artifact(Path(original["arms"][name][field]))
            for field in ("metrics_path", "summary_path", "evaluation_path")
        }
        for name in ("old_A", "old_B")
    }
    manifest = {
        "schema": "eap_detection_gpu_numerical_recovery_v1",
        "status": "READY_OPTIMIZED_NOT_EXECUTED",
        "original_aggregate_status": "NONPASS_INTERRUPTED_RESOURCE_ADMISSION",
        "candidate_decision": None,
        "performance_comparison": "INCONCLUSIVE",
        "memory_comparison": "INCONCLUSIVE",
        "original_plan": _artifact(original_plan_path),
        "original_plan_canonical_sha256": original["plan_sha256"],
        "original_failure": {
            "receipt": None,
            "reason": "NEW_UNPREREGISTERED_C_PLUS_G_PROCESS_AFTER_OLD_B_METRICS",
            "harness_session_id": 47813,
            "harness_exit_code": 1,
            "harness_output_transcription": (
                '{"status":"BUDGET_STOPPED","final_update":1040}\n'
                "RuntimeError: Concurrent compute appeared during old_B; run is "
                "inadmissible: "
                "['{\"name\": \"\", \"pid\": 72640, \"type\": \"C+G\", "
                "\"used_memory\": \"N/A\"}']"
            ),
            "transcription_captured_after_execution": True,
            "old_b_last_metric_observed_at": "2026-10-29T22:39:00.875+01:00",
            "old_b_summary_written_at": "2026-10-29T22:42:24.550123+01:00",
            "cim_observation": {
                "observed_at": "2026-10-29T22:56:53+01:00",
                "pid": 72640,
                "name": "LogonUI.exe",
                "parent_pid": 2304,
                "creation_time": "2026-10-29T22:40:27.146243+01:00",
                "command": None,
            },
            "optimized_was_executed": False,
            "aggregate_receipt_available": False,
        },
        "completed_control_evidence": old_evidence,
        "recovery_source": _artifact(Path(__file__)),
        "harness_source": _artifact(ROOT / "src/e_jepa_ttc/detection/gpu_admission_harness.py"),
        "serial_executor_source": _artifact(ROOT / "scripts/detection_gpu_admission.py"),
        "recovery_plan_sha256": recovered["plan_sha256"],
        "initial_state": initial,
        "optimized_command": arm["command"],
        "execution_receipt_path": str((output_dir / "execution_receipt.json").resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "plan.json", recovered)
    _write_once(output_dir / "recovery_preregistration.json", manifest)
    return manifest


def _load_preregistration(
    path: Path, expected_sha256: str, plan: dict[str, Any]
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Recovery preregistration hash differs")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "eap_detection_gpu_numerical_recovery_v1",
        "status": "READY_OPTIMIZED_NOT_EXECUTED",
        "original_aggregate_status": "NONPASS_INTERRUPTED_RESOURCE_ADMISSION",
        "candidate_decision": None,
        "performance_comparison": "INCONCLUSIVE",
        "memory_comparison": "INCONCLUSIVE",
        "recovery_plan_sha256": plan["plan_sha256"],
        "optimized_command": plan["arms"]["optimized"]["command"],
    }
    if any(prereg.get(key) != value for key, value in required.items()):
        raise ValueError("Recovery preregistration contract differs")
    if prereg.get("initial_state") != validate_initial_state(plan):
        raise ValueError("Recovery initial state differs from preregistration")
    evidence = prereg.get("completed_control_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"old_A", "old_B"}:
        raise ValueError("Completed control evidence is malformed")
    for fields in evidence.values():
        for artifact in fields.values():
            if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("Completed control evidence changed after prepare")
    return prereg


def verify(
    plan_path: Path,
    preregistration_path: Path,
    preregistration_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Verify only numerical parity; explicitly leave performance inconclusive."""
    plan = load_plan(plan_path)
    prereg = _load_preregistration(preregistration_path, preregistration_sha256, plan)
    receipt_path = Path(prereg["execution_receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "eap_detection_gpu_numerical_recovery_execution_v1"
        or receipt.get("status") != "OPTIMIZED_COMPLETED_PENDING_NUMERICAL_VERIFICATION"
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("preregistration_sha256") != preregistration_sha256
        or receipt.get("original_aggregate_status")
        != "NONPASS_INTERRUPTED_RESOURCE_ADMISSION"
        or receipt.get("performance_comparison") != "INCONCLUSIVE"
        or receipt.get("memory_comparison") != "INCONCLUSIVE"
    ):
        raise ValueError("Recovery execution receipt contract differs")
    arm = receipt.get("optimized_arm")
    if (
        not isinstance(arm, dict)
        or arm.get("name") != "optimized"
        or arm.get("returncode") != 0
        or arm.get("metric_records_observed") != int(plan["updates"])
    ):
        raise ValueError("Optimized execution receipt is incomplete")
    for key in ("initial_resources", "final_resources"):
        resources = receipt.get(key)
        if (
            not isinstance(resources, dict)
            or not isinstance(resources.get("gpu_processes_by_driver_type"), list)
            or isinstance(resources.get("available_physical_ram_bytes"), bool)
            or not isinstance(resources.get("available_physical_ram_bytes"), int)
            or resources["available_physical_ram_bytes"] <= 0
            or isinstance(resources.get("free_vram_bytes"), bool)
            or not isinstance(resources.get("free_vram_bytes"), int)
            or resources["free_vram_bytes"] <= 0
        ):
            raise ValueError("Recovery execution resource evidence is malformed")
    validate_checkpoint_lineage(plan)
    validate_initial_state(plan)
    result = verify_numerical_metrics(plan) | {
        "schema": "eap_detection_gpu_numerical_recovery_result_v1",
        "original_aggregate_status": "NONPASS_INTERRUPTED_RESOURCE_ADMISSION",
        "recovery_plan_sha256": plan["plan_sha256"],
        "preregistration_sha256": sha256_file(preregistration_path),
        "execution_receipt_sha256": sha256_file(receipt_path),
    }
    _write_once(output, result)
    return result


def execute(
    plan_path: Path,
    preregistration_path: Path,
    preregistration_sha256: str,
    approval_token: str,
) -> dict[str, Any]:
    """Execute only the missing optimized arm under the reviewed serial gates."""
    plan = load_plan(plan_path)
    if approval_token != plan["plan_sha256"]:
        raise PermissionError("Approval token must exactly equal the recovery plan hash")
    prereg = _load_preregistration(preregistration_path, preregistration_sha256, plan)
    receipt = Path(prereg["execution_receipt_path"])
    if receipt.exists():
        raise FileExistsError(f"Recovery execution receipt already exists: {receipt}")
    validate_checkpoint_lineage(plan)
    validate_initial_state(plan)
    initial_resources = resource_admission(plan)
    arm_receipt = _run_arm(plan, "optimized")
    final_resources = resource_admission(plan)
    result = {
        "schema": "eap_detection_gpu_numerical_recovery_execution_v1",
        "status": "OPTIMIZED_COMPLETED_PENDING_NUMERICAL_VERIFICATION",
        "plan_sha256": plan["plan_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "original_aggregate_status": "NONPASS_INTERRUPTED_RESOURCE_ADMISSION",
        "performance_comparison": "INCONCLUSIVE",
        "memory_comparison": "INCONCLUSIVE",
        "initial_resources": initial_resources,
        "optimized_arm": arm_receipt,
        "final_resources": final_resources,
    }
    _write_once(receipt, result)
    return result


def continuation_receipt(output: Path) -> dict[str, Any]:
    """Write the non-executed scientific R0 continuation command from update 3840."""
    base = Path("D:/PHASE_TTC_DETECTION_CAMPAIGN_20261029/migration_latest_r0_v2")
    resume = base / "migrated/last.pt"
    selected = base / "migrated/selected.pt"
    target = Path("D:/PHASE_TTC_DETECTION_CAMPAIGN_20261029/r0_optimized_continuation_from_3840_v1")
    optimized = ROOT / "output/detection_campaign/optimized_preregistration_cpu_v2"
    config = ROOT / "output/detection_campaign/rgb_screen_v5_seed7_v1/configs/r0.yaml"
    python = ROOT.parent / "phase-ttc-detection-local/.venv-detector/Scripts/python.exe"
    command = [
        str(python.resolve(strict=True)),
        str(
            (optimized / "code/scripts/train_detection_campaign_optimized.py").resolve(
                strict=True
            )
        ),
        "--config",
        str(config.resolve(strict=True)),
        "--performance-config",
        str(
            (
                optimized
                / "code/configs/detection_campaign/performance/indexed_shared_v1.json"
            ).resolve(strict=True)
        ),
        "--output",
        str(target.resolve()),
        "--resume",
        str(resume.resolve(strict=True)),
    ]
    value = {
        "schema": "eap_detection_r0_continuation_argv_v1",
        "status": "PREPARED_NOT_EXECUTED",
        "start_update": 3840,
        "nominal_stop_update": 8192,
        "resume_checkpoint": _artifact(resume),
        "selected_checkpoint": _artifact(selected),
        "migration_receipt": _artifact(base / "migration_receipt.json"),
        "runtime_preregistration": _artifact(optimized / "runtime_preregistration.json"),
        "command": command,
        "precondition": "PASS_NUMERICAL_PARITY_ONLY recovery and separate serial GPU admission",
    }
    value["canonical_sha256"] = canonical_sha256(value)
    _write_once(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--original-plan", type=Path, required=True)
    prepare_cmd.add_argument("--output-dir", type=Path, required=True)
    prepare_cmd.add_argument("--optimized-output", type=Path, required=True)
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--plan", type=Path, required=True)
    verify_cmd.add_argument("--preregistration", type=Path, required=True)
    verify_cmd.add_argument("--preregistration-sha256", required=True)
    verify_cmd.add_argument("--output", type=Path, required=True)
    execute_cmd = sub.add_parser("execute")
    execute_cmd.add_argument("--plan", type=Path, required=True)
    execute_cmd.add_argument("--preregistration", type=Path, required=True)
    execute_cmd.add_argument("--preregistration-sha256", required=True)
    execute_cmd.add_argument("--execute-gpu", action="store_true", required=True)
    execute_cmd.add_argument("--approval-token", required=True)
    continuation_cmd = sub.add_parser("continuation-argv")
    continuation_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.original_plan, args.output_dir, args.optimized_output)
    elif args.command == "verify":
        verify(
            args.plan,
            args.preregistration,
            args.preregistration_sha256,
            args.output,
        )
    elif args.command == "execute":
        execute(
            args.plan,
            args.preregistration,
            args.preregistration_sha256,
            args.approval_token,
        )
    else:
        continuation_receipt(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
