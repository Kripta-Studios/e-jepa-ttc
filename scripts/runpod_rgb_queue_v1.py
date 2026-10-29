"""Run only PAIR_R and R_CTX after their admitted RGB producer dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json
from operational.runpod_migration import runner
from operational.runpod_migration.inventory import digest, read
from operational.runpod_migration.rgb_queue import RGBQueuePauseError, run_rgb_downstream
from operational.runpod_migration.rgb_queue_endpoints import route_endpoint

TASK = "RGB_DOWNSTREAM"
ORIGIN_GIT_BASE = "771c96a23784e1f3955cec21ba1f4d9a063a9145"
RGB_PILOT_ARCHIVE_SHA256 = "a3dadc7e37e35735c9dad2d70bae5d3e28634a2dd6b802fe10d57ae3f2821cf0"


def _validate_authority(
    assignment_path: Path, run: Path, rotation: int
) -> tuple[datetime, dict[str, str]]:
    assignment = read(assignment_path)
    budget_path = assignment_path.parent / "USER_BUDGET_24USD.json"
    exclusion_path = assignment_path.parent / "LOCAL_QUEUE_EXCLUSION.json"
    budget, exclusion = read(budget_path), read(exclusion_path)
    migration_path = run / "MIGRATION_BINDING.json"
    migration = read(migration_path)
    task = [{"rotation": rotation, "fit": TASK}]
    assigned = assignment.get("maximum_compute_usd")
    total = budget.get("maximum_total_usd")
    deadline = datetime.fromisoformat(str(assignment.get("execution_deadline_utc")))
    cap = datetime.fromisoformat(str(budget.get("execution_deadline_utc")))
    if (
        assignment.get("schema") != "ejepa_cloud_assignment_v1"
        or assignment.get("execution_authorized") is not True
        or assignment.get("tasks") != task
        or assignment.get("budget_is_global_same_pod_not_additive") is not True
        or assignment.get("migration_binding_sha256") != digest(migration_path)
        or budget.get("schema") != "ejepa_cloud_budget_authorization_v1"
        or not isinstance(total, (int, float))
        or not isinstance(assigned, (int, float))
        or not 0 < assigned <= total <= 24
        or deadline.tzinfo is None
        or cap.tzinfo is None
        or not datetime.now(UTC) < deadline <= cap
    ):
        raise ValueError("RGB assignment, global budget, binding, or deadline is invalid")
    if (
        exclusion.get("schema") != "ejepa_cloud_rgb_dependency_exclusion_v1"
        or exclusion.get("status") != "EXCLUDED_FROM_LOCAL_EXECUTION"
        or exclusion.get("tasks") != task
        or exclusion.get("producer_publication_forbidden_until_downstream_return_verified")
        is not True
        or set(exclusion.get("producer_stop_sha256", {})) != {"R_A5", "R_C2F"}
        or digest(exclusion_path) != assignment.get("local_queue_exclusion_receipt_sha256")
        or exclusion.get("scientific_config_sha256") != migration.get("original_authority_sha256")
    ):
        raise ValueError("RGB local/cloud dependency ownership is absent or changed")
    return deadline, {
        "assignment_sha256": digest(assignment_path),
        "budget_sha256": digest(budget_path),
        "exclusion_sha256": digest(exclusion_path),
        "migration_binding_sha256": digest(migration_path),
        "queue_entrypoint_sha256": digest(Path(__file__)),
        "queue_library_sha256": digest(
            Path(__file__).resolve().parents[1] / "operational/runpod_migration/rgb_queue.py"
        ),
        "queue_endpoint_router_sha256": digest(
            Path(__file__).resolve().parents[1]
            / "operational/runpod_migration/rgb_queue_endpoints.py"
        ),
    }


def _admit(run: Path, rotation: int, authority: dict[str, str]) -> Path:
    path = run / "RGB_DOWNSTREAM_ADMISSION.json"
    value: dict[str, Any] = {
        "schema": "ejepa_cloud_rgb_downstream_admission_v1",
        "status": "PASS",
        "task": {"rotation": rotation, "fit": TASK},
        **authority,
        "event_branch_authorized": False,
        "event_inputs_authorized": False,
        "scientific_execution_allowed": True,
        "scientific_pipeline_edited": False,
        "required_producers": ["R_A5", "R_C2F"],
        "allowed_new_endpoints": ["PAIR_R", "R_CTX"],
        "allowed_generation_interface": "PHASE17_RAW_RGB",
    }
    runner._write_once(path, value)
    return path


def _install_origin_attribution(run: Path, bundle_receipt: Path) -> dict[str, str]:
    """Restore only the recorded base-commit attribution omitted from the tar."""
    receipt = read(bundle_receipt)
    if (
        receipt.get("profile") != "rgb_pilot"
        or receipt.get("sha256") != RGB_PILOT_ARCHIVE_SHA256
        or not isinstance(receipt.get("bytes"), int)
        or receipt["bytes"] <= 0
    ):
        raise ValueError("RGB pilot archive receipt is absent or differs from the frozen tar")
    from operational.final_refit import training

    training._git_commit = lambda: ORIGIN_GIT_BASE
    provenance = {
        "schema": "ejepa_rgb_head_git_attribution_v1",
        "origin_git_commit": ORIGIN_GIT_BASE,
        "origin_git_commit_is_base_only": True,
        "clean_checkout_claimed": False,
        "exact_execution_identified_by_scientific_and_bundle_hashes": True,
        "rgb_pilot_archive_sha256": RGB_PILOT_ARCHIVE_SHA256,
        "bundle_archive_receipt_sha256": digest(bundle_receipt),
        "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
    }
    path = run / "RGB_HEAD_GIT_ATTRIBUTION.json"
    runner._write_once(path, provenance)
    return {
        "git_attribution_sha256": digest(path),
        "bundle_archive_receipt_sha256": digest(bundle_receipt),
    }


def _validate_coverage(run: Path, path: Path) -> dict[str, str]:
    coverage = read(path)
    claimed = coverage.get("receipt_sha256")
    payload = {key: value for key, value in coverage.items() if key != "receipt_sha256"}
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    migration = read(run / "MIGRATION_BINDING.json")
    if (
        coverage.get("schema") != "ejepa_rgb_downstream_bundle_coverage_v1"
        or coverage.get("status") != "PASS"
        or claimed != actual
        or coverage.get("scientific_config_sha256") != migration.get("original_authority_sha256")
        or set(coverage.get("rotations", {})) != {"2", "3", "4"}
        or coverage.get("pair_r", {}).get("covered") is not True
        or coverage.get("r_ctx", {}).get("covered") is not True
        or coverage.get("r_ctx", {}).get("event_metadata_required") is not False
        or coverage.get("gpu_used") is not False
        or coverage.get("model_loaded") is not False
        or coverage.get("optimizer_updates") != 0
    ):
        raise ValueError("RGB downstream input coverage receipt is absent or changed")
    checker = (
        Path(__file__).resolve().parents[1]
        / "operational/runpod_migration/rgb_downstream_coverage.py"
    )
    return {
        "rgb_downstream_coverage_sha256": digest(path),
        "rgb_downstream_coverage_checker_sha256": digest(checker),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--a5-run", type=Path)
    parser.add_argument("--c2f-run", type=Path)
    parser.add_argument("--rotation", type=int, required=True, choices=(2, 3, 4))
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--bundle-receipt", type=Path, required=True)
    parser.add_argument("--coverage-receipt", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0 or args.budget_bytes <= 0:
        raise ValueError("Poll interval and memory budget must be positive")
    if __package__:
        from . import runpod_technical_overlay_v3 as technical
    else:
        import runpod_technical_overlay_v3 as technical

    runner.prepare = technical.prepare
    runner.prepare(args.bundle, args.run, args.budget_bytes)
    from operational.final_refit.continuation_queue import queue_lock
    from operational.final_refit.scientific_oof_pipeline import ResourcePause

    run = args.run.resolve(strict=True)
    provenance = _install_origin_attribution(run, args.bundle_receipt.resolve(strict=True))
    provenance.update(_validate_coverage(run, args.coverage_receipt.resolve(strict=True)))
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "optimizer_updates": 0,
                    "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
                    **provenance,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.assignment is None or args.a5_run is None or args.c2f_run is None:
        raise ValueError("Execution requires assignment plus both RGB producer run roots")
    deadline, authority = _validate_authority(
        args.assignment.resolve(strict=True), run, args.rotation
    )
    authority.update(
        {
            "r_a5_producer_run": str(args.a5_run.resolve(strict=True)),
            "r_c2f_producer_run": str(args.c2f_run.resolve(strict=True)),
        }
    )
    authority.update(provenance)
    admission_path = _admit(run, args.rotation, authority)
    authority = {**authority, "rgb_downstream_admission_sha256": digest(admission_path)}
    while datetime.now(UTC) < deadline:
        try:
            with queue_lock(run / f"CLOUD_RGB_R{args.rotation}.lock"):
                route_endpoint(run, args.a5_run, args.rotation, "R_A5")
                route_endpoint(run, args.c2f_run, args.rotation, "R_C2F")
                result = run_rgb_downstream(
                    run,
                    args.rotation,
                    deadline_utc=deadline,
                    authority=authority,
                )
        except (RGBQueuePauseError, ResourcePause) as error:
            atomic_write_json(
                run / "RGB_DOWNSTREAM_PAUSE.json",
                {
                    "schema": "ejepa_cloud_rgb_downstream_pause_v1",
                    "status": "PAUSED_RECOVERABLE",
                    "rotation": args.rotation,
                    "reason": str(error),
                    "manual": bool(getattr(error, "manual", False)),
                    "resources": dict(getattr(error, "resources", {})),
                    "utc": datetime.now(UTC).isoformat(),
                    "optimizer_state_durable_at_last_scientific_checkpoint": True,
                },
            )
            return 4
        if result["status"] == "COMPLETE" and result["stage"] == "RGB_OOF":
            result_path = run / "RGB_DOWNSTREAM_RESULT.json"
            runner._write_once(result_path, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.once:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 3
        time.sleep(min(args.poll_seconds, max(0.0, deadline.timestamp() - time.time())))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
