"""Prepare an additive local queue plan; install only with no live queue or launcher."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import psutil

from .inventory import digest, read


def prepare(repository: Path, run: Path, output: Path) -> dict:
    """Retain every historical binding and add only the explicit return consumer."""
    from operational.final_refit.recipe import canonical_sha256
    from operational.final_refit.scientific_oof_queue import validate_plan
    from operational.rgb_port.accounting import atomic_write_json

    repository, run = repository.resolve(strict=True), run.resolve(strict=True)
    original = read(run / "SCIENTIFIC_OOF_QUEUE_PLAN.json")
    validate_plan(original)
    plan = {**original, "source_bindings": dict(original["source_bindings"])}
    launcher = repository / "scripts/run_cloud_return_oof_stage.ps1"
    paths = [launcher, *sorted((repository / "operational/runpod_migration").glob("*.py"))]
    authorizations = repository / "artifacts/runpod_migration_20261026"
    for rotation in (2, 3, 4):
        for fit in ("R_A5", "R_C2F"):
            folder = authorizations
            if fit == "R_C2F":
                folder /= f"r{rotation}_c2f"
                name = f"SCIENTIFIC_R{rotation}_C2F.json"
            else:
                if rotation != 4:
                    folder /= f"r{rotation}"
                name = f"SCIENTIFIC_R{rotation}.json"
            paths += [folder / name, folder / "LOCAL_QUEUE_EXCLUSION.json"]
    for path in paths:
        resolved = path.resolve(strict=True)
        plan["source_bindings"][str(resolved)] = digest(resolved)
    plan["launcher"] = str(launcher.resolve(strict=True))
    plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(plan)
    validate_plan(plan)
    atomic_write_json(output, plan)
    return {
        "status": "PREPARED_NOT_INSTALLED",
        "original_plan_sha256": original["plan_sha256"],
        "proposed_plan_sha256": plan["plan_sha256"],
        "pending_plan": str(output.resolve()),
        "original_scientific_config_sha256": digest(run / "SCIENTIFIC_CONFIG.json"),
        "active_queue_modified": False,
        "active_training_interrupted": False,
    }


def install(run: Path, pending: Path) -> dict:
    """Refuse active queue/launcher ownership; preserve both previous plan and state."""
    from operational.final_refit.continuation_queue import queue_lock
    from operational.final_refit.scientific_oof_queue import validate_plan
    from operational.rgb_port.accounting import atomic_write_json

    run = run.resolve(strict=True)
    with queue_lock(run / "SCIENTIFIC_OOF_QUEUE.lock"):
        plan = read(pending)
        validate_plan(plan)
        if plan["run"] != str(run):
            raise ValueError("Prepared cloud consumer plan belongs to another run")
        state_path = run / "SCIENTIFIC_OOF_QUEUE_STATE.json"
        state = read(state_path)
        running = state.get("running", {})
        pid = running.get("pid")
        if isinstance(pid, int) and psutil.pid_exists(pid):
            process = psutil.Process(pid)
            if process.create_time() == running.get("create_time"):
                raise ValueError("Cannot change consumer plan while an isolated launcher is alive")
        if (run / "STOP_REQUEST").exists():
            raise ValueError("User manual stop remains active; installation cannot resume it")
        registry = run / "CLOUD_RETURN_REGISTRY.json"
        if (
            not registry.is_file()
            or read(registry).get("schema") != "ejepa_cloud_return_registry_v1"
        ):
            raise ValueError("A verified cloud return registry is required before installation")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        plan_path = run / "SCIENTIFIC_OOF_QUEUE_PLAN.json"
        shutil.copyfile(
            plan_path, run / f"SCIENTIFIC_OOF_QUEUE_PLAN.before_cloud_return_{stamp}.json"
        )
        shutil.copyfile(
            state_path, run / f"SCIENTIFIC_OOF_QUEUE_STATE.before_cloud_return_{stamp}.json"
        )
        state["plan_sha256"] = plan["plan_sha256"]
        state.pop("running", None)
        state.pop("error", None)
        state["status"] = "WAITING_RESOURCES"
        for entry in state["stages"].values():
            if entry["status"] != "COMPLETE":
                entry["status"] = "WAITING_RESOURCES"
                entry["eligible_monotonic"] = 0
                break
        # If interrupted between the two publications, the original queue
        # rejects the plan/state mismatch. It cannot silently launch updates.
        atomic_write_json(plan_path, plan)
        atomic_write_json(state_path, state)
        return {
            "status": "INSTALLED_NOT_STARTED",
            "plan_sha256": plan["plan_sha256"],
            "training_started": False,
            "prior_state_preserved": True,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "install"))
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pending-plan", type=Path, required=True)
    args = parser.parse_args()
    result = (
        prepare(args.repository, args.run, args.pending_plan)
        if args.command == "prepare"
        else install(args.run, args.pending_plan)
    )
    print(json.dumps(result, indent=2))
