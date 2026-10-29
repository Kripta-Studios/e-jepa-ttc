"""Event-only Pod2 preflight, bounded restart proof, and producer entrypoint."""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.final_refit.recipe import canonical_sha256
from operational.rgb_port.accounting import atomic_write_json
from operational.runpod_migration.event_handoff import activated, install_handoff
from operational.runpod_migration.event_resources import EventResourceMonitor
from operational.runpod_migration.event_runner import preflight_event, prepare_event
from operational.runpod_migration.inventory import digest, read

TASKS = [{"rotation": r, "fit": f} for r in (2, 3, 4) for f in ("E_A5", "E_C2F")]


def assignment(path: Path, migration: str, rotation: int, fit: str) -> dict[str, Any]:
    value = read(path.resolve(strict=True))
    deadline = datetime.fromisoformat(str(value.get("execution_deadline_utc")))
    exclusion_path = path.parent / "EVENT_LOCAL_QUEUE_EXCLUSION.json"
    if (
        value.get("schema") != "ejepa_event_cloud_assignment_v1"
        or value.get("execution_authorized") is not True
        or value.get("tasks") != TASKS
        or value.get("migration_binding_sha256") != migration
        or value.get("technical_update_cap_per_fit") != 8
        or value.get("technical_update_cap_total") != 48
        or value.get("maximum_total_usd") != 24
        or deadline.tzinfo is None
        or deadline <= datetime.now(UTC)
        or {"rotation": rotation, "fit": fit} not in value["tasks"]
        or digest(exclusion_path) != value.get("local_queue_exclusion_sha256")
    ):
        raise ValueError("Event assignment/exclusion is absent, expired, or differs")
    excluded = read(exclusion_path)
    if excluded.get("status") != "EXCLUDED_FROM_LOCAL_EXECUTION" or excluded.get("tasks") != TASKS:
        raise ValueError("All six cloud tuples must be excluded locally")
    EventResourceMonitor.deadline_timestamp = deadline.timestamp()
    return value


def proof_path(run: Path, rotation: int, fit: str) -> Path:
    return run / "technical_event_restart_probe" / f"r{rotation}" / fit / "RESULT.json"


def validate_proof(path: Path, migration: str, rotation: int, fit: str) -> dict[str, Any]:
    proof = read(path.resolve(strict=True))
    valid = (
        proof.get("schema") == "ejepa_linux_resume_probe_v1"
        and proof.get("status") == "PASS"
        and proof.get("rotation") == rotation
        and proof.get("fit") == fit
        and proof.get("technical_optimizer_updates") == 8
        and proof.get("scientific_optimizer_updates") == 0
        and set(proof.get("cases", {})) == {"uninterrupted", "restored"}
        and proof.get("cross_hardware_bitwise_parity_claimed") is False
    )
    if (rotation, fit) != (2, "E_A5"):
        valid = valid and proof.get("migration_binding_sha256") == migration
    states = [case.get("state") for case in proof.get("cases", {}).values()]
    if not valid or len(states) != 2 or states[0] != states[1]:
        raise ValueError("Matching eight-update event restart proof is required")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "bind-assignment", "install-handoff", "preflight", "probe", "producer"),
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--fit", choices=("E_A5", "E_C2F"), default="E_A5")
    parser.add_argument("--budget-bytes", type=int, default=60_999_999_488)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--handoff-root", type=Path)
    parser.add_argument("--pause-receipt", type=Path)
    parser.add_argument("--additional-updates", type=int)
    args = parser.parse_args()
    setup = prepare_event(args.bundle, args.run, args.budget_bytes)
    if args.command == "prepare":
        print(json.dumps({"status": "PASS", **setup["resources"]}, indent=2))
        return 0
    if args.command == "bind-assignment":
        if args.authorization is None or args.assignment is None:
            raise ValueError("--authorization and --assignment are required")
        authorization = read(args.authorization.resolve(strict=True))
        exclusion = args.authorization.parent / "EVENT_LOCAL_QUEUE_EXCLUSION.json"
        budget = args.authorization.parent / "POD2_BUDGET_ACTIVE.json"
        if not (
            authorization.get("schema") == "ejepa_event_execution_authorization_v1"
            and authorization.get("status") == "AUTHORIZED"
            and authorization.get("tasks") == TASKS
            and authorization.get("technical_update_cap_per_fit") == 8
            and authorization.get("technical_update_cap_total") == 48
            and authorization.get("maximum_total_usd") == 24
            and authorization.get("local_queue_exclusion_sha256") == digest(exclusion)
            and authorization.get("budget_receipt_sha256") == digest(budget)
        ):
            raise ValueError("Local event execution authorization differs")
        value = {
            "schema": "ejepa_event_cloud_assignment_v1",
            "execution_authorized": True,
            "tasks": TASKS,
            "migration_binding_sha256": setup["migration_binding_sha256"],
            "authorization_sha256": digest(args.authorization),
            "local_queue_exclusion_sha256": digest(exclusion),
            "budget_receipt_sha256": digest(budget),
            "maximum_total_usd": 24,
            "technical_update_cap_per_fit": 8,
            "technical_update_cap_total": 48,
            "execution_deadline_utc": authorization["execution_deadline_utc"],
        }
        atomic_write_json(args.assignment, value)
        print(json.dumps(value, indent=2))
        return 0
    if args.command == "install-handoff":
        if args.handoff_root is None:
            raise ValueError("--handoff-root is required")
        print(json.dumps(install_handoff(args.handoff_root, args.run), indent=2))
        return 0
    source, recipe, report = preflight_event(setup, args.run, args.rotation, args.fit)
    try:
        if args.command == "preflight":
            print(json.dumps(report, indent=2))
            return 0
        if args.assignment is None:
            raise ValueError("Optimizer execution requires --assignment")
        assignment(args.assignment, setup["migration_binding_sha256"], args.rotation, args.fit)
        from operational.final_refit import scientific_fold_producers as science
        from operational.final_refit.continuation_queue import queue_lock

        monitor = EventResourceMonitor(budget_bytes=args.budget_bytes)
        result_path = proof_path(args.run, args.rotation, args.fit)
        lock = args.run / f"EVENT_WORKER_R{args.rotation}_{args.fit}.lock"
        if args.command == "probe":
            with queue_lock(lock):
                if (args.rotation, args.fit) == (2, "E_A5"):
                    if args.pause_receipt is None:
                        raise ValueError("Imported E_A5/R2 probe requires --pause-receipt")
                    from operational.runpod_migration.event_restart_probe import probe

                    result = probe(
                        science,
                        source,
                        recipe,
                        setup["config"],
                        args.run,
                        fit=args.fit,
                        assignment=args.assignment,
                        pause_receipt=args.pause_receipt,
                        monitor=monitor,
                    )
                else:
                    from operational.runpod_migration.probe import probe

                    case_root = result_path.parent / "_cases"
                    case_root.mkdir(parents=True)
                    shutil.copyfile(
                        args.run / "SCIENTIFIC_CONFIG.json",
                        case_root / "SCIENTIFIC_CONFIG.json",
                    )
                    result = probe(
                        source,
                        science,
                        setup["config"],
                        case_root,
                        args.fit,
                        monitor,
                        setup["migration_binding_sha256"],
                    )
                    atomic_write_json(result_path, result)
            print(json.dumps(result, indent=2))
            return 0
        validate_proof(result_path, setup["migration_binding_sha256"], args.rotation, args.fit)
        original_identity = science.identity_for
        binding: dict[str, Any] = {}
        if (args.rotation, args.fit) == (2, "E_A5"):
            if args.pause_receipt is None:
                raise ValueError("Imported E_A5/R2 producer requires --pause-receipt")
            context = activated(
                science,
                args.run / "rotations/2/fits/E_A5",
                source,
                recipe,
                fit=args.fit,
                assignment=args.assignment,
                pause_receipt=args.pause_receipt,
                technical=False,
            )
        else:

            def cloud_identity(*pos: Any, **kw: Any) -> dict[str, Any]:
                identity = original_identity(*pos, **kw)
                identity.pop("identity_sha256")
                identity.update(
                    cloud_migration_sha256=setup["migration_binding_sha256"],
                    linux_resume_proof_sha256=digest(result_path),
                    assignment_sha256=digest(args.assignment),
                )
                identity["identity_sha256"] = canonical_sha256(identity)
                return identity

            science.identity_for = cloud_identity
            context = nullcontext(binding)
        try:
            with context as binding, queue_lock(lock):
                receipt = science.train(
                    source=source,
                    fit=args.fit,
                    config=Path(setup["config"]["config"]),
                    contract=Path(setup["config"]["fold_contract"]),
                    run=args.run,
                    monitor=monitor,
                    device="cuda",
                    max_updates=args.additional_updates,
                )
            if receipt["status"] == "COMPLETE":
                admission = {
                    "schema": "ejepa_event_scientific_cloud_admission_v1",
                    "status": "ADMITTED",
                    "rotation": args.rotation,
                    "fit": args.fit,
                    "assignment_sha256": digest(args.assignment),
                    "execution_binding_sha256": (
                        digest(
                            args.run
                            / f"rotations/{args.rotation}/fits/{args.fit}"
                            / "EVENT_EXECUTION_BINDING.json"
                        )
                        if binding
                        else None
                    ),
                    "probe_result_sha256": digest(result_path),
                    "input_identity_sha256": canonical_sha256(report["batch_identity"]),
                    "migration_binding_sha256": setup["migration_binding_sha256"],
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "completed_updates": receipt["completed_updates"],
                    "endpoint_updates": recipe.updates,
                    "cross_hardware_bitwise_continuity_claimed": False,
                }
                atomic_write_json(
                    args.run
                    / f"rotations/{args.rotation}/fits/{args.fit}"
                    / "SCIENTIFIC_CLOUD_ADMISSION.json",
                    admission,
                )
            print(json.dumps(receipt, indent=2))
            return 0 if receipt["status"] == "COMPLETE" else 3
        finally:
            science.identity_for = original_identity
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
