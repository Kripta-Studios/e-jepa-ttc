"""Durable continuation graph; never substitute in-sample features for OOF.

The deployed executor can wait for RGB endpoints and validate fold membership.
Scientific OOF execution is deliberately unavailable in the current repository.
Missing adapters are visible blockers, not successful no-op training jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from .auto_resume import atomic_json, file_digest
from .recipe import canonical_sha256

SCHEMA = "final_refit_continuation_queue_v1"
BLOCKERS = [
    {
        "id": "scientific_fold_producers",
        "source": "operational/final_refit/fold_producers.py",
        "reason": "Only bounded technical fits exist; scientific=True raises RuntimeError.",
    },
    {
        "id": "scientific_fold_heads",
        "source": "operational/final_refit/fold_training.py",
        "reason": "Non-technical PAIR/CTX fits are explicitly rejected.",
    },
    {
        "id": "oof_generation_and_deployment",
        "source": "docs/final_refit_20261017/OOF_TRAINING_UNIT_LIMITS.md",
        "reason": (
            "Missing admitted end-to-end generation, transitive endpoint ancestry, "
            "OOF cache-to-final-training adapters and deployment sealing."
        ),
    },
]


def read(path: Path) -> dict[str, Any]:
    """Read one atomic JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def topology(nodes: list[dict[str, Any]]) -> None:
    """Require a unique, topologically ordered graph with explicit capabilities."""
    seen: set[str] = set()
    for node in nodes:
        name = node["id"]
        if name in seen or not set(node["depends_on"]).issubset(seen):
            raise ValueError(f"Duplicate node, cycle or forward/missing dependency: {name}")
        if node["executor"] not in ("wait_rgb", "verify_roles", "unimplemented"):
            raise ValueError(f"Unregistered executor: {node['executor']}")
        seen.add(name)
    if not seen:
        raise ValueError("Empty queue")


def build_plan(repo: Path, runtime: Path, locator: Path, folds: Path) -> dict[str, Any]:
    """Freeze dependencies and current limitations without enabling missing runners."""
    nodes: list[dict[str, Any]] = []

    def add(name: str, parents: list[str], executor: str = "unimplemented") -> None:
        nodes.append({"id": name, "depends_on": parents, "executor": executor})

    add("RGB_ENDPOINTS", [], "wait_rgb")
    add("VERIFY_OOF_ROLES", ["RGB_ENDPOINTS"], "verify_roles")
    add("OOF_SCIENTIFIC_IMPLEMENTATION", ["VERIFY_OOF_ROLES"])
    generations = []
    for rotation in range(5):
        prefix = f"ROTATION_{rotation}"
        producers = [f"{prefix}_{fit}" for fit in ("E_A5", "E_C2F", "R_A5", "R_C2F")]
        for producer in producers:
            add(producer, ["OOF_SCIENTIFIC_IMPLEMENTATION"])
        for modality, a5, c2f in (
            ("E", producers[0], producers[1]),
            ("R", producers[2], producers[3]),
        ):
            pair_cache = f"{prefix}_PAIR_{modality}_CACHE"
            pair = f"{prefix}_PAIR_{modality}"
            contexts = f"{prefix}_{modality}_PHASE17"
            ctx = f"{prefix}_{modality}_CTX"
            generated = f"{prefix}_{modality}_GENERATION"
            add(pair_cache, [a5])
            add(pair, [pair_cache])
            add(contexts, [a5, c2f, pair])
            add(ctx, [contexts])
            add(generated, [ctx, contexts])
            generations.append(generated)
    add("STITCH_OOF_PHASE17_PHASE20", generations)
    add("OOF_NORMALIZERS", ["STITCH_OOF_PHASE17_PHASE20"])
    for modality in ("E", "R"):
        add(f"FINAL_PAIR_{modality}_CACHE", ["RGB_ENDPOINTS", "OOF_SCIENTIFIC_IMPLEMENTATION"])
        add(f"FINAL_PAIR_{modality}", [f"FINAL_PAIR_{modality}_CACHE"])
        add(f"FINAL_{modality}_CTX", ["OOF_NORMALIZERS"])
    add("FINAL_PHASE_ONLY", ["STITCH_OOF_PHASE17_PHASE20"])
    add(
        "SEAL_DEPLOYMENT_CHAIN",
        ["FINAL_PAIR_E", "FINAL_PAIR_R", "FINAL_E_CTX", "FINAL_R_CTX", "FINAL_PHASE_ONLY"],
    )
    add("EXPORT_AND_PARITY", ["SEAL_DEPLOYMENT_CHAIN"])
    add("TEST12_INPUT_ADMISSION", ["EXPORT_AND_PARITY"])
    add("TEST12_PREDICTIONS", ["TEST12_INPUT_ADMISSION"])
    add("PACKAGE_SUBMISSION_ZIP", ["TEST12_PREDICTIONS"])
    topology(nodes)
    bindings = {
        str(p.resolve()): file_digest(p)
        for p in (
            Path(__file__),
            repo / "operational/final_refit/auto_resume.py",
            repo / "operational/final_refit/oof_lineage.py",
            repo / "operational/final_refit/fold_sources.py",
            repo / "operational/final_refit/fold_producers.py",
            repo / "operational/final_refit/fold_training.py",
            runtime / "PRODUCER_PLAN.json",
            folds,
        )
    }
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": str(repo.resolve()),
        "runtime": str(runtime.resolve()),
        "locator": str(locator.resolve()),
        "fold_contract": str(folds.resolve()),
        "source_bindings": bindings,
        "nodes": nodes,
        "implementation_blockers": BLOCKERS,
        "end_to_end_executable": False,
        "scientific_updates_enabled_by_this_executor": 0,
        "authorization": "User requested automatic continuation through OOF and test12 predictions",
        "submission_authorized": False,
        "recipe_policy": "seed7; exact grouped ancestry; no in-sample fallback; no test tuning",
        "rotation0_policy": "Reuse requires endpoint and generation ancestry admission",
        "resource_policy": {
            "physical_min_bytes": 2 * 1024**3,
            "commit_min_bytes": 2 * 1024**3,
            "checkpoint_updates": 100,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject changed plans or implementation bytes, including edits while waiting."""
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("schema") != SCHEMA or canonical_sha256(payload) != plan.get("plan_sha256"):
        raise ValueError("Queue plan digest/schema mismatch")
    if plan.get("submission_authorized") is not False:
        raise ValueError("This queue cannot submit to CodaBench")
    topology(plan["nodes"])
    for path, expected in plan["source_bindings"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Bound queue implementation/input changed: {path}")


def rgb_endpoints(plan: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Wait for durable endpoints and supervisor exit, then verify both checkpoints."""
    runtime = Path(plan["runtime"])
    locator_path = Path(plan["locator"])
    locator = read(locator_path)
    active = (locator_path.parent / locator["active_run_relative_to_this_file"]).resolve()
    if active != runtime.resolve():
        raise ValueError("Active runtime changed; explicit queue rebind required")
    launcher = read(locator_path.parent / locator["launcher_receipt_relative_to_this_file"])
    details = {"fit_states": launcher["fit_states"], "updates": {}}
    complete = True
    for fit in ("R_A5", "R_C2F"):
        directory = runtime / "fits" / fit
        journal = read(directory / "UPDATE_JOURNAL.json")
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        details["updates"][fit] = journal["completed_updates"]
        if launcher["fit_states"].get(fit) != "COMPLETE":
            complete = False
            continue
        if not (
            journal["completed_updates"]
            == journal["durable_updates"]
            == receipt["completed_updates"]
            == 49_932
            and journal["pending_update_upper"] == 0
            and receipt["complete_state"] is True
            and receipt["scientific_endpoint"] is True
            and receipt["status"] == "COMPLETE"
            and isinstance(journal.get("identity_sha256"), str)
            and len(journal["identity_sha256"]) == 64
            and isinstance(journal.get("execution_binding_sha256"), str)
            and len(journal["execution_binding_sha256"]) == 64
            and journal.get("identity_sha256") == receipt.get("identity_sha256")
            and journal.get("execution_binding_sha256") == receipt.get("execution_binding_sha256")
        ):
            raise ValueError(f"Incomplete or inconsistent scientific endpoint: {fit}")
    if not complete or launcher["status"] != "SUPERVISOR_EXITED":
        return False, details
    # Do not trust a stale COMPLETE status while a producer still owns the runtime.
    for process in psutil.process_iter(["pid", "cmdline"]):
        if "operational.final_refit.producers" in (process.info["cmdline"] or []):
            return False, {**details, "waiting_for_producer_pid": process.pid}
    sources = read(runtime / "PRODUCER_PLAN.json")["executed_source_sha256"]
    for name, expected in sources.items():
        if file_digest(Path(plan["repository"]) / name) != expected:
            raise ValueError(f"Scientific runtime source changed: {name}")
    for fit in ("R_A5", "R_C2F"):
        directory = runtime / "fits" / fit
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        checkpoint = Path(receipt["checkpoint_path"]).resolve(strict=True)
        if not checkpoint.is_relative_to(directory.resolve()):
            raise ValueError(f"Checkpoint outside fit: {fit}")
        if file_digest(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {fit}")
    return True, details


def verify_roles(contract: Path) -> dict[str, Any]:
    """Validate all group splits and exact once-only generation coverage, without targets."""
    import pandas as pd

    from .oof_lineage import load_rotation_identity

    rotations = [load_rotation_identity(contract, index) for index in range(5)]
    frames = [
        pd.read_parquet(
            item.roles["generation"].rows_path,
            columns=["sample_token", "group_id", "source_ordinal"],
        )
        for item in rotations
    ]
    generation = pd.concat(frames, ignore_index=True)
    if (
        len(generation) != 88_744
        or generation.group_id.nunique() != 40
        or generation.sample_token.duplicated().any()
        or generation.source_ordinal.duplicated().any()
        or set(generation.source_ordinal.astype(int)) != set(range(88_744))
    ):
        raise ValueError("OOF generation does not cover TRAIN40 exactly once")
    universe = pd.concat(
        [
            pd.read_parquet(role.rows_path, columns=["sample_token", "group_id", "source_ordinal"])
            for role in rotations[0].roles.values()
        ],
        ignore_index=True,
    )
    columns = ["sample_token", "group_id", "source_ordinal"]
    left = pd.DataFrame(generation[columns]).sort_values("source_ordinal").reset_index(drop=True)
    right = pd.DataFrame(universe[columns]).sort_values("source_ordinal").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("OOF token/group/ordinal mapping differs from TRAIN40")
    return {
        "rotations": [asdict(r) for r in rotations],
        "queries": len(generation),
        "groups": 40,
        "targets_read": False,
        "optimizer_updates": 0,
        "endpoint_ancestry_verified": False,
    }


@contextmanager
def queue_lock(path: Path) -> Iterator[None]:
    """OS-owned lock, released after a crash; never reclaim by a stale PID alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def tick(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    """Revalidate idempotent steps on restart; never trust stale completion labels."""
    validate_plan(plan)
    status: dict[str, Any] = {
        "schema": SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "observed_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "process_create_time": psutil.Process().create_time(),
        "end_to_end_executable": False,
        "implementation_blockers": BLOCKERS,
        "stages": {node["id"]: "WAITING_DEPENDENCY" for node in plan["nodes"]},
        "scientific_updates_executed": 0,
        "submission_performed": False,
    }
    runtime = Path(plan["runtime"])
    stops = [output / "STOP_REQUEST", runtime / "AUTO_RESUME_STOP"]
    stops += [runtime / "fits" / f / "STOP_REQUEST" for f in ("R_A5", "R_C2F")]
    if any(path.exists() for path in stops):
        status["status"] = "MANUAL_STOP"
        return status
    ready, evidence = rgb_endpoints(plan)
    status["upstream"] = evidence
    status["stages"]["RGB_ENDPOINTS"] = "COMPLETE" if ready else "WAITING_UPSTREAM"
    if not ready:
        status["status"] = "WAITING_UPSTREAM"
        return status
    role_evidence = verify_roles(Path(plan["fold_contract"]))
    atomic_json(output / "OOF_ROLE_VERIFICATION.json", role_evidence)
    status["stages"]["VERIFY_OOF_ROLES"] = "COMPLETE"
    status["stages"]["OOF_SCIENTIFIC_IMPLEMENTATION"] = "BLOCKED_IMPLEMENTATION"
    status["status"] = "BLOCKED_IMPLEMENTATION"
    return status


def run_queue(plan_path: Path, *, once: bool = False) -> int:
    """Observe upstream automatically; stop explicitly at unavailable scientific work."""
    output = plan_path.parent
    with queue_lock(output / "QUEUE.lock"):
        plan = read(plan_path)
        state_path = output / "QUEUE_STATUS.json"
        if state_path.exists() and read(state_path).get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Queue state belongs to another plan; use a separate directory")
        while True:
            try:
                state = tick(plan, output)
            except (OSError, ValueError, KeyError, psutil.Error) as error:
                state = {
                    "schema": SCHEMA,
                    "plan_sha256": plan.get("plan_sha256"),
                    "observed_utc": datetime.now(UTC).isoformat(),
                    "status": "BLOCKED_INTEGRITY",
                    "error": f"{type(error).__name__}: {error}",
                    "scientific_updates_executed": 0,
                }
            atomic_json(state_path, state)
            if once or state["status"] != "WAITING_UPSTREAM":
                return 0 if state["status"] == "WAITING_UPSTREAM" else 3
            time.sleep(30)


def main() -> int:
    """Prepare an inspectable dependency graph or run its currently admitted prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("runtime", "locator", "folds", "output"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        return run_queue(args.plan.resolve(strict=True), once=args.once)
    repo = Path(__file__).resolve().parents[2]
    plan = build_plan(
        repo,
        args.runtime.resolve(strict=True),
        args.locator.resolve(strict=True),
        args.folds.resolve(strict=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and read(args.output) != plan:
        raise ValueError("Refusing to overwrite a different queue plan")
    atomic_json(args.output, plan)
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "nodes": len(plan["nodes"]),
                "end_to_end_executable": False,
                "blockers": BLOCKERS,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
