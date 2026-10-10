"""Durable dependency supervisor for the authorized RGB-PORT campaign."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .accounting import (
    OwnerLease,
    UpdateLedger,
    atomic_write_json,
    identity_is_live,
    process_create_time,
    read_json_shared,
    sha256_file,
)

if TYPE_CHECKING:
    from e_jepa_ttc.rgb_port.data import RGBProducerSource

FIT_IDS = (
    "E_A5_MATCHED",
    "E_C2F_MATCHED",
    "R_A5",
    "R_C2F",
    "PAIR_E_MATCHED",
    "PAIR_R",
    "E_H1_MATCHED",
    "E_CTX_MATCHED",
    "R_H1",
    "R_CTX",
    "F_TRUE",
    "F_ZERO",
)
HEAD_IDS = FIT_IDS[-6:]
TERMINAL = {"COMPLETE", "FAILED", "SKIPPED_DEPENDENCY", "BLOCKED_EXTERNAL"}
PHASES = ("P0", "P1", "P2", "P3", "P4")
TRANSIENT_RETURN_CODE = 3


def make_bound_rgb_producer_source(manifest_path: str | Path) -> RGBProducerSource:
    """Build the production RGB source and bind its complete prepared-cache identity."""
    from e_jepa_ttc.rgb_port.data import make_rgb_producer_source

    manifest = Path(manifest_path).resolve(strict=True)
    binding_path = manifest.with_name("P_RGB_CACHE_BINDING.json")
    binding = read_json_shared(binding_path)
    if binding.get("schema") != "rgb_port_cache_binding_v1" or binding.get("status") != "COMPLETE":
        raise ValueError("RGB producer training requires the complete prepared-cache binding")
    cache_manifest = Path(str(binding["cache_manifest_path"])).resolve(strict=True)
    cache_manifest_sha256 = sha256_file(cache_manifest)
    if cache_manifest_sha256 != binding.get("cache_manifest_sha256"):
        raise ValueError("RGB prepared-cache manifest changed before source construction")
    source = make_rgb_producer_source(manifest)
    source.identity = {
        **dict(source.identity),
        "prepared_cache_binding_path": str(binding_path),
        "prepared_cache_binding_sha256": sha256_file(binding_path),
        "prepared_cache_manifest_path": str(cache_manifest),
        "prepared_cache_manifest_sha256": cache_manifest_sha256,
        "prepared_cache_equivalence": binding.get("equivalence"),
    }
    return source


def _load_config(path: Path) -> dict[str, Any]:
    config = read_json_shared(path.resolve())
    if config.get("schema") != "rgb_port_execution_v1":
        raise ValueError("Unknown RGB-PORT execution schema")
    repository = path.resolve().parent.parent.parent
    config["run_root"] = str(_resolve(repository, str(config["run_root"])))
    if (Path(config["run_root"]) / "AUDIT_CODE_MIGRATION.json").is_file():
        members = config["package"]["members"]
        additions = [
            "AUDIT_CODE_MIGRATION.json",
            "repo-tree:operational/rgb_port_revision",
            "repo:tests/test_rgb_port_revision.py",
            "repo:docs/decisions/ADR-0002-rgb-port-audit-corrections.md",
        ]
        audit = Path(config["run_root"]) / "audit_fixes_20261009"
        additions.extend(
            str(item.relative_to(Path(config["run_root"]))).replace("\\", "/")
            for item in sorted(audit.rglob("*"))
            if item.is_file()
        )
        additions.extend(
            str(item.relative_to(Path(config["run_root"]))).replace("\\", "/")
            for item in Path(config["run_root"]).glob("fits/*/AUDIT_REVISION_RUNTIME.json")
        )
        members.extend(item for item in additions if item not in members)
    return config


def _resolve(base: Path, value: str) -> Path:
    path = Path(os.path.expandvars(value))
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_command(command: list[Any], repository: Path) -> list[str]:
    resolved: list[str] = []
    for value in command:
        argument = str(value)
        if argument.startswith("sha256://"):
            path = _resolve(repository, argument.removeprefix("sha256://"))
            if not path.is_file():
                raise FileNotFoundError(f"Command hash binding is missing: {path}")
            argument = sha256_file(path)
        resolved.append(argument)
    return resolved


def _task_limits(config: Mapping[str, Any]) -> dict[str, int]:
    limits: dict[str, int] = {"__technical__": 500}
    for task in config["tasks"]:
        if task.get("fit_id"):
            limits[str(task["fit_id"])] = int(task["max_updates"])
    return limits


def _reconcile_technical_journal(run_root: Path, ledger: UpdateLedger) -> None:
    path = run_root / "TECHNICAL_JOURNAL.json"
    if not path.is_file():
        return
    journal = read_json_shared(path)
    if journal.get("schema") != "rgb_port_technical_journal_v1":
        raise ValueError("Unknown RGB-PORT technical journal")
    desired = int(journal.get("completed", 0)) + int(journal.get("pending_upper", 0))
    current = ledger.totals(ledger.initialize())["technical"]
    if desired < current or desired > 500:
        raise RuntimeError("Technical journal moved backwards or exceeded cap")
    if desired > current:
        digest = sha256_file(path)
        ledger.charge(
            event_id=f"technical_journal:{digest}",
            fit_id="__technical__",
            category="technical",
            charged_updates=desired - current,
            evidence={"journal_sha256": digest, "charged_upper": desired},
        )


def _validate_config(config: dict[str, Any]) -> None:
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Execution config requires tasks")
    names = [str(task.get("id")) for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("Task IDs must be unique")
    known = set(names)
    fit_ids = {str(task["fit_id"]) for task in tasks if task.get("fit_id")}
    if fit_ids != set(FIT_IDS):
        raise ValueError(
            f"Execution config must contain exactly the twelve authorized fits: {sorted(fit_ids)}"
        )
    if any(set(task.get("depends", [])) - known for task in tasks):
        raise ValueError("Task dependency names must exist")
    if any(set(task.get("soft_depends", [])) - known for task in tasks):
        raise ValueError("Soft task dependency names must exist")
    if any(task.get("phase") not in PHASES[1:] for task in tasks):
        raise ValueError("Each DAG task must belong to P1, P2, P3, or P4")
    limits = _task_limits(config)
    if any(limits[name] > 49_932 for name in FIT_IDS[:4]):
        raise ValueError("Producer per-arm cap exceeded")
    if any(limits[name] > 6_840 for name in FIT_IDS[4:6]):
        raise ValueError("PAIR per-arm cap exceeded")
    if any(limits[name] != 2_500 for name in HEAD_IDS):
        raise ValueError("Every authorized head has exactly 2500 scientific updates")
    if sum(limits[name] for name in FIT_IDS) > 228_408:
        raise ValueError("Scientific campaign cap exceeded")
    if any(
        (not isinstance(task.get("command"), list) or not task["command"])
        and not task.get("external_only")
        for task in tasks
    ):
        raise ValueError("Each task needs an argv command list")
    for task in tasks:
        if (
            not task.get("fit_id")
            and not task.get("external_only")
            and not task.get("completion_artifact")
        ):
            raise ValueError(f"Non-fit task {task['id']} needs a completion artifact")
    for task in tasks:
        admission = task.get("head_admission")
        if admission:
            command = [str(value) for value in task["command"]]
            if "--device" not in command:
                raise ValueError("Head command must bind its admitted device")
            index = command.index("--device")
            if index + 1 == len(command) or command[index + 1] != admission.get("device"):
                raise ValueError("Head command and frozen admission device differ")
    prohibited = {"KD", "REFIT", "SEED13", "SEED23"}
    if any(any(token in name.upper() for token in prohibited) for name in names):
        raise ValueError("Unauthorized replication/KD/refit task")


def _freeze_inputs(config_path: Path, config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    freeze_path = run_root / "SOURCE_FREEZE.json"
    source_root = config_path.parent.parent.parent
    files = [config_path.resolve()]
    files.extend(_resolve(source_root, item) for item in config.get("source_files", []))
    files.extend(_resolve(source_root, item) for item in config.get("contract_files", []))
    for item in config.get("source_roots", []):
        directory = _resolve(source_root, item)
        if not directory.is_dir():
            raise FileNotFoundError(f"Scientific source root is absent: {directory}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in {".py", ".json", ".yaml", ".yml"}
        )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing scientific inputs: {missing}")
    candidate = {
        "schema": "rgb_port_source_freeze_v1",
        "base_commit": config["base_commit"],
        "files": {str(path): sha256_file(path) for path in sorted(set(files))},
    }
    if freeze_path.exists():
        frozen = read_json_shared(freeze_path)
        if frozen != candidate:
            from operational.rgb_port_revision.migration import inventory_matches

            if {key: value for key, value in frozen.items() if key != "files"} != {
                key: value for key, value in candidate.items() if key != "files"
            } or not inventory_matches(frozen["files"], candidate["files"], run_root):
                raise RuntimeError(
                    "Scientific source/config identity changed without audited migration"
                )
            return frozen
    else:
        atomic_write_json(freeze_path, candidate)
    return candidate


def _check_artifacts(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    root = config_path.parent.parent.parent
    results: dict[str, str] = {}
    for entry in config.get("required_artifacts", []):
        path = _resolve(root, entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")
        digest = sha256_file(path)
        expected = entry.get("sha256")
        if expected and digest != expected:
            raise RuntimeError(f"Required artifact identity changed: {path}")
        results[str(path)] = digest
    return results


def _validate_technical_qa(run_root: Path, *, require_global_admission: bool) -> dict[str, int]:
    journal_path = run_root / "TECHNICAL_JOURNAL.json"
    receipt_path = run_root / "ROOT_QA_RECEIPT.json"
    journal = read_json_shared(journal_path)
    receipt = read_json_shared(receipt_path)
    if journal.get("schema") != "rgb_port_technical_journal_v1":
        raise ValueError("Unknown RGB-PORT technical QA journal")
    completed = int(journal.get("completed", -1))
    pending = int(journal.get("pending_upper", -1))
    if completed < 0 or pending not in {0, 1} or completed + pending > 500:
        raise RuntimeError("Technical QA is incomplete or exceeds its 500-update cap")
    if require_global_admission and (
        receipt.get("schema") != "rgb_port_root_qa_v1"
        or int(receipt.get("exit_code", -1)) != 0
        or int(receipt.get("pending_upper", -1)) != 0
        or int(receipt.get("technical_updates_after", -1)) != completed
    ):
        raise RuntimeError("Root QA receipt does not admit scientific execution")
    return {"completed": completed, "pending_upper": pending}


def _snapshot_global_qa(run_root: Path) -> None:
    """Seal the pre-scientific global QA evidence before recovery QA mutates it."""
    qa_root = run_root / "qa"
    snapshots = {
        qa_root / "GLOBAL_QA_ADMISSION.json": run_root / "ROOT_QA_RECEIPT.json",
        qa_root / "GLOBAL_TECHNICAL_JOURNAL.json": run_root / "TECHNICAL_JOURNAL.json",
    }
    for target, source in snapshots.items():
        document = read_json_shared(source)
        if target.exists():
            if read_json_shared(target) != document:
                raise RuntimeError(f"Global QA snapshot changed: {target}")
            continue
        atomic_write_json(target, document)


def _sync_completion_copy(task: Mapping[str, Any], repository: Path) -> None:
    """Copy a mutable producer receipt into the task's stable typed receipt."""
    source_value = task.get("completion_copy_from")
    target_value = task.get("completion_artifact")
    if not source_value or not target_value:
        return
    source = _resolve(repository, str(source_value))
    if not source.is_file():
        return
    target = _resolve(repository, str(target_value))
    document = read_json_shared(source)
    if not target.exists() or read_json_shared(target) != document:
        atomic_write_json(target, document)


def _resource_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    import psutil

    aggregate = 0
    processes: list[dict[str, Any]] = []
    matched_roots: list[psutil.Process] = []
    seen: set[tuple[int, float]] = set()
    markers = [str(item).lower() for item in config["resources"].get("project_command_markers", [])]
    for process in psutil.process_iter(["pid", "create_time", "memory_info", "cmdline", "name"]):
        try:
            arguments = [str(value).lower() for value in (process.info.get("cmdline") or [])]
            # Match executable command lines only. Test names and documentation are not owners.
            if markers and _module_matches(arguments, markers):
                rss = int(process.info["memory_info"].rss)
                identity = (process.pid, float(process.info["create_time"]))
                seen.add(identity)
                aggregate += rss
                matched_roots.append(process)
                processes.append(
                    {
                        "pid": process.pid,
                        "create_time": process.info["create_time"],
                        "rss": rss,
                        "name": process.info["name"],
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    for root in matched_roots:
        try:
            descendants = root.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
        for process in descendants:
            try:
                created = float(process.create_time())
                identity = (process.pid, created)
                if identity in seen:
                    continue
                rss = int(process.memory_info().rss)
                seen.add(identity)
                aggregate += rss
                processes.append(
                    {
                        "pid": process.pid,
                        "create_time": created,
                        "rss": rss,
                        "name": process.name(),
                        "descendant_of": root.pid,
                    }
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
    memory = psutil.virtual_memory()
    commit_available = int(memory.available)
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("load", ctypes.c_uint32),
                ("total_phys", ctypes.c_uint64),
                ("avail_phys", ctypes.c_uint64),
                ("total_page", ctypes.c_uint64),
                ("avail_page", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("avail_virtual", ctypes.c_uint64),
                ("avail_extended", ctypes.c_uint64),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        commit_available = int(status.avail_page)
    disk_roots = [Path(config["run_root"])] + [
        Path(value) for value in config["resources"].get("disk_roots", [])
    ]
    disk_free = {str(path): int(shutil.disk_usage(path.anchor or path).free) for path in disk_roots}
    return {
        "host_available_bytes": int(memory.available),
        "commit_available_bytes": commit_available,
        "disk_free_bytes": min(disk_free.values()),
        "disk_free_by_root": disk_free,
        "project_tree_rss_bytes": aggregate,
        "project_processes": processes,
    }


def _resources_admit(
    config: Mapping[str, Any],
    *,
    recovery: bool,
    reservation_bytes: int = 0,
    rss_reservation_bytes: int = 0,
    host_reservation_bytes: int = 0,
) -> tuple[bool, dict[str, Any]]:
    snapshot = _resource_snapshot(config)
    policy = config["resources"]
    required_free = int((3 if recovery else 2) * 1024**3)
    reasons = []
    snapshot["host_reservation_bytes"] = int(host_reservation_bytes)
    if snapshot["host_available_bytes"] - host_reservation_bytes < required_free:
        reasons.append("host_memory")
    if snapshot["commit_available_bytes"] - host_reservation_bytes < required_free:
        reasons.append("windows_commit")
    snapshot["reservation_bytes"] = int(reservation_bytes)
    if snapshot["disk_free_bytes"] - reservation_bytes < int(policy["disk_margin_decimal_bytes"]):
        reasons.append("disk_margin")
    snapshot["rss_reservation_bytes"] = int(rss_reservation_bytes)
    if snapshot["project_tree_rss_bytes"] + rss_reservation_bytes > int(
        policy["project_tree_rss_max_decimal_bytes"]
    ):
        reasons.append("aggregate_project_rss")
    snapshot["admission_reasons"] = reasons
    return not reasons, snapshot


def _external_heavy_owner(config: Mapping[str, Any]) -> dict[str, Any] | None:
    import psutil

    markers = [str(value).lower() for value in config["resources"].get("heavy_command_markers", [])]
    for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
        try:
            arguments = [str(value).lower() for value in (process.info.get("cmdline") or [])]
            if _module_matches(arguments, markers):
                return {
                    "source": "live_process_scan",
                    "pid": process.pid,
                    "create_time": process.info["create_time"],
                    "matched_marker": next(
                        marker
                        for marker in markers
                        if any(
                            argument == marker
                            for argument in arguments[arguments.index("-m") + 1 :][:1]
                        )
                    ),
                }
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    for entry in config.get("external_heavy_owners", []):
        path = Path(entry["path"])
        if path.is_file():
            document = read_json_shared(path)
            identities = document.get("owners", [document])
            for identity in identities:
                arguments = [str(value).lower() for value in identity.get("cmdline", [])]
                selectors = [str(value).lower() for value in entry.get("command_markers", [])]
                selected = not selectors or _module_matches(arguments, selectors)
                if selected and identity_is_live(identity):
                    return {"path": str(path), "identity": identity}
    return None


def _module_matches(arguments: list[str], modules: list[str]) -> bool:
    return any(
        argument == "-m" and index + 1 < len(arguments) and arguments[index + 1] in modules
        for index, argument in enumerate(arguments)
    )


def _live_module_owner(
    modules: list[str], required_arguments: list[str] | None = None
) -> dict[str, Any] | None:
    """Return one exact ``python -m`` owner for an externally started task."""
    import psutil

    expected = [value.lower() for value in modules]
    required = [value.lower() for value in (required_arguments or [])]
    for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
        try:
            arguments = [str(value).lower() for value in (process.info.get("cmdline") or [])]
            if _module_matches(arguments, expected) and all(
                value in arguments for value in required
            ):
                return {
                    "pid": process.pid,
                    "create_time": process.info["create_time"],
                    "module": arguments[arguments.index("-m") + 1],
                }
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, ValueError):
            continue
    return None


def _find_matching_child(command: list[str]) -> dict[str, Any] | None:
    import psutil

    if not command:
        return None
    expected = [value.lower() for value in command]
    for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
        try:
            actual = [str(value).lower() for value in (process.info.get("cmdline") or [])]
            if actual == expected:
                return {"pid": process.pid, "create_time": process.info["create_time"]}
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return None


def _initial_state(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "rgb_port_state_v1",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "tasks": {
            task["id"]: {"status": "WAITING", "attempts": 0, "fit_id": task.get("fit_id")}
            for task in config["tasks"]
        },
        "phases": {
            "P0": {"status": "COMPLETE"},
            **{name: {"status": "WAITING"} for name in PHASES[1:]},
        },
        "campaign_status": "INCOMPLETE",
    }


def _refresh_phase_state(config: Mapping[str, Any], state: dict[str, Any]) -> None:
    phases: dict[str, dict[str, Any]] = {"P0": {"status": "COMPLETE"}}
    for phase in PHASES[1:]:
        task_ids = [str(task["id"]) for task in config["tasks"] if task.get("phase") == phase]
        statuses = [str(state["tasks"][task_id]["status"]) for task_id in task_ids]
        if statuses and all(status == "COMPLETE" for status in statuses):
            status = "COMPLETE"
        elif statuses and all(item in TERMINAL for item in statuses):
            status = "TERMINAL_WITH_BLOCKERS"
        elif any(item == "RUNNING" for item in statuses):
            status = "RUNNING"
        elif any(item == "PAUSED_RESOURCE" for item in statuses):
            status = "PAUSED_RESOURCE"
        else:
            status = "WAITING"
        phases[phase] = {"status": status, "tasks": task_ids}
    state["phases"] = phases


def _write_next_decision(
    config_path: Path, run_root: Path, state: Mapping[str, Any], ledger: UpdateLedger
) -> None:
    tasks = state["tasks"]
    blockers = {
        task_id: {"status": item["status"], "reason": item.get("reason")}
        for task_id, item in tasks.items()
        if item["status"] != "COMPLETE"
    }
    payload = {
        "schema": "rgb_port_next_decision_v1",
        "status": state["campaign_status"],
        "decision": (
            "RESULTS_READY_NO_AUTOMATIC_REPLICA"
            if not blockers
            else "DEPENDENCIES_OR_FAILURES_PRESERVED"
        ),
        "blockers": blockers,
        "phases": state.get("phases", {}),
        "accounting": ledger.totals(ledger.initialize()),
        "resume_command": [
            sys.executable,
            "-m",
            "operational.rgb_port.run",
            "resume",
            "--run",
            str(run_root),
        ],
        "config_path": str(config_path),
        "automatic_replica_or_refit_authorized": False,
    }
    atomic_write_json(run_root / "NEXT_DECISION.json", payload)


def _read_completed_updates(task: Mapping[str, Any], run_root: Path) -> tuple[int, dict[str, Any]]:
    if not task.get("fit_id"):
        return 0, {}
    receipt = run_root / "fits" / str(task["fit_id"]) / "CHECKPOINT_RECEIPT.json"
    journal = run_root / "fits" / str(task["fit_id"]) / "UPDATE_JOURNAL.json"
    evidence: dict[str, Any] = {}
    completed = 0
    if journal.is_file():
        data = read_json_shared(journal)
        completed = int(
            data.get(
                "durable_updates",
                data.get("completed_updates", data.get("committed_updates", 0)),
            )
        )
        evidence["journal_sha256"] = sha256_file(journal)
        evidence["journal_completed_updates"] = int(
            data.get("completed_updates", data.get("committed_updates", completed))
        )
        evidence["recovery_upper"] = int(data.get("recovery_upper", 0))
        evidence["pending_update_upper"] = int(data.get("pending_update_upper", 0))
    if receipt.is_file():
        data = read_json_shared(receipt)
        completed = max(completed, int(data.get("completed_updates", 0)))
        evidence["receipt_sha256"] = sha256_file(receipt)
        evidence["receipt_status"] = data.get("status")
    return completed, evidence


def _reconcile(
    task: Mapping[str, Any],
    run_root: Path,
    ledger: UpdateLedger,
    before: int,
    attempt: int,
    accounted_recovery: int,
) -> int:
    completed, evidence = _read_completed_updates(task, run_root)
    if completed < before or completed > int(task.get("max_updates", 0)):
        raise RuntimeError(f"Invalid update journal for {task['id']}: {before} -> {completed}")
    delta = completed - before
    if delta:
        ledger.charge(
            event_id=f"{task['id']}:attempt:{attempt}:scientific",
            fit_id=str(task["fit_id"]),
            category="scientific",
            charged_updates=delta,
            evidence={**evidence, "before": before, "after": completed},
        )
    recovery_upper = _recovery_bound(evidence, completed)
    if recovery_upper < accounted_recovery:
        raise RuntimeError("Recovery upper bound moved backwards")
    if recovery_upper > accounted_recovery:
        ledger.charge(
            event_id=f"{task['id']}:attempt:{attempt}:recovery",
            fit_id=str(task["fit_id"]),
            category="recovery",
            charged_updates=recovery_upper - accounted_recovery,
            evidence={**evidence, "accounted_before": accounted_recovery},
        )
    return completed


def _recovery_bound(evidence: Mapping[str, Any], durable: int) -> int:
    return (
        int(evidence.get("recovery_upper", 0))
        + int(evidence.get("pending_update_upper", 0))
        + max(0, int(evidence.get("journal_completed_updates", durable)) - durable)
    )


def _fit_complete(task: Mapping[str, Any], run_root: Path) -> bool:
    completed, _ = _read_completed_updates(task, run_root)
    if completed != int(task.get("max_updates", 0)):
        return False
    if not task.get("fit_id"):
        return True
    receipt = run_root / "fits" / str(task["fit_id"]) / "CHECKPOINT_RECEIPT.json"
    if not receipt.is_file():
        return False
    value = read_json_shared(receipt)
    return (
        value.get("status") in {"COMPLETE", "COMPLETED"}
        and value.get("complete_state") is not False
    )


def _artifact_task_complete(task: Mapping[str, Any], repository: Path) -> bool:
    value = task.get("completion_artifact")
    if not value:
        return False
    path = _resolve(repository, str(value))
    if not path.is_file():
        return False
    if path.suffix.lower() == ".json":
        document = read_json_shared(path)
        expected_schema = task.get("completion_schema")
        if expected_schema and document.get("schema") != expected_schema:
            return False
        status = document.get("status")
        if status is None and task.get("completion_status_optional"):
            pass
        elif status not in {"COMPLETE", "COMPLETED", "VERIFIED", "PASSED"}:
            return False
        qa_contract = task.get("technical_qa_contract")
        if qa_contract:
            journal_path = _resolve(repository, str(qa_contract["journal"]))
            journal = read_json_shared(journal_path)
            before = int(document.get("technical_updates_before", -1))
            after = int(document.get("technical_updates_after", -1))
            expected_test = str(Path(str(qa_contract["test"])))
            if (
                int(document.get("exit_code", -1)) != 0
                or int(document.get("pending_upper", -1)) != 0
                or after - before != int(qa_contract["expected_updates"])
                or int(journal.get("completed", -1)) != after
                or int(journal.get("pending_upper", -1)) != 0
                or set(document.get("tests", {})) != {expected_test}
            ):
                return False
    expected = task.get("completion_sha256")
    return not expected or sha256_file(path) == expected


def _terminal_artifact_blocker(task: Mapping[str, Any], repository: Path) -> str | None:
    expected_status = task.get("terminal_artifact_status")
    value = task.get("completion_artifact")
    if not expected_status or not value:
        return None
    path = _resolve(repository, str(value))
    if not path.is_file() or path.suffix.lower() != ".json":
        return None
    document = read_json_shared(path)
    if document.get("status") != expected_status:
        return None
    expected_schema = task.get("completion_schema")
    if expected_schema and document.get("schema") != expected_schema:
        raise ValueError(f"Terminal blocker artifact has another schema: {path}")
    return str(document.get("reason", expected_status))


def _failure_artifact_reason(task: Mapping[str, Any], repository: Path) -> str | None:
    value = task.get("failure_artifact")
    if not value:
        return None
    path = _resolve(repository, str(value))
    if not path.is_file():
        return None
    document = read_json_shared(path)
    exit_code = document.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return f"admission QA receipt exit code {exit_code}"
    return None


def _freeze_fit_endpoint(fit_id: str, run_root: Path) -> None:
    fit_root = run_root / "fits" / fit_id
    files = [
        fit_root / "checkpoint_last.pt",
        fit_root / "CHECKPOINT_RECEIPT.json",
        fit_root / "UPDATE_JOURNAL.json",
        fit_root / "RUN_PROVENANCE.json",
    ]
    if any(not path.is_file() for path in files):
        raise RuntimeError(f"Complete fit {fit_id} lacks its full-state/provenance files")
    candidate = {
        "schema": "rgb_port_fit_endpoint_freeze_v1",
        "fit_id": fit_id,
        "files": {path.name: sha256_file(path) for path in files},
    }
    path = fit_root / "ENDPOINT_FREEZE.json"
    if path.exists() and read_json_shared(path) != candidate:
        raise RuntimeError(f"Frozen endpoint changed: {fit_id}")
    if not path.exists():
        atomic_write_json(path, candidate)


def _materialize_head_admission(task: Mapping[str, Any], config_path: Path, run_root: Path) -> None:
    specification = task.get("head_admission")
    if not specification:
        return
    repository = config_path.parent.parent.parent

    def resolve(value: object) -> Path:
        return _resolve(repository, str(value))

    cache = resolve(specification["cache"])
    output = run_root / "fits" / str(task["fit_id"])
    sources = [resolve(value) for value in specification["executed_sources"]]
    paths = {
        "role_manifest_sha256": resolve(specification["role_manifest"]),
        "split_sha256": resolve(specification["split"]),
        "parent_sha256": resolve(specification["parent"]),
    }
    normalizer_path = (
        resolve(specification["normalizer"]) if specification.get("normalizer") else None
    )
    required = [cache, *sources, *paths.values()]
    if normalizer_path is not None:
        required.append(normalizer_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Head admission inputs are not fully materialized")
    payload = {
        "source_sha256": sha256_file(cache),
        "git_commit": str(read_json_shared(run_root / "SOURCE_FREEZE.json")["base_commit"]),
        "config_sha256": sha256_file(config_path),
        **{name: sha256_file(path) for name, path in paths.items()},
        "normalizer_sha256": (
            sha256_file(normalizer_path)
            if normalizer_path is not None
            else str(specification["normalizer_value"])
        ),
        "precision": "float32",
        "device": str(specification["device"]),
        "output": str(output.resolve()),
        "executed_source_sha256": {str(path.resolve()): sha256_file(path) for path in sources},
    }
    admission = resolve(specification["output"])
    if admission.exists() and read_json_shared(admission) != payload:
        raise RuntimeError(f"Head admission changed for {task['fit_id']}")
    if not admission.exists():
        atomic_write_json(admission, payload)


def _round_endpoint_entry(fit_id: str, fit_root: Path, expected_updates: int) -> dict[str, Any]:
    receipt_path = fit_root / "CHECKPOINT_RECEIPT.json"
    checkpoint_path = fit_root / "checkpoint_last.pt"
    endpoint_path = fit_root / "ENDPOINT_FREEZE.json"
    receipt = read_json_shared(receipt_path)
    checkpoint = Path(str(receipt.get("checkpoint_path", ""))).resolve(strict=True)
    canonical_checkpoint = checkpoint_path.resolve(strict=True)
    checkpoint_sha256 = sha256_file(canonical_checkpoint)
    if (
        receipt.get("fit_id") != fit_id
        or receipt.get("status") != "COMPLETE"
        or int(receipt.get("completed_updates", -1)) != expected_updates
        or not receipt.get("scientific_endpoint")
        or not receipt.get("complete_state")
        or int(receipt.get("accumulation_index", -1)) != 0
        or checkpoint != canonical_checkpoint
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or not isinstance(receipt.get("identity_sha256"), str)
    ):
        raise RuntimeError(f"Fit {fit_id} receipt is not a complete immutable endpoint")
    endpoint = read_json_shared(endpoint_path)
    files = endpoint.get("files")
    if (
        endpoint.get("schema") != "rgb_port_fit_endpoint_freeze_v1"
        or endpoint.get("fit_id") != fit_id
        or not isinstance(files, dict)
        or files.get("checkpoint_last.pt") != checkpoint_sha256
        or files.get("CHECKPOINT_RECEIPT.json") != sha256_file(receipt_path)
    ):
        raise RuntimeError(f"Fit {fit_id} endpoint freeze does not bind its full-state receipt")
    return {
        "fit_id": fit_id,
        "status": "COMPLETE",
        "scientific_endpoint": True,
        "completed_updates": expected_updates,
        "complete_state": True,
        "accumulation_index": 0,
        "checkpoint_path": str(canonical_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "identity_sha256": str(receipt["identity_sha256"]),
        "receipt_sha256": sha256_file(receipt_path),
        "endpoint_freeze_sha256": sha256_file(endpoint_path),
    }


def _write_round_freeze(config: Mapping[str, Any], run_root: Path) -> None:
    path = run_root / "rounds" / "HEADS_V_ROUND_FREEZE.json"
    endpoints: dict[str, Any] = {}
    limits = _task_limits(config)
    for fit_id in HEAD_IDS:
        fit_root = run_root / "fits" / fit_id
        required = [
            fit_root / "CHECKPOINT_RECEIPT.json",
            fit_root / "checkpoint_last.pt",
            fit_root / "ENDPOINT_FREEZE.json",
        ]
        if any(not item.is_file() for item in required):
            return
        endpoints[fit_id] = _round_endpoint_entry(fit_id, fit_root, limits[fit_id])
    candidate = {
        "schema": "rgb_port_head_round_freeze_v1",
        "endpoints": endpoints,
        "selection": "fixed_seed7_final_endpoints",
    }
    if path.exists() and read_json_shared(path) != candidate:
        raise RuntimeError("Head endpoint changed after the immutable V round freeze")
    if not path.exists():
        atomic_write_json(path, candidate)


def _write_parent_round_freeze(config: Mapping[str, Any], run_root: Path) -> None:
    endpoints: dict[str, Any] = {}
    limits = _task_limits(config)
    for fit_id in FIT_IDS[:6]:
        fit_root = run_root / "fits" / fit_id
        receipt_path = fit_root / "CHECKPOINT_RECEIPT.json"
        endpoint_path = fit_root / "ENDPOINT_FREEZE.json"
        checkpoint_path = fit_root / "checkpoint_last.pt"
        if (
            not receipt_path.is_file()
            or not endpoint_path.is_file()
            or not checkpoint_path.is_file()
        ):
            return
        endpoints[fit_id] = _round_endpoint_entry(fit_id, fit_root, limits[fit_id])
    candidate = {"schema": "rgb_port_parent_round_freeze_v1", "endpoints": endpoints}
    path = run_root / "rounds" / "PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    if path.exists() and read_json_shared(path) != candidate:
        raise RuntimeError("Producer/PAIR endpoints changed after round freeze")
    if not path.exists():
        atomic_write_json(path, candidate)


def _one_cycle(
    config_path: Path,
    config: dict[str, Any],
    run_root: Path,
    state: dict[str, Any],
    ledger: UpdateLedger,
) -> bool:
    progress = False
    _reconcile_technical_journal(run_root, ledger)
    if (run_root / "PAUSE").exists():
        state["supervisor_status"] = "PAUSED_REQUESTED"
        _refresh_phase_state(config, state)
        atomic_write_json(run_root / "RGB_PORT_STATE.json", state)
        return False
    state["supervisor_status"] = "RUNNING"
    task_map = {task["id"]: task for task in config["tasks"]}
    if all(
        state["tasks"].get(name, {}).get("status") == "COMPLETE"
        for name in config.get("parent_fit_task_ids", [])
    ):
        _write_parent_round_freeze(config, run_root)
    if all(
        state["tasks"].get(name, {}).get("status") == "COMPLETE"
        for name in config.get("head_task_ids", [])
    ):
        _write_round_freeze(config, run_root)
    for task_id, task in task_map.items():
        record = state["tasks"][task_id]
        if record["status"] == "COMPLETE":
            if task.get("fit_id"):
                _freeze_fit_endpoint(str(task["fit_id"]), run_root)
            elif task.get("completion_artifact"):
                artifact = _resolve(
                    config_path.parent.parent.parent, str(task["completion_artifact"])
                )
                if not _artifact_task_complete(task, config_path.parent.parent.parent):
                    raise RuntimeError(f"Completed task artifact changed or disappeared: {task_id}")
                if record.get("completion_sha256") != sha256_file(artifact):
                    raise RuntimeError(f"Completed task artifact identity changed: {task_id}")
            continue
        if record["status"] in {"FAILED", "SKIPPED_DEPENDENCY"}:
            continue
        _sync_completion_copy(task, config_path.parent.parent.parent)
        if (
            record["status"] == "BLOCKED_EXTERNAL"
            and not task.get("external_only")
            and not task.get("terminal_artifact_status")
        ):
            record["status"] = "WAITING"
            record.pop("reason", None)
        if record["status"] == "PAUSED_RESOURCE" and time.time() < float(
            record.get("retry_after_unix", 0)
        ):
            continue
        if record["status"] == "RUNNING":
            child = record.get("child", {})
            if identity_is_live(child):
                continue
            adopted = _find_matching_child(list(record.get("resolved_command", [])))
            if adopted is not None:
                record["child"] = adopted
                atomic_write_json(run_root / "RGB_PORT_STATE.json", state)
                continue
            before = int(record.get("started_from_update", 0))
            after = _reconcile(
                task,
                run_root,
                ledger,
                before,
                int(record["attempts"]),
                int(record.get("accounted_recovery_upper", 0)),
            )
            _, evidence = _read_completed_updates(task, run_root)
            _reconcile_technical_journal(run_root, ledger)
            _sync_completion_copy(task, config_path.parent.parent.parent)
            failure_reason = _failure_artifact_reason(task, config_path.parent.parent.parent)
            if failure_reason is not None:
                record.update(
                    status="FAILED",
                    reason=failure_reason,
                    completed_updates=after,
                    accounted_recovery_upper=_recovery_bound(evidence, after),
                )
            else:
                record.update(
                    status="PAUSED_RESOURCE",
                    reason="orphaned child reconciled for durable resume",
                    completed_updates=after,
                    accounted_recovery_upper=_recovery_bound(evidence, after),
                    retry_after_unix=time.time()
                    + min(900, 30 * 2 ** (int(record["attempts"]) - 1)),
                )
            record.pop("child", None)
            progress = True
            if record["status"] == "FAILED":
                continue
        dependency_states = [state["tasks"][name]["status"] for name in task.get("depends", [])]
        if any(status in {"FAILED", "SKIPPED_DEPENDENCY"} for status in dependency_states):
            record["status"] = "SKIPPED_DEPENDENCY"
            record["reason"] = "required scientific dependency failed"
            progress = True
            continue
        if not all(status == "COMPLETE" for status in dependency_states):
            continue
        soft_dependency_states = [
            state["tasks"][name]["status"] for name in task.get("soft_depends", [])
        ]
        if not all(status in TERMINAL for status in soft_dependency_states):
            continue
        if task.get("fit_id") and _fit_complete(task, run_root):
            completed = _reconcile(
                task,
                run_root,
                ledger,
                int(record.get("completed_updates", 0)),
                int(record.get("attempts", 0)),
                int(record.get("accounted_recovery_upper", 0)),
            )
            _freeze_fit_endpoint(str(task["fit_id"]), run_root)
            record["status"] = "COMPLETE"
            record["completed_updates"] = completed
            progress = True
            continue
        if task.get("fit_id"):
            observed = _reconcile(
                task,
                run_root,
                ledger,
                int(record.get("completed_updates", 0)),
                int(record.get("attempts", 0)),
                int(record.get("accounted_recovery_upper", 0)),
            )
            _, observed_evidence = _read_completed_updates(task, run_root)
            record["completed_updates"] = observed
            record["accounted_recovery_upper"] = _recovery_bound(observed_evidence, observed)
        if not task.get("fit_id") and _artifact_task_complete(
            task, config_path.parent.parent.parent
        ):
            record["status"] = "COMPLETE"
            record["completion_artifact"] = str(task["completion_artifact"])
            record["completion_sha256"] = sha256_file(
                _resolve(config_path.parent.parent.parent, str(task["completion_artifact"]))
            )
            progress = True
            continue
        artifact_blocker = _terminal_artifact_blocker(task, config_path.parent.parent.parent)
        if artifact_blocker is not None:
            artifact = _resolve(config_path.parent.parent.parent, str(task["completion_artifact"]))
            digest = sha256_file(artifact)
            if (
                record.get("status") != "BLOCKED_EXTERNAL"
                or record.get("reason") != artifact_blocker
                or record.get("completion_sha256") != digest
            ):
                progress = True
            record.update(
                status="BLOCKED_EXTERNAL",
                reason=artifact_blocker,
                completion_artifact=str(task["completion_artifact"]),
                completion_sha256=digest,
            )
            continue
        if task.get("external_only"):
            missing = [
                required
                for required in task.get("requires", [])
                if not _resolve(config_path.parent.parent.parent, required).is_file()
            ]
            reason = (
                f"missing external contract(s): {missing}"
                if missing
                else f"awaiting externally produced result: {task.get('completion_artifact')}"
            )
            if record.get("status") != "BLOCKED_EXTERNAL" or record.get("reason") != reason:
                progress = True
            record["status"] = "BLOCKED_EXTERNAL"
            record["reason"] = reason
            continue
        for required in task.get("requires", []):
            if not _resolve(config_path.parent.parent.parent, required).exists():
                record["status"] = "BLOCKED_EXTERNAL"
                record["reason"] = f"missing external contract: {required}"
                progress = True
                break
        if record["status"] == "BLOCKED_EXTERNAL":
            continue
        if (
            task.get("requires_head_round_freeze")
            and not (run_root / "rounds" / "HEADS_V_ROUND_FREEZE.json").is_file()
        ):
            continue
        if (
            task.get("requires_parent_round_freeze")
            and not (run_root / "rounds" / "PRODUCERS_PAIR_ENDPOINT_FREEZE.json").is_file()
        ):
            continue
        existing_owner = _live_module_owner(
            [str(value) for value in task.get("adopt_live_modules", [])],
            [str(value) for value in task.get("adopt_argv_values", [])],
        )
        if existing_owner is not None:
            record.update(
                status="PAUSED_RESOURCE",
                reason="waiting for externally started identical task",
                external_owner=existing_owner,
                retry_after_unix=time.time() + 30,
            )
            continue
        admitted, snapshot = _resources_admit(
            config,
            recovery=record["attempts"] > 0,
            reservation_bytes=int(task.get("disk_reservation_bytes", 0)),
            rss_reservation_bytes=int(task.get("rss_reservation_bytes", 0)),
            host_reservation_bytes=int(task.get("host_reservation_bytes", 0)),
        )
        if not admitted:
            record.update(status="PAUSED_RESOURCE", resource_snapshot=snapshot)
            continue
        if task.get("heavy"):
            owner = _external_heavy_owner(config)
            if owner:
                record.update(
                    status="PAUSED_RESOURCE",
                    reason="external heavy owner is live",
                    external_owner=owner,
                )
                continue
        _materialize_head_admission(task, config_path, run_root)
        if task.get("fit_id"):
            before = int(record.get("completed_updates", 0))
        else:
            before = 0
        if task.get("fit_id"):
            ledger.admit_fit_attempt(
                fit_id=str(task["fit_id"]),
                scientific_upper=int(task["max_updates"]) - before,
                recovery_upper=int(task.get("recovery_attempt_upper", 100)),
            )
        attempt = int(record["attempts"]) + 1
        record.update(status="RUNNING", attempts=attempt, started_from_update=before)
        atomic_write_json(run_root / "RGB_PORT_STATE.json", state)
        env = os.environ.copy()
        repository = config_path.parent.parent.parent
        python_path = os.pathsep.join((str(repository / "src"), str(repository)))
        env.update(
            {
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "RGB_PORT_RUN_ROOT": str(run_root),
                "RGB_PORT_TASK_ID": task_id,
                "PYTHONPATH": python_path,
            }
        )
        log_path = run_root / "logs" / f"{task_id}.attempt_{attempt:03d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_command = _resolve_command(list(task["command"]), config_path.parent.parent.parent)
        record["resolved_command"] = resolved_command
        atomic_write_json(run_root / "RGB_PORT_STATE.json", state)
        with log_path.open("ab", buffering=0) as log:
            child_process = subprocess.Popen(
                resolved_command,
                cwd=config_path.parent.parent.parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            child_created = process_create_time(child_process.pid)
            if child_created is None:
                child_process.terminate()
                raise RuntimeError("Could not bind child PID creation time")
            record["child"] = {"pid": child_process.pid, "create_time": child_created}
            record["log"] = str(log_path.resolve())
            atomic_write_json(run_root / "RGB_PORT_STATE.json", state)
            return_code = child_process.wait()
            os.fsync(log.fileno())
        _reconcile_technical_journal(run_root, ledger)
        _sync_completion_copy(task, config_path.parent.parent.parent)
        after = _reconcile(
            task,
            run_root,
            ledger,
            before,
            attempt,
            int(record.get("accounted_recovery_upper", 0)),
        )
        _, progress_evidence = _read_completed_updates(task, run_root)
        record["accounted_recovery_upper"] = _recovery_bound(progress_evidence, after)
        record.pop("child", None)
        if return_code == 0:
            if task.get("fit_id") and not _fit_complete(task, run_root):
                record.update(
                    status="FAILED",
                    reason="command exited zero without complete bound receipt",
                    completed_updates=after,
                )
            elif (
                not task.get("fit_id")
                and (blocker := _terminal_artifact_blocker(task, config_path.parent.parent.parent))
                is not None
            ):
                artifact = _resolve(
                    config_path.parent.parent.parent, str(task["completion_artifact"])
                )
                record.update(
                    status="BLOCKED_EXTERNAL",
                    reason=blocker,
                    completion_artifact=str(task["completion_artifact"]),
                    completion_sha256=sha256_file(artifact),
                )
            elif not task.get("fit_id") and not _artifact_task_complete(
                task, config_path.parent.parent.parent
            ):
                record.update(
                    status="FAILED",
                    reason="command exited zero without its complete bound artifact",
                    completed_updates=after,
                )
            else:
                record.update(status="COMPLETE", completed_updates=after)
                if task.get("completion_artifact"):
                    artifact = _resolve(
                        config_path.parent.parent.parent, str(task["completion_artifact"])
                    )
                    record["completion_artifact"] = str(task["completion_artifact"])
                    record["completion_sha256"] = sha256_file(artifact)
                if task.get("fit_id"):
                    _freeze_fit_endpoint(str(task["fit_id"]), run_root)
        elif return_code == TRANSIENT_RETURN_CODE:
            record.update(
                status="PAUSED_RESOURCE",
                reason="child reported a recoverable resource pause",
                completed_updates=after,
                retry_after_unix=time.time() + min(900, 30 * 2 ** (attempt - 1)),
            )
        else:
            record.update(
                status="FAILED",
                reason=f"child exit code {return_code}",
                completed_updates=after,
            )
        progress = True
        atomic_write_json(run_root / "RGB_PORT_STATE.json", state)
        break
    if all(item["status"] in TERMINAL for item in state["tasks"].values()):
        state["campaign_status"] = (
            "COMPLETE"
            if all(item["status"] == "COMPLETE" for item in state["tasks"].values())
            else "TERMINAL_WITH_BLOCKERS"
        )
    _refresh_phase_state(config, state)
    atomic_write_json(run_root / "RGB_PORT_STATE.json", state)
    if state["campaign_status"] != "INCOMPLETE":
        _write_next_decision(config_path, run_root, state, ledger)
    return progress


def preflight(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    _validate_config(config)
    repository = config_path.parent.parent.parent
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != config["base_commit"]:
        raise RuntimeError(f"Git base changed: expected {config['base_commit']}, found {head}")
    run_root = Path(config["run_root"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "RGB_PORT_STATE.json"
    initial_admission = not state_path.exists()
    technical_qa = _validate_technical_qa(run_root, require_global_admission=initial_admission)
    if initial_admission:
        _snapshot_global_qa(run_root)
    _freeze_inputs(config_path, config, run_root)
    artifacts = _check_artifacts(config_path, config)
    ledger = UpdateLedger(run_root / "ACCOUNTING.json", _task_limits(config))
    ledger.initialize()
    _reconcile_technical_journal(run_root, ledger)
    state = (
        read_json_shared(state_path) if state_path.exists() else _initial_state(config_path, config)
    )
    if state["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Execution config changed after state creation")
    atomic_write_json(state_path, state)
    result = {
        "schema": "rgb_port_preflight_v1",
        "status": "PASSED",
        "source_freeze_sha256": sha256_file(run_root / "SOURCE_FREEZE.json"),
        "required_artifacts": artifacts,
        "scientific_updates_max": sum(_task_limits(config)[name] for name in FIT_IDS),
        "technical_qa": technical_qa,
        "resource_snapshot": _resource_snapshot(config),
    }
    atomic_write_json(run_root / "PREFLIGHT.json", result)
    return result


def supervise(config_path: Path, *, once: bool = False) -> int:
    preflight(config_path)
    config = _load_config(config_path.resolve())
    run_root = Path(config["run_root"]).resolve()
    lease = OwnerLease(run_root / "OWNER.json")
    lease.acquire()
    try:
        ledger = UpdateLedger(run_root / "ACCOUNTING.json", _task_limits(config))
        while True:
            state = read_json_shared(run_root / "RGB_PORT_STATE.json")
            _freeze_inputs(config_path.resolve(), config, run_root)
            progressed = _one_cycle(config_path.resolve(), config, run_root, state, ledger)
            if state["campaign_status"] != "INCOMPLETE":
                return 0 if state["campaign_status"] == "COMPLETE" else 2
            if once:
                return 0
            time.sleep(2 if progressed else 30)
    finally:
        lease.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight, execute, resume, or package the durable RGB-PORT DAG."
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "execute"):
        item = sub.add_parser(action)
        item.add_argument("--config", type=Path, required=True)
        if action == "execute":
            item.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    resume = sub.add_parser("resume")
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    package = sub.add_parser("package")
    package.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "preflight":
        print(json.dumps(preflight(args.config), sort_keys=True))
        return 0
    if args.action in {"execute", "resume"}:
        config_path = (
            args.config
            if args.action == "execute"
            else Path(read_json_shared(args.run / "RGB_PORT_STATE.json")["config_path"])
        )
        return supervise(config_path, once=args.once)
    state = read_json_shared(args.run / "RGB_PORT_STATE.json")
    config = _load_config(Path(state["config_path"]))
    package_config = config["package"]
    from .package import build_bundle

    build_bundle(
        args.run,
        _resolve(args.run, package_config["output"]),
        list(package_config["members"]),
        repository_root=Path(state["config_path"]).parent.parent.parent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
