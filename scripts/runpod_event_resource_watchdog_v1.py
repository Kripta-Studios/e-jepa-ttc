"""Linux-only fail-closed watchdog for the nine admitted EVENT Pod2 tasks."""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - imported on Windows only by unit tests
    fcntl = None  # type: ignore[assignment]

SCHEMA = "ejepa_event_resource_resume_watchdog_config_v1"
STATUS_SCHEMA = "ejepa_event_resource_resume_watchdog_status_v1"
PRODUCER_ROLE_SCHEMA = "ejepa_event_watchdog_producer_task_v1"
DOWNSTREAM_ROLE_SCHEMA = "ejepa_event_watchdog_downstream_task_v1"
POD_BUDGET_SCHEMA = "ejepa_event_pod_budget_v1"
DOWNSTREAM_BUDGET_SCHEMA = "ejepa_event_cloud_budget_authorization_v1"
LEGACY_EXCLUSION_SCHEMA = "ejepa_event_legacy_controller_exclusion_v1"
PRODUCERS = ("E_A5", "E_C2F")
HEAD = "EVENT_DOWNSTREAM"
ROTATIONS = (2, 3, 4)


def digest(path: Path) -> str:
    """Return the SHA-256 of one local file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_linux_path(value: object) -> Path:
    if not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise ValueError(f"Unsafe Linux path: {value!r}")
    pure = PurePosixPath(value)
    if ".." in pure.parts:
        raise ValueError(f"Linux path traversal is forbidden: {value}")
    return Path(value)


def parse_deadline(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Deadline must be an ISO-8601 string")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Deadline must be timezone-aware")
    return result


def option(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise ValueError(f"Command requires exactly one {name}")
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"Command option has no value: {name}")
    return argv[index + 1]


def command_sha256(argv: list[str]) -> str:
    return hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_ids() -> set[str]:
    return {
        f"R{rotation}_{fit}"
        for rotation in ROTATIONS
        for fit in (*PRODUCERS, HEAD)
    }


def producer_assignments() -> list[dict[str, object]]:
    return [
        {"rotation": rotation, "fit": fit}
        for rotation in ROTATIONS
        for fit in PRODUCERS
    ]


def _validate_runtime_files(task: dict[str, Any]) -> None:
    files = task.get("runtime_files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Task lacks runtime file bindings: {task.get('id')}")
    observed: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Runtime file binding must be an object")
        path = safe_linux_path(item.get("path")).resolve(strict=True)
        sha = item.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"Invalid runtime SHA binding: {path}")
        if digest(path) != sha:
            raise ValueError(f"Runtime file bytes changed: {path}")
        observed.add(str(path))
    if str(Path(task["entrypoint"]).resolve(strict=True)) not in observed:
        raise ValueError(f"Entrypoint is absent from runtime bindings: {task['id']}")
    suffixes = {Path(item).as_posix() for item in observed}
    required = (
        {
            "scripts/runpod_event_scientific_v1.py",
            "operational/runpod_migration/event_runner.py",
            "operational/runpod_migration/event_handoff.py",
            "operational/runpod_migration/event_resources.py",
        }
        if task["fit"] in PRODUCERS
        else {
            "scripts/runpod_event_queue_v1.py",
            "operational/runpod_migration/event_queue.py",
        }
    )
    missing = {
        suffix for suffix in required if not any(path.endswith(suffix) for path in suffixes)
    }
    if missing:
        raise ValueError(f"Task runtime closure is incomplete: {task['id']} {sorted(missing)}")


def _validate_argv(task: dict[str, Any], bundle: str, run: str) -> None:
    identifier = task["id"]
    argv = task.get("argv")
    adopted = task.get("adopted_argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 4
        or any(not isinstance(item, str) or not item or "\0" in item for item in argv)
        or not isinstance(adopted, list)
        or not adopted
        or any(candidate == [] or not isinstance(candidate, list) for candidate in adopted)
        or any(
            not isinstance(item, str) or not item or "\0" in item
            for candidate in adopted
            for item in candidate
        )
    ):
        raise ValueError(f"Task argv bindings are invalid: {identifier}")
    if argv not in adopted:
        raise ValueError(f"Restart argv must be an admitted process argv: {identifier}")
    if not Path(argv[0]).is_absolute():
        raise ValueError(f"Python executable must be absolute: {identifier}")
    entrypoint = safe_linux_path(task.get("entrypoint")).resolve(strict=True)
    entrypoint_sha = task.get("entrypoint_sha256")
    if digest(entrypoint) != entrypoint_sha:
        raise ValueError(f"Restart entrypoint bytes changed: {identifier}")
    if str(entrypoint) not in argv:
        raise ValueError(f"Restart argv does not contain its entrypoint: {identifier}")
    for candidate in adopted:
        if candidate[0] != argv[0]:
            raise ValueError(f"Adopted Python executable differs: {identifier}")
        if "probe" in candidate or "--additional-updates" in candidate:
            raise ValueError(f"Technical or bounded-update argv is forbidden: {identifier}")
    if option(argv, "--bundle") != bundle or option(argv, "--run") != run:
        raise ValueError(f"Task bundle/run differs: {identifier}")
    if option(argv, "--rotation") != str(task["rotation"]):
        raise ValueError(f"Task rotation differs: {identifier}")
    command_index = argv.index(str(entrypoint)) + 1
    if task["fit"] in PRODUCERS:
        if command_index >= len(argv) or argv[command_index] != "producer":
            raise ValueError(f"Producer restart command differs: {identifier}")
        if option(argv, "--fit") != task["fit"]:
            raise ValueError(f"Producer fit differs: {identifier}")
        if task["rotation"] == 2 and task["fit"] == "E_A5":
            safe_linux_path(option(argv, "--pause-receipt"))
    elif "--fit" in argv or "producer" in argv:
        raise ValueError(f"Downstream argv injected a producer role: {identifier}")
    expected_cmdline = task.get("pid_cmdline_sha256")
    if not isinstance(expected_cmdline, str) or expected_cmdline not in {
        command_sha256(candidate) for candidate in adopted
    }:
        raise ValueError(f"Initial cmdline hash is not admitted: {identifier}")
    _validate_runtime_files(task)


def _validate_assignment(
    task: dict[str, Any],
    *,
    pod_budget_sha: str,
    downstream_budget_path: Path,
    downstream_budget_sha: str,
    deadline: str,
) -> None:
    path = safe_linux_path(task.get("assignment_path")).resolve(strict=True)
    if digest(path) != task.get("assignment_sha256"):
        raise ValueError(f"Assignment bytes changed: {task['id']}")
    assignment = read(path)
    if task["fit"] in PRODUCERS:
        valid = (
            assignment.get("schema") == "ejepa_event_cloud_assignment_v1"
            and assignment.get("execution_authorized") is True
            and assignment.get("tasks") == producer_assignments()
            and assignment.get("maximum_total_usd") == 24
            and assignment.get("budget_receipt_sha256") == pod_budget_sha
            and assignment.get("execution_deadline_utc") == deadline
        )
    else:
        singleton = [{"rotation": task["rotation"], "fit": HEAD}]
        valid = (
            assignment.get("schema") == "ejepa_cloud_assignment_v1"
            and assignment.get("execution_authorized") is True
            and assignment.get("tasks") == singleton
            and assignment.get("shared_budget_authorization_sha256")
            == downstream_budget_sha
            and assignment.get("execution_deadline_utc") == deadline
            and option(task["argv"], "--budget") == str(downstream_budget_path)
        )
    if not valid or option(task["argv"], "--assignment") != str(path):
        raise ValueError(f"Task assignment authority differs: {task['id']}")


def validate_config(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = read(path)
    authority, tasks = value.get("authority"), value.get("tasks")
    if (
        value.get("schema") != SCHEMA
        or not isinstance(authority, dict)
        or not isinstance(tasks, list)
    ):
        raise ValueError("Unknown or malformed EVENT watchdog config")
    if value.get("backoff_seconds") != 60 or value.get("max_retries") != 12:
        raise ValueError("EVENT restart policy must be backoff=60 and max_retries=12")
    poll = value.get("poll_seconds")
    if not isinstance(poll, int) or not 1 <= poll <= 60:
        raise ValueError("Watchdog poll_seconds must be within 1..60")
    environment = value.get("required_environment")
    if (
        not isinstance(environment, dict)
        or not environment
        or "PYTHONPATH" not in environment
        or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in environment.items()
        )
    ):
        raise ValueError("Required environment must be a string mapping")
    bundle = str(safe_linux_path(value.get("bundle_root")))
    run = str(safe_linux_path(value.get("run_root")))

    pod_path = safe_linux_path(authority.get("pod_budget_path")).resolve(strict=True)
    downstream_path = safe_linux_path(
        authority.get("downstream_budget_path")
    ).resolve(strict=True)
    pod_sha, downstream_sha = digest(pod_path), digest(downstream_path)
    if pod_sha != authority.get("pod_budget_sha256") or downstream_sha != authority.get(
        "downstream_budget_sha256"
    ):
        raise ValueError("EVENT budget bytes differ from watchdog authority")
    pod, downstream = read(pod_path), read(downstream_path)
    deadline = pod.get("execution_deadline_utc")
    if not (
        pod.get("schema") == POD_BUDGET_SCHEMA
        and pod.get("maximum_total_usd") == 24
        and downstream.get("schema") == DOWNSTREAM_BUDGET_SCHEMA
        and downstream.get("execution_authorized") is True
        and downstream.get("maximum_total_usd") == 24
        and downstream.get("source_budget_sha256") == pod_sha
        and downstream.get("execution_deadline_utc") == deadline
        and authority.get("global_deadline_utc") == deadline
        and parse_deadline(deadline) > datetime.now(UTC)
    ):
        raise ValueError("EVENT Pod2/downstream budget binding is invalid or expired")

    migration_path = safe_linux_path(
        authority.get("migration_binding_path")
    ).resolve(strict=True)
    if digest(migration_path) != authority.get("migration_binding_sha256"):
        raise ValueError("EVENT migration binding bytes changed")
    migration = read(migration_path)
    if not (
        migration.get("schema") == "ejepa_event_linux_migration_binding_v1"
        and migration.get("bundle_root") == bundle
        and migration.get("run_root") == run
        and migration.get("scientific_files_rewritten") is False
        and migration.get("network_files_rehashed_on_restart") is False
        and isinstance(migration.get("runtime_sources"), dict)
    ):
        raise ValueError("EVENT migration/runtime binding is invalid")

    exclusion_path = safe_linux_path(
        authority.get("legacy_controller_exclusion_path")
    ).resolve(strict=True)
    if digest(exclusion_path) != authority.get("legacy_controller_exclusion_sha256"):
        raise ValueError("Legacy-controller exclusion receipt bytes changed")
    exclusion = read(exclusion_path)
    if not (
        exclusion.get("schema") == LEGACY_EXCLUSION_SCHEMA
        and exclusion.get("status") == "PASS"
        and exclusion.get("controllers_disabled") is True
        and exclusion.get("shell_supervisors_running") is False
        and exclusion.get("task_ids") == sorted(task_ids())
        and exclusion.get("active_legacy_pids") == []
        and isinstance(exclusion.get("disabled_controller_paths"), list)
        and bool(exclusion["disabled_controller_paths"])
    ):
        raise ValueError("Legacy shell controllers are not explicitly disabled")

    if len(tasks) != 9:
        raise ValueError("Watchdog requires six EVENT producers and three downstream tasks")
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in tasks:
        if not isinstance(raw, dict):
            raise ValueError("Watchdog task must be an object")
        task = dict(raw)
        identifier, rotation, fit = task.get("id"), task.get("rotation"), task.get("fit")
        role_schema = (
            PRODUCER_ROLE_SCHEMA if fit in PRODUCERS else DOWNSTREAM_ROLE_SCHEMA
        )
        if (
            not isinstance(identifier, str)
            or rotation not in ROTATIONS
            or fit not in {*PRODUCERS, HEAD}
            or identifier != f"R{rotation}_{fit}"
            or task.get("role_schema") != role_schema
            or task.get("run") != run
        ):
            raise ValueError(f"EVENT task identity/role is invalid: {identifier}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate EVENT task: {identifier}")
        identifiers.add(identifier)
        pid, start_ticks, started_epoch = (
            task.get("pid"),
            task.get("pid_start_ticks"),
            task.get("pid_started_epoch"),
        )
        if (
            not isinstance(pid, int)
            or pid <= 1
            or not isinstance(start_ticks, int)
            or start_ticks < 0
            or not isinstance(started_epoch, (int, float))
            or started_epoch <= 0
        ):
            raise ValueError(f"Initial process identity is invalid: {identifier}")
        task["entrypoint"] = str(safe_linux_path(task.get("entrypoint")))
        task["log"] = str(safe_linux_path(task.get("log")))
        _validate_argv(task, bundle, run)
        _validate_assignment(
            task,
            pod_budget_sha=pod_sha,
            downstream_budget_path=downstream_path,
            downstream_budget_sha=downstream_sha,
            deadline=str(deadline),
        )
        normalized.append(task)
    if identifiers != task_ids():
        raise ValueError("EVENT watchdog task closure differs")
    for field in ("status_path", "lock_path"):
        safe_linux_path(value.get(field))
    return {**value, "tasks": normalized, "config_sha256": digest(path)}


def proc_identity(
    pid: int, proc_root: Path = Path("/proc"), clock_ticks: int | None = None
) -> dict[str, Any] | None:
    root = proc_root / str(pid)
    try:
        raw_stat = (root / "stat").read_text(encoding="utf-8")
        tail = raw_stat[raw_stat.rindex(")") + 2 :].split()
        state, start_ticks = tail[0], int(tail[19])
        argv = [
            item.decode("utf-8")
            for item in (root / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        btime = int(
            next(
                line for line in (proc_root / "stat").read_text().splitlines()
                if line.startswith("btime ")
            ).split()[1]
        )
        ticks = clock_ticks if clock_ticks is not None else int(os.sysconf("SC_CLK_TCK"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration, ValueError):
        return None
    return {
        "pid": pid,
        "state": state,
        "start_ticks": start_ticks,
        "started_epoch": btime + start_ticks / ticks,
        "argv": argv,
        "cmdline_sha256": command_sha256(argv),
    }


def command_matches(task: dict[str, Any], process: dict[str, Any]) -> bool:
    argv = process.get("argv")
    return bool(
        process.get("state") != "Z"
        and isinstance(argv, list)
        and argv in task["adopted_argv"]
        and process.get("cmdline_sha256") == command_sha256(argv)
    )


def wait_process_identity(
    pid: int,
    task: dict[str, Any],
    proc_root: Path,
    *,
    timeout_seconds: float = 2.0,
    poll_seconds: float = 0.01,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Wait briefly for fork/exec to expose the admitted argv without relaunching."""
    deadline = monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while True:
        observed = proc_identity(pid, proc_root)
        if observed is not None:
            last = observed
            if command_matches(task, observed):
                return observed, last
        if monotonic() >= deadline:
            return None, last
        sleeper(poll_seconds)


def fresh(path: Path, started_epoch: float) -> bool:
    return path.is_file() and path.stat().st_mtime >= started_epoch


def clean_checkpoint(directory: Path, started_epoch: float, pid: int) -> tuple[bool, str]:
    receipt_path, journal_path = (
        directory / "CHECKPOINT_RECEIPT.json",
        directory / "UPDATE_JOURNAL.json",
    )
    if not fresh(receipt_path, started_epoch) or not fresh(journal_path, started_epoch):
        return False, "STALE_OR_MISSING_CHECKPOINT_TRANSACTION"
    receipt, journal = read(receipt_path), read(journal_path)
    completed = receipt.get("completed_updates")
    if not (
        isinstance(completed, int)
        and completed == journal.get("completed_updates") == journal.get("durable_updates")
        and journal.get("pending_update_upper") == 0
        and receipt.get("complete_state") is True
    ):
        return False, "CHECKPOINT_TRANSACTION_NOT_DURABLE"
    progress_path = directory / "PROGRESS.json"
    if progress_path.is_file() and (
        not fresh(progress_path, started_epoch) or read(progress_path).get("pid") != pid
    ):
        return False, "PROGRESS_PID_OR_FRESHNESS_MISMATCH"
    return True, "DURABLE"


def immutable_complete_endpoint(
    directory: Path, rotation: int, fit: str
) -> tuple[bool, str]:
    """Admit an old sealed PAIR_E endpoint without imposing current-process mtimes."""
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    journal_path = directory / "UPDATE_JOURNAL.json"
    admission_path = directory / "ENDPOINT_ADMISSION.json"
    if any(not item.is_file() for item in (receipt_path, journal_path, admission_path)):
        return False, "SEALED_ENDPOINT_EVIDENCE_MISSING"
    receipt, journal, admission = map(read, (receipt_path, journal_path, admission_path))
    checkpoint_raw, completed = receipt.get("checkpoint_path"), receipt.get("completed_updates")
    if not isinstance(checkpoint_raw, str):
        return False, "SEALED_ENDPOINT_CHECKPOINT_PATH_INVALID"
    try:
        checkpoint = Path(checkpoint_raw).resolve(strict=True)
        contained = checkpoint.is_relative_to(directory.resolve(strict=True))
    except (FileNotFoundError, OSError):
        return False, "SEALED_ENDPOINT_CHECKPOINT_MISSING"
    checkpoint_sha = digest(checkpoint) if contained and checkpoint.is_file() else None
    if not (
        isinstance(completed, int)
        and receipt.get("status") == "COMPLETE"
        and receipt.get("scientific_endpoint") is True
        and receipt.get("complete_state") is True
        and completed == journal.get("completed_updates") == journal.get("durable_updates")
        and journal.get("pending_update_upper") == 0
        and checkpoint_sha == receipt.get("checkpoint_sha256")
        and admission.get("schema") == "final_refit_fold_endpoint_admission_v1"
        and admission.get("status") == "PASS"
        and admission.get("scientific_endpoint") is True
        and admission.get("rotation") == rotation
        and admission.get("fit_id") == fit
        and admission.get("updates") == completed
        and admission.get("checkpoint_path") == checkpoint_raw
        and admission.get("checkpoint_sha256") == checkpoint_sha
        and admission.get("checkpoint_receipt_path") == str(receipt_path.resolve())
        and admission.get("checkpoint_receipt_sha256") == digest(receipt_path)
    ):
        return False, "SEALED_ENDPOINT_IDENTITY_OR_DURABILITY_MISMATCH"
    return True, "SEALED_COMPLETE_ENDPOINT"


def producer_cloud_admitted(directory: Path, task: dict[str, Any]) -> tuple[bool, str]:
    path = directory / "SCIENTIFIC_CLOUD_ADMISSION.json"
    if not path.is_file():
        return False, "EVENT_CLOUD_ADMISSION_MISSING"
    admission = read(path)
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    receipt = read(receipt_path)
    if not (
        admission.get("schema") == "ejepa_event_scientific_cloud_admission_v1"
        and admission.get("status") == "ADMITTED"
        and admission.get("rotation") == task["rotation"]
        and admission.get("fit") == task["fit"]
        and admission.get("assignment_sha256") == task["assignment_sha256"]
        and admission.get("checkpoint_sha256") == receipt.get("checkpoint_sha256")
        and admission.get("completed_updates") == receipt.get("completed_updates")
        and receipt.get("status") == "COMPLETE"
        and receipt.get("scientific_endpoint") is True
    ):
        return False, "EVENT_CLOUD_ADMISSION_OR_ENDPOINT_MISMATCH"
    return True, "EVENT_CLOUD_ADMITTED"


def stop_paths(task: dict[str, Any]) -> list[Path]:
    run = Path(task["run"])
    root = run / f"rotations/{task['rotation']}"
    if task["fit"] in PRODUCERS:
        return [run / "STOP_REQUEST", root / f"fits/{task['fit']}/STOP_REQUEST"]
    return [
        run / "STOP_REQUEST",
        root / "STOP_REQUEST",
        root / "fits/PAIR_E/STOP_REQUEST",
        root / "fits/E_CTX/STOP_REQUEST",
    ]


def producer_terminal(task: dict[str, Any], managed: dict[str, Any]) -> tuple[str, str]:
    directory = Path(task["run"]) / f"rotations/{task['rotation']}/fits/{task['fit']}"
    started, pid = float(managed["started_epoch"]), int(managed["pid"])
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    if not receipt_path.is_file():
        return "BLOCKED", "CHECKPOINT_RECEIPT_MISSING"
    receipt = read(receipt_path)
    if receipt.get("status") == "COMPLETE":
        sealed, reason = immutable_complete_endpoint(
            directory, int(task["rotation"]), str(task["fit"])
        )
        if not sealed:
            return "BLOCKED", reason
        admitted, reason = producer_cloud_admitted(directory, task)
        return ("COMPLETE", reason) if admitted else ("BLOCKED", reason)
    clean, reason = clean_checkpoint(directory, started, pid)
    if not clean:
        return "BLOCKED", reason
    pause_path = directory / "RESOURCE_PAUSE.json"
    if not fresh(pause_path, started):
        return "BLOCKED", "RESOURCE_PAUSE_NOT_FRESH"
    pause = read(pause_path)
    if receipt.get("status") != "PAUSED_RESOURCE" or pause.get("manual") is not False:
        return "BLOCKED", "NOT_AN_AUTOMATIC_RESOURCE_PAUSE"
    if pause.get("completed_updates") != receipt.get("completed_updates"):
        return "BLOCKED", "PAUSE_CHECKPOINT_UPDATE_MISMATCH"
    return "RESTARTABLE", "FRESH_DURABLE_EVENT_RESOURCE_PAUSE"


def downstream_complete(task: dict[str, Any], result_path: Path) -> tuple[bool, str]:
    """Verify the admitted EVENT_OOF result and all declared durable outputs."""
    run, rotation = Path(task["run"]).resolve(strict=True), int(task["rotation"])
    result = read(result_path)
    payload = {key: value for key, value in result.items() if key != "receipt_sha256"}
    if not (
        result.get("schema") == "ejepa_cloud_event_downstream_queue_v1"
        and result.get("stage") == "EVENT_OOF"
        and result.get("status") == "COMPLETE"
        and result.get("rotation") == rotation
        and result.get("receipt_sha256") == canonical_sha256(payload)
        and result.get("dependencies")
        == {"E_A5": True, "E_C2F": True, "PAIR_E": True, "E_CTX": True}
        and result.get("emitted_interface") == "PHASE17_RAW_EVENT"
        and result.get("phase20_context_emitted") is False
        and result.get("final_ttc_predictions_emitted") is False
    ):
        return False, "MALFORMED_FRESH_EVENT_RESULT"

    admission_path = run / f"EVENT_DOWNSTREAM_ADMISSION_R{rotation}.json"
    if not admission_path.is_file():
        return False, "EVENT_DOWNSTREAM_ADMISSION_MISSING"
    admission = read(admission_path)
    migration_path = run / "EVENT_MIGRATION_BINDING.json"
    authority = result.get("authority")
    if not (
        isinstance(authority, dict)
        and admission.get("schema") == "ejepa_event_downstream_admission_v1"
        and admission.get("status") == "PASS"
        and admission.get("task") == {"rotation": rotation, "fit": HEAD}
        and admission.get("assignment_sha256") == task["assignment_sha256"]
        and migration_path.is_file()
        and admission.get("event_migration_binding_sha256") == digest(migration_path)
        and admission.get("allowed_new_endpoints") == ["PAIR_E", "E_CTX"]
        and admission.get("allowed_generation_interface") == "PHASE17_RAW_EVENT"
        and admission.get("rgb_model_inference_authorized") is False
        and admission.get("scientific_pipeline_edited") is False
        and authority.get("event_downstream_admission_sha256") == digest(admission_path)
        and authority.get("assignment_sha256") == task["assignment_sha256"]
        and authority.get("event_migration_binding_sha256") == digest(migration_path)
    ):
        return False, "EVENT_DOWNSTREAM_ADMISSION_OR_AUTHORITY_MISMATCH"

    for fit in ("PAIR_E", "E_CTX"):
        clean, reason = immutable_complete_endpoint(
            run / f"rotations/{rotation}/fits/{fit}", rotation, fit
        )
        if not clean:
            return False, f"{fit}_{reason}"

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False, "EVENT_RESULT_ARTIFACT_MANIFEST_MISSING"
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return False, "EVENT_RESULT_ARTIFACT_ENTRY_INVALID"
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in seen:
            return False, "EVENT_RESULT_ARTIFACT_PATH_INVALID"
        seen.add(item["path"])
        try:
            path = (run / Path(*relative.parts)).resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False, "EVENT_RESULT_ARTIFACT_MISSING"
        if not path.is_file() or not path.is_relative_to(run):
            return False, "EVENT_RESULT_ARTIFACT_PATH_INVALID"
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != digest(path):
            return False, "EVENT_RESULT_ARTIFACT_HASH_OR_SIZE_MISMATCH"
    return True, "EVENT_OOF_ADMITTED_COMPLETE"


def downstream_terminal(task: dict[str, Any], managed: dict[str, Any]) -> tuple[str, str]:
    run, rotation = Path(task["run"]), int(task["rotation"])
    started, pid = float(managed["started_epoch"]), int(managed["pid"])
    result_path = run / f"EVENT_DOWNSTREAM_RESULT_R{rotation}.json"
    if fresh(result_path, started):
        complete, reason = downstream_complete(task, result_path)
        return ("COMPLETE", reason) if complete else ("BLOCKED", reason)
    pause_path = run / f"EVENT_DOWNSTREAM_PAUSE_R{rotation}.json"
    if not fresh(pause_path, started):
        return "BLOCKED", "EVENT_DOWNSTREAM_PAUSE_NOT_FRESH"
    pause = read(pause_path)
    if not (
        pause.get("schema") == "ejepa_event_downstream_pause_v1"
        and pause.get("status") == "PAUSED_RECOVERABLE"
        and pause.get("rotation") == rotation
        and pause.get("manual") is False
    ):
        return "BLOCKED", "EVENT_PAUSE_NOT_AUTOMATIC"
    pipeline_path = run / f"rotations/{rotation}/EVENT_PIPELINE_RECEIPT.json"
    if not fresh(pipeline_path, started):
        return "BLOCKED", "EVENT_PIPELINE_RECEIPT_NOT_FRESH"
    pipeline = read(pipeline_path)
    if not (
        pipeline.get("schema") == "ejepa_cloud_event_downstream_queue_v1"
        and pipeline.get("rotation") == rotation
        and pipeline.get("pid") == pid
        and pipeline.get("stage") != "EVENT_OOF"
        and pipeline.get("status") != "COMPLETE"
    ):
        return "BLOCKED", "EVENT_PIPELINE_PID_OR_STAGE_MISMATCH"
    observed_pause = False
    for fit in ("PAIR_E", "E_CTX"):
        directory = run / f"rotations/{rotation}/fits/{fit}"
        receipt_path = directory / "CHECKPOINT_RECEIPT.json"
        if not receipt_path.is_file():
            continue
        receipt = read(receipt_path)
        if fit == "PAIR_E" and receipt.get("status") == "COMPLETE":
            clean, reason = immutable_complete_endpoint(directory, rotation, fit)
        else:
            clean, reason = clean_checkpoint(directory, started, pid)
            if clean and receipt.get("status") == "PAUSED_RESOURCE":
                resource_path = directory / "RESOURCE_PAUSE.json"
                clean = fresh(resource_path, started)
                if clean:
                    resource = read(resource_path)
                    clean = (
                        resource.get("manual") is False
                        and resource.get("completed_updates")
                        == receipt.get("completed_updates")
                    )
                reason = "FRESH_DURABLE_RESOURCE_PAUSE" if clean else "RESOURCE_PAUSE_INVALID"
                observed_pause = observed_pause or clean
            elif clean:
                clean, reason = False, "HEAD_CHECKPOINT_NOT_PAUSED_RESOURCE"
        if not clean:
            return "BLOCKED", f"{fit}_{reason}"
    if not observed_pause:
        return "BLOCKED", "NO_FRESH_PAUSED_RESOURCE_HEAD_BOUNDARY"
    return "RESTARTABLE", "FRESH_DURABLE_EVENT_DOWNSTREAM_RESOURCE_PAUSE"


def campaign_terminal_code(states: list[str]) -> int | None:
    """A blocked task never stops supervision of another live EVENT task."""
    terminal = {"COMPLETE", "DEADLINE_EXPIRED", "BLOCKED"}
    if not states or not all(state in terminal for state in states):
        return None
    return 5 if "BLOCKED" in states else 0


class Lock:
    def __init__(self, path: Path) -> None:
        self.path, self.stream = path, None

    def __enter__(self) -> Lock:
        if fcntl is None:
            raise RuntimeError("EVENT resource watchdog lock requires Linux fcntl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError("Another EVENT resource watchdog owns the lock") from error
        return self

    def __exit__(self, *_args: object) -> None:
        assert fcntl is not None and self.stream is not None
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


class Watchdog:
    def __init__(
        self,
        config_path: Path,
        *,
        proc_root: Path = Path("/proc"),
        now: Callable[[], float] = time.time,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        if os.name != "posix":
            raise RuntimeError("EVENT resource watchdog requires Linux")
        self.config = validate_config(config_path)
        if any(
            os.environ.get(key) != expected
            for key, expected in self.config["required_environment"].items()
        ):
            raise ValueError("Watchdog process did not inherit its admitted environment")
        self.proc_root, self.now, self.popen = proc_root, now, popen
        self.source_sha256 = digest(Path(__file__).resolve())
        self.status_path = Path(self.config["status_path"])
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            return {"tasks": {}}
        value = read(self.status_path)
        if (
            value.get("schema") != STATUS_SCHEMA
            or value.get("config_sha256") != self.config["config_sha256"]
            or value.get("source_sha256") != self.source_sha256
        ):
            raise ValueError("Existing EVENT watchdog status belongs to another config/source")
        return value

    def _initial(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "pid": task["pid"],
            "start_ticks": task["pid_start_ticks"],
            "started_epoch": task["pid_started_epoch"],
            "cmdline_sha256": task["pid_cmdline_sha256"],
            "retries": 0,
        }

    def _publish(self) -> dict[str, Any]:
        states = self.state.setdefault("tasks", {})
        summary = {
            name: sum(item.get("state") == name for item in states.values())
            for name in (
                "RUNNING",
                "RESTARTED",
                "RESTART_BACKOFF",
                "COMPLETE",
                "BLOCKED",
                "DEADLINE_EXPIRED",
            )
        }
        value = {
            "schema": STATUS_SCHEMA,
            "updated_utc": datetime.fromtimestamp(self.now(), UTC).isoformat(),
            "config_sha256": self.config["config_sha256"],
            "source_sha256": self.source_sha256,
            "global_deadline_utc": self.config["authority"]["global_deadline_utc"],
            "pod_budget_sha256": self.config["authority"]["pod_budget_sha256"],
            "downstream_budget_sha256": self.config["authority"][
                "downstream_budget_sha256"
            ],
            "tasks": states,
            "summary": summary,
            "optimizer_updates_performed_by_watchdog": 0,
            "processes_killed": 0,
            "stop_files_removed": 0,
        }
        atomic_json(self.status_path, value)
        self.state = value
        return value

    def _adopt(self, task: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
        pid = int(current["pid"])
        process = proc_identity(pid, self.proc_root)
        if process is None:
            return None
        if process["start_ticks"] != current["start_ticks"]:
            current.update(state="BLOCKED", reason="PID_REUSED")
            return process
        if process.get("state") == "Z":
            return None
        if (
            process.get("cmdline_sha256") != current["cmdline_sha256"]
            or not command_matches(task, process)
        ):
            current.update(state="BLOCKED", reason="PROCESS_IDENTITY_MISMATCH")
            return process
        current.update(state="RUNNING", reason="ADOPTED_OR_STILL_RUNNING")
        return process

    def _restart(self, task: dict[str, Any], current: dict[str, Any]) -> None:
        retries = int(current.get("retries", 0))
        if retries >= 12:
            current.update(state="BLOCKED", reason="MAX_RETRIES_EXHAUSTED")
            return
        if self.now() < float(current.get("next_retry_epoch", 0)):
            current.update(state="RESTART_BACKOFF", reason="WAITING_60_SECOND_BACKOFF")
            return
        log = Path(task["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab", buffering=0) as output:
            process = self.popen(
                task["argv"],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        identity, last = wait_process_identity(process.pid, task, self.proc_root)
        if identity is None:
            evidence: dict[str, Any] = {
                "pid": process.pid,
                "start_ticks": last.get("start_ticks") if last is not None else None,
                "started_epoch": last.get("started_epoch") if last is not None else None,
                "cmdline_sha256": (
                    last.get("cmdline_sha256") if last is not None else None
                ),
                "retries": retries + 1,
                "state": "BLOCKED",
                "reason": "RESTARTED_PROCESS_IDENTITY_TIMEOUT",
                "spawned_process_left_running": True,
            }
            if last is not None:
                evidence.update(
                    observed_process_state=last.get("state"),
                )
            current.update(evidence)
            return
        current.update(
            pid=process.pid,
            start_ticks=identity["start_ticks"],
            started_epoch=identity["started_epoch"],
            cmdline_sha256=identity["cmdline_sha256"],
            retries=retries + 1,
            next_retry_epoch=self.now() + 60,
            state="RESTARTED",
            reason="AUTOMATIC_EVENT_RESOURCE_PAUSE_RESUME",
        )

    def step(self) -> dict[str, Any]:
        states = self.state.setdefault("tasks", {})
        deadline = parse_deadline(
            self.config["authority"]["global_deadline_utc"]
        ).timestamp()
        for task in self.config["tasks"]:
            current = states.setdefault(task["id"], self._initial(task))
            if current.get("state") in {"COMPLETE", "BLOCKED", "DEADLINE_EXPIRED"}:
                continue
            if any(path.exists() for path in stop_paths(task)):
                current.update(state="BLOCKED", reason="STOP_REQUEST_PRESENT")
                continue
            process = self._adopt(task, current)
            if current.get("state") == "BLOCKED" or process is not None:
                continue
            terminal, reason = (
                producer_terminal(task, current)
                if task["fit"] in PRODUCERS
                else downstream_terminal(task, current)
            )
            if terminal == "COMPLETE":
                current.update(state="COMPLETE", reason=reason)
            elif terminal != "RESTARTABLE":
                current.update(state="BLOCKED", reason=reason)
            elif self.now() >= deadline:
                current.update(state="DEADLINE_EXPIRED", reason="AUTHORIZED_DEADLINE_REACHED")
            else:
                self._restart(task, current)
        return self._publish()

    def run(self, *, once: bool = False) -> int:
        with Lock(Path(self.config["lock_path"])):
            while True:
                value = self.step()
                code = campaign_terminal_code(
                    [item["state"] for item in value["tasks"].values()]
                )
                if code is not None:
                    return code
                if once:
                    return 3
                time.sleep(self.config["poll_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()
    if args.validate_config:
        config = validate_config(args.config)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "config_sha256": config["config_sha256"],
                    "tasks": len(config["tasks"]),
                    "global_deadline_utc": config["authority"]["global_deadline_utc"],
                    "optimizer_updates": 0,
                },
                indent=2,
            )
        )
        return 0
    return Watchdog(args.config).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
