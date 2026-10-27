"""Durable sequential scheduler for the admitted scientific OOF refit.

The queue waits for the already-running RGB endpoints, validates resources from
``SharedResourceMonitor``, and invokes one immutable stage at a time through the
bound PowerShell launcher.  It never submits benchmark predictions and never
retries scientific or integrity failures.  Only a receipt-proven resource pause
may be retried automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import psutil

from operational.rgb_port.accounting import atomic_write_json, read_json_shared

from .auto_resume import file_digest
from .continuation_queue import queue_lock, rgb_endpoints
from .recipe import canonical_sha256
from .shared_resources import (
    GIB,
    PROJECT_RSS_LIMIT_BYTES,
    SharedResourceMonitor,
    clear_stale_publisher_lock,
    create_shared_resource_session,
)

SCHEMA = "final_refit_scientific_oof_queue_plan_v1"
STATE_SCHEMA = "final_refit_scientific_oof_queue_state_v1"
CHECKPOINT_UPDATES = 100
MIN_PHYSICAL_BYTES = 2 * GIB
MIN_COMMIT_BYTES = 2 * GIB
CHECKPOINT_RESERVATION_BYTES = GIB
WAIT_SECONDS = 30.0
PUBLISHER_START_SECONDS = 30.0
PUBLISHER_RECOVERY_LIMIT = 3
RESOURCE_EXIT_CODE = 3
TERMINAL = {"COMPLETE", "MANUAL_STOP", "BLOCKED_ERROR", "BLOCKED_INTEGRITY"}


class _Child(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


PopenFactory = Callable[..., _Child]


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _stage_specs(run: Path) -> list[dict[str, Any]]:
    stages = []
    for rotation in range(5):
        stages.append(
            {
                "id": f"ROTATION_{rotation}",
                "module": "operational.final_refit.scientific_oof_pipeline",
                "arguments": ["--run", str(run), "--rotation", str(rotation)],
                "receipt": str(run / "rotations" / str(rotation) / "PIPELINE_RECEIPT.json"),
                "rotation": rotation,
            }
        )
    stages.append(
        {
            "id": "FINAL",
            "module": "operational.final_refit.scientific_oof_final",
            "arguments": ["--run", str(run)],
            "receipt": str(run / "final" / "FINAL_RECEIPT.json"),
            "rotation": None,
        }
    )
    return stages


def build_plan(
    repository: Path,
    run: Path,
    legacy_plan: Path,
    scientific_config: Path,
    launcher: Path,
    *,
    python_executable: Path,
    powershell_executable: Path,
) -> dict[str, Any]:
    """Freeze every executable source, argument and resource contract."""
    repository = repository.resolve(strict=True)
    run = run.resolve(strict=True)
    required = (
        Path(__file__),
        repository / "operational/final_refit/continuation_queue.py",
        repository / "operational/final_refit/shared_resources.py",
        repository / "operational/final_refit/scientific_oof_pipeline.py",
        repository / "operational/final_refit/scientific_oof_final.py",
        launcher.resolve(strict=True),
        legacy_plan.resolve(strict=True),
        scientific_config.resolve(strict=True),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze unavailable OOF executors: {missing}")
    config = read_json_shared(scientific_config)
    if config.get("checkpoint_updates") != CHECKPOINT_UPDATES:
        raise ValueError("SCIENTIFIC_CONFIG must fix checkpoint_updates=100")
    if config.get("physical_min_bytes") != MIN_PHYSICAL_BYTES:
        raise ValueError("SCIENTIFIC_CONFIG must fix physical_min_bytes=2GiB")
    if config.get("commit_min_bytes") != MIN_COMMIT_BYTES:
        raise ValueError("SCIENTIFIC_CONFIG must fix commit_min_bytes=2GiB")
    stages = _stage_specs(run)
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": str(repository),
        "run": str(run),
        "legacy_continuation_plan": str(legacy_plan.resolve(strict=True)),
        "scientific_config": str(scientific_config.resolve(strict=True)),
        "launcher": str(launcher.resolve(strict=True)),
        "python_executable": str(python_executable.resolve(strict=True)),
        "powershell_executable": str(powershell_executable.resolve(strict=True)),
        "resource_session": str(run / "RESOURCE_SESSION.json"),
        "resource_snapshot": str(run / "RESOURCE_SNAPSHOT.json"),
        "resource_publisher_lock": str(run / "RESOURCE_PUBLISHER.lock"),
        "stop_path": str(run / "STOP_REQUEST"),
        "state_path": str(run / "SCIENTIFIC_OOF_QUEUE_STATE.json"),
        "lock_path": str(run / "SCIENTIFIC_OOF_QUEUE.lock"),
        "result_directory": str(run / "launcher_results"),
        "stages": stages,
        "source_bindings": {
            str(path.resolve()): file_digest(path)
            for path in (
                *required,
                python_executable.resolve(strict=True),
                powershell_executable.resolve(strict=True),
            )
        },
        "resource_policy": {
            "physical_min_bytes": MIN_PHYSICAL_BYTES,
            "commit_min_bytes": MIN_COMMIT_BYTES,
            "checkpoint_reservation_bytes": CHECKPOINT_RESERVATION_BYTES,
            "project_rss_limit_bytes": PROJECT_RSS_LIMIT_BYTES,
            "checkpoint_updates": CHECKPOINT_UPDATES,
        },
        "sequential": True,
        "submission_authorized": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Reject plan, source or argument drift before observing or launching work."""
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("schema") != SCHEMA or canonical_sha256(payload) != plan.get("plan_sha256"):
        raise ValueError("Scientific OOF queue plan digest/schema mismatch")
    if plan.get("submission_authorized") is not False or plan.get("sequential") is not True:
        raise ValueError("Scientific OOF queue policy changed")
    policy = plan.get("resource_policy")
    expected_policy = {
        "physical_min_bytes": MIN_PHYSICAL_BYTES,
        "commit_min_bytes": MIN_COMMIT_BYTES,
        "checkpoint_reservation_bytes": CHECKPOINT_RESERVATION_BYTES,
        "project_rss_limit_bytes": PROJECT_RSS_LIMIT_BYTES,
        "checkpoint_updates": CHECKPOINT_UPDATES,
    }
    if policy != expected_policy:
        raise ValueError("Scientific OOF resource/checkpoint policy changed")
    if list(plan.get("stages", [])) != _stage_specs(Path(str(plan["run"]))):
        raise ValueError("Scientific OOF stage arguments changed")
    for path, expected in dict(plan.get("source_bindings", {})).items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Bound scientific OOF source/input changed: {path}")


def _resource_ready(allowed: bool, values: Mapping[str, Any]) -> bool:
    """Apply the frozen 2GiB policy to a fully validated shared snapshot."""
    del allowed  # SharedResourceMonitor is stricter on commit; this queue freezes 2GiB.
    commit = values.get("windows_commit_headroom_bytes")
    return bool(
        values.get("snapshot_valid") is True
        and values.get("run_root_binding_valid") is True
        and values.get("process_coverage_bounded") is True
        and values.get("publisher_scan_error") is None
        and type(values.get("project_tree_rss_upper_bound_bytes")) is int
        and int(values["project_tree_rss_upper_bound_bytes"]) <= PROJECT_RSS_LIMIT_BYTES
        and type(values.get("host_available_bytes")) is int
        and int(values["host_available_bytes"]) >= MIN_PHYSICAL_BYTES
        and (commit is None or (type(commit) is int and commit >= MIN_COMMIT_BYTES))
        and type(values.get("disk_free_after_reservation_bytes")) is int
        and int(values["disk_free_after_reservation_bytes"]) >= 10_000_000_000
    )


def _hash_map(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty hash map")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid {name} entry")
        int(digest, 16)
        result[key] = digest
    return result


def _validate_final_deliverables(receipt: Mapping[str, Any]) -> None:
    required = {
        "model_seal",
        "head_export",
        "head_export_receipt",
        "predictions",
        "prediction_receipt",
        "submission_zip",
        "package_receipt",
    }
    if (
        receipt.get("all_head_endpoints_complete") is not True
        or receipt.get("all_oof_inputs_complete") is not True
    ):
        raise ValueError("FINAL receipt does not prove all head endpoints and OOF inputs")
    expected_receipt_sha = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    if receipt.get("receipt_sha256") != expected_receipt_sha:
        raise ValueError("FINAL receipt_sha256 mismatch")
    paths = receipt.get("artifact_paths")
    hashes = receipt.get("artifact_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError("FINAL artifact maps are missing")
    if set(paths) != required or set(hashes) != required:
        raise ValueError("FINAL must bind the exact seven deliverables")
    resolved: dict[str, Path] = {}
    for name in sorted(required):
        raw_path = paths[name]
        expected = hashes[name]
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError("FINAL artifact map entry has an invalid type")
        path = Path(raw_path)
        if not path.is_absolute() or str(path.resolve(strict=True)) != raw_path:
            raise ValueError(f"FINAL artifact path is not canonical: {name}")
        if file_digest(path) != expected:
            raise ValueError(f"FINAL artifact digest mismatch: {name}")
        resolved[name] = path

    prediction = read_json_shared(resolved["prediction_receipt"])
    package = read_json_shared(resolved["package_receipt"])
    package_payload = {key: value for key, value in package.items() if key != "receipt_sha256"}
    seal_sha = hashes["model_seal"]
    if (
        prediction.get("schema") != "scientific_oof_test12_prediction_receipt_v1"
        or prediction.get("status") != "COMPLETE"
        or prediction.get("model_seal_sha256") != seal_sha
        or prediction.get("predictions_sha256") != hashes["predictions"]
        or prediction.get("private_targets_read") is not False
        or prediction.get("external_submission") is not False
    ):
        raise ValueError("FINAL prediction receipt is incomplete or detached")
    if (
        package.get("schema") != "scientific_oof_test12_package_v1"
        or package.get("status") != "PACKAGED_NOT_SUBMITTED"
        or package.get("model_seal_sha256") != seal_sha
        or package.get("prediction_receipt_sha256") != hashes["prediction_receipt"]
        or package.get("zip_path") != str(resolved["submission_zip"])
        or package.get("zip_sha256") != hashes["submission_zip"]
        or package.get("private_targets_read") is not False
        or package.get("external_submission") is not False
        or package.get("receipt_sha256") != canonical_sha256(package_payload)
    ):
        raise ValueError("FINAL package is incomplete, submitted, or detached")


def validate_receipt(stage: Mapping[str, Any], *, allow_pause: bool) -> dict[str, Any]:
    """Validate a durable endpoint or a resource-only pause receipt."""
    path = Path(str(stage["receipt"]))
    receipt = read_json_shared(path)
    if receipt.get("stage_id") != stage["id"]:
        raise ValueError(f"Stage receipt identity mismatch: {stage['id']}")
    if receipt.get("checkpoint_updates") != CHECKPOINT_UPDATES:
        raise ValueError("Stage receipt checkpoint cadence changed")
    for field in ("completed_updates", "durable_updates", "pending_update_upper"):
        if type(receipt.get(field)) is not int or int(receipt[field]) < 0:
            raise ValueError(f"Invalid stage receipt counter: {field}")
    if receipt["completed_updates"] != receipt["durable_updates"]:
        raise ValueError("Stage receipt is not at a durable update boundary")
    if receipt["pending_update_upper"] != 0:
        raise ValueError("Stage receipt retains a pending update")
    status = receipt.get("status")
    if status == "PAUSED_RESOURCE" and allow_pause:
        if receipt.get("manual_stop") is not False:
            raise ValueError("Manual stop cannot be retried as a resource pause")
        return receipt
    if status != "COMPLETE" or receipt.get("scientific_endpoint") is not True:
        raise ValueError(f"Stage has no complete scientific endpoint: {stage['id']}")
    admissions = _hash_map(receipt.get("admissions_sha256"), "admissions_sha256")
    fragments = _hash_map(receipt.get("fragment_files_sha256"), "fragment_files_sha256")
    for name, values in (("admissions_sha256", admissions), ("fragment_files_sha256", fragments)):
        for raw_path, expected in values.items():
            path = Path(raw_path)
            if not path.is_absolute() or str(path.resolve(strict=True)) != raw_path:
                raise ValueError(
                    f"{name} path must be an existing canonical absolute path: {raw_path}"
                )
            if file_digest(path) != expected:
                raise ValueError(f"{name} artifact digest mismatch: {raw_path}")
    fragment = receipt.get("fragment_sha256")
    if not isinstance(fragment, str) or len(fragment) != 64:
        raise ValueError("Invalid fragment_sha256")
    int(fragment, 16)
    if fragment != canonical_sha256(fragments):
        raise ValueError("fragment_sha256 does not seal fragment_files_sha256")
    if stage["id"] == "FINAL":
        _validate_final_deliverables(receipt)
    return receipt


def _pid_live(pid: object, created: object) -> bool:
    if type(pid) is not int or isinstance(created, bool) or not isinstance(created, (int, float)):
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and float(process.create_time()) == float(created)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, OSError):
        return False


def _initial_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "status": "WAITING_RGB",
        "observed_utc": _utc(),
        "stages": {
            stage["id"]: {"status": "WAITING_DEPENDENCY", "attempts": 0} for stage in plan["stages"]
        },
        "scientific_updates_claimed": 0,
        "submission_performed": False,
    }


class ScientificOOFQueue:
    """One durable queue owner; dependencies are injectable for unit tests."""

    def __init__(
        self,
        plan: dict[str, Any],
        *,
        monitor: SharedResourceMonitor | None = None,
        popen_factory: PopenFactory = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_plan(plan)
        self.plan = plan
        self.run_root = Path(plan["run"])
        self.state_path = Path(plan["state_path"])
        self.stop_path = Path(plan["stop_path"])
        self.popen_factory = popen_factory
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.monitor = monitor
        self._injected_monitor = monitor is not None
        self._publisher_child: _Child | None = None

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return _initial_state(self.plan)
        state = read_json_shared(self.state_path)
        if (
            state.get("schema") != STATE_SCHEMA
            or state.get("plan_sha256") != self.plan["plan_sha256"]
        ):
            raise ValueError("Queue state belongs to another plan")
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        state["observed_utc"] = _utc()
        atomic_write_json(self.state_path, state)

    def _result_path(self, stage: Mapping[str, Any], attempt: int) -> Path:
        return Path(self.plan["result_directory"]) / f"{stage['id']}.attempt_{attempt}.json"

    def _classify_result(
        self,
        state: dict[str, Any],
        stage: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        entry = state["stages"][stage["id"]]
        if result.get("stage_id") != stage["id"] or type(result.get("exit_code")) is not int:
            raise ValueError("Launcher result identity/counter mismatch")
        running = state.get("running")
        if isinstance(running, dict) and result.get("intent_nonce") != running.get("intent_nonce"):
            raise ValueError("Launcher result intent mismatch")
        code = int(result["exit_code"])
        entry["exit_code"] = code
        entry["launcher_result"] = dict(result)
        if code == 0:
            entry["receipt"] = validate_receipt(stage, allow_pause=False)
            entry["status"] = "COMPLETE"
            return
        if code == RESOURCE_EXIT_CODE:
            if self.stop_path.exists():
                state["status"] = "MANUAL_STOP"
                entry["status"] = "MANUAL_STOP"
                return
            pause_receipt = validate_receipt(stage, allow_pause=True)
            if pause_receipt.get("status") != "PAUSED_RESOURCE":
                raise ValueError("Resource exit code requires a PAUSED_RESOURCE receipt")
            entry["receipt"] = pause_receipt
            entry["status"] = "WAITING_RESOURCES"
            state["status"] = "WAITING_RESOURCES"
            entry["eligible_monotonic"] = self.monotonic() + min(
                600.0, 60.0 * 2 ** min(int(entry["attempts"]) - 1, 4)
            )
            return
        entry["status"] = "BLOCKED_ERROR"
        state["status"] = "BLOCKED_ERROR"

    def _recover_running(self, state: dict[str, Any]) -> bool:
        running = state.get("running")
        if not isinstance(running, dict):
            return False
        if running.get("pid") is None:
            intent_path = Path(str(running.get("intent_path", "")))
            if not intent_path.is_file():
                begun_ns = running.get("launch_intent_wall_time_ns")
                if type(begun_ns) is int and time.time_ns() - begun_ns <= 30_000_000_000:
                    state["status"] = "WAITING_LAUNCH_HANDSHAKE"
                    return True
                raise ValueError("Expired launch intent has no launcher handshake")
            intent = read_json_shared(intent_path)
            if (
                intent.get("schema") != "final_refit_scientific_oof_launch_intent_v1"
                or intent.get("stage_id") != running.get("stage_id")
                or intent.get("intent_nonce") != running.get("intent_nonce")
                or intent.get("repository") != self.plan["repository"]
                or intent.get("run") != self.plan["run"]
            ):
                raise ValueError("Launcher handshake does not match the durable launch intent")
            running["pid"] = intent.get("launcher_pid")
            running["create_time"] = intent.get("launcher_create_time")
            state["running"] = running
        if _pid_live(running.get("pid"), running.get("create_time")):
            state["status"] = "WAITING_CHILD"
            return True
        result_path = Path(str(running.get("result_path", "")))
        if not result_path.is_file():
            raise ValueError("Recorded launcher died without a durable result")
        stage = next(item for item in self.plan["stages"] if item["id"] == running["stage_id"])
        self._classify_result(state, stage, read_json_shared(result_path))
        state.pop("running", None)
        return False

    def _publisher_command(self) -> list[str]:
        return [
            self.plan["python_executable"],
            "-m",
            "operational.final_refit.shared_resources",
            "publish",
            "--repository",
            self.plan["repository"],
            "--run-root",
            self.plan["run"],
            "--session",
            self.plan["resource_session"],
            "--snapshot",
            self.plan["resource_snapshot"],
            "--lock",
            self.plan["resource_publisher_lock"],
        ]

    def _runtime_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        repository = self.plan["repository"]
        source = str(Path(repository) / "src")
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (source, repository, existing) if item
        )
        environment["PYTHONUTF8"] = "1"
        return environment

    def _ensure_publisher(self, state: dict[str, Any]) -> None:
        if self._injected_monitor:
            return
        session = Path(self.plan["resource_session"])
        if not session.exists():
            create_shared_resource_session(
                session,
                repository=Path(self.plan["repository"]),
                run_root=self.run_root,
            )
        if self.monitor is None:
            self.monitor = SharedResourceMonitor(
                Path(self.plan["repository"]),
                session_path=session,
                snapshot_path=Path(self.plan["resource_snapshot"]),
                run_root=self.run_root,
            )
        _allowed, values = self.monitor.guard(
            self.run_root,
            checkpoint_reservation=CHECKPOINT_RESERVATION_BYTES,
            completed_updates=0,
        )
        if values.get("snapshot_valid") is True:
            state["publisher"] = {
                "status": "HEALTHY",
                "owned": True,
                "publisher_pid": values.get("publisher_pid"),
                "publisher_create_time": values.get("publisher_create_time"),
            }
            return
        lock = Path(self.plan["resource_publisher_lock"])
        if lock.exists():
            try:
                clear_stale_publisher_lock(
                    lock,
                    session_path=session,
                    repository=Path(self.plan["repository"]),
                    run_root=self.run_root,
                )
            except RuntimeError as error:
                failures = int(state.get("publisher_recovery_failures", 0)) + 1
                state["publisher_recovery_failures"] = failures
                state["publisher"] = {"status": "WAITING_EXISTING", "error": str(error)}
                if failures >= PUBLISHER_RECOVERY_LIMIT:
                    raise ValueError(
                        "Publisher identity could not be recovered within the bounded limit"
                    ) from error
                return
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        child = self.popen_factory(
            self._publisher_command(),
            cwd=self.plan["repository"],
            env=self._runtime_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self._publisher_child = child
        state["publisher"] = {"status": "STARTING", "owned": True, "launcher_pid": child.pid}
        deadline = self.monotonic() + PUBLISHER_START_SECONDS
        while self.monotonic() < deadline and child.poll() is None:
            _allowed, values = self.monitor.guard(
                self.run_root,
                checkpoint_reservation=CHECKPOINT_RESERVATION_BYTES,
                completed_updates=0,
            )
            if values.get("snapshot_valid") is True:
                state["publisher"] = {
                    "status": "HEALTHY",
                    "owned": True,
                    "launcher_pid": child.pid,
                    "publisher_pid": values.get("publisher_pid"),
                    "publisher_create_time": values.get("publisher_create_time"),
                }
                state["publisher_recovery_failures"] = 0
                return
            self.sleeper(0.25)
        raise ValueError("Owned publisher did not produce a valid authenticated snapshot")

    def _cleanup_owned_publisher(self, state: Mapping[str, Any]) -> None:
        publisher = state.get("publisher")
        child = self._publisher_child
        if not isinstance(publisher, dict) or publisher.get("owned") is not True:
            return
        if child is not None and child.poll() is None and hasattr(child, "terminate"):
            child.terminate()  # type: ignore[attr-defined]
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                return
        lock = Path(self.plan["resource_publisher_lock"])
        if child is None and lock.exists():
            lock_value = read_json_shared(lock)
            pid = publisher.get("publisher_pid")
            created = publisher.get("publisher_create_time")
            session = read_json_shared(Path(self.plan["resource_session"]))
            if (
                type(pid) is not int
                or isinstance(created, bool)
                or not isinstance(created, (int, float))
                or lock_value.get("pid") != pid
                or lock_value.get("create_time") != created
                or lock_value.get("session_sha256") != session.get("descriptor_sha256")
            ):
                return
            try:
                process = psutil.Process(int(pid))
                if float(process.create_time()) != float(created):
                    return
                process.terminate()
                process.wait(timeout=10.0)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (psutil.AccessDenied, psutil.TimeoutExpired, OSError, TypeError, ValueError):
                return
        if lock.exists():
            clear_stale_publisher_lock(
                lock,
                session_path=Path(self.plan["resource_session"]),
                repository=Path(self.plan["repository"]),
                run_root=self.run_root,
            )

    def _revalidate_prefix(self, state: dict[str, Any]) -> dict[str, Any] | None:
        first_incomplete = None
        blocked = False
        for stage in self.plan["stages"]:
            entry = state["stages"][stage["id"]]
            if entry["status"] == "COMPLETE":
                if blocked:
                    raise ValueError("Completed stage appears after an incomplete dependency")
                entry["receipt"] = validate_receipt(stage, allow_pause=False)
            elif first_incomplete is None:
                first_incomplete = stage
                blocked = True
            else:
                blocked = True
        return first_incomplete

    def _spawn_stage(self, state: dict[str, Any], stage: dict[str, Any]) -> None:
        entry = state["stages"][stage["id"]]
        attempt = int(entry["attempts"]) + 1
        result_path = self._result_path(stage, attempt)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists():
            raise ValueError(f"Refusing to overwrite launcher result: {result_path}")
        intent_nonce = uuid.uuid4().hex
        intent_path = result_path.with_suffix(".intent.json")
        stage_lock = self.run_root / "stage_locks" / f"{stage['id']}.lock"
        command = [
            self.plan["powershell_executable"],
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            self.plan["launcher"],
            "-Repository",
            self.plan["repository"],
            "-PythonExecutable",
            self.plan["python_executable"],
            "-Module",
            stage["module"],
            "-Run",
            self.plan["run"],
            "-StageId",
            stage["id"],
            "-ResultPath",
            str(result_path),
            "-IntentPath",
            str(intent_path),
            "-IntentNonce",
            intent_nonce,
            "-StageLockPath",
            str(stage_lock),
        ]
        if stage["rotation"] is not None:
            command += ["-Rotation", str(stage["rotation"])]
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        entry["attempts"] = attempt
        entry["status"] = "LAUNCHING"
        state["status"] = "LAUNCHING"
        state["running"] = {
            "stage_id": stage["id"],
            "attempt": attempt,
            "pid": None,
            "create_time": None,
            "result_path": str(result_path),
            "intent_path": str(intent_path),
            "intent_nonce": intent_nonce,
            "launch_intent_wall_time_ns": time.time_ns(),
            "stage_lock_path": str(stage_lock),
            "command": command,
        }
        self.save_state(state)
        child = self.popen_factory(
            command,
            cwd=self.plan["repository"],
            env=self._runtime_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        created = psutil.Process(child.pid).create_time()
        entry["status"] = "RUNNING"
        state["status"] = "RUNNING"
        state["running"] = {
            "stage_id": stage["id"],
            "attempt": attempt,
            "pid": child.pid,
            "create_time": created,
            "result_path": str(result_path),
            "intent_path": str(intent_path),
            "intent_nonce": intent_nonce,
            "stage_lock_path": str(stage_lock),
            "command": command,
        }
        self.save_state(state)
        while child.poll() is None:
            self.sleeper(5.0)
        child.wait()
        if not result_path.is_file():
            raise ValueError("Launcher exited without a durable result")
        self._classify_result(state, stage, read_json_shared(result_path))
        state.pop("running", None)

    def tick(self) -> dict[str, Any]:
        """Advance by at most one scientific stage and persist every transition."""
        state = self.load_state()
        if self.stop_path.exists():
            state["status"] = "MANUAL_STOP"
            self.save_state(state)
            return state
        if self._recover_running(state):
            self.save_state(state)
            return state
        legacy = read_json_shared(Path(self.plan["legacy_continuation_plan"]))
        ready, evidence = rgb_endpoints(legacy)
        state["rgb_endpoints"] = evidence
        if not ready:
            state["status"] = "WAITING_RGB"
            self.save_state(state)
            return state
        self._ensure_publisher(state)
        stage = self._revalidate_prefix(state)
        if stage is None:
            state["status"] = "COMPLETE"
            self.save_state(state)
            return state
        entry = state["stages"][stage["id"]]
        if entry["status"] in {"BLOCKED_ERROR", "MANUAL_STOP"}:
            state["status"] = entry["status"]
            self.save_state(state)
            return state
        if self.monotonic() < float(entry.get("eligible_monotonic", 0.0)):
            state["status"] = "WAITING_RESOURCES"
            self.save_state(state)
            return state
        if self.monitor is None:
            raise ValueError("Shared resource monitor was not initialized")
        allowed, resources = self.monitor.guard(
            self.run_root,
            checkpoint_reservation=CHECKPOINT_RESERVATION_BYTES,
            completed_updates=0,
        )
        state["resources"] = resources
        if not _resource_ready(allowed, resources):
            state["status"] = "WAITING_RESOURCES"
            self.save_state(state)
            return state
        self._spawn_stage(state, stage)
        if state["status"] == "RUNNING":
            state["status"] = "STAGE_COMPLETE"
        self.save_state(state)
        return state

    def run(self, *, once: bool = False) -> int:
        """Run durably until complete, manual stop or a non-retryable failure."""
        with queue_lock(Path(self.plan["lock_path"])):
            try:
                while True:
                    try:
                        state = self.tick()
                    except (OSError, ValueError, KeyError, psutil.Error) as error:
                        state = self.load_state()
                        state["status"] = "BLOCKED_INTEGRITY"
                        state["error"] = f"{type(error).__name__}: {error}"
                        self.save_state(state)
                    if once or state["status"] in TERMINAL:
                        return 0 if state["status"] == "COMPLETE" else 3
                    self.sleeper(WAIT_SECONDS)
            finally:
                self._cleanup_owned_publisher(self.load_state())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("repository", "run", "legacy-plan", "config", "launcher"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--python-executable", type=Path, required=True)
    prepare.add_argument("--powershell-executable", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        plan = read_json_shared(args.plan.resolve(strict=True))
        return ScientificOOFQueue(plan).run(once=args.once)
    plan = build_plan(
        args.repository,
        args.run,
        args.legacy_plan,
        args.config,
        args.launcher,
        python_executable=args.python_executable,
        powershell_executable=args.powershell_executable,
    )
    if args.output.exists() and read_json_shared(args.output) != plan:
        raise ValueError("Refusing to overwrite a different scientific OOF queue plan")
    atomic_write_json(args.output, plan)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "stages": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
