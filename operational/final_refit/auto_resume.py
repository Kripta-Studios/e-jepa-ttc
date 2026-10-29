"""External resource-pause supervisor; preserves the hash-bound training runtime."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from operational.final_refit.shared_resources import SharedResourceMonitor
from operational.rgb_port.accounting import read_json_shared

GIB = 1024**3
FITS = ("R_A5", "R_C2F")
WAITING_STATES = ("WAITING_MEMORY", "WAITING_MONITOR", "WAITING_RESOURCES")
RECOVERABLE_MONITOR_ERRORS = (
    "ValueError: snapshot is stale",
    "ValueError: publisher PID identity is not live",
)


def monitor_unavailable(resources: dict[str, Any]) -> bool:
    """Recognize availability failures, never signature/identity/clock corruption."""
    return (
        resources.get("snapshot_valid") is False
        and resources.get("run_root_binding_valid") is True
        and resources.get("snapshot_error") in RECOVERABLE_MONITOR_ERRORS
    )


def inventory_unreadable(resources: dict[str, Any]) -> bool:
    """Wait for a fresh complete scan after authenticated AccessDenied coverage gaps."""
    if not (
        resources.get("snapshot_valid") is True
        and resources.get("run_root_binding_valid") is True
        and resources.get("process_coverage_bounded") is False
        and resources.get("publisher_scan_error") is None
    ):
        return False
    failures = resources.get("inspection_failures")
    if not isinstance(failures, list) or not failures:
        return False
    if any(not isinstance(item, dict) for item in failures):
        return False
    unbounded = [item for item in failures if item.get("rss_upper_bound") == -1]
    return bool(unbounded) and all(item.get("error") == "AccessDenied" for item in unbounded)


def classify_exit(
    code: int,
    journal: dict[str, Any],
    receipt: dict[str, Any],
    pause: dict[str, Any],
    *,
    manual: bool,
) -> str:
    """Retry clean memory/monitor pauses; fail closed for every other exit."""
    if manual:
        return "MANUAL_STOP"
    completed = journal.get("completed_updates")
    clean = (
        isinstance(completed, int)
        and completed == journal.get("durable_updates") == receipt.get("completed_updates")
        and journal.get("pending_update_upper") == 0
        and receipt.get("complete_state") is True
    )
    if code == 0 and clean and receipt.get("scientific_endpoint") is True:
        return "COMPLETE" if receipt.get("status") == "COMPLETE" else "BLOCKED_ERROR"
    memory_low = any(
        isinstance(pause.get(key), int) and 0 <= pause[key] < 2 * GIB
        for key in ("host_available_bytes", "windows_commit_headroom_bytes")
    )
    if (
        code == 3
        and clean
        and receipt.get("status") == "PAUSED_RESOURCE"
        and pause.get("completed_updates") == completed
    ):
        if monitor_unavailable(pause) or inventory_unreadable(pause):
            return "WAITING_MONITOR"
        if pause.get("snapshot_valid") is True and memory_low:
            return "WAITING_MEMORY"
    return "BLOCKED_ERROR"


@dataclass
class ResumeGate:
    """Require continuous valid observations, not a single high memory sample."""

    stable_since: float | None = None

    def reset(self) -> None:
        self.stable_since = None

    def ready(self, now: float, allowed: bool, physical: int, commit: int | None) -> bool:
        if not allowed or physical < 2 * GIB or commit is None or commit < 2 * GIB:
            self.reset()
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now - self.stable_since >= 30


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish status atomically so readers never see truncated JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers/antivirus can briefly deny DELETE sharing on the target.
    # Keep the old complete JSON visible and retry only this transient failure.
    for attempt in range(41):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 40:
                raise
            time.sleep(0.25)


def file_digest(path: Path) -> str:
    """Hash a checkpoint without loading its tensors or allocating its size."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def owner_mutex() -> Iterator[None]:
    """Share the original launcher's Windows mutex and reject duplicate owners."""
    if os.name != "nt":
        raise RuntimeError("This supervisor requires Windows")
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, "Global\\EJEPA_FINAL_REFIT_RGB_PRODUCERS")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    held = False
    try:
        held = kernel.WaitForSingleObject(handle, 0) in (0, 0x80)
        if not held:
            raise RuntimeError("Another producer launcher owns the mutex")
        yield
    finally:
        if held:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def reject_live_producers() -> None:
    """Reject existing training processes; never terminate unrelated processes."""
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if "operational.final_refit.producers" in command:
            raise RuntimeError(f"Producer already active: {process.pid}")


class Supervisor:
    """Own one publisher and restart each cleanly paused producer independently."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run = args.run.resolve(strict=True)
        self.repo = Path(__file__).resolve().parents[2]
        self.stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.path = self.run / f"LAUNCHER_AUTO_{self.stamp}.json"
        self.env = {
            **os.environ,
            "PYTHONPATH": f"{self.repo / 'src'};{self.repo}",
            "PYTHONUTF8": "1",
        }
        self.children: dict[str, subprocess.Popen[bytes]] = {}
        self.states: dict[str, str] = {}
        self.retries: dict[str, int] = dict.fromkeys(FITS, 0)
        self.eligible_at: dict[str, float] = dict.fromkeys(FITS, 0.0)
        self.start_updates: dict[str, int] = {}
        self.logs: dict[str, str] = {}
        self.pids: dict[str, int] = {}
        self.exit_codes: dict[str, int] = {}
        self.publisher: subprocess.Popen[bytes] | None = None
        self.publisher_restarts = 0
        self.publisher_consecutive_restarts = 0
        self.monitor_unavailable_since: float | None = None
        self.gate = ResumeGate()

    def spawn(self, label: str, arguments: list[str]) -> subprocess.Popen[bytes]:
        stamp = time.time_ns()
        output = self.run / "logs" / f"{label}_auto_{stamp}.out.log"
        error = output.with_suffix(".err.log")
        with output.open("wb") as stdout, error.open("wb") as stderr:
            process = subprocess.Popen(
                [str(self.args.python), *arguments],
                cwd=self.repo,
                env=self.env,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        self.logs[f"{label}_stdout"] = str(output)
        self.logs[f"{label}_stderr"] = str(error)
        return process

    def evidence(self, fit: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        directory = self.run / "fits" / fit
        return (
            read_json_shared(directory / "UPDATE_JOURNAL.json"),
            read_json_shared(directory / "CHECKPOINT_RECEIPT.json"),
            read_json_shared(directory / "RESOURCE_PAUSE.json"),
        )

    def manual(self, fit: str) -> bool:
        return (self.run / "AUTO_RESUME_STOP").exists() or (
            self.run / "fits" / fit / "STOP_REQUEST"
        ).exists()

    def observation(self, event: str, fit: str) -> None:
        now = datetime.now(UTC)
        journal = read_json_shared(self.run / "fits" / fit / "UPDATE_JOURNAL.json")
        atomic_json(
            self.run / f"STATUS_AUTO_{time.time_ns()}.json",
            {
                "observed_utc": now.isoformat(),
                "event": event,
                "fits": {fit: {"current_update": journal["completed_updates"], **journal}},
            },
        )

    def publish_status(self, resources: dict[str, Any], *, exited: bool = False) -> None:
        active = [f for f, p in self.children.items() if p.poll() is None]
        value = {
            "schema": "final_refit_resource_auto_resume_v2",
            "status": "SUPERVISOR_EXITED" if exited else "SUPERVISING",
            "observed_utc": datetime.now(UTC).isoformat(),
            "launcher_pid": os.getpid(),
            "publisher_pid": self.publisher.pid if self.publisher else None,
            "producer_pids": self.pids,
            "producer_exit_codes": self.exit_codes,
            "fit_states": self.states,
            "logs": self.logs,
            "automatic_restart": "CLEAN_MEMORY_AND_MONITOR_AVAILABILITY_PAUSES",
            "publisher_restarts": self.publisher_restarts,
            "resume_origins": self.start_updates,
            "minimum_memory_bytes": 2 * GIB,
            "stable_seconds": 30,
            "poll_seconds": 10,
            "retry_counts": self.retries,
            "eligible_in_seconds": {
                f: max(0, t - time.monotonic()) for f, t in self.eligible_at.items()
            },
            "checkpoint_interval_updates": 100,
            "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "resources": {
                k: resources.get(k)
                for k in (
                    "host_available_bytes",
                    "windows_commit_headroom_bytes",
                    "snapshot_valid",
                    "snapshot_error",
                    "snapshot_age_ns",
                    "run_root_binding_valid",
                    "disk_free_after_reservation_bytes",
                    "publisher_scan_error",
                    "process_coverage_bounded",
                    "project_tree_rss_upper_bound_bytes",
                )
            },
            "unreadable_processes": [
                {"pid": item.get("pid"), "error": item.get("error")}
                for item in resources.get("inspection_failures", [])
                if item.get("rss_upper_bound") == -1
            ],
        }
        atomic_json(self.path, value)
        locator = read_json_shared(self.args.locator)
        locator.update(
            status="SUPERVISOR_EXITED" if exited else "AUTO_SUPERVISING",
            active_fits=active,
            paused_fits={f: {"status": s} for f, s in self.states.items() if f not in active},
            launcher_receipt_relative_to_this_file=os.path.relpath(
                self.path, self.args.locator.parent
            ).replace("\\", "/"),
            resume_policy=(
                "Automatic after clean memory/monitor pause and 30s valid resources >=2GiB; "
                "manual/error stops retained"
            ),
            status_verified_utc=value["observed_utc"],
            automatic_resume_stop_file=str(self.run / "AUTO_RESUME_STOP"),
        )
        atomic_json(self.args.locator, locator)

    def start_fit(self, fit: str) -> None:
        if fit in self.children and self.children[fit].poll() is None:
            raise RuntimeError(f"Duplicate launch attempt: {fit}")
        journal, _, _ = self.evidence(fit)
        self.start_updates[fit] = journal["completed_updates"]
        self.observation("START_FROM_CONFIRMED_CHECKPOINT", fit)
        process = self.spawn(
            fit,
            [
                "-m",
                "operational.final_refit.producers",
                "--source",
                str(self.args.source),
                "--teacher-manifest",
                str(self.args.teacher_manifest),
                "--config",
                str(self.args.config),
                "--run",
                str(self.run),
                "--recipe",
                str(self.run / "PRODUCER_PLAN.json"),
                "--device",
                "cuda",
                "--microbatch",
                "32",
                "--fit-id",
                fit,
            ],
        )
        self.children[fit] = process
        self.pids[fit] = process.pid
        self.exit_codes.pop(fit, None)
        self.states[fit] = "STARTING"
        self.gate.reset()

    def observe_children(self) -> None:
        for fit, process in self.children.items():
            if self.states[fit] not in ("STARTING", "RUNNING"):
                continue
            code = process.poll()
            if code is None:
                journal = read_json_shared(self.run / "fits" / fit / "UPDATE_JOURNAL.json")
                if journal["completed_updates"] > self.start_updates[fit]:
                    self.states[fit] = "RUNNING"
                continue
            self.states[fit] = classify_exit(code, *self.evidence(fit), manual=self.manual(fit))
            self.exit_codes[fit] = code
            self.observation(f"EXIT_{code}_{self.states[fit]}", fit)
            if self.states[fit] in WAITING_STATES:
                self.retries[fit] += 1
                self.eligible_at[fit] = time.monotonic() + min(
                    600, 60 * 2 ** min(self.retries[fit] - 1, 4)
                )
            self.gate.reset()

    def stop_publisher(self) -> None:
        """Stop only the owned publisher tree, only after every trainer exits."""
        if any(p.poll() is None for p in self.children.values()):
            raise RuntimeError("Cannot replace publisher while a producer is alive")
        if self.publisher is None or self.publisher.poll() is not None:
            return
        parent = psutil.Process(self.publisher.pid)
        for child in reversed(parent.children(recursive=True)):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        self.publisher.terminate()
        self.publisher.wait()

    def start_publisher(self) -> SharedResourceMonitor:
        """Clear a proven-dead lock, start publisher and reset client sequence state."""
        common = [
            "--repository",
            str(self.repo),
            "--run-root",
            str(self.run),
            "--session",
            str(self.run / "RESOURCE_SESSION.json"),
        ]
        clear = self.spawn(
            "clear_lock",
            [
                "-m",
                "operational.final_refit.shared_resources",
                "clear-stale-lock",
                *common,
                "--lock",
                str(self.run / "RESOURCE_PUBLISHER.lock"),
            ],
        )
        if clear.wait(timeout=30) != 0:
            raise RuntimeError("Publisher lock cannot be cleared; inspect clear_lock stderr")
        self.publisher = self.spawn(
            "publisher",
            [
                "-m",
                "operational.final_refit.shared_resources",
                "publish",
                *common,
                "--snapshot",
                str(self.run / "RESOURCE_SNAPSHOT.json"),
                "--lock",
                str(self.run / "RESOURCE_PUBLISHER.lock"),
                "--interval-seconds",
                "5",
            ],
        )
        return SharedResourceMonitor(
            self.repo,
            session_path=self.run / "RESOURCE_SESSION.json",
            snapshot_path=self.run / "RESOURCE_SNAPSHOT.json",
            run_root=self.run,
        )

    def recover_publisher(
        self, resources: dict[str, Any], now: float
    ) -> SharedResourceMonitor | None:
        """Recover dead/stalled inventory after trainers save; retain the original TTL."""
        dead = self.publisher is None or self.publisher.poll() is not None
        unavailable = dead or monitor_unavailable(resources)
        if not unavailable:
            self.monitor_unavailable_since = None
            return None
        self.gate.reset()
        if self.monitor_unavailable_since is None:
            self.monitor_unavailable_since = now
        if any(p.poll() is None for p in self.children.values()):
            return None
        if not dead and now - self.monitor_unavailable_since < 60:
            return None
        if self.publisher_consecutive_restarts >= 3:
            raise RuntimeError("Publisher failed three recoveries without stable valid resources")
        self.stop_publisher()
        self.publisher_restarts += 1
        self.publisher_consecutive_restarts += 1
        self.monitor_unavailable_since = None
        return self.start_publisher()

    def resume_manual(self, previous: dict[str, Any]) -> None:
        """Consume an explicit one-time manual resume after validating both saved fits."""
        plan = read_json_shared(self.run / "PRODUCER_PLAN.json")
        for name, expected in plan["executed_source_sha256"].items():
            if file_digest(self.repo / name) != expected:
                raise RuntimeError(f"Executed source changed: {name}")
        checkpoints = {}
        resumed_states = {}
        for fit in FITS:
            journal, receipt, pause = self.evidence(fit)
            valid_complete = (
                previous.get("fit_states", {}).get(fit) == "COMPLETE"
                and classify_exit(
                    previous.get("producer_exit_codes", {}).get(fit, -1),
                    journal,
                    receipt,
                    pause,
                    manual=False,
                )
                == "COMPLETE"
                and journal.get("identity_sha256") == receipt.get("identity_sha256")
                and journal.get("execution_binding_sha256")
                == receipt.get("execution_binding_sha256")
            )
            healthy_or_unavailable = (
                pause.get("snapshot_valid") is True
                and pause.get("run_root_binding_valid") is True
                and pause.get("publisher_scan_error") is None
            ) or monitor_unavailable(pause)
            valid_manual = (
                previous.get("fit_states", {}).get(fit) == "MANUAL_STOP"
                and previous.get("producer_exit_codes", {}).get(fit) == 3
                and self.manual(fit)
                and receipt.get("status") == "PAUSED_RESOURCE"
                and receipt.get("complete_state") is True
                and isinstance(journal.get("completed_updates"), int)
                and journal["completed_updates"]
                == journal.get("durable_updates")
                == receipt.get("completed_updates")
                == pause.get("completed_updates")
                and journal.get("pending_update_upper") == 0
                and journal.get("identity_sha256") == receipt.get("identity_sha256")
                and journal.get("execution_binding_sha256")
                == receipt.get("execution_binding_sha256")
                and healthy_or_unavailable
            )
            if not (valid_complete or valid_manual):
                raise RuntimeError(f"Cannot explicitly resume unverified manual stop: {fit}")
            checkpoint = Path(receipt["checkpoint_path"]).resolve(strict=True)
            if not checkpoint.is_relative_to((self.run / "fits" / fit).resolve()):
                raise RuntimeError(f"Checkpoint outside its fit: {fit}")
            if file_digest(checkpoint) != receipt["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch: {fit}")
            checkpoints[fit] = {
                "completed_updates": journal["completed_updates"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
            resumed_states[fit] = "COMPLETE" if valid_complete else "WAITING_RESOURCES"
        # All validations precede removing any stop marker. Retain each marker as evidence.
        archived = []
        markers = [self.run / "AUTO_RESUME_STOP"] + [
            self.run / "fits" / fit / "STOP_REQUEST" for fit in FITS
        ]
        for marker in markers:
            if marker.exists():
                destination = marker.with_name(f"{marker.name}.resumed_{time.time_ns()}")
                marker.rename(destination)
                archived.append(str(destination))
        atomic_json(
            self.run / f"MANUAL_RESUME_{self.stamp}.json",
            {
                "authorized_by": "Explicit invocation with --resume-manual",
                "observed_utc": datetime.now(UTC).isoformat(),
                "checkpoints": checkpoints,
                "fit_states": resumed_states,
                "archived_stop_markers": archived,
                "previous_launcher_pid": previous.get("launcher_pid"),
            },
        )
        self.states = resumed_states

    def execute(self) -> None:
        """Run until terminal/manual states; resource waits keep the supervisor alive."""
        previous = read_json_shared(self.args.locator)
        previous_receipt = read_json_shared(
            self.args.locator.parent / previous["launcher_receipt_relative_to_this_file"]
        )
        for fit in FITS:
            codes = previous_receipt.get("producer_exit_codes", {})
            self.exit_codes[fit] = codes.get(fit, -1)
            self.states[fit] = classify_exit(
                codes.get(fit, -1), *self.evidence(fit), manual=self.manual(fit)
            )
        if self.args.resume_manual:
            self.resume_manual(previous_receipt)
        try:
            monitor = self.start_publisher()
            while True:
                self.observe_children()
                for fit in FITS:
                    if self.states[fit] in WAITING_STATES and self.manual(fit):
                        self.states[fit] = "MANUAL_STOP"
                if (self.run / "AUTO_RESUME_STOP").exists():
                    break
                allowed, resources = monitor.guard(
                    self.run, checkpoint_reservation=GIB, completed_updates=0
                )
                replacement = self.recover_publisher(resources, time.monotonic())
                if replacement is not None:
                    monitor = replacement
                    self.publish_status(resources)
                    time.sleep(10)
                    continue
                ready = self.gate.ready(
                    time.monotonic(),
                    allowed,
                    resources["host_available_bytes"],
                    resources["windows_commit_headroom_bytes"],
                )
                if ready:
                    self.publisher_consecutive_restarts = 0
                if ready and "STARTING" not in self.states.values():
                    for fit in FITS:
                        if (
                            self.states[fit] in WAITING_STATES
                            and time.monotonic() >= self.eligible_at[fit]
                        ):
                            self.start_fit(fit)
                            break  # Stagger cold starts; eventual training remains concurrent.
                self.publish_status(resources)
                if all(
                    s in ("COMPLETE", "MANUAL_STOP", "BLOCKED_ERROR") for s in self.states.values()
                ):
                    break
                time.sleep(10)
        finally:
            # Never terminate a training process. Ask it to save, then wait for its exit.
            for fit, process in self.children.items():
                if process.poll() is None:
                    stop = self.run / "fits" / fit / "STOP_REQUEST"
                    if not stop.exists():
                        stop.write_text("Supervisor shutdown: save at update boundary.\n")
            for process in self.children.values():
                process.wait()
            self.observe_children()
            self.stop_publisher()
            self.publish_status({}, exited=True)


def main() -> int:
    """Launch with explicit existing data/config bindings and an explicit venv."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "locator", "python", "source", "teacher-manifest", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--resume-manual",
        action="store_true",
        help="Explicitly resume both verified manual stops; archive their stop markers once.",
    )
    args = parser.parse_args()
    for value in vars(args).values():
        if isinstance(value, Path) and not value.exists():
            parser.error(f"Required path missing: {value}")
    with owner_mutex():
        reject_live_producers()
        Supervisor(args).execute()
    return 0


if __name__ == "__main__":
    sys.exit(main())
