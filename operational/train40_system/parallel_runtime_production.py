"""Authenticated two-arm runtime guards for the TRAIN40 parallel production."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import psutil

from operational.efficient_context.common import digest
from operational.train40_system import resource_monitor
from operational.train40_system.durable_io import atomic_json

AGGREGATE_RSS_LIMIT_BYTES = 24 * 1024**3
HOST_AVAILABLE_MIN_BYTES = 2 * 1024**3
DISK_FREE_MIN_BYTES = 20 * 1024**3
PHYSICAL_UPDATE_LIMIT = 240_000
SCIENTIFIC_ENDPOINTS = {"a5": 49_932, "c2f": 49_932}
REGISTRY_NAME = "PARALLEL_PRODUCTION_REGISTRY.json"
LEDGER_NAME = "PARALLEL_PRODUCTION_LEDGER.json"
MODULE = "operational.train40_system.parallel_engine_production"
PARALLEL_ENGINE_FAMILY = "operational.train40_system.parallel_engine"


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object or fail closed."""
    for attempt in range(6):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.005 * (2**attempt))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def is_parallel_engine_command(parts: list[str] | tuple[str, ...]) -> bool:
    """Recognize every additive parallel-engine generation for exclusive allowlisting."""
    if "-m" not in parts:
        return False
    index = parts.index("-m") + 1
    if index >= len(parts):
        return False
    module = str(parts[index])
    return module == PARALLEL_ENGINE_FAMILY or module.startswith(PARALLEL_ENGINE_FAMILY + "_")


def require_exact_parallel_family(
    observed: set[tuple[int, float]], expected: set[tuple[int, float]]
) -> None:
    """Reject missing authenticated shim/workers and every additional family process."""
    if observed != expected:
        raise RuntimeError("Parallel process allowlist differs")


def process_identity(pid: int) -> tuple[int, float]:
    """Return a PID-reuse-safe live process identity."""
    process = psutil.Process(pid)
    return process.pid, process.create_time()


def command_identity(process: psutil.Process) -> tuple[str, str, str, str]:
    """Extract module, output and arm from an authenticated child command."""
    parts = process.cmdline()
    if (
        "-m" not in parts
        or "--output" not in parts
        or "--arm" not in parts
        or "--token" not in parts
    ):
        raise ValueError("Parallel child command is incomplete")
    return (
        parts[parts.index("-m") + 1],
        str(Path(parts[parts.index("--output") + 1]).resolve()),
        parts[parts.index("--arm") + 1],
        parts[parts.index("--token") + 1],
    )


def validate_child(row: dict[str, Any], output: Path, arm: str) -> psutil.Process:
    """Authenticate one exact child by PID, creation time and complete route."""
    if row.get("arm") != arm or row.get("module") != MODULE:
        raise ValueError("Parallel child route differs")
    process = psutil.Process(int(row["pid"]))
    if abs(process.create_time() - float(row["create_time"])) > 0.001:
        raise ValueError("Parallel child PID identity differs")
    if command_identity(process) != (MODULE, str(output.resolve()), arm, str(row["token"])):
        raise ValueError("Parallel child command differs")
    return process


def validate_launcher_identity(row: dict[str, Any], output: Path, arm: str) -> psutil.Process:
    """Authenticate one exact Windows venv launcher identity and route."""
    launcher_row = row.get("launcher")
    if not isinstance(launcher_row, dict):
        raise ValueError("Parallel launcher identity is missing")
    launcher = psutil.Process(int(launcher_row["pid"]))
    if abs(launcher.create_time() - float(launcher_row["create_time"])) > 0.001:
        raise ValueError("Parallel launcher PID identity differs")
    if command_identity(launcher) != (MODULE, str(output.resolve()), arm, str(row["token"])):
        raise ValueError("Parallel launcher command differs")
    return launcher


def validate_launcher(
    row: dict[str, Any], actual: psutil.Process, output: Path, arm: str
) -> psutil.Process:
    """Require the READY interpreter below its authenticated Windows venv shim."""
    launcher = validate_launcher_identity(row, output, arm)
    descendants = tree_identities(launcher)
    actual_identity = (actual.pid, actual.create_time())
    same_process = actual_identity == (launcher.pid, launcher.create_time())
    if not same_process and actual_identity not in descendants:
        raise ValueError("READY process is not descended from its authenticated launcher")
    return launcher


def tree_identities(root: psutil.Process) -> dict[tuple[int, float], psutil.Process]:
    """Collect one live process tree with PID-reuse-safe identities."""
    result: dict[tuple[int, float], psutil.Process] = {}
    for process in (root, *root.children(recursive=True)):
        try:
            fresh = psutil.Process(process.pid)
            identity = (fresh.pid, fresh.create_time())
            if identity == (process.pid, process.create_time()):
                result[identity] = fresh
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return result


def physical_budget(output: Path) -> dict[str, int]:
    """Read the same global physical accounting used by frozen trainers."""
    authorization = read_object(output / "AUTHORIZATION.json")
    technical = read_object(output / "TECHNICAL_ACCOUNTING.json")
    physical = int(authorization["previous_physical_execution_upper"])
    physical += int(technical["synthetic_optimizer_updates"])
    recovery = 0
    pending = 0
    for path in (output / "fits").glob("*/UPDATE_JOURNAL.json"):
        journal = read_object(path)
        physical += int(journal["committed_updates"])
        physical += int(journal["recovery_upper"])
        physical += int(journal["pending_update_upper"])
        recovery += int(journal["recovery_upper"])
        pending += int(journal["pending_update_upper"])
    return {"physical": physical, "recovery": recovery, "pending": pending}


def arm_journal(output: Path, arm: str) -> dict[str, Any]:
    """Read one arm's optimizer-boundary journal."""
    return read_object(output / "fits" / f"{arm}_seed7" / "UPDATE_JOURNAL.json")


def validate_full_baseline(output: Path, arm: str, journal: dict[str, Any]) -> str:
    """Bind a resumable full checkpoint to the current durable journal boundary."""
    fit = output / "fits" / f"{arm}_seed7"
    receipt = read_object(fit / "CHECKPOINT_RECEIPT.json")
    checkpoint = fit / "checkpoint_last.pt"
    committed = int(journal["committed_updates"])
    recovery = int(journal["recovery_upper"])
    if (
        int(journal["durable_updates"]) != committed
        or int(journal["pending_update_upper"]) != 0
        or int(receipt["committed_updates"]) != committed
        or int(receipt["recovery_upper"]) != recovery
        or receipt.get("status") not in {"PAUSED_RESOURCE", "COMPLETE"}
        or receipt.get("full_optimizer_scheduler_sampler_and_all_RNG") is not True
        or not checkpoint.is_file()
        or digest(checkpoint) != receipt.get("sha256")
    ):
        raise ValueError(f"{arm} baseline is not an authenticated full checkpoint")
    return str(receipt["sha256"])


def arm_target_complete(output: Path, arm: str, token: str) -> bool:
    """Authenticate a normally exited arm by its exact target and full checkpoint."""
    ledger_path = output / LEDGER_NAME
    if not ledger_path.is_file():
        return False
    ledger = read_object(ledger_path)
    if ledger.get("token") != token:
        return False
    target = int(ledger["arms"][arm]["target_committed"])
    journal = arm_journal(output, arm)
    receipt_path = output / "fits" / f"{arm}_seed7" / "CHECKPOINT_RECEIPT.json"
    if not receipt_path.is_file():
        return False
    receipt = read_object(receipt_path)
    checkpoint = output / "fits" / f"{arm}_seed7" / "checkpoint_last.pt"
    baseline_recovery = int(ledger["arms"][arm]["baseline_recovery_upper"])
    return (
        int(journal["committed_updates"]) == target
        and int(journal["durable_updates"]) == target
        and int(journal["pending_update_upper"]) == 0
        and int(receipt["committed_updates"]) == target
        and int(journal["recovery_upper"]) == baseline_recovery
        and int(receipt["recovery_upper"]) == baseline_recovery
        and receipt.get("status") in {"PAUSED_RESOURCE", "COMPLETE"}
        and receipt.get("full_optimizer_scheduler_sampler_and_all_RNG") is True
        and checkpoint.is_file()
        and digest(checkpoint) == receipt.get("sha256")
    )


def publish_upfront_reservation(output: Path, token: str) -> dict[str, Any]:
    """Reserve each arm's exact remaining work through its scientific endpoint."""
    registry = read_object(output / REGISTRY_NAME)
    if registry.get("token") != token or registry.get("phase") != "BOTH_AT_BARRIER":
        raise ValueError("Both authenticated arms must be at the pre-update barrier")
    budget = physical_budget(output)
    if budget["pending"]:
        raise ValueError("Cannot reserve over a pending optimizer update")
    arms: dict[str, Any] = {}
    combined_remaining = 0
    for arm in ("a5", "c2f"):
        journal = arm_journal(output, arm)
        committed = int(journal["committed_updates"])
        if int(journal["pending_update_upper"]):
            raise ValueError(f"{arm} has a pending optimizer update")
        if int(journal["durable_updates"]) != committed:
            raise ValueError(f"{arm} baseline is not a full durable checkpoint")
        recovery = int(journal["recovery_upper"])
        if recovery < 0 or budget["recovery"] > resource_monitor.RECOVERY_LIMIT:
            raise ValueError("Accounted recovery exceeds the campaign recovery budget")
        checkpoint_sha256 = validate_full_baseline(output, arm, journal)
        limit = int(journal["contract"]["updates_limit"])
        target = SCIENTIFIC_ENDPOINTS[arm]
        if limit != target or committed > target:
            raise ValueError(f"{arm} scientific endpoint contract differs")
        combined_remaining += target - committed
        arms[arm] = {
            "baseline_committed": committed,
            "baseline_durable": int(journal["durable_updates"]),
            "baseline_recovery_upper": recovery,
            "baseline_checkpoint_sha256": checkpoint_sha256,
            "target_committed": target,
        }
    if budget["physical"] + combined_remaining > PHYSICAL_UPDATE_LIMIT:
        raise ValueError("Parallel reservation exceeds the physical update cap")
    ledger = {
        "schema": "train40_parallel_production_ledger_v1",
        "token": token,
        "status": "ARMED",
        "scientific_endpoints": SCIENTIFIC_ENDPOINTS,
        "combined_remaining_updates": combined_remaining,
        "physical_updates_before_reservation": budget["physical"],
        "physical_updates_reserved_upper": budget["physical"] + combined_remaining,
        "recovery_upper_at_reservation": budget["recovery"],
        "arms": arms,
    }
    atomic_json(output / LEDGER_NAME, ledger)
    return ledger


class ParallelResourceMonitor(resource_monitor.ResourceMonitor):
    """Retain frozen guards and add authenticated peer, quota and aggregate-RAM checks."""

    def __init__(self, output: Path) -> None:
        super().__init__(output)
        self.arm = os.environ.get("TRAIN40_PARALLEL_ARM", "")
        self.token = os.environ.get("TRAIN40_PARALLEL_TOKEN", "")
        if self.arm not in {"a5", "c2f"} or not self.token:
            raise ValueError("Parallel monitor requires scoped arm and token")
        self._parallel_last_scan = 0.0
        self._parallel_latest: dict[str, Any] | None = None
        self._parallel_latest_arms: tuple[str, ...] = ()
        self._guard_invocations = 0
        self._parallel_guard_total_seconds = 0.0

    def _paired_sample_once(self, required_arms: tuple[str, ...]) -> dict[str, Any]:
        now = time.monotonic()
        if (
            self._parallel_latest is not None
            and self._parallel_latest_arms == required_arms
            and now - self._parallel_last_scan < 5.0
        ):
            return self._parallel_latest
        registry = read_object(self.output / REGISTRY_NAME)
        if registry.get("token") != self.token:
            raise ValueError("Parallel registry token differs")
        children = registry.get("children", {})
        expected: dict[tuple[int, float], psutil.Process] = {}
        completed: list[str] = []
        for arm in required_arms:
            try:
                process = validate_child(children[arm], self.output, arm)
                launcher = validate_launcher(children[arm], process, self.output, arm)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                if not arm_target_complete(self.output, arm, self.token):
                    raise
                completed.append(arm)
                try:
                    launcher = validate_launcher_identity(children[arm], self.output, arm)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                else:
                    expected[(launcher.pid, launcher.create_time())] = launcher
            else:
                expected[(process.pid, process.create_time())] = process
                expected[(launcher.pid, launcher.create_time())] = launcher
        observed_parallel: set[tuple[int, float]] = set()
        for process in psutil.process_iter(["name"]):
            if not (process.info.get("name") or "").lower().startswith("python"):
                continue
            try:
                parts = process.cmdline()
                if is_parallel_engine_command(parts):
                    observed_parallel.add((process.pid, process.create_time()))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied as error:
                raise RuntimeError("Parallel process inventory is incomplete") from error
        require_exact_parallel_family(observed_parallel, set(expected))
        trees: dict[tuple[int, float], psutil.Process] = {}
        for process in expected.values():
            trees.update(tree_identities(process))
        aggregate_rss = sum(process.memory_info().rss for process in trees.values())
        sample = {
            "aggregate_tree_rss_bytes": aggregate_rss,
            "aggregate_tree_processes": len(trees),
            "host_available_bytes": int(psutil.virtual_memory().available),
            "disk_free_bytes": int(shutil.disk_usage(self.output).free),
            "peer_identities": [
                {"pid": pid, "create_time": created} for pid, created in sorted(expected)
            ],
            "completed_arms": completed,
            "sampled_monotonic": now,
        }
        self._parallel_latest = sample
        self._parallel_latest_arms = required_arms
        self._parallel_last_scan = now
        return sample

    def _paired_sample(self, required_arms: tuple[str, ...]) -> dict[str, Any]:
        """Retry fresh inventories across legitimate Windows process-exit races."""
        last_error: BaseException | None = None
        for attempt in range(4):
            try:
                return self._paired_sample_once(required_arms)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, RuntimeError) as error:
                if isinstance(error.__cause__, psutil.AccessDenied):
                    raise
                last_error = error
                if attempt < 3:
                    time.sleep(0.01 * (2**attempt))
        assert last_error is not None
        raise last_error

    def _repair_transient_frozen_inventory(
        self, allowed: bool, telemetry: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Replace only a stale dead-Python inventory result with a fresh clean scan."""
        reasons = list(telemetry.get("reasons", []))
        if "RESOURCE_MONITOR_INVENTORY_INCOMPLETE" not in reasons:
            return allowed, telemetry
        root = psutil.Process()
        fresh: dict[str, Any] | None = None
        for attempt in range(4):
            fresh = resource_monitor._collect_sample(root.pid, root.create_time(), self.output)
            fresh_admissible = (
                not fresh.get("inventory_unreadable_python_pids")
                and not fresh.get("other_heavy_processes")
                and int(fresh.get("tree_rss_bytes", resource_monitor.TREE_RSS_LIMIT_BYTES))
                < resource_monitor.TREE_RSS_LIMIT_BYTES
                and int(fresh.get("host_available_bytes", 0))
                >= resource_monitor.HOST_AVAILABLE_MIN_BYTES
                and int(fresh.get("disk_free_bytes", 0))
                >= resource_monitor.DISK_FREE_MIN_BYTES
            )
            if fresh_admissible:
                reasons.remove("RESOURCE_MONITOR_INVENTORY_INCOMPLETE")
                repaired = {
                    **telemetry,
                    "reasons": reasons,
                    "inventory_repair": {
                        "attempt": attempt + 1,
                        "processes_seen": fresh.get("processes_seen"),
                        "python_cmdline_reads": fresh.get("python_cmdline_reads"),
                    },
                }
                return not reasons, repaired
            if attempt < 3:
                time.sleep(0.01 * (2**attempt))
        return allowed, {**telemetry, "inventory_repair": fresh}

    def guard(self, output: Path) -> tuple[bool, dict[str, Any]]:
        """Run frozen admission, then enforce the paired barrier and fixed arm target."""
        begun = time.perf_counter()
        allowed, telemetry = super().guard(output)
        allowed, telemetry = self._repair_transient_frozen_inventory(allowed, telemetry)
        self._guard_invocations += 1
        # A5's first guard is the admitted zero-optimizer CUDA-graph warmup gate.
        prewarm = self.arm == "a5" and self._guard_invocations == 1
        reasons = list(telemetry.get("reasons", []))
        ledger_path = self.output / LEDGER_NAME
        if not prewarm:
            if not ledger_path.is_file():
                atomic_json(
                    self.output / f"PARALLEL_{self.arm.upper()}_PRODUCTION_BARRIER.json",
                    {"token": self.token, "arm": self.arm, "pid": os.getpid()},
                )
                deadline = time.monotonic() + 180.0
                while time.monotonic() < deadline and not ledger_path.is_file():
                    time.sleep(0.05)
                if not ledger_path.is_file():
                    reasons.append("PARALLEL_BARRIER_TIMEOUT")
                else:
                    # The initial frozen snapshot can age while waiting at the barrier.
                    allowed, telemetry = super().guard(output)
                    allowed, telemetry = self._repair_transient_frozen_inventory(allowed, telemetry)
                    reasons = list(telemetry.get("reasons", []))
        try:
            paired = self._paired_sample(("a5",) if prewarm else ("a5", "c2f"))
        except BaseException as error:
            paired = {"error": f"{type(error).__name__}: {error}"}
            reasons.append("PARALLEL_AUTHENTICATION_FAILED")
        else:
            if int(paired["aggregate_tree_rss_bytes"]) >= AGGREGATE_RSS_LIMIT_BYTES:
                reasons.append("PARALLEL_AGGREGATE_RAM_BOUNDARY")
            if int(paired["host_available_bytes"]) < HOST_AVAILABLE_MIN_BYTES:
                reasons.append("RAM_RESOURCE_BOUNDARY")
            if int(paired["disk_free_bytes"]) < DISK_FREE_MIN_BYTES:
                reasons.append("CHECKPOINT_DISK_RESERVE")
        if not prewarm and ledger_path.is_file():
            ledger = read_object(ledger_path)
            if ledger.get("token") != self.token or ledger.get("status") != "ARMED":
                reasons.append("PARALLEL_LEDGER_INVALID")
            else:
                journal = arm_journal(self.output, self.arm)
                target = int(ledger["arms"][self.arm]["target_committed"])
                committed = int(journal["committed_updates"])
                if committed >= target:
                    reasons.append("PARALLEL_ARM_TARGET_REACHED")
        telemetry = {**telemetry, "parallel": paired, "parallel_reasons": reasons}
        telemetry["reasons"] = list(dict.fromkeys(reasons))
        elapsed = time.perf_counter() - begun
        self._parallel_guard_total_seconds += elapsed
        telemetry["parallel_guard_seconds"] = elapsed
        return allowed and not telemetry["reasons"], telemetry

    def snapshot(self) -> dict[str, Any]:
        """Add paired state to the frozen monitor snapshot."""
        return {
            **super().snapshot(),
            "parallel_arm": self.arm,
            "parallel_token": self.token,
            "parallel_guard_invocations": self._guard_invocations,
            "parallel_guard_total_seconds": self._parallel_guard_total_seconds,
            "parallel_latest": self._parallel_latest,
        }


__all__ = [
    "LEDGER_NAME",
    "MODULE",
    "SCIENTIFIC_ENDPOINTS",
    "REGISTRY_NAME",
    "ParallelResourceMonitor",
    "arm_journal",
    "physical_budget",
    "process_identity",
    "publish_upfront_reservation",
    "read_object",
    "is_parallel_engine_command",
    "require_exact_parallel_family",
    "validate_launcher",
    "validate_launcher_identity",
    "validate_full_baseline",
]
