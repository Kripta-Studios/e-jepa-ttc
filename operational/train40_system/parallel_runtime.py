"""Authenticated two-arm runtime guards for the bounded TRAIN40 parallel pilot."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import psutil

from operational.efficient_context.common import digest
from operational.train40_system import resource_monitor
from operational.train40_system.durable_io import atomic_json

AGGREGATE_RSS_LIMIT_BYTES = 24 * 1024**3
HOST_AVAILABLE_MIN_BYTES = 2 * 1024**3
DISK_FREE_MIN_BYTES = 20_000_000_000
PHYSICAL_UPDATE_LIMIT = 240_000
PILOT_UPDATES_PER_ARM = 1_000
REGISTRY_NAME = "PARALLEL_PILOT_REGISTRY.json"
LEDGER_NAME = "PARALLEL_PILOT_LEDGER.json"
MODULE = "operational.train40_system.parallel_engine"


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object or fail closed."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


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
        and receipt.get("status") == "PAUSED_RESOURCE"
        and receipt.get("full_optimizer_scheduler_sampler_and_all_RNG") is True
        and checkpoint.is_file()
        and digest(checkpoint) == receipt.get("sha256")
    )


def publish_upfront_reservation(output: Path, token: str) -> dict[str, Any]:
    """Reserve exactly 1,000 future updates per restored arm before either advances."""
    registry = read_object(output / REGISTRY_NAME)
    if registry.get("token") != token or registry.get("phase") != "BOTH_AT_BARRIER":
        raise ValueError("Both authenticated arms must be at the pre-update barrier")
    budget = physical_budget(output)
    if budget["pending"]:
        raise ValueError("Cannot reserve over a pending optimizer update")
    if budget["physical"] + 2 * PILOT_UPDATES_PER_ARM > PHYSICAL_UPDATE_LIMIT:
        raise ValueError("Parallel reservation exceeds the physical update cap")
    arms: dict[str, Any] = {}
    for arm in ("a5", "c2f"):
        journal = arm_journal(output, arm)
        committed = int(journal["committed_updates"])
        if int(journal["pending_update_upper"]):
            raise ValueError(f"{arm} has a pending optimizer update")
        if int(journal["durable_updates"]) != committed:
            raise ValueError(f"{arm} baseline is not a full durable checkpoint")
        recovery = int(journal["recovery_upper"])
        if arm == "a5" and recovery != 30:
            raise ValueError("A5 external-I/O recovery must be exactly accounted as 30")
        limit = int(journal["contract"]["updates_limit"])
        target = committed + PILOT_UPDATES_PER_ARM
        if target > limit:
            raise ValueError(f"{arm} fixed pilot target exceeds its scientific endpoint")
        arms[arm] = {
            "baseline_committed": committed,
            "baseline_durable": int(journal["durable_updates"]),
            "baseline_recovery_upper": recovery,
            "target_committed": target,
        }
    ledger = {
        "schema": "train40_parallel_pilot_ledger_v1",
        "token": token,
        "status": "ARMED",
        "updates_per_arm": PILOT_UPDATES_PER_ARM,
        "combined_updates": 2 * PILOT_UPDATES_PER_ARM,
        "physical_updates_before_reservation": budget["physical"],
        "physical_updates_reserved_upper": budget["physical"] + 2 * PILOT_UPDATES_PER_ARM,
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

    def _paired_sample(self, required_arms: tuple[str, ...]) -> dict[str, Any]:
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
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                if not arm_target_complete(self.output, arm, self.token):
                    raise
                completed.append(arm)
            else:
                expected[(process.pid, process.create_time())] = process
        observed_parallel: set[tuple[int, float]] = set()
        for process in psutil.process_iter(["name"]):
            if not (process.info.get("name") or "").lower().startswith("python"):
                continue
            try:
                parts = process.cmdline()
                if MODULE in parts:
                    observed_parallel.add((process.pid, process.create_time()))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                raise RuntimeError("Parallel process inventory is incomplete") from error
        if observed_parallel != set(expected):
            raise RuntimeError("Parallel process allowlist differs")
        trees: dict[tuple[int, float], psutil.Process] = {}
        for process in expected.values():
            trees.update(tree_identities(process))
        aggregate_rss = sum(process.memory_info().rss for process in trees.values())
        sample = {
            "aggregate_tree_rss_bytes": aggregate_rss,
            "aggregate_tree_processes": len(trees),
            "host_available_bytes": int(psutil.virtual_memory().available),
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

    def guard(self, output: Path) -> tuple[bool, dict[str, Any]]:
        """Run frozen admission, then enforce the paired barrier and fixed arm target."""
        begun = time.perf_counter()
        allowed, telemetry = super().guard(output)
        self._guard_invocations += 1
        # A5's first guard is the admitted zero-optimizer CUDA-graph warmup gate.
        prewarm = self.arm == "a5" and self._guard_invocations == 1
        reasons = list(telemetry.get("reasons", []))
        ledger_path = self.output / LEDGER_NAME
        if not prewarm:
            if not ledger_path.is_file():
                atomic_json(
                    self.output / f"PARALLEL_{self.arm.upper()}_BARRIER.json",
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
    "PILOT_UPDATES_PER_ARM",
    "REGISTRY_NAME",
    "ParallelResourceMonitor",
    "arm_journal",
    "physical_budget",
    "process_identity",
    "publish_upfront_reservation",
    "read_object",
]
