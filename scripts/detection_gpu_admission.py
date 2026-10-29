"""Prepare, execute manually, or verify the serial detection GPU admission."""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.campaign_training import sha256_file
from e_jepa_ttc.detection.gpu_admission_harness import (
    ARM_NAMES,
    SCHEMA,
    Tolerances,
    bind_plan,
    load_plan,
    validate_checkpoint_lineage,
    validate_initial_state,
    verify_metrics,
)


def _write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite admission artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(draft_path: Path, output: Path) -> dict[str, Any]:
    """Bind an explicit draft to current immutable artifact bytes."""
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if not isinstance(draft, dict) or draft.get("schema") != SCHEMA:
        raise ValueError("Draft has unknown admission schema")
    if draft.get("arm_order") != list(ARM_NAMES) or draft.get("updates") not in (16, 32):
        raise ValueError("Draft arm order or update budget is unsupported")
    if "tolerances" not in draft:
        draft["tolerances"] = Tolerances().__dict__
    if "graphics_baseline" in draft:
        raise ValueError("Graphics baseline is captured by prepare, not supplied by the draft")
    draft["graphics_baseline"] = _graphics_baseline()
    Tolerances(**draft["tolerances"]).validate()
    binding_paths: dict[str, str] = {}
    for binding in draft.get("bindings", []):
        path = Path(binding["path"]).resolve(strict=True)
        binding_paths[str(binding["role"])] = str(path)
        binding["sha256"] = sha256_file(path)
    for name in ARM_NAMES:
        arm = draft["arms"][name]
        for field in ("resume_checkpoint", "selected_checkpoint"):
            path = Path(arm[field]["path"]).resolve(strict=True)
            arm[field] = {"path": str(path), "sha256": sha256_file(path)}
        start = int(arm["start_update"])
        stop = int(arm["stop_after_update"])
        if stop - start != int(draft["updates"]):
            raise ValueError(f"{name} does not execute the exact admission budget")
        command = arm["command"]
        if str(stop) not in command or str(arm["resume_checkpoint"]["path"]) not in command:
            raise ValueError(f"{name} command does not bind exact resume and stop update")
        required_command_paths = {
            binding_paths["config"],
            binding_paths[
                "optimized_producer" if name == "optimized" else "old_producer"
            ],
            str(Path(arm["output"]).resolve()),
        }
        if name == "optimized":
            required_command_paths.add(binding_paths["performance_config"])
        if not required_command_paths.issubset(set(command)):
            raise ValueError(f"{name} command omits a bound producer/config/output path")
        if Path(arm["output"]).exists():
            raise FileExistsError(f"Admission arm output already exists: {name}")
    plan = bind_plan(draft)
    with tempfile.TemporaryDirectory(prefix="phase_ttc_gpu_plan_") as temporary:
        probe = Path(temporary) / "plan.json"
        probe.write_text(json.dumps(plan, allow_nan=False), encoding="utf-8")
        load_plan(probe)
    _write_once(output, plan)
    return plan


def _physical_available_bytes() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.available_physical)


def _nvidia_query(arguments: list[str]) -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", *arguments], check=True, capture_output=True, text=True
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _gpu_processes() -> list[dict[str, str | int]]:
    """Read driver process types without treating WDDM graphics as pure compute."""
    completed = subprocess.run(
        ["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True
    )
    root = ET.fromstring(completed.stdout)
    result = []
    for item in root.findall(".//process_info"):
        pid = item.findtext("pid")
        if pid is None:
            raise ValueError("NVIDIA XML process entry lacks PID")
        result.append(
            {
                "pid": int(pid),
                "type": item.findtext("type") or "UNKNOWN",
                "name": item.findtext("process_name") or "",
                "used_memory": item.findtext("used_memory") or "UNKNOWN",
            }
        )
    return result


def _process_snapshot() -> dict[int, tuple[int, int]]:
    """Return PID -> (parent PID, creation FILETIME) for PID-reuse-safe ancestry."""
    if os.name != "nt":
        raise RuntimeError("Serial GPU admission currently requires Windows process identity")

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("usage", ctypes.c_ulong),
            ("pid", ctypes.c_ulong),
            ("default_heap", ctypes.c_void_p),
            ("module_id", ctypes.c_ulong),
            ("threads", ctypes.c_ulong),
            ("parent_pid", ctypes.c_ulong),
            ("priority", ctypes.c_long),
            ("flags", ctypes.c_ulong),
            ("exe_file", ctypes.c_wchar * 260),
        ]

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel.OpenProcess.restype = ctypes.c_void_p
    snapshot_handle = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot_handle in (0, ctypes.c_void_p(-1).value):
        raise OSError("CreateToolhelp32Snapshot failed")
    result: dict[int, tuple[int, int]] = {}
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        success = kernel.Process32FirstW(snapshot_handle, ctypes.byref(entry))
        while success:
            process_handle = kernel.OpenProcess(0x1000, False, int(entry.pid))
            if process_handle:
                created, exited, kernel_time, user_time = (FileTime() for _ in range(4))
                if kernel.GetProcessTimes(
                    process_handle,
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                ):
                    filetime = (int(created.high) << 32) | int(created.low)
                    creation = ((filetime - 116_444_736_000_000_000) // 10_000) * 1000
                    result[int(entry.pid)] = (int(entry.parent_pid), creation)
                kernel.CloseHandle(process_handle)
            success = kernel.Process32NextW(snapshot_handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot_handle)
    return result


def _descendant_pids(root_pid: int, root_creation: int) -> set[int]:
    snapshot = _process_snapshot()
    allowed = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, creation) in snapshot.items():
            if pid not in allowed and parent in allowed and creation >= root_creation:
                allowed.add(pid)
                changed = True
    return allowed


def _cim_process_snapshot() -> dict[int, tuple[int, int]]:
    """Read identities for protected WDDM processes that deny OpenProcess."""
    command = (
        "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CreationDate "
        "| ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(completed.stdout)
    rows = raw if isinstance(raw, list) else [raw]
    result = {}
    for row in rows:
        creation_text = str(row["CreationDate"])
        serialized = re.fullmatch(r"/Date\((-?\d+)\)/", creation_text)
        creation_us = (
            int(serialized.group(1)) * 1000
            if serialized
            else int(datetime.fromisoformat(creation_text).timestamp() * 1_000_000)
        )
        result[int(row["ProcessId"])] = (
            int(row["ParentProcessId"]),
            creation_us,
        )
    return result


def _graphics_baseline() -> list[dict[str, str | int]]:
    snapshot = _process_snapshot()
    cim_snapshot = _cim_process_snapshot()
    result = []
    for row in _gpu_processes():
        if row["type"] != "C+G":
            continue
        identity = snapshot.get(int(row["pid"])) or cim_snapshot.get(int(row["pid"]))
        if identity is None:
            raise RuntimeError("Graphics process vanished during baseline capture")
        result.append(row | {"creation_time_us": identity[1]})
    return sorted(result, key=lambda item: int(item["pid"]))


def _gpu_competitors(plan: dict[str, Any], allowed_compute_pids: set[int]) -> list[dict[str, Any]]:
    snapshot = _process_snapshot()
    cim_snapshot = _cim_process_snapshot()
    baseline = {
        (int(row["pid"]), int(row["creation_time_us"]))
        for row in plan["graphics_baseline"]
    }
    competitors = []
    for row in _gpu_processes():
        pid = int(row["pid"])
        kind = str(row["type"])
        if kind == "C" and pid not in allowed_compute_pids:
            competitors.append(dict(row))
        elif kind == "C+G":
            identity = snapshot.get(pid) or cim_snapshot.get(pid)
            planned_creation = next(
                (creation for planned_pid, creation in baseline if planned_pid == pid), None
            )
            if planned_creation is None or (
                identity is not None and identity[1] != planned_creation
            ):
                competitors.append(dict(row))
        elif kind not in {"C", "C+G", "G"}:
            competitors.append(dict(row))
    return competitors


def _verify_graphics_baseline_live(plan: dict[str, Any]) -> None:
    """Recheck PID creation identities once at each admission boundary."""
    current = _cim_process_snapshot()
    for row in plan["graphics_baseline"]:
        identity = current.get(int(row["pid"]))
        if identity is not None and identity[1] != int(row["creation_time_us"]):
            raise RuntimeError("Preregistered WDDM graphics identity changed")


def resource_admission(plan: dict[str, Any]) -> dict[str, object]:
    """Reject competing compute processes and insufficient physical/GPU memory."""
    _verify_graphics_baseline_live(plan)
    processes = _gpu_processes()
    competitors = _gpu_competitors(plan, set())
    if competitors:
        raise RuntimeError(f"Unapproved GPU processes are active: {competitors}")
    free_rows = _nvidia_query(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    if len(free_rows) != 1:
        raise RuntimeError("Admission requires exactly one visible NVIDIA GPU")
    available_ram = _physical_available_bytes()
    free_vram = int(free_rows[0]) * 1024 * 1024
    resources = plan["resources"]
    if available_ram < int(resources["minimum_available_physical_ram_bytes"]):
        raise RuntimeError("Physical RAM admission gate failed")
    if free_vram < int(resources["minimum_free_vram_bytes"]):
        raise RuntimeError("VRAM admission gate failed")
    return {
        "gpu_processes_by_driver_type": processes,
        "available_physical_ram_bytes": available_ram,
        "free_vram_bytes": free_vram,
    }


def _run_arm(plan: dict[str, Any], name: str) -> dict[str, object]:
    arm = plan["arms"][name]
    metrics = Path(arm["metrics_path"])
    started = time.perf_counter()
    first_record_at: float | None = None
    last_record_at: float | None = None
    evaluation_artifact_at: float | None = None
    last_count = 0
    concurrent_compute_observed: list[str] = []
    next_process_check = started
    process = subprocess.Popen(arm["command"], cwd=ROOT)
    launch_snapshot = _process_snapshot()
    if process.pid not in launch_snapshot:
        raise RuntimeError("Launcher PID vanished before process-tree binding")
    launch_creation = launch_snapshot[process.pid][1]
    while process.poll() is None:
        if metrics.is_file():
            count = sum(1 for line in metrics.read_text(encoding="utf-8").splitlines() if line)
            if count and first_record_at is None:
                first_record_at = time.perf_counter()
            if count > last_count:
                last_record_at = time.perf_counter()
                last_count = count
        if evaluation_artifact_at is None and Path(arm["evaluation_path"]).is_file():
            evaluation_artifact_at = time.perf_counter()
        if time.perf_counter() >= next_process_check:
            allowed_pids = _descendant_pids(process.pid, launch_creation)
            concurrent_compute_observed.extend(
                json.dumps(row, sort_keys=True)
                for row in _gpu_competitors(plan, allowed_pids)
            )
            next_process_check = time.perf_counter() + 2.0
        time.sleep(0.2)
    ended = time.perf_counter()
    if process.returncode != 0:
        raise RuntimeError(f"Admission arm failed without automatic retry: {name}")
    if concurrent_compute_observed:
        raise RuntimeError(
            f"Concurrent compute appeared during {name}; run is inadmissible: "
            f"{sorted(set(concurrent_compute_observed))}"
        )
    if first_record_at is None or last_record_at is None or evaluation_artifact_at is None:
        raise RuntimeError(f"Admission arm emitted no metric records: {name}")
    if not Path(arm["summary_path"]).is_file() or not Path(arm["evaluation_path"]).is_file():
        raise RuntimeError(f"Admission arm lacks required summary or final evaluation: {name}")
    first_row = json.loads(metrics.read_text(encoding="utf-8").splitlines()[0])
    first_measured_work = float(first_row["preprocessing_seconds"]) + float(
        first_row["forward_backward_update_seconds"]
    )
    startup_through_first = first_record_at - started
    return {
        "name": name,
        "returncode": process.returncode,
        "wall_seconds": ended - started,
        "startup_through_first_record_seconds": startup_through_first,
        "startup_excluding_first_measured_prepare_compute_seconds": max(
            0.0, startup_through_first - first_measured_work
        ),
        "evaluation_after_last_record_seconds": evaluation_artifact_at - last_record_at,
        "finalize_after_evaluation_artifact_seconds": ended - evaluation_artifact_at,
        "metric_records_observed": last_count,
        "resource_admission": resource_admission(plan),
    }


def execute(plan_path: Path, receipt_path: Path, approval_token: str) -> dict[str, object]:
    """Run exactly three arms serially after an explicit plan-hash token."""
    plan = load_plan(plan_path)
    if approval_token != plan["plan_sha256"]:
        raise ValueError("Execution token must equal the reviewed plan SHA-256")
    if receipt_path.resolve() != Path(plan["execution_receipt_path"]).resolve():
        raise ValueError("Execution receipt path differs from the reviewed plan")
    for name in ARM_NAMES:
        if Path(plan["arms"][name]["output"]).exists():
            raise FileExistsError(f"Admission output already exists: {name}")
    admission = resource_admission(plan)
    validate_checkpoint_lineage(plan)
    initial = validate_initial_state(plan)
    results = []
    for name in ARM_NAMES:
        results.append(_run_arm(plan, name))
    receipt = {
        "schema": "eap_detection_gpu_admission_execution_v1",
        "status": "COMPLETED_ENGINEERING_ONLY",
        "plan_sha256": plan["plan_sha256"],
        "initial_state": initial,
        "initial_resource_admission": admission,
        "arms": results,
        "automatic_process_stop_or_kill": False,
        "candidate_outputs": False,
    }
    _write_once(receipt_path, receipt)
    return receipt


def verify(plan_path: Path, output: Path) -> dict[str, object]:
    """Write evidence-only verification without choosing a continuation."""
    plan = load_plan(plan_path)
    execution_path = Path(plan["execution_receipt_path"])
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Execution receipt belongs to a different plan")
    if (
        execution.get("schema") != "eap_detection_gpu_admission_execution_v1"
        or execution.get("status") != "COMPLETED_ENGINEERING_ONLY"
        or execution.get("automatic_process_stop_or_kill") is not False
        or execution.get("candidate_outputs") is not False
    ):
        raise ValueError("Execution receipt status or safety contract is invalid")
    executed_arms = execution.get("arms")
    if not isinstance(executed_arms, list) or [row.get("name") for row in executed_arms] != list(
        ARM_NAMES
    ):
        raise ValueError("Execution receipt arm order is invalid")
    if any(row.get("returncode") != 0 for row in executed_arms):
        raise ValueError("Execution receipt contains a failed arm")
    timing_fields = (
        "wall_seconds",
        "startup_through_first_record_seconds",
        "startup_excluding_first_measured_prepare_compute_seconds",
        "evaluation_after_last_record_seconds",
        "finalize_after_evaluation_artifact_seconds",
    )
    def validate_resource_receipt(resource: object) -> None:
        if not isinstance(resource, dict) or not isinstance(
            resource.get("gpu_processes_by_driver_type"), list
        ):
            raise ValueError("Execution receipt resource evidence is malformed")
        for field, threshold_field in (
            ("available_physical_ram_bytes", "minimum_available_physical_ram_bytes"),
            ("free_vram_bytes", "minimum_free_vram_bytes"),
        ):
            value = resource.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < int(plan["resources"][threshold_field])
            ):
                raise ValueError("Execution receipt resource gate evidence is invalid")

    validate_resource_receipt(execution.get("initial_resource_admission"))
    for row in executed_arms:
        if row.get("metric_records_observed") != plan["updates"]:
            raise ValueError("Execution receipt metric count differs from the plan")
        if any(
            isinstance(row.get(field), bool)
            or not isinstance(row.get(field), (int, float))
            or not math.isfinite(float(row[field]))
            or float(row[field]) < 0
            for field in timing_fields
        ):
            raise ValueError("Execution receipt timing evidence is malformed")
        validate_resource_receipt(row.get("resource_admission"))
    validate_checkpoint_lineage(plan)
    initial = validate_initial_state(plan)
    if execution.get("initial_state") != initial:
        raise ValueError("Execution receipt initial-state evidence differs from current artifacts")
    result = {
        "schema": "eap_detection_gpu_admission_verification_v1",
        "plan_sha256": plan["plan_sha256"],
        "initial_state": initial,
        "comparison": verify_metrics(plan),
        "execution_receipt_sha256": sha256_file(execution_path),
        "serial_wall_timings": execution.get("arms"),
        "scientific_candidate": False,
        "continuation_selected": None,
    }
    _write_once(output, result)
    return result


def diagnose(output: Path) -> dict[str, object]:
    """Record read-only WDDM classification without launching or stopping GPU work."""
    baseline = _graphics_baseline()
    plan = {"graphics_baseline": baseline}
    processes = _gpu_processes()
    competitors = _gpu_competitors(plan, set())
    result = {
        "schema": "eap_detection_gpu_identity_diagnostic_v1",
        "status": "PASS",
        "graphics_baseline": baseline,
        "observed_gpu_processes": processes,
        "unapproved_processes": competitors,
        "accepted_existing_c_plus_g_count": len(baseline),
        "pure_compute_rejected_count": sum(row["type"] == "C" for row in competitors),
        "gpu_work_launched": False,
        "processes_stopped_or_killed": False,
        "diagnostic_script_sha256": sha256_file(Path(__file__)),
    }
    _write_once(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--draft", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--receipt", type=Path, required=True)
    execute_parser.add_argument("--execute-gpu", action="store_true", required=True)
    execute_parser.add_argument("--approval-token", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    diagnose_parser = subparsers.add_parser("diagnose")
    diagnose_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.draft, args.output)
    elif args.command == "execute":
        execute(args.plan, args.receipt, args.approval_token)
    elif args.command == "verify":
        verify(args.plan, args.output)
    else:
        diagnose(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
