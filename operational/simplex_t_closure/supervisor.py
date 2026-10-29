"""Lightweight T6-only supervisor. It never starts a training entrypoint."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/simplex_t/closure_20261002"
LOCK = ROOT / "artifacts/simplex_t/T0/CHECKPOINTED_REMAINING.lock"


def write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".pending")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory() -> dict[str, int | None]:
    class Performance(ctypes.Structure):
        _fields_ = (
            [("cb", ctypes.c_ulong)]
            + [
                (k, ctypes.c_size_t)
                for k in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ]
            + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")]
        )

    counter = Performance()
    counter.cb = ctypes.sizeof(counter)
    ok = ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(counter), counter.cb)
    process = psutil.Process()
    rss = process.memory_info().rss + sum(
        p.memory_info().rss for p in process.children(recursive=True)
    )
    return {
        "host_available_bytes": psutil.virtual_memory().available,
        "process_tree_rss_bytes": rss,
        "windows_commit_headroom_bytes": (counter.CommitLimit - counter.CommitTotal)
        * counter.PageSize
        if ok
        else None,
        "written_volume_free_bytes": psutil.disk_usage(str(ROOT)).free,
    }


def import_memory_failure(output: str) -> bool:
    return any(
        token in output.lower()
        for token in (
            "1455",
            "paging file is too small",
            "archivo de paginación es demasiado pequeño",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-reserved-bytes", type=int, default=1070596096)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    policy = json.loads(Path(__file__).with_name("resource_policy.json").read_text())
    ART.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        owner = json.loads(LOCK.read_text(encoding="utf-8"))
        try:
            p = psutil.Process(owner["pid"])
            command = " ".join(p.cmdline())
            raise RuntimeError(f"T6 lease PID still alive; do not steal: {p.pid}: {command}")
        except psutil.NoSuchProcess:
            pass
        # Preserve the stale lease bytes before recovering this namespace.
        write(ART / "STALE_OWNER.json", owner)
        LOCK.unlink()
    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "command": str(Path(__file__)),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    try:
        snapshot = memory()
        write(ART / "RESOURCE_ADMISSION.json", snapshot)
        missing = []
        template = json.loads(
            (ROOT / "artifacts/simplex_t/scientific_campaign/launches/T6.json").read_text(
                encoding="utf-8"
            )
        )
        for role, value in template["roots"].items():
            try:
                Path(value).stat()
            except OSError as error:
                missing.append(
                    {"role": role, "path": value, "operation": "stat", "error": repr(error)}
                )
        if missing:
            write(
                ART / "STATUS.json",
                {
                    "status": "BLOCKED_REQUIRED_ROOT",
                    "missing": missing,
                    "optimizer_updates_executed": 0,
                },
            )
            return 3
        reasons = []
        if snapshot["host_available_bytes"] < policy["host_available_ram_floor_bytes"]:
            reasons.append("host RAM below user-amended 2 GiB floor")
        if int(snapshot["process_tree_rss_bytes"] or 0) > 4 * 1024**3:
            reasons.append("process tree exceeds active 4 GiB RSS ceiling")
        if (
            snapshot["written_volume_free_bytes"] - args.own_reserved_bytes
            < policy["written_volume_emergency_floor_bytes"]
        ):
            reasons.append("own outputs plus user-amended 10 GB emergency floor do not fit")
        if reasons:
            write(
                ART / "STATUS.json",
                {
                    "status": "PAUSED_MEASURED_RESOURCE",
                    "reasons": reasons,
                    "resources": snapshot,
                    "own_reserved_bytes": args.own_reserved_bytes,
                    "optimizer_updates_executed": 0,
                },
            )
            print(
                json.dumps(
                    {
                        "status": "PAUSED_MEASURED_RESOURCE",
                        "reasons": reasons,
                        "resources": snapshot,
                    }
                ),
                flush=True,
            )
            return 3
        # The exact physical journal relocation was an already valid repair.
        sys.path.insert(0, str(ROOT / "artifacts/simplex_t/T0"))
        historical_resume = importlib.import_module("resume_remaining_checkpointed")
        template = historical_resume.bind_t6_accounting_journal(template)
        declared = template["delivery"]["resource_attempts"]
        covered = {
            json.loads(Path(row["receipt"]).read_text(encoding="utf-8"))["stage"]
            for row in declared
        }
        for stage in sorted({"T2", "T3", "T4", "T5"} - covered):
            receipts = []
            for receipt_path in (ROOT / "artifacts/simplex_t/resource_observations").glob(
                "*.receipt.json"
            ):
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("stage") == stage
                    and receipt.get("freeze_sha256") == template["freeze_sha256"]
                ):
                    receipts.append(
                        (
                            receipt.get("execution_result_status")
                            != "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS",
                            receipt_path,
                            receipt,
                        )
                    )
            if not receipts:
                raise FileNotFoundError(f"missing actual terminal resource observation for {stage}")
            _, receipt_path, receipt = sorted(receipts, key=lambda row: (row[0], str(row[1])))[0]
            launch_path = receipt_path.with_name(
                receipt_path.name.replace(".receipt.json", ".launch.json")
            )
            if sha(launch_path) != receipt["launch_sha256"]:
                raise ValueError("historical resource launch differs from receipt")
            declared.append(
                {
                    "launch": str(launch_path),
                    "launch_sha256": sha(launch_path),
                    "receipt": str(receipt_path),
                    "receipt_sha256": sha(receipt_path),
                }
            )
        write(
            ART / "RESOURCE_RECEIPTS_RECONCILIATION.json",
            {
                "actual_historical_observations": declared,
                "exhaustive_attempt_history_claimed": False,
                "missing_hard_termination_measurements_invented": False,
                "T2_terminal_receipt_is_failed_invocation_not_complete_fit": True,
            },
        )
        attempt = ART / "canonical_attempt"
        attempt.mkdir(exist_ok=True)
        template["output"] = str(attempt / "analysis")
        template["delivery"]["output"] = str(attempt / "delivery")
        # Preserve operational/scientific identity separation; no checkout.
        template["delivery"]["analysis_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        launch = attempt / "launch_reconciled_resources.json"
        if launch.exists() and json.loads(launch.read_text(encoding="utf-8")) != template:
            raise ValueError("closure launch inputs differ; preserve the prior attempt")
        if not launch.exists():
            write(launch, template)
        command = [
            sys.executable,
            "-B",
            "-u",
            str(Path(__file__).with_name("worker.py")),
            "--launch",
            str(launch),
            "--launch-sha256",
            sha(launch),
            "--other-reserved-bytes",
            "0",
            "--own-reserved-bytes",
            str(args.own_reserved_bytes),
        ]
        if args.verify_only:
            command.append("--verify-only")
        write(
            ART / "COMMAND.json",
            {
                "command": command,
                "optimizer_updates_authorized": 0,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        log = ART / "worker.log"
        offset = log.stat().st_size if log.exists() else 0
        write(
            ART / "STATUS.json",
            {
                "status": "CANONICAL_VERIFY_RUNNING" if args.verify_only else "T6_CLOSURE_RUNNING",
                "owner_pid": os.getpid(),
                "log": str(log),
                "log_offset": offset,
                "optimizer_updates_executed": 0,
            },
        )
        with log.open("a", encoding="utf-8") as stream:
            code = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            ).returncode
        with log.open("rb") as stream:
            stream.seek(offset)
            output = stream.read().decode("utf-8", errors="replace")
        status = (
            "WORKER_RETURNED_CANONICAL_VERIFICATION_REQUIRED"
            if code == 0
            else "WORKER_PAUSED"
            if code == 3
            else "WORKER_ERROR"
        )
        if import_memory_failure(output):
            status = "PAUSED_WINDOWS_IMPORT_MEMORY_1455"
        write(
            ART / "STATUS.json",
            {
                "status": status,
                "exit_code": code,
                "resources_after": memory(),
                "log": str(log),
                "log_offset": offset,
                "optimizer_updates_executed": 0,
            },
        )
        print(json.dumps({"status": status, "exit_code": code, "log": str(log)}), flush=True)
        # No blind restart. The next invocation must remeasure admission.
        return code
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
