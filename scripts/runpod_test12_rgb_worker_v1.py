"""Stage one frozen Test12 RGB sequence with byte identity and shared quota guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path("/workspace/ejepa_test12_rgb")
SHARED = Path("/workspace/ejepa_test12_probe")
QUOTA = Path("/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json")


def atomic(path: Path, value: dict) -> None:
    """Publish one complete JSON object durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def quota_available(snapshot: dict, now: float, reserved: int, reserve: int) -> int:
    """Reject unknown/stale authorities and deduct shared in-flight commitments."""
    if (
        snapshot.get("schema") != "ejepa_event_volume_snapshot_v1"
        or snapshot.get("run_root") != "/workspace/ejepa_event_runs"
        or snapshot.get("volume_root") != "/workspace"
        or snapshot.get("quota_bytes") != 600_000_000_000
        or not 0 <= now - float(snapshot["observed_timestamp"]) <= 120
        or not 0 <= int(snapshot["used_bytes"]) <= 600_000_000_000
        or reserved < 0
    ):
        raise ValueError("Invalid or stale Test12 volume authority")
    return 600_000_000_000 - int(snapshot["used_bytes"]) - reserved - reserve


def digest(path: Path) -> str:
    """Hash bounded ranges and release clean read cache as verification advances."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        offset = 0
        while block := stream.read(8 * 1024**2):
            result.update(block)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)
    return result.hexdigest()


def main() -> int:
    """Download only the selected required shards; never delete or train."""
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--sequence", default="3RUZHwnCwK")
    args = parser.parse_args()
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != args.plan_sha256:
        raise ValueError("Download plan SHA differs")
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "ejepa_test12_rgb_download_plan_v1" or plan["raw_root"] != str(ROOT):
        raise ValueError("Unknown download plan authority")
    if plan["inputs_sha256"] != "b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526":
        raise ValueError("Frozen Test12 INPUTS differ")
    records = plan["sequences"][args.sequence]
    for record in records:
        rel = Path(record["path"])
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or rel.parts[:3] != ("data", "test", args.sequence)
        ):
            raise ValueError("RGB source escapes selected sequence")
    ROOT.mkdir(exist_ok=True)
    sys.path.insert(0, str(SHARED))
    from resource_reader import memory_snapshot

    status_path = ROOT / "status" / f"{args.sequence}.json"
    key = f"rgb_tars_{args.sequence}"
    ledger_path = SHARED / "DOWNLOAD_RESERVATIONS.json"
    lock_path = SHARED / "DOWNLOAD_RESERVATIONS.lock"

    def status(stage: str, **extra: object) -> None:
        atomic(
            status_path,
            {
                "schema": "ejepa_test12_rgb_worker_status_v1",
                "status": stage,
                "pid": os.getpid(),
                "sequence": args.sequence,
                "timestamp": time.time(),
                "optimizer_updates": 0,
                "plan_sha256": args.plan_sha256,
                **extra,
            },
        )

    def guard(remaining: int) -> None:
        now = time.time()
        if now >= plan["deadline_epoch"] or (ROOT / "STOP_REQUEST").exists():
            raise RuntimeError("Deadline/manual stop")
        headroom, _ = memory_snapshot(
            Path("/sys/fs/cgroup"), Path("/proc/self/cgroup").read_text(), 2**63 - 1
        )
        if min(headroom, psutil.virtual_memory().available) < plan["ram_reserve_bytes"]:
            raise RuntimeError("RAM reserve below 2 GiB")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            others = sum(int(v["bytes"]) for k, v in ledger.items() if k != key)
            available = quota_available(
                json.loads(QUOTA.read_text()), now, others, plan["reserve_bytes"]
            )
            if available < remaining:
                raise RuntimeError(f"Quota reserve unavailable: {available} < {remaining}")
            ledger[key] = {"bytes": remaining, "pid": os.getpid(), "updated_timestamp": now}
            atomic(ledger_path, ledger)

    with (ROOT / f"WORKER_{args.sequence}.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        receipts = {}
        try:
            for index, record in enumerate(records):
                remaining = sum(
                    x["size_bytes"] for x in records[index:] if not (ROOT / x["path"]).is_file()
                )
                guard(remaining)
                path = ROOT / record["path"]
                if not path.exists():
                    command = [
                        "/opt/ejepa_env/bin/hf",
                        "download",
                        plan["repository"],
                        record["path"],
                        "--repo-type",
                        "dataset",
                        "--revision",
                        plan["revision"],
                        "--local-dir",
                        str(ROOT),
                        "--max-workers",
                        "1",
                    ]
                    env = dict(
                        os.environ,
                        HF_HUB_DISABLE_XET="1",
                        HF_HUB_ENABLE_HF_TRANSFER="0",
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                    )
                    status("DOWNLOADING", relative_path=record["path"], remaining_bytes=remaining)
                    child = subprocess.Popen(command, env=env, start_new_session=True)
                    try:
                        while child.poll() is None:
                            guard(remaining)
                            status("DOWNLOADING", relative_path=record["path"], child_pid=child.pid)
                            time.sleep(5)
                        if child.returncode:
                            raise RuntimeError(f"hf download failed: {child.returncode}")
                    finally:
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                status("VERIFYING_SHA256", relative_path=record["path"])
                before = path.stat()
                if (
                    path.is_symlink()
                    or before.st_size != record["size_bytes"]
                    or digest(path) != record["sha256"]
                ):
                    raise ValueError(f"Frozen RGB media identity failed: {path}")
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("RGB shard changed during verification")
                receipt = {
                    "schema": "ejepa_test12_rgb_shard_receipt_v1",
                    "status": "PASS",
                    "relative_path": record["path"],
                    "path": str(path),
                    "sha256": record["sha256"],
                    "size_bytes": after.st_size,
                    "revision": plan["revision"],
                    "inputs_sha256": plan["inputs_sha256"],
                    "inventory_sha256": plan["inventory_sha256"],
                    "verified_utc": datetime.now(UTC).isoformat(),
                    "frozen_mtime_ns": record["mtime_ns"],
                    "actual_mtime_ns": after.st_mtime_ns,
                    "mtime_restored": False,
                }
                receipt_path = ROOT / "receipts" / args.sequence / (path.name + ".json")
                atomic(receipt_path, receipt)
                receipts[record["path"]] = str(receipt_path)
            ready = {
                "schema": "ejepa_test12_rgb_sequence_ready_v1",
                "status": "READY",
                "sequence": args.sequence,
                "source_receipts": receipts,
                "source_count": len(records),
                "bytes": sum(x["size_bytes"] for x in records),
                "inputs_sha256": plan["inputs_sha256"],
                "inventory_sha256": plan["inventory_sha256"],
                "revision": plan["revision"],
                "elapsed_seconds": time.time() - started,
                "completed_utc": datetime.now(UTC).isoformat(),
            }
            atomic(ROOT / "sequences" / f"{args.sequence}.json", ready)
            status(
                "COMPLETE", **{"ready_receipt": str(ROOT / "sequences" / f"{args.sequence}.json")}
            )
        except Exception as error:
            status("STOPPED_REQUIRES_REVIEW", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            with lock_path.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
                if ledger.get(key, {}).get("pid") == os.getpid():
                    ledger.pop(key)
                    atomic(ledger_path, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
