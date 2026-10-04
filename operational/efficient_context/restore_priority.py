"""Restore one blocked TRAIN profile file alongside the bounded primary downloader."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, digest, read
from .restore_data import atomic_json, plan


def restore(c: Campaign, sequence: str) -> int:
    """Select only an already authorized file; enforce the joint4GiB process RSS."""
    import psutil

    value = plan(c)
    row = next(v for v in value["files"] if v["sequence_id"] == sequence)
    root = Path(value["destination"])
    receipt_path = c.out / "data_recovery/files" / (sequence + ".json")
    if receipt_path.exists() and read(receipt_path).get("status") == "VERIFIED":
        return 0
    cli = shutil.which("hf")
    if cli is None:
        raise FileNotFoundError("existing hf CLI required")
    env = dict(
        os.environ,
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        HF_XET_HIGH_PERFORMANCE="0",
        HF_XET_NUM_CONCURRENT_RANGE_GETS="2",
        HF_XET_CHUNK_CACHE_SIZE_BYTES="0",
        HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY="1",
        HF_XET_CACHE=str(root / ".hf_xet_recovery"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        RAYON_NUM_THREADS="2",
        PYTHONUTF8="1",
    )
    command = [
        cli,
        "download",
        value["repo"],
        row["filename"],
        "--type",
        "dataset",
        "--revision",
        value["revision"],
        "--local-dir",
        str(root),
        "--max-workers",
        "1",
        "--quiet",
    ]
    log = c.out / "data_recovery/logs" / (sequence + "_priority.txt")
    started = datetime.now(UTC).isoformat()
    status_path = c.out / "data_recovery/PRIORITY_DOWNLOAD.json"
    with log.open("ab") as stream:
        worker = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        while worker.poll() is None:
            processes = {os.getpid(): psutil.Process()}
            for process in psutil.process_iter(["pid", "cmdline"]):
                line = " ".join(process.info["cmdline"] or [])
                if "operational.efficient_context.restore_data restore" in line:
                    try:
                        if Path(process.cwd()).resolve() == ROOT:
                            processes[process.pid] = process
                    except psutil.Error:
                        pass
            for process in list(processes.values()):
                try:
                    processes.update(
                        {child.pid: child for child in process.children(recursive=True)}
                    )
                except psutil.Error:
                    pass
            rss = 0
            for process in processes.values():
                try:
                    rss += process.memory_info().rss
                except psutil.Error:
                    pass
            admitted = c.check() and rss <= 4 * 1024**3
            atomic_json(
                status_path,
                {
                    "status": "DOWNLOADING" if admitted else "PAUSED_RESOURCE",
                    "file": row,
                    "joint_downloader_RSS": rss,
                    "observed_utc": datetime.now(UTC).isoformat(),
                    "worker_pid": worker.pid,
                    "command": command,
                },
            )
            if not admitted:
                owner = psutil.Process(worker.pid)
                for child in reversed(owner.children(recursive=True)):
                    child.terminate()
                worker.terminate()
                worker.wait(timeout=60)
                return 3
            time.sleep(5)
    if worker.returncode:
        atomic_json(
            status_path,
            {
                "status": "TRANSIENT_FAILURE_PRESERVED",
                "file": row,
                "returncode": worker.returncode,
                "log": str(log),
            },
        )
        return 3
    file = root / row["filename"]
    if file.stat().st_size != row["bytes"] or digest(file) != row["sha256"]:
        raise ValueError("priority TRAIN full SHA256 verification failed")
    atomic_json(
        receipt_path,
        dict(
            row,
            path=str(file),
            mtime_ns=file.stat().st_mtime_ns,
            start_utc=started,
            end_utc=datetime.now(UTC).isoformat(),
            status="VERIFIED",
            log_sha256=digest(log),
        ),
    )
    atomic_json(status_path, {"status": "VERIFIED", "file": row, "optimizer_updates": 0})
    print("RESTORE_PRIORITY_VERIFIED", sequence, flush=True)
    return 0


def main() -> int:
    """One explicitly chosen allowlisted TRAIN file; never a repository snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    return restore(Campaign(args.protocol), args.sequence)


if __name__ == "__main__":
    raise SystemExit(main())
