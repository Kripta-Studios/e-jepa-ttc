"""Restore only authorized TRAIN bytes from pinned public Hugging Face revisions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, digest, read
from .common import atomic_json as write_json


def atomic_json(path: Path, value: dict) -> None:
    """Retry transient Windows read/share contention; preserve failed temporary bytes."""
    for attempt in range(20):
        try:
            write_json(path, value)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def plan(c: Campaign) -> dict:
    """Select raw media by the existing input-only role manifest, never targets."""
    configuration = read(Path(c.launch["source_configuration"]))
    sequences = set(configuration["original_sequences"] + configuration["expansion_sequences"])
    remote = read(c.out / "data_recovery/eAP-dataset_REMOTE_METADATA.json")
    siblings = {row["rfilename"]: row for row in remote["siblings"]}
    route = read(c.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json")
    profile_names = {Path(row["path"]).parent.name for row in route["raw"]}
    entries = []
    for sequence in sorted(sequences):
        filename = f"data/train/{sequence}/events.h5"
        row = siblings[filename]
        entries.append(
            {
                "filename": filename,
                "bytes": row["size"],
                "sha256": row["lfs"]["sha256"],
                "sequence_id": sequence,
                "profile_required": sequence in profile_names,
            }
        )
    entries.sort(key=lambda row: (not row["profile_required"], row["bytes"], row["filename"]))
    value = {
        "repo": "NAIL-HNU/eAP-dataset",
        "revision": remote["sha"],
        "files": entries,
        "selected_sequences": len(sequences),
        "total_bytes": sum(row["bytes"] for row in entries),
        "destination": str(c.raw.parent.parent),
        "excluded_RGB_and_test": True,
        "write_authority": "explicit user request: restore lost E: TRAIN data from public HF",
        "source_configuration_sha256": digest(Path(c.launch["source_configuration"])),
    }
    path = c.out / "data_recovery/DOWNLOAD_PLAN.json"
    if path.exists() and read(path) != value:
        raise ValueError("restoration allowlist or pinned revision changed")
    if not path.exists():
        atomic_json(path, value)
    return value


def restore(c: Campaign) -> int:
    """One bounded downloader, resumable partial files, exact streamed SHA256 receipts."""
    value = plan(c)
    root = Path(value["destination"])
    remaining = sum(
        row["bytes"] for row in value["files"] if not (root / row["filename"]).is_file()
    )
    if shutil.disk_usage(root).free - remaining < 20_000_000_000:
        raise InterruptedError("raw restoration reservation leaves less than20GB on E:")
    cli = shutil.which("hf")
    if cli is None:
        raise FileNotFoundError(
            "existing hf CLI is required; global environment upgrade prohibited"
        )
    env = dict(os.environ)
    env.update(
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        HF_XET_HIGH_PERFORMANCE="0",
        HF_XET_NUM_CONCURRENT_RANGE_GETS="4",
        HF_XET_CHUNK_CACHE_SIZE_BYTES="0",
        HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY="1",
        HF_XET_CACHE=str(root / ".hf_xet_recovery"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUTF8="1",
    )
    verified = []
    failures = []
    for index, row in enumerate(value["files"]):
        c.require_resources()
        file = root / row["filename"]
        receipt = c.out / "data_recovery/files" / (row["sequence_id"] + ".json")
        if receipt.exists():
            previous = read(receipt)
            if (
                file.is_file()
                and previous["sha256"] == row["sha256"]
                and file.stat().st_size == row["bytes"]
                and file.stat().st_mtime_ns == previous["mtime_ns"]
            ):
                verified.append(previous)
                continue
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
        log = c.out / "data_recovery/logs" / f"{row['sequence_id']}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(UTC).isoformat()
        print(
            "RESTORE_START",
            index + 1,
            len(value["files"]),
            row["filename"],
            row["bytes"],
            flush=True,
        )
        with log.open("ab") as stream:
            worker = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            while worker.poll() is None:
                if not c.check():
                    worker.terminate()
                    worker.wait(timeout=60)
                    atomic_json(
                        c.out / "data_recovery/STATUS.json",
                        {"status": "PAUSED_RESOURCE", "file": row, "command": command},
                    )
                    return 3
                atomic_json(
                    c.out / "data_recovery/PROGRESS.json",
                    {
                        "status": "DOWNLOADING",
                        "file": row,
                        "position": index + 1,
                        "verified_files": len(verified),
                        "verified_bytes": sum(v["bytes"] for v in verified),
                        "worker_pid": worker.pid,
                        "observed_utc": datetime.now(UTC).isoformat(),
                    },
                )
                time.sleep(5)
        if worker.returncode:
            failure = {
                "status": "TRANSIENT_DOWNLOAD_FAILURE",
                "file": row,
                "command": command,
                "log": str(log),
                "returncode": worker.returncode,
            }
            failures.append(failure)
            atomic_json(
                receipt,
                failure,
            )
            print("RESTORE_FAILED_PRESERVED_CONTINUING", row["sequence_id"], flush=True)
            continue
        if (
            not file.is_file()
            or file.stat().st_size != row["bytes"]
            or digest(file) != row["sha256"]
        ):
            raise ValueError(
                "restored raw payload failed exact byte/SHA256 verification: " + str(file)
            )
        result = dict(
            row,
            path=str(file),
            mtime_ns=file.stat().st_mtime_ns,
            start_utc=started,
            end_utc=datetime.now(UTC).isoformat(),
            status="VERIFIED",
            log_sha256=digest(log),
        )
        atomic_json(receipt, result)
        verified.append(result)
        print("RESTORE_VERIFIED", row["sequence_id"], len(verified), flush=True)
    atomic_json(
        c.out / "data_recovery/STATUS.json",
        {
            "status": "COMPLETE" if not failures else "PARTIAL_RESUMABLE",
            "files": len(verified),
            "bytes": sum(v["bytes"] for v in verified),
            "revision": value["revision"],
            "optimizer_updates": 0,
            "failures": failures,
        },
    )
    return 0 if not failures else 3


def main() -> int:
    """Restore TRAIN or inspect the already pinned and authorized download plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "restore"))
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    c = Campaign(args.protocol)
    if args.action == "plan":
        print(json.dumps(plan(c), ensure_ascii=False, indent=2))
        return 0
    return restore(c)


if __name__ == "__main__":
    raise SystemExit(main())
