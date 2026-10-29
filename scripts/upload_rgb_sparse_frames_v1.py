"""Upload only locally verified, atomically published sparse RGB TAR inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.pending")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def checked_relative(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    if not re.fullmatch(r"data/train/[A-Za-z0-9_-]+/rgb_shards/[A-Za-z0-9_.-]+\.tar", value):
        raise ValueError(f"Unexpected sparse RGB path: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.host):
        raise ValueError("Unsafe SSH endpoint")
    if not re.fullmatch(r"/opt/[A-Za-z0-9_/-]+", args.destination):
        raise ValueError("Sparse input destination must be explicit /opt storage")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError("Upload deadline requires a timezone")
    source = args.source.resolve()
    ssh = [
        "ssh",
        "-i",
        str(args.key),
        "-p",
        str(args.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        args.host,
    ]
    scp = [
        "scp",
        "-i",
        str(args.key),
        "-P",
        str(args.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
    ]

    def remote(code: str) -> dict:
        command = "/opt/ejepa_env/bin/python -c " + shlex.quote(code)
        result = subprocess.run(
            ssh + [command], text=True, capture_output=True, check=True, timeout=180
        )
        return json.loads(result.stdout)

    completed: dict[str, dict] = {}
    if args.status.exists():
        old = json.loads(args.status.read_text(encoding="utf-8"))
        if old.get("source") != str(source) or old.get("destination") != args.destination:
            raise ValueError("Prior uploader belongs to another destination/source")
        completed = old.get("completed", {})
    base = {
        "schema": "ejepa_rgb_sparse_uploader_v1",
        "source": str(source),
        "destination": args.destination,
        "pid": os.getpid(),
        "scientific_execution_authorized_by_upload": False,
        "deadline": args.deadline,
        "remote_reserve_bytes": args.reserve_bytes,
    }

    def publish(stage: str, **details: Any) -> None:
        write(
            args.status,
            {
                **base,
                "utc": datetime.now(UTC).isoformat(),
                "stage": stage,
                "completed": completed,
                "files_uploaded": len(completed),
                "bytes_uploaded": sum(v["bytes"] for v in completed.values()),
                **details,
            },
        )

    publish("WAITING_VERIFIED_SPARSE_TARS")
    while datetime.now(UTC) < deadline:
        if args.status.with_name("UPLOAD_STOP_REQUEST").exists():
            publish("MANUAL_STOP")
            return
        for receipt_path in sorted(source.glob("data/train/*/rgb_shards/*.tar.json")):
            tar = receipt_path.with_suffix("")
            rel = checked_relative(tar, source)
            if rel in completed:
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("schema") != "ejepa_rgb_sparse_tar_receipt_v1":
                raise ValueError(f"Unexpected sparse receipt: {receipt_path}")
            sha, size = digest(tar), tar.stat().st_size
            if receipt.get("sparse_tar_sha256") != sha:
                raise ValueError(f"Local sparse TAR differs from verified receipt: {rel}")
            target = args.destination + "/" + rel
            temp = target + ".uploading"
            preflight = remote(
                "import pathlib,os,json,hashlib; "
                f"p=pathlib.Path({target!r});p.parent.mkdir(parents=True,exist_ok=True);"
                "v=os.statvfs(p.parent);"
                "print(json.dumps({'free':v.f_bavail*v.f_frsize,'exists':p.exists(),"
                "'sha':hashlib.file_digest(p.open('rb'),'sha256').hexdigest() "
                "if p.exists() else None}))"
            )
            if preflight["exists"] and preflight["sha"] != sha:
                raise ValueError(f"Remote sparse TAR already exists with different bytes: {rel}")
            if not preflight["exists"]:
                if preflight["free"] < size + args.reserve_bytes:
                    publish("BLOCKED_STORAGE", pending=rel, free_bytes=preflight["free"])
                    return
                publish("TRANSFERRING", current=rel, current_bytes=size)
                subprocess.run(scp + [str(tar), f"{args.host}:{temp}"], check=True, timeout=7200)
                verified = remote(
                    "import pathlib,os,json,hashlib;"
                    f"p=pathlib.Path({temp!r});d=pathlib.Path({target!r});"
                    "h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();"
                    f"assert h=={sha!r} and p.stat().st_size=={size};"
                    "os.replace(p,d);print(json.dumps({'status':'VERIFIED','sha256':h}))"
                )
            else:
                verified = {"status": "VERIFIED_EXISTING", "sha256": sha}
            if digest(tar) != sha:
                raise ValueError("Local published TAR changed during transfer")
            subprocess.run(
                scp + [str(receipt_path), f"{args.host}:{target}.json"], check=True, timeout=180
            )
            completed[rel] = {
                **verified,
                "bytes": size,
                "local_receipt_sha256": digest(receipt_path),
                "verified_utc": datetime.now(UTC).isoformat(),
            }
            publish("WAITING_VERIFIED_SPARSE_TARS")
        time.sleep(10)
    publish("DEADLINE_EXPIRED")


if __name__ == "__main__":
    main()
