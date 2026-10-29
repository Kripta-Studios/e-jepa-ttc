"""Continuously download and verify all six event checkpoints on Windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

TASKS = [(rotation, fit) for rotation in (2, 3, 4) for fit in ("E_A5", "E_C2F")]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def admit(archive: Path, root: Path, expected: str) -> dict[str, object]:
    if sha(archive) != expected:
        raise ValueError("Event snapshot archive SHA differs")
    with tarfile.open(archive) as source:
        members = source.getmembers()
        unsafe = any(
            not item.isfile() or Path(item.name).is_absolute() or ".." in Path(item.name).parts
            for item in members
        )
        if unsafe:
            raise ValueError("Unsafe event snapshot member")
        manifest_stream = source.extractfile("MANIFEST.json")
        if manifest_stream is None:
            raise ValueError("Event snapshot manifest missing")
        manifest = json.load(manifest_stream)
        name = f"{int(manifest['completed_updates']):06d}_{str(manifest['checkpoint_sha256'])[:12]}"
        target, pending = root / name, root / f".{name}.pending"
        if target.exists():
            return manifest
        pending.mkdir()
        try:
            for member in members:
                destination = pending / member.name
                if not destination.resolve().is_relative_to(pending.resolve()):
                    raise ValueError("Event snapshot escaped local root")
                destination.parent.mkdir(parents=True, exist_ok=True)
                incoming = source.extractfile(member)
                if incoming is None:
                    raise ValueError("Event snapshot payload missing")
                with destination.open("xb") as output:
                    shutil.copyfileobj(incoming, output)
            for relative, binding in manifest["files"].items():
                path = pending / relative
                if path.stat().st_size != binding["bytes"] or sha(path) != binding["sha256"]:
                    raise ValueError("Event snapshot member differs")
            os.replace(pending, target)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    root = args.output.resolve()
    allowed = Path(__file__).resolve().parents[1] / "artifacts/runpod_returns"
    if not root.is_relative_to(allowed.resolve()):
        raise ValueError("Event backups must remain under artifacts/runpod_returns")
    root.mkdir(parents=True, exist_ok=True)
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-i",
        str(args.key),
        "-p",
        str(args.port),
        args.host,
    ]
    latest = {task: -1 for task in TASKS}
    while not (root / "STOP_BACKUP").exists():
        records = []
        for rotation, fit in TASKS:
            record: dict[str, object] = {"rotation": rotation, "fit": fit}
            try:
                command = [
                    "/opt/ejepa_env/bin/python",
                    "/workspace/ejepa_event_operations/event_snapshot_v1.py",
                    "--run",
                    "/workspace/ejepa_event_runs",
                    "--rotation",
                    str(rotation),
                    "--fit",
                    fit,
                ]
                result = subprocess.run(
                    ssh + [shlex.join(command)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=180,
                )
                remote = json.loads(result.stdout)
                step = int(remote.get("updates", -1))
                if remote.get("status") == "READY" and step > latest[(rotation, fit)]:
                    folder = root / f"r{rotation}_{fit}"
                    folder.mkdir(exist_ok=True)
                    pending = folder / "download.tar.pending"
                    with pending.open("wb") as output:
                        subprocess.run(
                            ssh + [shlex.join(["cat", str(remote["archive"])])],
                            stdout=output,
                            check=True,
                            timeout=300,
                        )
                    manifest = admit(pending, folder, str(remote["archive_sha256"]))
                    pending.unlink()
                    latest[(rotation, fit)] = step
                    record.update(
                        status="VERIFIED_LOCAL",
                        updates=step,
                        checkpoint_sha256=manifest["checkpoint_sha256"],
                    )
                else:
                    record.update(status="WAITING", updates=latest[(rotation, fit)])
            except Exception as error:
                record.update(status="BACKUP_ERROR", error=f"{type(error).__name__}: {error}")
            records.append(record)
        status = {"utc": datetime.now(UTC).isoformat(), "tasks": records}
        pending_status = root / "BACKUP_STATUS.pending"
        pending_status.write_text(json.dumps(status, indent=2) + "\n")
        os.replace(pending_status, root / "BACKUP_STATUS.json")
        with (root / "BACKUP_HISTORY.jsonl").open("a") as history:
            history.write(json.dumps(status) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
