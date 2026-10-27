"""Download verified immutable snapshots into an isolated Windows return directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def admit_archive(archive: Path, root: Path, expected: str) -> dict:
    """Extract only regular files; verify every byte before publishing a return snapshot."""
    if digest(archive) != expected:
        raise ValueError("Downloaded archive SHA differs")
    root = root.resolve(strict=True)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Snapshot archive contains unsafe/special member")
        payload = source.extractfile("MANIFEST.json")
        if payload is None:
            raise ValueError("Snapshot manifest is missing")
        manifest = json.load(payload)
        name = manifest["snapshot"]
        if not re.fullmatch(r"\d{6}_[0-9a-f]{12}", name):
            raise ValueError("Unsafe snapshot identity")
        target, pending = root / name, root / f".{name}.pending"
        if target.exists():
            if json.loads((target / "MANIFEST.json").read_text()) != manifest:
                raise ValueError("Existing local snapshot differs")
            for relative, binding in manifest["files"].items():
                path = target / relative
                if not path.resolve().is_relative_to(target) or path.is_symlink():
                    raise ValueError("Existing snapshot file escaped its directory")
                if path.stat().st_size != binding["bytes"] or digest(path) != binding["sha256"]:
                    raise ValueError("Existing local snapshot file is corrupt")
            return manifest
        if pending.exists():
            if not pending.resolve().is_relative_to(root):
                raise ValueError("Unsafe pending directory")
            shutil.rmtree(pending)
        pending.mkdir()
        for member in members:
            path = pending / member.name
            if not path.resolve().is_relative_to(pending.resolve()):
                raise ValueError("Archive member escapes local snapshot")
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Missing archive member payload")
            with path.open("xb") as output:
                shutil.copyfileobj(stream, output)
                output.flush()
                os.fsync(output.fileno())
        if {member.name for member in members} != set(manifest["files"]) | {"MANIFEST.json"}:
            raise ValueError("Archive coverage differs from snapshot manifest")
        for relative, binding in manifest["files"].items():
            path = pending / relative
            if path.stat().st_size != binding["bytes"] or digest(path) != binding["sha256"]:
                raise ValueError("Snapshot file differs from its manifest")
        os.replace(pending, target)
        return manifest


def retain(root: Path) -> None:
    """Retain latest two verified snapshots plus every sealed endpoint."""
    root = root.resolve(strict=True)
    snapshots = []
    for child in root.iterdir():
        if child.is_dir() and re.fullmatch(r"\d{6}_[0-9a-f]{12}", child.name):
            manifest = json.loads((child / "MANIFEST.json").read_text())
            snapshots.append((manifest["completed_updates"], child, manifest))
    snapshots.sort(key=lambda item: item[0], reverse=True)
    for _, path, manifest in snapshots[2:]:
        if manifest["checkpoint_status"] == "COMPLETE":
            continue
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Unsafe retention target")
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    allowed = Path(__file__).resolve().parents[1] / "artifacts/runpod_returns"
    if not root.is_relative_to(allowed.resolve()):
        raise ValueError("Cloud backups must remain in the isolated runpod_returns tree")
    root.mkdir(parents=True, exist_ok=True)
    base = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=yes",
        "-i",
        str(args.key),
        "-p",
        str(args.port),
        args.host,
    ]
    previous = -1
    while not (root / "STOP_BACKUP").exists():
        record = {
            "utc": datetime.now(UTC).isoformat(),
            "rotation": args.rotation,
            "local_queue_modified": False,
            "training_interrupted": False,
        }
        try:
            command = [
                "python3",
                "/workspace/ejepa_operations/runpod_snapshot_v1.py",
                "--run",
                args.run,
                "--rotation",
                str(args.rotation),
            ]
            value = subprocess.run(
                base + [shlex.join(command)],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            remote = json.loads(value.stdout)
            record["remote"] = remote
            if remote["status"] == "READY" and remote["updates"] > previous:
                remote_path = remote["archive"]
                if not remote_path.startswith(args.run.rstrip("/") + "/cloud_snapshots/"):
                    raise ValueError("Remote archive escaped its assigned run")
                temporary = root / "download.tar.pending"
                with temporary.open("wb") as output:
                    subprocess.run(
                        base + [shlex.join(["cat", remote_path])],
                        stdout=output,
                        check=True,
                        timeout=180,
                    )
                    output.flush()
                    os.fsync(output.fileno())
                manifest = admit_archive(temporary, root, remote["archive_sha256"])
                record["missed_100_update_intervals"] = (
                    max(0, (remote["updates"] - previous) // 100 - 1) if previous >= 0 else 0
                )
                previous = remote["updates"]
                temporary.unlink()
                retain(root)
                record.update(
                    status="VERIFIED_LOCAL",
                    latest_updates=previous,
                    checkpoint_sha256=manifest["checkpoint_sha256"],
                )
            else:
                record.update(status="WAITING_NEXT_DURABLE_CHECKPOINT", latest_updates=previous)
        except Exception as error:
            record.update(status="BACKUP_ERROR", error=f"{type(error).__name__}: {error}")
        temporary_status = root / "BACKUP_STATUS.pending"
        temporary_status.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary_status, root / "BACKUP_STATUS.json")
        with (root / "BACKUP_HISTORY.jsonl").open("a") as history:
            history.write(json.dumps(record) + "\n")
        if args.once:
            print(json.dumps(record))
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
