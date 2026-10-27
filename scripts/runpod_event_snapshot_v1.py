"""Publish one immutable event checkpoint snapshot without pausing training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot(run: Path, rotation: int, fit: str) -> dict[str, object]:
    directory = run / "rotations" / str(rotation) / "fits" / fit
    names = ("CHECKPOINT_POINTER.json", "CHECKPOINT_RECEIPT.json", "UPDATE_JOURNAL.json")
    try:
        raw = {name: (directory / name).read_bytes() for name in names}
    except FileNotFoundError:
        return {"status": "NOT_READY"}
    pointer, receipt, journal = (json.loads(raw[name]) for name in names)
    step = int(pointer["completed_updates"])
    if (
        receipt.get("completed_updates") != step
        or journal.get("durable_updates") != step
        or pointer.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
        or (step % 100 and receipt.get("status") != "COMPLETE")
    ):
        return {"status": "NOT_READY", "updates": step}
    identity = f"{step:06d}_{pointer['checkpoint_sha256'][:12]}"
    root = run / "event_cloud_snapshots" / f"r{rotation}_{fit}"
    root.mkdir(parents=True, exist_ok=True)
    target, archive = root / identity, root / f"{identity}.tar"
    if archive.is_file():
        return {
            "status": "READY", "updates": step, "archive": str(archive),
            "archive_sha256": sha(archive),
        }
    pending = root / f".{identity}.{os.getpid()}.pending"
    pending.mkdir()
    try:
        relative = Path("rotations") / str(rotation) / "fits" / fit
        copied = pending / relative
        (copied / "checkpoint_versions").mkdir(parents=True)
        source = directory / "checkpoint_versions" / str(pointer["version"])
        version = copied / "checkpoint_versions" / str(pointer["version"])
        try:
            os.link(source, version)
        except OSError:
            shutil.copyfile(source, version)
        if sha(version) != pointer["checkpoint_sha256"]:
            raise ValueError("Pinned event checkpoint differs")
        for name, value in raw.items():
            (copied / name).write_bytes(value)
        for name in ("PROGRESS.json", "EVENT_EXECUTION_BINDING.json"):
            if (directory / name).is_file():
                shutil.copyfile(directory / name, copied / name)
        if any((directory / name).read_bytes() != raw[name] for name in names[:2]):
            return {"status": "NOT_READY", "reason": "POINTER_ADVANCED"}
        if receipt.get("status") == "COMPLETE":
            for name in ("ENDPOINT_ADMISSION.json", "SCIENTIFIC_CLOUD_ADMISSION.json"):
                shutil.copyfile(directory / name, copied / name)
        files = {
            path.relative_to(pending).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
            for path in pending.rglob("*")
            if path.is_file()
        }
        manifest = {
            "schema": "ejepa_event_checkpoint_snapshot_v1",
            "snapshot_exporter_sha256": sha(Path(__file__)),
            "rotation": rotation, "fit": fit, "completed_updates": step,
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "checkpoint_status": receipt["status"], "files": files,
            "training_paused_for_snapshot": False,
        }
        (pending / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(pending, target)
        temporary = archive.with_suffix(".tar.pending")
        with tarfile.open(temporary, "w", dereference=True) as output:
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    output.add(path, arcname=path.relative_to(target).as_posix())
        os.replace(temporary, archive)
        return {
            "status": "READY", "updates": step, "archive": str(archive),
            "archive_sha256": sha(archive),
        }
    finally:
        if pending.exists():
            shutil.rmtree(pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--fit", choices=("E_A5", "E_C2F"), required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run.resolve(strict=True), args.rotation, args.fit)))
