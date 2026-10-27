"""Snapshot named Pod2 outputs with SHA manifests; never mutate source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

OUTPUT = Path("/workspace/ejepa_preservation")


def publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pending")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def groups() -> dict[str, list[str]]:
    result = {
        "final_fits": ["/workspace/ejepa_final_runs/final/fits"],
        "final_generated": ["/workspace/ejepa_final_runs/final"],
        "final_inputs": ["/workspace/ejepa_final_inputs"],
        "event_runtime": ["/workspace/ejepa_event_operations"],
        "final_runtime": ["/workspace/ejepa_final_operations"],
        "event_top_metadata": ["/workspace/ejepa_event_runs"],
        "test12_probe_code": ["/workspace/ejepa_test12_probe"],
        "final_diagnostics": ["/workspace/ejepa_final_diagnostics"],
        "final_delivery": ["/workspace/ejepa_final_runs/final"],
        "test12_event_metadata": ["/workspace/ejepa_test12_events"],
        "test12_rgb_metadata": ["/workspace/ejepa_test12_rgb"],
        "rgb_fold_closure": ["/workspace/ejepa_rgb_handoff/extracted_v10"],
        "final_auxiliary_runtime": [
            "/workspace/ejepa_test12_cache_controller", "/workspace/ejepa_preservation"
        ],
    }
    for rotation in (2, 3, 4):
        root = f"/workspace/ejepa_event_runs/rotations/{rotation}"
        for part in ("fits", "generation", "cache"):
            result[f"event_r{rotation}_{part}"] = [f"{root}/{part}"]
        result[f"event_r{rotation}_support"] = [root]
    base = "/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement"
    result["frozen_code"] = [
        f"{base}/{name}" for name in ("operational", "src", "tests", "configs")
    ]
    result["frozen_project_metadata"] = [
        f"{base}/{name}"
        for name in (
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "AGENTS.md",
            "README.md",
            "LICENSE",
            "requirements.txt",
            "requirements-linux.txt",
        )
    ]
    for path in Path("/workspace/ejepa_test12_events/sequences").glob("*.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "COMPLETE":
            result["test12_events_" + path.stem] = [str(path)]
    for path in Path("/workspace/ejepa_test12_rgb/consumed").glob("*.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "PASS":
            result["test12_predictions_" + path.stem] = [str(path)]
    return result


def selected(group: str) -> Iterator[tuple[Path, str]]:
    if group.startswith("test12_predictions_"):
        receipt = Path(groups()[group][0])
        value = json.loads(receipt.read_text())
        roots = (
            Path("/workspace/ejepa_final_runs/final/test12"),
            Path("/workspace/ejepa_test12_rgb"),
        )
        paths = [
            receipt,
            *(Path(v) for v in value["source_receipts"].values()),
            *(Path(v["path"]) for v in value["outputs"]),
        ]
        for path in paths:
            resolved = path.resolve(strict=True)
            if not any(resolved.is_relative_to(root) for root in roots):
                raise ValueError("Prediction preservation path escaped output roots")
            yield path, "file"
        return
    if group.startswith("test12_events_"):
        receipt = Path(groups()[group][0])
        value = json.loads(receipt.read_text())
        root = Path("/workspace/ejepa_test12_events").resolve()
        paths = [receipt, Path(value["raw_receipt_path"])]
        for item in value["tensors"].values():
            paths.extend((Path(item["path"]), Path(item["receipt_path"])))
        for path in paths:
            path.resolve(strict=True).relative_to(root)
            yield path, "file"
        return
    for root_name in groups()[group]:
        root = Path(root_name)
        if not root.exists():
            continue
        if root.is_file():
            yield root, "symlink" if root.is_symlink() else "file"
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            current = Path(directory)
            relative = current.relative_to(root)
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            if group == "event_top_metadata":
                dirs[:] = []
            if group.startswith("event_r") and group.endswith("_support") and not relative.parts:
                dirs[:] = [d for d in dirs if d not in ("fits", "generation")]
            if group == "final_generated" and not relative.parts:
                dirs[:] = [d for d in dirs if d != "fits"]
            if group == "final_delivery" and not relative.parts:
                dirs[:] = [d for d in dirs if d == "test12"]
            if group == "test12_event_metadata" and not relative.parts:
                dirs[:] = [d for d in dirs if d != "tensors"]
            if group == "test12_rgb_metadata" and not relative.parts:
                dirs[:] = [d for d in dirs if d not in ("data", ".cache")]
            if group == "test12_probe_code" and not relative.parts:
                dirs[:] = [d for d in dirs if d not in ("raw", "result")]
            for d in list(dirs):
                path = current / d
                if path.is_symlink():
                    yield path, "symlink"
                    dirs.remove(d)
            entries = {entry.name: entry for entry in os.scandir(current)}
            for name in sorted(files):
                path = current / name
                if path.suffix in (".lock", ".pyc", ".pending", ".partial", ".tmp"):
                    continue
                if group in ("event_runtime", "final_runtime") and path.suffix in (
                    ".tar",
                    ".gz",
                    ".zip",
                ):
                    continue
                yield path, "symlink" if entries[name].is_symlink() else "file"


def inventory() -> None:
    output = {"schema": "pod2_preservation_inventory_v1", "started": time.time(), "groups": {}}
    for group in groups():
        entries = []
        for path, kind in selected(group):
            stat = path.lstat()
            item = {
                "path": str(path),
                "kind": kind,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if kind == "symlink":
                item["target"] = os.readlink(path)
            entries.append(item)
        output["groups"][group] = {
            "files": len(entries),
            "bytes": sum(e["bytes"] for e in entries),
            "entries": entries,
        }
        output["updated"] = time.time()
        publish(OUTPUT / "INVENTORY.json", output)
    output["status"] = "COMPLETE"
    publish(OUTPUT / "INVENTORY.json", output)


class HashedReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        value = self.stream.read(size)
        self.sha.update(value)
        return value


def uncached_sha256(path: Path) -> str:
    """Hash a temporary backup archive without retaining its pages in Pod RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        offset = 0
        for data in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(data)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), offset, len(data), os.POSIX_FADV_DONTNEED)
            offset += len(data)
    return digest.hexdigest()


def archive(group: str) -> None:
    if group not in groups():
        raise ValueError("Unregistered preservation group")
    folder = Path("/tmp/ejepa_preservation")
    folder.mkdir(exist_ok=True)
    compressed = group == "event_runtime"
    destination = folder / (f"{group}.tar.gz" if compressed else f"{group}.tar")
    pending = destination.with_suffix(".pending")
    inventory_path = OUTPUT / "INVENTORY.json"
    known = (
        json.loads(inventory_path.read_text()).get("groups", {}).get(group)
        if inventory_path.exists()
        else None
    )
    required = (
        known["bytes"]
        if known
        else sum(p.lstat().st_size for p, kind in selected(group) if kind == "file")
    )
    if shutil.disk_usage(folder).free < required + 10 * 1024**3:
        raise RuntimeError("Temporary archive exceeds container disk reserve; split this group")
    manifest = {
        "schema": "pod2_preservation_archive_v1",
        "group": group,
        "started": time.time(),
        "files": {},
        "symlinks": {},
    }
    expected_files = {}
    if group.startswith("test12_predictions_"):
        source = json.loads(Path(groups()[group][0]).read_text())
        expected_files = {item["path"]: item["sha256"] for item in source["outputs"]}
    elif group.startswith("test12_events_"):
        source = json.loads(Path(groups()[group][0]).read_text())
        expected_files = {item["path"]: item["sha256"] for item in source["tensors"].values()}
    options = {"compresslevel": 1} if compressed else {}
    with tarfile.open(
        pending,
        "w:gz" if compressed else "w",
        format=tarfile.PAX_FORMAT,
        dereference=True,
        **options,
    ) as tar:
        flushed_bytes = 0
        for path, kind in selected(group):
            relative = str(path).lstrip("/")
            if kind == "symlink":
                manifest["symlinks"][relative] = os.readlink(path)
                continue
            before = path.stat()
            info = tarfile.TarInfo(relative)
            info.size = before.st_size
            info.mode = stat.S_IMODE(before.st_mode)
            info.uid = before.st_uid
            info.gid = before.st_gid
            info.mtime = before.st_mtime
            with path.open("rb") as stream:
                reader = HashedReader(stream)
                tar.addfile(info, reader)
                if (
                    group.startswith("event_r") or group in ("event_runtime", "event_top_metadata")
                ) and hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            if str(path) in expected_files and reader.sha.hexdigest() != expected_files[str(path)]:
                raise ValueError("Completed sequence output differs from scientific receipt")
            immutable = group.startswith("event_r") and group.endswith(("_fits", "_generation"))
            after = before if immutable else path.stat()
            manifest["files"][relative] = {
                "bytes": before.st_size,
                "sha256": reader.sha.hexdigest(),
                "source_mtime_ns": before.st_mtime_ns,
                "source_changed_during_snapshot": None
                if immutable
                else ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)),
                "snapshot_policy": (
                    "completed_fit_or_generation_content_hash"
                    if immutable
                    else "content_hash_and_source_stat_recheck"
                ),
            }
            flushed_bytes += before.st_size
            if flushed_bytes >= 64 * 1024**2 and hasattr(os, "posix_fadvise"):
                tar.fileobj.flush()
                os.fsync(tar.fileobj.fileno())
                os.posix_fadvise(tar.fileobj.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                flushed_bytes = 0
        manifest["finished"] = time.time()
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("PRESERVATION_MANIFEST.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    os.replace(pending, destination)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    digest = uncached_sha256(destination)
    result = {
        "schema": "pod2_preservation_archive_receipt_v1",
        "status": "READY",
        "group": group,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "files": len(manifest["files"]),
        "symlinks": manifest["symlinks"],
        "timestamp": time.time(),
    }
    publish(OUTPUT / f"{group}.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "archive"])
    parser.add_argument("--group")
    args = parser.parse_args()
    if args.action == "inventory":
        inventory()
    else:
        archive(args.group)
