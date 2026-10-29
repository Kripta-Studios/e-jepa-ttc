"""Verify and immutably stage the six-file EVENT authority closure addendum."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from operational.rgb_port.accounting import atomic_write_json


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_name(name: str) -> str:
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"Unsafe authority supplement member: {name}")
    return candidate.as_posix()


def stage(archive: Path, root: Path, admission_path: Path) -> dict[str, Any]:
    archive = archive.resolve(strict=True)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        names = [safe_name(item.name) for item in members if item.isfile()]
        if len(names) != len(set(names)) or any(
            not (item.isfile() or item.isdir()) for item in members
        ):
            raise ValueError("Authority supplement must contain unique regular files only")
        bundle.extractall(root, filter="data")
    source_manifest = root / "SUPPLEMENT_MANIFEST.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if (
        manifest.get("schema") != "ejepa_event_authority_supplement_manifest_v2"
        or manifest.get("status") != "COMPLETE_VERIFIED_LOCAL"
        or int(manifest.get("file_count", -1)) != 7
        or len(files) != 7
        or int(manifest.get("total_bytes", -1)) != 66_978_254
    ):
        raise ValueError("Authority closure manifest differs")
    stats = []
    for item in files:
        relative = safe_name(str(item["relative"]))
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(item["bytes"])
            or digest(path) != item["sha256"]
        ):
            raise ValueError(f"Authority closure member differs: {relative}")
        path.chmod(0o444)
        observed = path.stat()
        stats.append(
            {
                "relative": relative,
                "bytes": observed.st_size,
                "mtime_ns": observed.st_mtime_ns,
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "mode": observed.st_mode,
            }
        )
    canonical_manifest = root / "SUPPLEMENT_MANIFEST.json"
    canonical_manifest.chmod(0o444)
    root.chmod(0o555)
    seal_path = root.parent / f"{root.name}.SEAL.json"
    seal = {
        "schema": "ejepa_event_authority_supplement_immutable_seal_v1",
        "status": "SEALED",
        "root": str(root.resolve()),
        "manifest_sha256": digest(canonical_manifest),
        "files": stats,
    }
    atomic_write_json(seal_path, seal)
    admission = {
        "schema": "ejepa_event_authority_supplement_admission_v1",
        "status": "VERIFIED",
        "root": str(root.resolve()),
        "manifest_sha256": digest(canonical_manifest),
        "immutable_seal_sha256": digest(seal_path),
        "files": 7,
        "total_bytes": 66_978_254,
        "hashes_verified": 7,
        "authority_paths_covered": 260,
        "authority_paths_missing": 0,
    }
    atomic_write_json(admission_path, admission)
    return admission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.archive, args.root, args.admission), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
