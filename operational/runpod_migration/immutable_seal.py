"""Fast restart barrier for a bundle fully hashed during local staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .bundle import safe_member
from .inventory import digest, read


SCHEMA = "ejepa_immutable_local_bundle_seal_v1"


def _identity(files: list[dict[str, Any]]) -> str:
    rows = [
        {
            "relative": safe_member(str(item["relative"])),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in files
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def create(root: Path, readiness: Path, output: Path) -> dict[str, Any]:
    """Seal stat identities after the staging command's complete SHA verification."""
    root = root.resolve(strict=True)
    manifest_path = root / "BUNDLE_MANIFEST.json"
    manifest = read(manifest_path)
    ready = read(readiness.resolve(strict=True))
    content = _identity(list(manifest["files"]))
    if (
        ready.get("status") != "READY"
        or ready.get("stage_root") != str(root)
        or ready.get("bundle_manifest_sha256") != digest(manifest_path)
        or ready.get("content_identity_sha256") != content
    ):
        raise ValueError("Local staging receipt does not bind this verified bundle")
    stats = []
    for item in manifest["files"]:
        relative = safe_member(str(item["relative"]))
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Staged member differs before sealing: {relative}")
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
    value = {
        "schema": SCHEMA,
        "status": "SEALED",
        "root": str(root),
        "manifest_sha256": digest(manifest_path),
        "readiness_sha256": digest(readiness),
        "content_identity_sha256": content,
        "files": stats,
        "full_content_hash_completed_by_staging": True,
        "restart_barrier_uses_local_stat_identity": True,
        "network_files_rehashed_on_restart": False,
    }
    pending = output.with_name(f".{output.name}.{os.getpid()}.pending")
    output.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, output)
    return value


def verify(root: Path, seal_path: Path) -> dict[str, Any]:
    """Validate a sealed local mirror without re-reading every file body."""
    root, seal_path = root.resolve(strict=True), seal_path.resolve(strict=True)
    manifest_path = root / "BUNDLE_MANIFEST.json"
    manifest, seal = read(manifest_path), read(seal_path)
    if (
        seal.get("schema") != SCHEMA
        or seal.get("status") != "SEALED"
        or seal.get("root") != str(root)
        or seal.get("manifest_sha256") != digest(manifest_path)
        or seal.get("content_identity_sha256") != _identity(list(manifest["files"]))
        or len(seal.get("files", [])) != len(manifest["files"])
    ):
        raise ValueError("Immutable local bundle seal differs")
    for item in seal["files"]:
        relative = safe_member(str(item["relative"]))
        path = root / relative
        observed = path.stat()
        expected = (
            int(item["bytes"]),
            int(item["mtime_ns"]),
            int(item["device"]),
            int(item["inode"]),
            int(item["mode"]),
        )
        actual = (
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
        )
        if path.is_symlink() or not path.is_file() or actual != expected or observed.st_mode & 0o222:
            raise ValueError(f"Sealed local member changed: {relative}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--root", type=Path, required=True)
    create_parser.add_argument("--readiness", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    result = (
        create(args.root, args.readiness, args.output)
        if args.command == "create"
        else verify(args.root, args.seal)
    )
    print(json.dumps({"status": "PASS", "files": len(result["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
