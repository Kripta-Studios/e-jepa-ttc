"""Verify or selectively restore the deduplicated JEPA pod backup (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


def digest(stream: BinaryIO) -> str:
    """Compute SHA-256 without loading a large tensor file into memory."""
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def verify(entries: list[dict[str, Any]]) -> dict[str, int | str]:
    """Verify every referenced file or archive member once, including reused data."""
    files: dict[str, str] = {}
    archives: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for entry in entries:
        source = entry["local_source"]
        if source["kind"] == "file":
            files[source["path"]] = entry["sha256"]
        else:
            archives[(source["path"], source["kind"])][source["member"]] = entry["sha256"]
    for name, expected in files.items():
        with Path(name).open("rb") as stream:
            if digest(stream) != expected:
                raise ValueError(f"Hash mismatch: {name}")
    for (name, kind), expected_members in archives.items():
        found = set()
        if kind == "zip_member":
            with zipfile.ZipFile(name) as archive:
                for member in expected_members:
                    with archive.open(member) as stream:
                        if digest(stream) != expected_members[member]:
                            raise ValueError(f"Hash mismatch: {name}:{member}")
                    found.add(member)
        else:
            with tarfile.open(name, "r:*") as archive:
                for item in archive:
                    if item.name not in expected_members:
                        continue
                    stream = archive.extractfile(item)
                    if stream is None:
                        raise ValueError(f"Not a regular member: {item.name}")
                    with stream:
                        if digest(stream) != expected_members[item.name]:
                            raise ValueError(f"Hash mismatch: {name}:{item.name}")
                    found.add(item.name)
        if found != set(expected_members):
            raise ValueError(f"Missing members in {name}: {set(expected_members) - found}")
    return {
        "status": "all_referenced_content_sha256_verified",
        "remote_paths": len(entries),
        "remote_bytes": sum(e["bytes"] for e in entries),
        "unique_regular_files": len(files),
        "archives": len(archives),
        "archive_members": sum(len(v) for v in archives.values()),
    }


def restore(entries: list[dict[str, Any]], destination: Path) -> None:
    """Restore selected paths without overwriting divergent existing files."""
    destination = destination.resolve()
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative path: {relative}")
        target = destination.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Destination escape: {relative}")
        if target.exists():
            with target.open("rb") as stream:
                if digest(stream) != entry["sha256"]:
                    raise ValueError(f"Existing file differs; refusing overwrite: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = entry["local_source"]
        # Exclusive creation avoids overwriting existing work. A failed copy remains
        # visibly invalid and is detected on the next verification/restore attempt.
        with target.open("xb") as output:
            if source["kind"] == "file":
                with Path(source["path"]).open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            elif source["kind"] == "zip_member":
                with zipfile.ZipFile(source["path"]) as archive:
                    with archive.open(source["member"]) as stream:
                        shutil.copyfileobj(stream, output)
            else:
                with tarfile.open(source["path"], "r:*") as archive:
                    stream = archive.extractfile(source["member"])
                    if stream is None:
                        raise ValueError(f"Invalid member: {source['member']}")
                    with stream:
                        shutil.copyfileobj(stream, output)
        with target.open("rb") as stream:
            if digest(stream) != entry["sha256"]:
                raise ValueError(f"Restored hash differs: {target}")


def main() -> None:
    """List, verify, or restore a subset; default is a read-only summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix", default="", help="Original relative-path prefix to select")
    parser.add_argument(
        "--storage",
        choices=("all", "existing", "new"),
        default="all",
        help="Select existing local references or newly downloaded archive members",
    )
    parser.add_argument("--verify", action="store_true", help="Rehash all selected source bytes")
    parser.add_argument(
        "--destination", type=Path, help="Restore selected paths into this directory"
    )
    parser.add_argument("--receipt", type=Path, help="Write the verification result as JSON")
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [e for e in plan["files"] if e["path"].startswith(args.prefix)]
    if args.storage != "all":
        want_new = args.storage == "new"
        entries = [
            e for e in entries if (e["local_source"]["kind"] == "new_payload_member") == want_new
        ]
    if not entries:
        raise ValueError("Prefix selects no files")
    result: dict[str, Any] = {
        "selected_paths": len(entries),
        "selected_bytes": sum(e["bytes"] for e in entries),
        "action": "summary_only",
    }
    if args.verify:
        result = verify(entries)
    if args.destination:
        restore(entries, args.destination)
        result["restored_to"] = str(args.destination.resolve())
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
