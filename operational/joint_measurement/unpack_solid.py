"""Extract the lossless V18 ZIP payload with stdlib only and verify every original byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

MANIFEST = "BUNDLE_MANIFEST.json"


def safe_name(name: str) -> str:
    """Reject traversal, drive paths and Windows aliases before writing any member."""
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or any(part.rstrip(" .") != part for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive member: {name}")
    return name


def stream_digest(stream: BinaryIO, writer: BinaryIO | None = None) -> dict:
    """Hash streamed bytes while optionally writing their exact unmodified content."""
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        if writer is not None:
            writer.write(block)
        digest.update(block)
        size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def verify(path: Path, destination: Path | None = None) -> dict:
    """Check outer CRC/hashes and all original TAR members; never overwrite a destination."""
    if destination is not None:
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {
            "README.md",
            "unpack.py",
            "PAYLOAD.tar.xz",
            MANIFEST,
        }:
            raise ValueError("Unexpected outer ZIP membership")
        outer = json.loads(archive.read(MANIFEST))
        if outer["schema"] != "v18_essential_bundle_v1" or set(outer["files"]) != (
            set(names) - {MANIFEST}
        ):
            raise ValueError("Unknown outer ZIP manifest")
        if outer["context"].get("packaging") != "lossless_solid_tar_xz_inside_zip_v1":
            raise ValueError("Unknown solid payload format")
        for name, expected in outer["files"].items():
            with archive.open(name) as stream:
                if stream_digest(stream) != expected:
                    raise ValueError(f"Changed ZIP member: {name}")
        observed = {}
        seen = set()
        inner = None
        with archive.open("PAYLOAD.tar.xz") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|xz") as payload:
                for member in payload:
                    name = safe_name(member.name)
                    if not member.isfile() or name.casefold() in seen:
                        raise ValueError("Duplicate, linked or non-regular payload member")
                    seen.add(name.casefold())
                    stream = payload.extractfile(member)
                    if stream is None:
                        raise ValueError("Missing regular payload bytes")
                    if name == MANIFEST:
                        content = stream.read()
                        inner = json.loads(content)
                        if destination is not None:
                            (destination / MANIFEST).write_bytes(content)
                        continue
                    if destination is None:
                        observed[name] = stream_digest(stream)
                    else:
                        target = destination / name
                        if not target.resolve().is_relative_to(destination):
                            raise ValueError("Payload escaped its new extraction directory")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("xb") as output:
                            observed[name] = stream_digest(stream, output)
                    if observed[name]["bytes"] != member.size:
                        raise ValueError("Truncated payload member")
        if inner is None or inner["schema"] != "v18_essential_bundle_v1":
            raise ValueError("Missing original manifest")
        if inner["files"] != observed:
            raise ValueError("Expanded payload differs from the original byte manifest")
    with path.open("rb") as stream:
        zip_hash = stream_digest(stream)
    return {
        "status": "PASS",
        "zip_sha256": zip_hash["sha256"],
        "zip_bytes": zip_hash["bytes"],
        "payload_members": len(observed),
        "payload_bytes": sum(item["bytes"] for item in observed.values()),
        "context": inner["context"],
        "packaging": "lossless_solid_tar_xz_inside_zip_v1",
        "scope": "Outer ZIP CRC/hash and exact expanded membership/size/hash; no scientific replay",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.zip, args.destination), indent=2))
