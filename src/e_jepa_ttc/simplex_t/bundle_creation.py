"""Create an essential-results ZIP from explicit source pins, then verify every byte."""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .bundle_integrity import validate_bundle_inventory, verify_bundle


@dataclass(frozen=True)
class BundleMember:
    """Caller-authorized file, expected byte count and content digest."""

    path: Path
    sha256: str
    bytes: int


def create_verified_bundle(
    archive: Path,
    members: dict[str, BundleMember],
    *,
    validate_inventory_authority: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict:
    """Stream immutable inventory into a new ZIP and publish only after verification.

    The mandatory authority callback must establish inventory completeness and
    authorization. Resource admission must reserve the pending archive separately
    from other outputs. No directory scanning, implicit data discovery, uploads,
    deletion, refitting or scientific-completion claim occurs here. A failed
    partial ZIP is retained for audit; retry requires a new archive path.
    """
    expected = {name: member.sha256 for name, member in members.items()}
    validate_bundle_inventory(expected)
    if archive.suffix.lower() != ".zip":
        raise ValueError("explicit ZIP destination required")
    partial = archive.with_suffix(".zip.partial")
    if archive.exists() or partial.exists():
        raise FileExistsError("preserve existing final or partial bundle")
    if any(type(member.bytes) is not int or member.bytes < 0 for member in members.values()):
        raise ValueError("exact nonnegative source byte counts required")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: essential bundle creation")

    validate_inventory_authority()
    boundary()
    for member in members.values():
        if member.path.resolve() in {archive.resolve(), partial.resolve()}:
            raise ValueError("bundle cannot contain its own output")
        if not member.path.is_file() or member.path.stat().st_size != member.bytes:
            raise ValueError("bundle source size or file identity differs")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as destination:
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            for name in sorted(members):
                boundary()
                member = members[name]
                digest = hashlib.sha256()
                count = 0
                with (
                    member.path.open("rb") as source,
                    bundle.open(name, "w", force_zip64=True) as target,
                ):
                    while True:
                        boundary()
                        block = source.read(1_048_576)
                        if not block:
                            break
                        count += len(block)
                        if count > member.bytes:
                            raise ValueError("bundle source grew during packaging")
                        digest.update(block)
                        target.write(block)
                if count != member.bytes or digest.hexdigest() != member.sha256:
                    raise ValueError("bundle source differs from frozen inventory")
        destination.flush()
        os.fsync(destination.fileno())
    validate_inventory_authority()
    verified = verify_bundle(partial, expected, resource_ok=resource_ok)
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        while True:
            boundary()
            block = stream.read(1_048_576)
            if not block:
                break
            digest.update(block)
    boundary()
    # Rename, not replace: an existing final path is never intentionally overwritten.
    if archive.exists():
        raise FileExistsError("final bundle appeared during verification")
    partial.rename(archive)
    return {
        "status": "PINNED_BUNDLE_CREATED_AND_VERIFIED_NOT_SCIENTIFIC_COMPLETION",
        "path": str(archive),
        "sha256": digest.hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "members": verified["members"],
        "uncompressed_bytes": verified["uncompressed_bytes"],
        "optimizer_updates": 0,
    }
