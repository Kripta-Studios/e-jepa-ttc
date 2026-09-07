"""Verify essential ZIP payload bytes against an explicit pinned member inventory."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath


def verify_bundle(
    archive: Path, inventory: dict[str, str], *, resource_ok: Callable[[], bool]
) -> dict:
    """Read every archived byte in bounded chunks; do not extract or trust CRC alone.

    Inventory authority and scientific completeness belong to the caller. This
    verifies transport integrity only, including the inventory document itself
    when its digest is supplied. No member may be omitted, duplicated or added.
    """
    if not inventory:
        raise ValueError("nonempty pinned bundle inventory required")
    for name, digest in inventory.items():
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or ":" in name
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError("canonical relative member name and SHA256 required")
    if len({name.casefold() for name in inventory}) != len(inventory):
        raise ValueError("case-aliased bundle members")
    checked_bytes = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(inventory) or {member.filename for member in members} != set(
            inventory
        ):
            raise ValueError("bundle has missing, extra or duplicate members")
        for member in members:
            if member.is_dir() or member.flag_bits & 1:
                raise ValueError("directory or encrypted bundle member")
            digest = hashlib.sha256()
            count = 0
            with bundle.open(member) as stream:
                while True:
                    if not resource_ok():
                        raise InterruptedError("PAUSED_RESOURCE: essential bundle verification")
                    data = stream.read(1024 * 1024)
                    if not data:
                        break
                    digest.update(data)
                    count += len(data)
            if count != member.file_size or digest.hexdigest() != inventory[member.filename]:
                raise ValueError("archived payload differs from pinned inventory")
            checked_bytes += count
    return {
        "status": "BUNDLE_PAYLOAD_SHA256_VERIFIED_NOT_SCIENTIFIC_ADMISSION",
        "members": len(inventory),
        "uncompressed_bytes": checked_bytes,
        "optimizer_updates": 0,
    }
