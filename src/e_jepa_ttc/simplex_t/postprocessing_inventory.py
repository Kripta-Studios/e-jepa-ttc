"""Content pins for the newly owned postprocessing directory, not a final delivery."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from .bundle_integrity import validate_bundle_inventory


class _FilePin(TypedDict):
    sha256: str
    bytes: int


def inventory_postprocessing(
    output: Path, *, resource_ok: Callable[[], bool], sealed_manifest_present: bool = False
) -> dict:
    """Hash regular outputs in bounded reads, refusing links and unfinished files.

    Call only on the fresh directory owned by the postprocessor, before writing
    POSTPROCESSING.json, or with sealed_manifest_present after its separate validation.
    Scientific graph authority is established by its caller;
    this inventory does not include source publications or certify completion.
    """

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: postprocessing inventory")

    boundary()
    root = output.resolve(strict=True)
    if (
        output.is_symlink()
        or getattr(output.lstat(), "st_file_attributes", 0) & 1024
        or not root.is_dir()
    ):
        raise ValueError("owned regular output directory required")
    members: dict[str, _FilePin] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        boundary()
        for path in sorted(directory.iterdir()):
            boundary()
            before = path.lstat()
            if path.is_symlink() or getattr(before, "st_file_attributes", 0) & 1024:
                raise ValueError("linked postprocessing output forbidden")
            if stat.S_ISDIR(before.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("regular postprocessing files required")
            name = path.relative_to(root).as_posix()
            if name == "POSTPROCESSING.json" and sealed_manifest_present:
                continue
            if name == "POSTPROCESSING.json" or path.suffix.lower() in {".partial", ".tmp"}:
                raise ValueError("unfinished or already inventoried postprocessing output")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while True:
                    boundary()
                    block = stream.read(1_048_576)
                    if not block:
                        break
                    size += len(block)
                    if size > before.st_size:
                        raise ValueError("postprocessing file grew during inventory")
                    digest.update(block)
            after = path.lstat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or size != before.st_size:
                raise ValueError("postprocessing file changed during inventory")
            members[name] = {"sha256": digest.hexdigest(), "bytes": size}
    validate_bundle_inventory({name: entry["sha256"] for name, entry in members.items()})
    return {
        "schema": "simplex_t_postprocessing_inventory_v1",
        "scope": "OWNED_POSTPROCESSING_OUTPUTS_EXCLUDING_SELF_MANIFEST",
        "members": members,
        "files": len(members),
        "bytes": sum(entry["bytes"] for entry in members.values()),
    }
