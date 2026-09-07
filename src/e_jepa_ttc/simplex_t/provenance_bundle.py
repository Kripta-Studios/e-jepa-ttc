"""Transport scientific freeze references and its pinned metadata without following links."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from .bundle_creation import BundleMember
from .bundle_integrity import validate_bundle_inventory
from .scientific_freeze import read_scientific_freeze

METADATA_SUFFIXES = frozenset(
    {
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".py",
        ".ps1",
        ".psm1",
        ".md",
        ".txt",
        ".csv",
    }
)


def provenance_bundle_members(
    freeze: Path,
    *,
    freeze_sha256: str,
    roots: dict[str, Path],
    validate_scientific_authority: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict[str, BundleMember]:
    """Include the validated freeze and every explicitly pinned text/code artifact.

    Binary producer weights and feature arrays remain exact references in the
    freeze, not copies of old experts in the essential-results archive. New heads,
    predictions and history payloads have separate mandatory bundle providers.
    Paths mentioned *inside* metadata are never traversed here. The authority
    callback must establish allowed roles before the freeze reader opens pins.
    This does not authorize a holdout or assert final scientific completion.
    """

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: scientific provenance transport")

    def verify() -> dict:
        boundary()
        return read_scientific_freeze(
            freeze,
            expected_sha256=freeze_sha256,
            roots=roots,
            validate_prerequisites=validate_scientific_authority,
        )

    record = verify()
    members: dict[str, BundleMember] = {}

    def add(name: str, path: Path, digest: str) -> None:
        boundary()
        before = path.stat()
        hashed = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            while True:
                boundary()
                block = stream.read(1_048_576)
                if not block:
                    break
                count += len(block)
                if count > before.st_size:
                    raise ValueError("scientific provenance file grew")
                hashed.update(block)
        after = path.stat()
        if (
            count != before.st_size
            or (before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_ino, after.st_size, after.st_mtime_ns)
            or hashed.hexdigest() != digest
        ):
            raise ValueError("scientific provenance source bytes changed")
        if name in members:
            raise ValueError("duplicate scientific provenance archive member")
        members[name] = BundleMember(path, digest, count)

    work = roots["work"].resolve(strict=True)
    target = freeze.resolve(strict=True)
    if not target.is_relative_to(work):
        raise ValueError("scientific freeze must remain inside companion worktree")
    add("provenance/SCIENTIFIC_FREEZE.json", target, freeze_sha256)
    for pin in record["files"]:
        relative = Path(pin["relative_path"])
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("invalid provenance relative path")
        root = roots[pin["root"]].resolve(strict=True)
        source = (root / relative).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("scientific provenance source escapes its root")
        if source.suffix.lower() not in METADATA_SUFFIXES:
            continue
        name = f"provenance/{pin['root']}/{relative.as_posix()}"
        validate_bundle_inventory({name: pin["sha256"]})
        add(name, source, pin["sha256"])
    if verify() != record:
        raise ValueError("scientific freeze changed during transport binding")
    validate_bundle_inventory({name: member.sha256 for name, member in members.items()})
    return members
