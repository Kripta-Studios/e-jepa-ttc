"""Transport pins for input-only history indices, independent of feature-cache progress."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .bundle_creation import BundleMember


@dataclass(frozen=True)
class HistoryPoolPins:
    """Explicit index and dedup manifests acknowledged for one permitted pool."""

    index: Path
    index_sha256: str
    dedup: Path
    dedup_sha256: str


def history_bundle_members(
    pools: dict[str, HistoryPoolPins],
    *,
    work_root: Path,
    validate_authority: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict[str, BundleMember]:
    """Verify and bind six input-index files per pool, never opening referenced media.

    The caller binds exact pool availability and role/time/producer authority.
    This proves transport bytes, not loader parity, scientific freeze or replay.
    Source paths embedded in manifests are retained as references, not traversed.
    """
    if "D0" not in pools or set(pools) - {"D0", "D1", "DENSE_OLD"}:
        raise ValueError("registered history pools including D0 required")
    validate_authority()
    work = work_root.resolve(strict=True)
    members = {}

    def add(name: str, path: Path, expected: str) -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: history bundle inventory")
        target = path.resolve(strict=True)
        if not target.is_relative_to(work) or not target.is_file():
            raise ValueError("history payload outside companion worktree")
        digest = hashlib.sha256()
        size = 0
        expected_size = target.stat().st_size
        with target.open("rb") as stream:
            while True:
                if not resource_ok():
                    raise InterruptedError("PAUSED_RESOURCE: history bundle inventory")
                block = stream.read(1_048_576)
                if not block:
                    break
                size += len(block)
                if size > expected_size:
                    raise ValueError("history payload grew")
                digest.update(block)
        if size != expected_size or digest.hexdigest() != expected:
            raise ValueError("history payload hash or size differs")
        members[name] = BundleMember(target, expected, size)

    def read(path: Path, expected: str) -> dict:
        with path.open("rb") as stream:
            data = stream.read(8_388_609)
        if len(data) > 8_388_608 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("history metadata changed or exceeds bound")
        return json.loads(data)

    for pool, pins in sorted(pools.items()):
        prefix = f"history/{pool}/"
        add(prefix + "INDEX_MANIFEST.json", pins.index, pins.index_sha256)
        add(prefix + "DEDUP_MANIFEST.json", pins.dedup, pins.dedup_sha256)
        index, dedup = read(pins.index, pins.index_sha256), read(pins.dedup, pins.dedup_sha256)
        linked = (
            dedup.get("identity", {}).get("index_manifest_sha256")
            if pool == "D1"
            else dedup.get("input_manifest_sha256")
        )
        rows = dedup["outputs"]
        if (
            linked != pins.index_sha256
            or len(rows) != 3
            or {row["path"] for row in rows} != {f"outer{fold}.npz" for fold in range(3)}
        ):
            raise ValueError("history dedup lineage or three-fold coverage differs")
        add(
            prefix + "query_context_index.npz",
            pins.index.parent / "query_context_index.npz",
            index["index_sha256"],
        )
        for row in rows:
            add(prefix + row["path"], pins.dedup.parent / row["path"], row["sha256"])
    validate_authority()
    return members
