"""Bounded page-cache controller for sealed EVENT producer/teacher NPZ inputs.

The controller is an operational sidecar.  It never writes an input, changes a
training guard, or invokes global cache controls.  ``POSIX_FADV_DONTNEED`` is
only advisory and is issued on read-only descriptors for immutable stage files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from operational.rgb_port.accounting import atomic_write_json
from scripts.runpod_resources_v3 import memory_snapshot

GIB = 1024**3
BUDGET_BYTES = 60_999_999_488
DEFAULT_TARGET_HEADROOM_BYTES = 8 * GIB
DEFAULT_MAX_BYTES_PER_CYCLE = 4 * GIB
DEFAULT_MAX_FILES_PER_CYCLE = 256
STAGE_ROOT = Path("/opt/ejepa_event_bundle")
ALLOWED_PREFIXES = (
    "drives/E/EJEPA_TRAIN40_CACHE_20261005/producer_inputs/",
    "workspace/e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005/teacher/",
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _safe_relative(value: object) -> str:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe sealed cache path: {value}")
    return str(relative)


@dataclass(frozen=True)
class CacheFile:
    relative: str
    path: Path
    bytes: int
    sha256: str
    mtime_ns: int
    inode: int
    device: int
    mode: int


def build_inventory(
    root: Path,
    seal_path: Path,
    *,
    expected_root: Path = STAGE_ROOT,
) -> tuple[list[CacheFile], dict[str, Any]]:
    """Bind eligible files to both the content manifest and immutable stat seal."""
    root = root.resolve(strict=True)
    if root != expected_root.resolve(strict=True):
        raise ValueError("Cache controller is restricted to the admitted /opt stage")
    seal_path = seal_path.resolve(strict=True)
    if seal_path != root.parent / f"{root.name}.SEAL.json":
        raise ValueError("Cache controller seal does not belong to the stage")
    manifest_path = root / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("profile") == "event_producers"
        and seal.get("schema") == "ejepa_immutable_local_bundle_seal_v1"
        and seal.get("status") == "SEALED"
        and seal.get("root") == str(root)
        and seal.get("manifest_sha256") == _digest(manifest_path)
        and len(seal.get("files", [])) == len(manifest.get("files", []))
    ):
        raise ValueError("Stage manifest/seal binding differs")

    manifest_rows = {_safe_relative(item["relative"]): item for item in manifest["files"]}
    seal_rows = {_safe_relative(item["relative"]): item for item in seal["files"]}
    selected: list[CacheFile] = []
    for relative, item in sorted(manifest_rows.items()):
        if not relative.endswith(".npz") or not relative.startswith(ALLOWED_PREFIXES):
            continue
        sealed = seal_rows.get(relative)
        if sealed is None or int(sealed["bytes"]) != int(item["bytes"]):
            raise ValueError(f"Eligible cache lacks matching immutable seal: {relative}")
        path = root / relative
        observed = path.lstat()
        expected = (
            int(sealed["bytes"]),
            int(sealed["mtime_ns"]),
            int(sealed["inode"]),
            int(sealed["device"]),
            int(sealed["mode"]),
        )
        actual = (
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ino,
            observed.st_dev,
            observed.st_mode,
        )
        if (
            path.is_symlink()
            or not path.is_file()
            or actual != expected
            or observed.st_mode & 0o222
        ):
            raise ValueError(f"Eligible sealed cache changed: {relative}")
        selected.append(
            CacheFile(
                relative=relative,
                path=path,
                bytes=observed.st_size,
                sha256=str(item["sha256"]),
                mtime_ns=observed.st_mtime_ns,
                inode=observed.st_ino,
                device=observed.st_dev,
                mode=observed.st_mode,
            )
        )
    if not selected:
        raise ValueError("No sealed producer/teacher NPZ inputs were selected")
    provenance = {
        "manifest_sha256": _digest(manifest_path),
        "seal_sha256": _digest(seal_path),
        "selected_identity_sha256": hashlib.sha256(
            json.dumps(
                [
                    {"relative": item.relative, "bytes": item.bytes, "sha256": item.sha256}
                    for item in selected
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "selected_files": len(selected),
        "selected_bytes": sum(item.bytes for item in selected),
        "allowed_prefixes": list(ALLOWED_PREFIXES),
    }
    return selected, provenance


def cgroup_observation(
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> dict[str, Any]:
    headroom, reports = memory_snapshot(
        cgroup_root,
        membership_path.read_text(encoding="utf-8"),
        BUDGET_BYTES,
    )
    leaf = reports[0]
    stats = leaf["memory_stat"]
    anon = int(stats.get("anon", stats.get("total_rss", stats.get("rss", 0))))
    file_cache = int(stats.get("file", stats.get("total_cache", stats.get("cache", 0))))
    return {
        "effective_headroom_bytes": int(headroom),
        "charged_current_bytes": int(leaf["current"]),
        "anon_or_rss_bytes": anon,
        "file_cache_bytes": file_cache,
        "cgroup_version": int(leaf["version"]),
        "cgroup_path": str(leaf["path"]),
        "kernel_limit_bytes": int(leaf["kernel_limit"]),
        "effective_limit_bytes": int(leaf["effective_limit"]),
    }


def _assert_same_file(item: CacheFile, descriptor: int) -> None:
    observed = os.fstat(descriptor)
    actual = (
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ino,
        observed.st_dev,
        observed.st_mode,
    )
    expected = (item.bytes, item.mtime_ns, item.inode, item.device, item.mode)
    if actual != expected or observed.st_mode & 0o222:
        raise ValueError(f"Sealed cache changed before advice: {item.relative}")


def advise_file(
    item: CacheFile,
    *,
    advise: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Advise the complete clean range through an O_RDONLY/O_NOFOLLOW descriptor."""
    fadvise = advise or getattr(os, "posix_fadvise", None)
    if fadvise is None:
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(item.path, flags)
    try:
        _assert_same_file(item, descriptor)
        fadvise(descriptor, 0, 0, int(getattr(os, "POSIX_FADV_DONTNEED", 4)))
        _assert_same_file(item, descriptor)
    finally:
        os.close(descriptor)
    return {"relative": item.relative, "offset": 0, "length": 0, "range_bytes": item.bytes}


def advise_cycle(
    inventory: list[CacheFile],
    cursor: int,
    *,
    target_headroom_bytes: int = DEFAULT_TARGET_HEADROOM_BYTES,
    max_bytes: int = DEFAULT_MAX_BYTES_PER_CYCLE,
    max_files: int = DEFAULT_MAX_FILES_PER_CYCLE,
    observe: Callable[[], dict[str, Any]] = cgroup_observation,
    advise: Callable[[int, int, int, int], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    if not inventory or not 0 <= cursor < len(inventory):
        raise ValueError("Invalid cache inventory/cursor")
    if target_headroom_bytes < 2 * GIB or max_bytes <= 0 or max_files <= 0:
        raise ValueError("Invalid cache-controller bounds")
    before = observe()
    ranges: list[dict[str, Any]] = []
    advised_bytes = 0
    while int(before["effective_headroom_bytes"]) < target_headroom_bytes and len(ranges) < min(
        max_files, len(inventory)
    ):
        item = inventory[cursor]
        if ranges and advised_bytes + item.bytes > max_bytes:
            break
        if item.bytes > max_bytes:
            cursor = (cursor + 1) % len(inventory)
            if cursor == 0:
                break
            continue
        ranges.append(advise_file(item, advise=advise))
        advised_bytes += item.bytes
        cursor = (cursor + 1) % len(inventory)
        after_each = observe()
        if int(after_each["effective_headroom_bytes"]) >= target_headroom_bytes:
            break
    after = observe()
    return cursor, {
        "schema": "ejepa_event_scoped_cache_cycle_v1",
        "status": "ADVISED" if ranges else "IDLE_HEADROOM_OK",
        "target_headroom_bytes": target_headroom_bytes,
        "max_bytes_per_cycle": max_bytes,
        "max_files_per_cycle": max_files,
        "advised_files": len(ranges),
        "advised_bytes": advised_bytes,
        "ranges": ranges,
        "before": before,
        "after": after,
        "posix_fadvise_dontneed": True,
        "read_only_descriptors": True,
        "global_drop_caches_used": False,
        "memory_force_empty_used": False,
        "training_guard_changed": False,
        "scientific_files_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--seal", type=Path, default=Path("/opt/ejepa_event_bundle.SEAL.json"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--target-headroom-bytes", type=int, default=DEFAULT_TARGET_HEADROOM_BYTES)
    parser.add_argument("--max-bytes-per-cycle", type=int, default=DEFAULT_MAX_BYTES_PER_CYCLE)
    parser.add_argument("--max-files-per-cycle", type=int, default=DEFAULT_MAX_FILES_PER_CYCLE)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if os.name != "posix" or args.interval_seconds < 2:
        raise ValueError("Controller requires POSIX and interval >=2 seconds")
    inventory, provenance = build_inventory(args.root, args.seal)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    cursor = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        while not args.stop_file.exists():
            cursor, cycle = advise_cycle(
                inventory,
                cursor,
                target_headroom_bytes=args.target_headroom_bytes,
                max_bytes=args.max_bytes_per_cycle,
                max_files=args.max_files_per_cycle,
            )
            value = {
                **cycle,
                "controller_source_sha256": _digest(Path(__file__)),
                "inventory": provenance,
                "cursor": cursor,
                "pid": os.getpid(),
                "observed_timestamp": time.time(),
            }
            atomic_write_json(args.receipt, value)
            print(json.dumps(value, separators=(",", ":")), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval_seconds)
        return 0
    finally:
        args.lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
