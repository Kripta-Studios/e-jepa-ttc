"""Reclaim clean cache pages only from sealed, complete RGB input TARs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from operational.rgb_port.accounting import atomic_write_json
from scripts.runpod_resources_v3 import memory_snapshot

GIB = 1024**3
MEDIA_ROOT = Path("/opt/ejepa_rgb_frames")
TARGET_HEADROOM_BYTES = 8 * GIB
POLL_SECONDS = 15.0
MEMORY_BUDGET_BYTES = 60_999_999_488
DEADLINE = datetime(2026, 10, 27, 12, 2, 19, tzinfo=UTC)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CacheTar:
    relative: str
    path: Path
    bytes: int
    sha256: str
    stat_identity: tuple[int, int, int, int, int]


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_size,
        value.st_mtime_ns,
        value.st_ino,
        value.st_dev,
        value.st_mode,
    )


def _relative(value: object) -> str:
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 5
        or path.parts[:2] != ("data", "train")
        or path.parts[-2] != "rgb_shards"
        or path.suffix != ".tar"
    ):
        raise ValueError(f"Unsafe RGB TAR path: {value}")
    return path.as_posix()


def build_inventory(
    root: Path,
    manifests: list[tuple[Path, str]],
    *,
    expected_root: Path = MEDIA_ROOT,
) -> tuple[list[CacheTar], dict[str, Any]]:
    """Hash-bind complete manifests, transfer receipts, and immutable TAR stats."""
    root = root.resolve(strict=True)
    if root != expected_root.resolve(strict=True) or len(manifests) != 2:
        raise ValueError("RGB cache reclaim requires exact /opt root and two manifests")
    selected: dict[str, CacheTar] = {}
    manifest_evidence: list[dict[str, Any]] = []
    for manifest_path, expected_sha in manifests:
        manifest_path = manifest_path.resolve(strict=True)
        if digest(manifest_path) != expected_sha:
            raise ValueError("RGB cache manifest bytes differ")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in manifest.items() if key != "receipt_sha256"}
        records = manifest.get("tars")
        if not (
            manifest.get("schema") == "ejepa_rgb_frame_supplement_v1"
            and manifest.get("status") == "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
            and manifest.get("payload_recompressed") is False
            and manifest.get("decoded_pixel_parity") is True
            and manifest.get("optimizer_updates") == 0
            and manifest.get("receipt_sha256") == canonical(payload)
            and isinstance(records, list)
            and len(records) == manifest.get("tar_count")
        ):
            raise ValueError("RGB cache manifest is not a complete immutable endpoint")
        for record in records:
            relative = _relative(record.get("relative"))
            if relative in selected:
                raise ValueError("RGB cache manifests overlap")
            path = root.joinpath(*PurePosixPath(relative).parts)
            sidecar = path.with_name(path.name + ".json")
            if (
                path.is_symlink()
                or sidecar.is_symlink()
                or not path.is_file()
                or not sidecar.is_file()
            ):
                raise ValueError("RGB cache TAR/receipt is missing or linked")
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("RGB cache TAR escapes media root")
            observed, side_stat = path.stat(), sidecar.stat()
            record_payload = {
                key: value for key, value in record.items() if key != "receipt_sha256"
            }
            if (
                observed.st_mode & 0o222
                or side_stat.st_mode & 0o222
                or observed.st_size != int(record.get("sparse_tar_bytes", -1))
                or not isinstance(record.get("sparse_tar_sha256"), str)
                or len(record["sparse_tar_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in record["sparse_tar_sha256"]
                )
                or json.loads(sidecar.read_text(encoding="utf-8")) != record
                or record.get("receipt_sha256") != canonical(record_payload)
            ):
                raise ValueError("RGB cache TAR/transfer seal differs")
            selected[relative] = CacheTar(
                relative,
                path,
                observed.st_size,
                str(record["sparse_tar_sha256"]),
                _identity(observed),
            )
        manifest_evidence.append(
            {
                "path": str(manifest_path),
                "sha256": expected_sha,
                "tar_count": manifest["tar_count"],
                "sparse_tar_bytes": manifest["sparse_tar_bytes"],
            }
        )
    inventory = [selected[key] for key in sorted(selected)]
    return inventory, {
        "manifests": manifest_evidence,
        "selected_files": len(inventory),
        "selected_bytes": sum(item.bytes for item in inventory),
        "selected_identity_sha256": canonical(
            [
                {"relative": item.relative, "bytes": item.bytes, "sha256": item.sha256}
                for item in inventory
            ]
        ),
    }


def observe_memory() -> dict[str, Any]:
    headroom, reports = memory_snapshot(
        Path("/sys/fs/cgroup"),
        Path("/proc/self/cgroup").read_text(encoding="utf-8"),
        MEMORY_BUDGET_BYTES,
    )
    leaf = reports[0]
    stats = leaf["memory_stat"]
    return {
        "effective_headroom_bytes": int(headroom),
        "charged_current_bytes": int(leaf["current"]),
        "anon_or_rss_bytes": int(stats.get("anon", stats.get("rss", 0))),
        "file_cache_bytes": int(stats.get("file", stats.get("cache", 0))),
        "effective_limit_bytes": int(leaf["effective_limit"]),
    }


def advise_file(
    item: CacheTar, *, advise: Callable[[int, int, int, int], None] | None = None
) -> None:
    fadvise = advise or getattr(os, "posix_fadvise", None)
    if fadvise is None:
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(item.path, flags)
    try:
        if _identity(os.fstat(descriptor)) != item.stat_identity:
            raise ValueError(f"RGB cache TAR changed before advice: {item.relative}")
        fadvise(descriptor, 0, 0, int(getattr(os, "POSIX_FADV_DONTNEED", 4)))
        if _identity(os.fstat(descriptor)) != item.stat_identity:
            raise ValueError(f"RGB cache TAR changed during advice: {item.relative}")
    finally:
        os.close(descriptor)


def cycle(
    inventory: list[CacheTar],
    *,
    observe: Callable[[], dict[str, Any]] = observe_memory,
    advise: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    before = observe()
    advised = 0
    if int(before["effective_headroom_bytes"]) < TARGET_HEADROOM_BYTES:
        for item in inventory:
            advise_file(item, advise=advise)
            advised += item.bytes
    after = observe()
    return {
        "schema": "ejepa_rgb_input_cache_reclaim_cycle_v1",
        "status": "ADVISED" if advised else "IDLE_HEADROOM_OK",
        "target_headroom_bytes": TARGET_HEADROOM_BYTES,
        "advised_files": len(inventory) if advised else 0,
        "advised_bytes": advised,
        "before": before,
        "after": after,
        "method": "POSIX_FADV_DONTNEED",
        "read_only_no_follow_descriptors": True,
        "global_cache_controls_used": False,
        "scientific_files_modified": False,
        "training_guard_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--manifest-sha256", action="append", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if os.name != "posix" or len(args.manifest) != 2 or len(args.manifest_sha256) != 2:
        raise ValueError("RGB cache reclaim requires POSIX and two manifest bindings")
    inventory, provenance = build_inventory(
        args.root, list(zip(args.manifest, args.manifest_sha256, strict=True))
    )
    config = {
        "root": str(args.root),
        "manifest_sha256": args.manifest_sha256,
        "target_headroom_bytes": TARGET_HEADROOM_BYTES,
        "poll_seconds": POLL_SECONDS,
        "deadline_utc": DEADLINE.isoformat(),
    }
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(args.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(lock)
        while not args.stop_file.exists() and datetime.now(UTC) < DEADLINE:
            result = cycle(inventory)
            value = {
                **result,
                "controller_source_sha256": digest(Path(__file__)),
                "config": config,
                "config_sha256": canonical(config),
                "inventory": provenance,
                "pid": os.getpid(),
                "observed_utc": datetime.now(UTC).isoformat(),
            }
            atomic_write_json(args.receipt, value)
            print(json.dumps(value, separators=(",", ":")), flush=True)
            if args.once:
                return 0
            time.sleep(POLL_SECONDS)
        return 0
    finally:
        args.lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
