"""Evict only verified immutable event-bundle pages from this Pod's cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def memory_current() -> int:
    return int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    logical, stage, seal_path = (
        args.logical.resolve(strict=True),
        args.stage.resolve(strict=True),
        args.seal.resolve(strict=True),
    )
    manifest = json.loads((stage / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    sealed = {str(item["relative"]): item for item in seal["files"]}
    if (
        seal.get("schema") != "ejepa_immutable_local_bundle_seal_v1"
        or seal.get("status") != "SEALED"
        or seal.get("root") != str(stage)
        or len(sealed) != len(manifest["files"])
    ):
        raise ValueError("Immutable local event seal differs before cache reclaim")
    before = memory_current()
    advised_files = advised_bytes = 0
    os.sync()
    for item in manifest["files"]:
        relative, size = str(item["relative"]), int(item["bytes"])
        expected = sealed.get(relative)
        local = stage / relative
        observed = local.stat()
        if (
            expected is None
            or local.is_symlink()
            or observed.st_size != size
            or observed.st_mtime_ns != int(expected["mtime_ns"])
            or observed.st_ino != int(expected["inode"])
            or observed.st_dev != int(expected["device"])
            or observed.st_mode != int(expected["mode"])
        ):
            raise ValueError(f"Sealed stage member changed: {relative}")
        for root in (logical, stage):
            path = root / relative
            if path.is_symlink() or path.stat().st_size != size:
                raise ValueError(f"Immutable reclaim member differs: {path}")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
            advised_files += 1
            advised_bytes += size
    time.sleep(2)
    value = {
        "schema": "ejepa_event_scoped_cache_reclaim_v1",
        "status": "PASS",
        "method": "POSIX_FADV_DONTNEED",
        "scope": [str(logical), str(stage)],
        "manifest_members": len(manifest["files"]),
        "advised_files": advised_files,
        "advised_bytes": advised_bytes,
        "cgroup_memory_before_bytes": before,
        "cgroup_memory_after_bytes": memory_current(),
        "host_drop_caches_used": False,
        "memory_force_empty_used": False,
        "scientific_processes_active": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    pending = args.receipt.with_name(f".{args.receipt.name}.{os.getpid()}.pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, args.receipt)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
