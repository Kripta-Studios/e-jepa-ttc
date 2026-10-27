"""Advise away clean pages of sealed producer NPZ inputs without touching host caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json

PREFIX = "drives/E/EJEPA_TRAIN40_CACHE_20261005/producer_inputs/"


def cgroup_usage() -> int:
    return int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())


def reclaim(root: Path, seal_path: Path, receipt: Path) -> dict[str, Any]:
    root, seal_path = root.resolve(strict=True), seal_path.resolve(strict=True)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != "ejepa_immutable_local_bundle_seal_v1" or seal.get(
        "status"
    ) != "SEALED":
        raise ValueError("Immutable bundle seal is required")
    selected = [
        item
        for item in seal["files"]
        if str(item["relative"]).startswith(PREFIX) and str(item["relative"]).endswith(".npz")
    ]
    if not selected:
        raise ValueError("No sealed producer NPZ inputs were selected")
    before = cgroup_usage()
    advised_bytes = 0
    os.sync()
    for item in selected:
        path = root / str(item["relative"])
        observed = path.stat()
        if (
            path.is_symlink()
            or not path.is_file()
            or observed.st_size != int(item["bytes"])
            or observed.st_mtime_ns != int(item["mtime_ns"])
            or observed.st_ino != int(item["inode"])
            or observed.st_dev != int(item["device"])
            or observed.st_mode != int(item["mode"])
            or observed.st_mode & 0o222
        ):
            raise ValueError(f"Sealed producer input changed: {path}")
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
        advised_bytes += observed.st_size
    value = {
        "schema": "ejepa_event_input_cache_reclaim_v1",
        "status": "PASS",
        "method": "POSIX_FADV_DONTNEED",
        "scope": PREFIX,
        "selected_files": len(selected),
        "advised_bytes": advised_bytes,
        "cgroup_memory_before_bytes": before,
        "cgroup_memory_after_bytes": cgroup_usage(),
        "host_drop_caches_used": False,
        "memory_force_empty_used": False,
        "scientific_files_modified": False,
    }
    atomic_write_json(receipt, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reclaim(args.root, args.seal, args.receipt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
