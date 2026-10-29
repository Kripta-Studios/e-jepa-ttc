"""Scoped page-cache eviction for completed, durable EVENT raw downloads."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_CHUNK_BYTES = 64 * 1024**2


def durable_sha256_and_evict(
    path: Path,
    *,
    allowed_root: Path,
    expected_bytes: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    fsync: Callable[[int], None] = os.fsync,
    advise: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Hash one closed raw file and evict only ranges already read after fsync."""
    path = path.absolute()
    root = allowed_root.resolve(strict=True)
    if (
        os.name != "posix"
        and advise is None
        or path.is_symlink()
        or not path.resolve(strict=True).is_relative_to(root)
        or expected_bytes <= 0
        or chunk_bytes <= 0
    ):
        raise ValueError("Page-cache eviction is limited to regular raw files under its root")
    fadvise = advise or getattr(os, "posix_fadvise", None)
    if fadvise is None:
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    sequential = int(getattr(os, "POSIX_FADV_SEQUENTIAL", 2))
    dontneed = int(getattr(os, "POSIX_FADV_DONTNEED", 4))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    offset = 0
    calls = 0
    try:
        stat = os.fstat(descriptor)
        if not os.path.isfile(path) or stat.st_size != expected_bytes:
            raise ValueError("Raw file size changed before durable hashing")
        # The downloader has closed its writer before this function is called.
        # fsync establishes durability before any clean cache pages are advised away.
        fsync(descriptor)
        fadvise(descriptor, 0, 0, sequential)
        while True:
            block = os.read(descriptor, chunk_bytes)
            if not block:
                break
            digest.update(block)
            length = len(block)
            fadvise(descriptor, offset, length, dontneed)
            offset += length
            calls += 1
        if offset != expected_bytes:
            raise ValueError("Raw file size changed during durable hashing")
        # Drop readahead/tail pages not covered by aligned chunk advisories.
        fadvise(descriptor, 0, 0, dontneed)
        calls += 1
    finally:
        os.close(descriptor)
    return {
        "sha256": digest.hexdigest(),
        "bytes": offset,
        "fsync_before_eviction": True,
        "posix_fadvise_dontneed": True,
        "fadvise_calls": calls,
        "chunk_bytes": chunk_bytes,
        "scope": "completed EVENT raw file only",
        "global_drop_caches_used": False,
    }


__all__ = ["DEFAULT_CHUNK_BYTES", "durable_sha256_and_evict"]
