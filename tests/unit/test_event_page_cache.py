from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from operational.runpod_migration.event_page_cache import durable_sha256_and_evict


def test_durable_hash_advises_only_consumed_ranges_after_fsync(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "events.h5"
    payload = b"0123456789abcdef" * 1024
    path.write_bytes(payload)
    order: list[tuple[str, int, int]] = []

    def fsync(_descriptor: int) -> None:
        order.append(("fsync", 0, 0))

    def advise(_descriptor: int, offset: int, length: int, advice: int) -> None:
        order.append((f"advise:{advice}", offset, length))

    result = durable_sha256_and_evict(
        path,
        allowed_root=raw,
        expected_bytes=len(payload),
        chunk_bytes=4096,
        fsync=fsync,
        advise=advise,
    )
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["bytes"] == len(payload)
    assert order[0][0] == "fsync"
    chunk_ranges = [(offset, length) for _, offset, length in order[2:-1]]
    assert chunk_ranges == [(offset, 4096) for offset in range(0, len(payload), 4096)]
    assert order[-1][1:] == (0, 0)


def test_page_cache_scope_rejects_outside_file_and_symlink(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "outside.h5"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="limited"):
        durable_sha256_and_evict(
            outside,
            allowed_root=raw,
            expected_bytes=7,
            advise=lambda *_args: None,
        )
    link = raw / "link.h5"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlink privilege unavailable: {error}")
    with pytest.raises(ValueError, match="limited"):
        durable_sha256_and_evict(
            link,
            allowed_root=raw,
            expected_bytes=7,
            advise=lambda *_args: None,
        )
