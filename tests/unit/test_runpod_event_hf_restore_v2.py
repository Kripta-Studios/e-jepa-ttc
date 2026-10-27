from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.runpod_event_hf_restore_v2 import _stream_durable_chunks


class _Response:
    def __init__(self, blocks: list[bytes], order: list[str]) -> None:
        self._blocks = iter(blocks)
        self._order = order
        self._reads = 0

    def read(self, _size: int) -> bytes:
        self._reads += 1
        self._order.append(f"read:{self._reads}")
        return next(self._blocks, b"")


def test_stream_evicts_each_durable_chunk_before_reading_next(tmp_path: Path) -> None:
    order: list[str] = []
    ranges: list[tuple[int, int]] = []
    blocks = [b"a" * 7, b"b" * 5]
    response = _Response(blocks, order)
    partial = tmp_path / "events.h5.partial"

    def sync(_descriptor: int) -> None:
        order.append("sync")

    def advise(_descriptor: int, offset: int, length: int, _advice: int) -> None:
        ranges.append((offset, length))
        order.append("dontneed")

    def commit(offset: int, _prefix_sha256: str) -> None:
        order.append(f"commit:{offset}")

    digest = hashlib.sha256()
    with partial.open("wb", buffering=0) as stream:
        result = _stream_durable_chunks(
            response,
            stream,
            digest,
            0,
            12,
            commit,
            chunk_bytes=7,
            sync=sync,
            advise=advise,
        )

    assert order == [
        "read:1",
        "sync",
        "dontneed",
        "commit:7",
        "read:2",
        "sync",
        "dontneed",
        "commit:12",
        "read:3",
    ]
    assert ranges == [(0, 7), (7, 5)]
    assert result == {
        "offset": 12,
        "fadvise_calls": 2,
        "page_cache_bytes_advised": 12,
    }
    assert digest.hexdigest() == hashlib.sha256(b"".join(blocks)).hexdigest()


def test_stream_rejects_bytes_past_pinned_size_before_write(tmp_path: Path) -> None:
    partial = tmp_path / "events.h5.partial"
    response = _Response([b"too-large"], [])
    with partial.open("wb", buffering=0) as stream:
        with pytest.raises(ValueError, match="exceeds pinned raw size"):
            _stream_durable_chunks(
                response,
                stream,
                hashlib.sha256(),
                0,
                3,
                lambda *_args: None,
                sync=lambda _descriptor: None,
                advise=lambda *_args: None,
            )
    assert partial.read_bytes() == b""
