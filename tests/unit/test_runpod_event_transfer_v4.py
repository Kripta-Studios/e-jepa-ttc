from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import runpod_event_transfer_v3 as v3
from scripts import runpod_event_transfer_v4 as v4


def test_receiver_is_exact_v3_and_accepts_original_wire(tmp_path: Path) -> None:
    assert v4.REMOTE_RECEIVER is v3.REMOTE_RECEIVER
    payload = b"original-v3-wire\x00payload"
    header = {
        "relative": "events/part.bin",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    process = subprocess.run(
        [sys.executable, "-u", "-c", v4.REMOTE_RECEIVER, str(tmp_path)],
        input=json.dumps(header).encode() + b"\n" + payload + b'{"command":"END"}\n',
        capture_output=True,
        timeout=15,
        check=True,
    )
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert [row["status"] for row in replies] == ["OFFSET", "OK", "COMPLETE"]
    assert (tmp_path / "events" / "part.bin").read_bytes() == payload


def test_read_ahead_respects_count_and_byte_bounds(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(bytes([index]) * 6)
        paths.append(path)
    with v4.BoundedReadAhead(max_cache_bytes=12, max_buffer_bytes=8) as ahead:
        assert ahead.schedule(0, paths[0], 6)
        assert ahead.schedule(1, paths[1], 6)
        assert not ahead.schedule(2, paths[2], 6)
        assert ahead.pending_count == 2
        assert ahead.reserved_bytes == 12
        assert ahead.take(0).data == bytes([0]) * 6
        assert ahead.reserved_bytes == 6
        assert ahead.schedule(2, paths[2], 6)
        assert ahead.take(1).data == bytes([1]) * 6
        assert ahead.take(2).data == bytes([2]) * 6
        assert ahead.reserved_bytes == 0


def test_file_above_current_buffer_falls_back_to_streaming(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"123456789")
    with v4.BoundedReadAhead(max_cache_bytes=128, max_buffer_bytes=8) as ahead:
        assert not ahead.schedule(0, path, 9)
        assert ahead.pending_count == 0
        assert ahead.reserved_bytes == 0


def test_source_change_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "events.bin"
    path.write_bytes(b"before")
    snapshot = v4._source_stat(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed while transferring"):
        v4._assert_unchanged(path, snapshot)


def test_prefetch_worker_error_propagates(tmp_path: Path) -> None:
    path = tmp_path / "short.bin"
    path.write_bytes(b"short")
    with v4.BoundedReadAhead(max_cache_bytes=20, max_buffer_bytes=20) as ahead:
        assert ahead.schedule(0, path, 10)
        with pytest.raises(ValueError, match="differs"):
            ahead.take(0)
        assert ahead.reserved_bytes == 0
