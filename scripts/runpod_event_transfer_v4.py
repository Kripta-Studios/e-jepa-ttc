"""Resume manifest-bound event files with bounded local read-ahead telemetry.

This is an additive sender overlay.  The remote program is exactly the v3
receiver; only local reads and observations differ.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

try:
    from scripts import runpod_event_transfer_v3 as transfer_v3
except ImportError:  # Direct ``python scripts/runpod_event_transfer_v4.py`` invocation.
    import runpod_event_transfer_v3 as transfer_v3

# Keep the deployed wire peer byte-for-byte identical to v3.
REMOTE_RECEIVER = transfer_v3.REMOTE_RECEIVER

MIB = 1024**2
PREFETCH_CACHE_BYTES = 128 * MIB
CURRENT_BUFFER_BYTES = 40 * MIB
PREFETCH_WORKERS = 2
COPY_BLOCK_BYTES = 8 * MIB


@dataclass(frozen=True)
class SourceStat:
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class BufferedSource:
    path: Path
    data: bytes
    stat: SourceStat


def _source_stat(path: Path) -> SourceStat:
    value = path.stat()
    return SourceStat(
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        device=value.st_dev,
        inode=value.st_ino,
    )


def _checked_source(value: object, expected_size: int) -> tuple[Path, SourceStat]:
    unresolved = Path(str(value))
    if unresolved.is_symlink():
        raise ValueError(f"Local source is a symlink: {unresolved}")
    source = unresolved.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Local source is not a regular file: {source}")
    snapshot = _source_stat(source)
    if snapshot.size != expected_size:
        raise ValueError(f"Local source differs: {source}")
    return source, snapshot


def _assert_unchanged(path: Path, expected: SourceStat) -> None:
    try:
        actual = _source_stat(path)
    except OSError as exc:
        raise ValueError(f"Local source disappeared while transferring: {path}") from exc
    if actual != expected:
        raise ValueError(f"Local source changed while transferring: {path}")


def _read_checked(value: object, expected_size: int) -> BufferedSource:
    source, before = _checked_source(value, expected_size)
    data = source.read_bytes()
    _assert_unchanged(source, before)
    if len(data) != expected_size:
        raise ValueError(f"Local source changed while reading: {source}")
    return BufferedSource(source, data, before)


class BoundedReadAhead:
    """At most two reads, bounded by explicit byte reservations."""

    def __init__(
        self,
        *,
        max_cache_bytes: int = PREFETCH_CACHE_BYTES,
        max_buffer_bytes: int = CURRENT_BUFFER_BYTES,
        workers: int = PREFETCH_WORKERS,
        reader: Callable[[object, int], BufferedSource] = _read_checked,
    ) -> None:
        if max_cache_bytes <= 0 or max_buffer_bytes <= 0 or workers <= 0:
            raise ValueError("Read-ahead limits must be positive")
        self.max_cache_bytes = max_cache_bytes
        self.max_buffer_bytes = max_buffer_bytes
        self.max_pending = workers
        self._reader = reader
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="event-read")
        self._pending: dict[int, tuple[int, Future[BufferedSource]]] = {}
        self._reserved = 0
        self._lock = threading.Lock()

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def schedule(self, index: int, source: object, size: int) -> bool:
        if size < 0:
            raise ValueError("Manifest file size must be nonnegative")
        # Larger files use the streaming path, keeping the current-file buffer
        # within 40 MiB even though the aggregate cache allowance is 128 MiB.
        if size > self.max_buffer_bytes:
            return False
        with self._lock:
            if index in self._pending:
                return True
            if (
                len(self._pending) >= self.max_pending
                or self._reserved + size > self.max_cache_bytes
            ):
                return False
            self._reserved += size
            try:
                future = self._executor.submit(self._reader, source, size)
            except BaseException:
                self._reserved -= size
                raise
            self._pending[index] = (size, future)
        return True

    def take(self, index: int) -> BufferedSource | None:
        with self._lock:
            pending = self._pending.pop(index, None)
        if pending is None:
            return None
        size, future = pending
        try:
            return future.result()
        finally:
            with self._lock:
                self._reserved -= size

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._pending.clear()
            self._reserved = 0

    def __enter__(self) -> BoundedReadAhead:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TransferTelemetry:
    def __init__(self, interval_seconds: float = 5.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("Telemetry interval must be positive")
        self.interval_seconds = interval_seconds
        self.started = self.last = time.monotonic()
        self.bytes_total = self.bytes_last = 0

    def add_bytes(self, count: int, *, files_done: int, files_total: int) -> None:
        self.bytes_total += count
        now = time.monotonic()
        interval = now - self.last
        if interval >= self.interval_seconds:
            print(
                json.dumps(
                    {
                        "event": "transfer_rate",
                        "files_done": files_done,
                        "files_total": files_total,
                        "bytes": self.bytes_total,
                        "interval_bytes": self.bytes_total - self.bytes_last,
                        "interval_seconds": interval,
                        "bytes_per_second": (self.bytes_total - self.bytes_last) / interval,
                        "elapsed_seconds": now - self.started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            self.last, self.bytes_last = now, self.bytes_total


def _write_buffered(
    output: BinaryIO,
    data: bytes,
    offset: int,
    telemetry: TransferTelemetry,
    number: int,
    total: int,
) -> None:
    view = memoryview(data)
    for start in range(offset, len(view), COPY_BLOCK_BYTES):
        block = view[start : start + COPY_BLOCK_BYTES]
        output.write(block)
        telemetry.add_bytes(len(block), files_done=number - 1, files_total=total)


def _write_streamed(
    output: BinaryIO,
    source: Path,
    offset: int,
    telemetry: TransferTelemetry,
    number: int,
    total: int,
) -> None:
    with source.open("rb") as stream:
        stream.seek(offset)
        while block := stream.read(COPY_BLOCK_BYTES):
            output.write(block)
            telemetry.add_bytes(len(block), files_done=number - 1, files_total=total)


def transfer(
    manifest_path: Path,
    host: str,
    user: str,
    port: int,
    identity_file: Path,
    remote_root: str,
    *,
    telemetry_interval_seconds: float = 5.0,
) -> dict:
    manifest = transfer_v3._read(manifest_path.resolve(strict=True))
    files = list(manifest.get("files", []))
    if not files or manifest.get("all_files_content_bound") is not True:
        raise ValueError("Transfer requires a nonempty, fully content-bound manifest")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", host) or host.startswith("-"):
        raise ValueError("Invalid SSH host")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", user) or user.startswith("-"):
        raise ValueError("Invalid SSH user")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_root) or ".." in remote_root.split("/"):
        raise ValueError("Invalid remote root")

    encoded = base64.b64encode(REMOTE_RECEIVER.encode()).decode()
    remote = f"python3 -u -c \"import base64;exec(base64.b64decode('{encoded}'))\" {remote_root}"
    command = [
        os.environ.get("EJEPA_TRANSFER_SSH", "ssh"),
        "-T",
        "-p",
        str(port),
        "-i",
        str(identity_file.resolve(strict=True)),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        f"{user}@{host}",
        remote,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("SSH transfer pipes are unavailable")

    sent = skipped = 0
    telemetry = TransferTelemetry(telemetry_interval_seconds)
    try:
        with BoundedReadAhead() as ahead:
            for initial in range(min(PREFETCH_WORKERS, len(files))):
                item = files[initial]
                ahead.schedule(initial, item["source"], int(item["bytes"]))

            for index, item in enumerate(files):
                number = index + 1
                size = int(item["bytes"])
                relative = transfer_v3._safe_relative(str(item["relative"]))
                sha256 = str(item["sha256"])

                read_started = time.monotonic()
                buffered = ahead.take(index)
                read_wait = time.monotonic() - read_started
                if buffered is None:
                    source, source_snapshot = _checked_source(item["source"], size)
                else:
                    source, source_snapshot = buffered.path, buffered.stat
                    _assert_unchanged(source, source_snapshot)

                next_index = index + PREFETCH_WORKERS
                if next_index < len(files):
                    next_item = files[next_index]
                    ahead.schedule(next_index, next_item["source"], int(next_item["bytes"]))

                header = {"relative": relative, "bytes": size, "sha256": sha256}
                ack_started = time.monotonic()
                process.stdin.write((json.dumps(header, separators=(",", ":")) + "\n").encode())
                process.stdin.flush()
                response = json.loads(process.stdout.readline())
                header_ack = time.monotonic() - ack_started
                payload_write = 0.0
                final_ack = 0.0
                offset = 0

                if response.get("status") == "SKIP":
                    skipped += size
                else:
                    if response.get("status") != "OFFSET":
                        raise RuntimeError(f"Unexpected receiver response: {response}")
                    offset = int(response["offset"])
                    if not 0 <= offset <= size:
                        raise ValueError("Receiver returned an invalid offset")
                    write_started = time.monotonic()
                    if buffered is None:
                        _write_streamed(
                            process.stdin, source, offset, telemetry, number, len(files)
                        )
                    else:
                        _write_buffered(
                            process.stdin,
                            buffered.data,
                            offset,
                            telemetry,
                            number,
                            len(files),
                        )
                    process.stdin.flush()
                    payload_write = time.monotonic() - write_started
                    final_started = time.monotonic()
                    response = json.loads(process.stdout.readline())
                    final_ack = time.monotonic() - final_started
                    if response.get("status") != "OK" or response.get("relative") != relative:
                        raise RuntimeError(f"Receiver rejected {relative}: {response}")
                    sent += size - offset

                _assert_unchanged(source, source_snapshot)
                print(
                    json.dumps(
                        {
                            "event": "transfer_file",
                            "file_number": number,
                            "files_total": len(files),
                            "relative": relative,
                            "status": response.get("status"),
                            "bytes": size,
                            "offset": offset,
                            "buffered": buffered is not None,
                            "read_wait_seconds": read_wait,
                            "header_ack_seconds": header_ack,
                            "payload_write_seconds": payload_write,
                            "final_ack_seconds": final_ack,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

        process.stdin.write(b'{"command":"END"}\n')
        process.stdin.flush()
        terminal = json.loads(process.stdout.readline())
        process.stdin.close()
        code = process.wait()
        if code != 0 or terminal.get("status") != "COMPLETE":
            raise RuntimeError(f"SSH receiver failed: exit={code}, terminal={terminal}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    return {
        "status": "COMPLETE",
        "manifest": str(manifest_path.resolve()),
        "files": len(files),
        "bytes_sent_this_call": sent,
        "bytes_reused": skipped,
        "remote_root": remote_root,
        "sender": "runpod_event_transfer_v4",
        "prefetch_cache_bytes": PREFETCH_CACHE_BYTES,
        "current_buffer_bytes": CURRENT_BUFFER_BYTES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--remote-root", default="/workspace/ejepa_event_bundle")
    parser.add_argument("--telemetry-interval-seconds", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        manifest = transfer_v3._read(args.manifest.resolve(strict=True))
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "files": manifest.get("file_count"),
                    "bytes": manifest.get("total_bytes"),
                    "remote_root": args.remote_root,
                    "sender": "runpod_event_transfer_v4",
                    "prefetch_cache_bytes": PREFETCH_CACHE_BYTES,
                    "current_buffer_bytes": CURRENT_BUFFER_BYTES,
                },
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            transfer(
                args.manifest,
                args.host,
                args.user,
                args.port,
                args.identity_file,
                args.remote_root,
                telemetry_interval_seconds=args.telemetry_interval_seconds,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
