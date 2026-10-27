"""Bounded public-HF EVENT restore with durable hashing and scoped cache eviction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

from operational.runpod_migration.event_page_cache import durable_sha256_and_evict
from scripts.runpod_event_hf_restore_v1 import (
    EXPECTED_BYTES,
    EXPECTED_FILES,
    MIN_FINAL_RESERVE,
    REPO,
    REVISION,
    _canonical,
    _digest,
    _read,
    _rows,
)

STREAM_CHUNK_BYTES = 64 * 1024**2


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_coverage(path: Path, manifest_sha256: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read(path)
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema") == "ejepa_event_raw_coverage_v1"
        and value.get("status") == "VERIFIED"
        and value.get("raw_deferred_manifest_sha256") == manifest_sha256
        and value.get("files") == EXPECTED_FILES
        and value.get("total_bytes") == EXPECTED_BYTES
        and value.get("hashes_verified") == EXPECTED_FILES
        and value.get("availability_verified") is True
        and value.get("receipt_sha256") == _canonical(payload)
    ):
        return value
    raise ValueError("Existing EVENT raw coverage receipt is invalid or changed")


def _validate_remote(remote: dict[str, Any], manifest_sha256: str) -> None:
    payload = {key: value for key, value in remote.items() if key != "receipt_sha256"}
    if (
        remote.get("schema") != "ejepa_event_hf_remote_identity_v1"
        or remote.get("status") != "PASS"
        or remote.get("repository") != REPO
        or remote.get("revision") != REVISION
        or remote.get("raw_deferred_manifest_sha256") != manifest_sha256
        or remote.get("exact_size_and_lfs_sha256_matches") != EXPECTED_FILES
        or remote.get("receipt_sha256") != _canonical(payload)
    ):
        raise ValueError("Pinned public HF identity receipt is absent or changed")


def _volume_headroom(root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    remaining = 0
    for row in rows:
        destination = root / row["relative"]
        if destination.is_file() and destination.stat().st_size == row["bytes"]:
            continue
        remaining += row["bytes"] - _validated_resume_committed(root, row)
    largest = max((row["bytes"] for row in rows), default=0)
    volume = _read(Path("/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json"))
    age = time.time() - float(volume["observed_timestamp"])
    if not (
        volume.get("schema") == "ejepa_event_volume_snapshot_v1"
        and volume.get("quota_bytes") == 600_000_000_000
        and 0 <= int(volume.get("used_bytes", -1)) <= 600_000_000_000
        and 0 <= age <= 120
    ):
        raise RuntimeError("Fresh 600 GB event-volume snapshot is required")
    declared = 600_000_000_000 - int(volume["used_bytes"])
    usable = min(shutil.disk_usage(root).free, declared)
    if usable - remaining - largest < MIN_FINAL_RESERVE:
        raise RuntimeError("Raw restore cannot retain 50 GiB plus its largest in-flight file")
    return {
        "usable_free_bytes": usable,
        "remaining_payload_bytes": remaining,
        "largest_inflight_reserve_bytes": largest,
        "final_reserve_bytes": MIN_FINAL_RESERVE,
    }


def _resume_state_path(root: Path, filename: str) -> Path:
    key = sha256(filename.encode()).hexdigest()
    return root / ".event_raw_transfer" / f"{key}.json"


def _validated_resume_committed(root: Path, row: dict[str, Any]) -> int:
    state_path = _resume_state_path(root, row["relative"])
    partial = (root / row["relative"]).with_name(
        (root / row["relative"]).name + ".partial"
    )
    if not state_path.exists():
        return 0
    state = _read(state_path)
    payload = {key: value for key, value in state.items() if key != "receipt_sha256"}
    committed = int(state.get("committed_bytes", -1))
    valid = (
        state.get("schema") == "ejepa_event_raw_partial_v2"
        and state.get("repository") == REPO
        and state.get("revision") == REVISION
        and state.get("relative") == row["relative"]
        and state.get("expected_bytes") == row["bytes"]
        and state.get("expected_sha256") == row["sha256"]
        and state.get("receipt_sha256") == _canonical(payload)
        and 0 <= committed <= row["bytes"]
        and partial.is_file()
        and partial.stat().st_size >= committed
    )
    if not valid:
        raise ValueError(f"Raw resume state changed: {row['relative']}")
    return committed


def _stream_durable_chunks(
    response: Any,
    stream: Any,
    digest: Any,
    offset: int,
    expected_bytes: int,
    commit: Callable[[int, str], None],
    *,
    chunk_bytes: int = STREAM_CHUNK_BYTES,
    sync: Callable[[int], None] | None = None,
    advise: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, int]:
    """Write, sync, and evict every owned partial-file chunk before reading more."""
    sync_file = sync or getattr(os, "fdatasync", os.fsync)
    fadvise = advise or getattr(os, "posix_fadvise", None)
    if fadvise is None:
        raise RuntimeError("POSIX_FADV_DONTNEED is required for bounded raw restore")
    dontneed = int(getattr(os, "POSIX_FADV_DONTNEED", 4))
    calls = 0
    advised = 0
    while block := response.read(chunk_bytes):
        if offset + len(block) > expected_bytes:
            raise ValueError("Public HF response exceeds pinned raw size")
        start = offset
        stream.write(block)
        stream.flush()
        sync_file(stream.fileno())
        digest.update(block)
        offset += len(block)
        fadvise(stream.fileno(), start, len(block), dontneed)
        calls += 1
        advised += len(block)
        commit(offset, digest.hexdigest())
    return {
        "offset": offset,
        "fadvise_calls": calls,
        "page_cache_bytes_advised": advised,
    }


def _scan_committed_prefix(
    partial: Path,
    committed_bytes: int,
    expected_prefix_sha256: str,
) -> tuple[Any, int]:
    digest = sha256()
    calls = 0
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        raise RuntimeError("POSIX_FADV_DONTNEED is required for bounded raw restore")
    dontneed = int(getattr(os, "POSIX_FADV_DONTNEED", 4))
    with partial.open("rb", buffering=0) as stream:
        offset = 0
        while offset < committed_bytes:
            block = stream.read(min(STREAM_CHUNK_BYTES, committed_bytes - offset))
            if not block:
                raise ValueError("Partial raw file is shorter than its durable resume state")
            digest.update(block)
            fadvise(stream.fileno(), offset, len(block), dontneed)
            offset += len(block)
            calls += 1
    if digest.hexdigest() != expected_prefix_sha256:
        raise ValueError("Partial raw prefix differs from its durable resume state")
    return digest, calls


def _urllib_download(
    filename: str,
    root: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[Path, str, str, dict[str, Any]]:
    destination = root / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    state_path = _resume_state_path(root, filename)
    empty_sha = sha256(b"").hexdigest()
    if not state_path.exists():
        if partial.exists():
            raise ValueError("Raw partial exists without a durable resume state")
        state = {
            "schema": "ejepa_event_raw_partial_v2",
            "repository": REPO,
            "revision": REVISION,
            "relative": filename,
            "expected_bytes": expected_bytes,
            "expected_sha256": expected_sha256,
            "committed_bytes": 0,
            "prefix_sha256": empty_sha,
        }
        state["receipt_sha256"] = _canonical(state)
        _atomic_json(state_path, state)
    state = _read(state_path)
    state_payload = {key: value for key, value in state.items() if key != "receipt_sha256"}
    identity = (
        state.get("schema") == "ejepa_event_raw_partial_v2"
        and state.get("repository") == REPO
        and state.get("revision") == REVISION
        and state.get("relative") == filename
        and state.get("expected_bytes") == expected_bytes
        and state.get("expected_sha256") == expected_sha256
        and state.get("receipt_sha256") == _canonical(state_payload)
    )
    if not identity:
        raise ValueError("Raw durable resume state has different pinned identity")
    offset = int(state.get("committed_bytes", -1))
    if not 0 <= offset <= expected_bytes:
        raise ValueError("Raw durable resume offset is outside pinned size")
    if offset == 0:
        if partial.exists() and partial.stat().st_size:
            with partial.open("r+b") as stream:
                stream.truncate(0)
                stream.flush()
                getattr(os, "fdatasync", os.fsync)(stream.fileno())
        digest = sha256()
        prefix_calls = 0
    else:
        if not partial.is_file() or partial.stat().st_size < offset:
            raise ValueError("Raw partial is absent or shorter than durable resume state")
        if partial.stat().st_size > offset:
            with partial.open("r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                getattr(os, "fdatasync", os.fsync)(stream.fileno())
        digest, prefix_calls = _scan_committed_prefix(
            partial, offset, str(state.get("prefix_sha256", ""))
        )
    url = (
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
        + urllib.parse.quote(filename, safe="/")
    )
    headers = {"Accept-Encoding": "identity", "User-Agent": "ejepa-event-restore-v2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        if error.code != 416 or offset != expected_bytes:
            raise
        response = None
    etag = ""
    write_result = {"offset": offset, "fadvise_calls": 0, "page_cache_bytes_advised": 0}
    if response is not None:
        status = int(getattr(response, "status", response.getcode()))
        if (offset == 0 and status != 200) or (offset > 0 and status != 206):
            response.close()
            raise RuntimeError(f"HF did not honor pinned resume offset: status {status}")
        content_range = response.headers.get("Content-Range", "")
        if offset and (
            not content_range.startswith(f"bytes {offset}-")
            or not content_range.endswith(f"/{expected_bytes}")
        ):
            response.close()
            raise RuntimeError("HF Range response does not resume at the requested byte")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != expected_bytes - offset:
            response.close()
            raise RuntimeError("HF response length differs from pinned remaining bytes")
        etag = response.headers.get("ETag", "").strip('"')
        def commit(committed: int, prefix_sha: str) -> None:
            state["committed_bytes"] = committed
            state["prefix_sha256"] = prefix_sha
            state_payload = {
                key: value for key, value in state.items() if key != "receipt_sha256"
            }
            state["receipt_sha256"] = _canonical(state_payload)
            _atomic_json(state_path, state)

        with response, partial.open("ab", buffering=0) as stream:
            write_result = _stream_durable_chunks(
                response,
                stream,
                digest,
                offset,
                expected_bytes,
                commit,
            )
    if partial.stat().st_size != expected_bytes:
        raise ValueError("Public HF download size differs before admission")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Public HF download SHA-256 differs before admission")
    os.replace(partial, destination)
    directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    result = {
        "sha256": digest.hexdigest(),
        "bytes": expected_bytes,
        "fsync_before_eviction": True,
        "posix_fadvise_dontneed": True,
        "fadvise_calls": prefix_calls + int(write_result["fadvise_calls"]),
        "page_cache_bytes_advised": (
            offset + int(write_result["page_cache_bytes_advised"])
        ),
        "chunk_bytes": STREAM_CHUNK_BYTES,
        "scope": "owned EVENT raw partial during download",
        "global_drop_caches_used": False,
    }
    return destination.resolve(strict=True), "urllib_public_range_stream_v2", etag, result


def restore(manifest_path: Path, remote_receipt: Path, root: Path) -> dict[str, Any]:
    if os.name != "posix":
        raise RuntimeError("EVENT raw restore requires Linux POSIX fadvise")
    manifest_path = manifest_path.resolve(strict=True)
    remote_receipt = remote_receipt.resolve(strict=True)
    manifest, remote = _read(manifest_path), _read(remote_receipt)
    manifest_sha = _digest(manifest_path)
    _validate_remote(remote, manifest_sha)
    root = root.resolve()
    if not root.is_relative_to(Path("/workspace")):
        raise ValueError("EVENT raw restoration must stay on /workspace")
    root.mkdir(parents=True, exist_ok=True)
    coverage_path = root / "EVENT_RAW_COVERAGE_RECEIPT.json"
    completed = _valid_coverage(coverage_path, manifest_sha)
    if completed is not None:
        # Once admitted, never advise pages from raw files potentially in scientific use.
        return completed
    rows = sorted(_rows(manifest), key=lambda item: item["bytes"], reverse=True)
    headroom = _volume_headroom(root, rows)
    os.environ.update(
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        HF_HUB_DISABLE_UPDATE_CHECK="1",
        HF_HUB_DISABLE_XET="1",
        HF_HUB_DOWNLOAD_TIMEOUT="300",
        HF_HOME=str(root / ".hf_control"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    receipts = root / ".event_raw_receipts"
    page_cache_bytes = 0
    page_cache_calls = 0
    for row in rows:
        destination = root / row["relative"]
        backend = "existing_file"
        etag = ""
        if not destination.is_file() or destination.stat().st_size != row["bytes"]:
            observed, backend, etag, result = _urllib_download(
                row["relative"], root, row["bytes"], row["sha256"]
            )
            if observed != destination.resolve(strict=True):
                raise ValueError("HF local-dir output escaped the exact raw destination")
        else:
            result = durable_sha256_and_evict(
                destination,
                allowed_root=root,
                expected_bytes=row["bytes"],
            )
        if result["sha256"] != row["sha256"]:
            raise ValueError(f"Downloaded EVENT raw SHA-256 differs: {row['relative']}")
        page_cache_bytes += int(result.get("page_cache_bytes_advised", result["bytes"]))
        page_cache_calls += int(result["fadvise_calls"])
        receipt = {
            "schema": "ejepa_event_raw_file_receipt_v2",
            "relative": row["relative"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "revision": REVISION,
            "mtime_ns": destination.stat().st_mtime_ns,
            "fsync_before_eviction": result["fsync_before_eviction"],
            "posix_fadvise_dontneed": result["posix_fadvise_dontneed"],
            "global_drop_caches_used": False,
            "download_backend": backend,
            "http_etag": etag,
        }
        receipt["receipt_sha256"] = _canonical(receipt)
        receipt_path = receipts / (row["relative"].replace("/", "__") + ".json")
        if receipt_path.exists() and _read(receipt_path) != receipt:
            raise ValueError(f"Raw file receipt changed: {row['relative']}")
        if not receipt_path.exists():
            _atomic_json(receipt_path, receipt)
    coverage = {
        "schema": "ejepa_event_raw_coverage_v1",
        "status": "VERIFIED",
        "raw_deferred_manifest_sha256": manifest_sha,
        "content_identity_sha256": manifest["content_identity_sha256"],
        "repository": REPO,
        "revision": REVISION,
        "files": EXPECTED_FILES,
        "sequences": 40,
        "total_bytes": EXPECTED_BYTES,
        "hashes_verified": EXPECTED_FILES,
        "availability_verified": True,
        "download_backend": "per_file_public_http_receipts",
        "implicit_token_used": False,
        "fsync_before_page_cache_eviction": True,
        "posix_fadvise_dontneed": True,
        "page_cache_bytes_advised": page_cache_bytes,
        "page_cache_advice_calls": page_cache_calls,
        "global_drop_caches_used": False,
        "scientific_active_inputs_advised": False,
        "initial_headroom": headroom,
    }
    coverage["receipt_sha256"] = _canonical(coverage)
    _atomic_json(coverage_path, coverage)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.manifest, args.remote_receipt, args.root), indent=2))


if __name__ == "__main__":
    main()
