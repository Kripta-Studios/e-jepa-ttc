"""Resumable, quota-gated local handoff of immutable RGB returns to Pod2."""

# ruff: noqa: E501 -- the embedded remote helper is intentionally self-contained.

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "ejepa_rgb_pod2_handoff_config_v1"
CHUNK_BYTES = 1024 * 1024
RESERVE_BYTES = 2 * 1024**3
REMOTE_ROOT = "/workspace/ejepa_rgb_handoff"
EVENT_RESERVED_BYTES = 53_687_091_200
SYSTEM_RESERVED_BYTES = 4_294_967_296
QUOTA_SCHEMA = "ejepa_event_volume_snapshot_v1"
QUOTA_RUN_ROOT = "/workspace/ejepa_event_runs"
QUOTA_VOLUME_ROOT = "/workspace"


class LocalPendingError(RuntimeError):
    """Required verified local handoff inputs do not exist yet."""

REMOTE_HELPER = r'''
import datetime, hashlib, json, os, pathlib, sys
def emit(**value):
    print(json.dumps(value, sort_keys=True), flush=True)
def hash_drop(path):
    hasher = hashlib.sha256(); position = 0
    with path.open('rb') as stream:
        while True:
            block = stream.read(67108864)
            if not block: break
            hasher.update(block)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), position, len(block), os.POSIX_FADV_DONTNEED)
            position += len(block)
    return hasher.hexdigest()
def main():
    action, root_raw, name, offset_raw, total_raw, file_sha, chunk_sha, quota_raw, age_raw, deadline_raw, reserve_raw = sys.argv[1:]
    now = datetime.datetime.now(datetime.timezone.utc)
    deadline = datetime.datetime.fromisoformat(deadline_raw)
    if now >= deadline:
        raise TimeoutError('Pod2 handoff deadline reached')
    root = pathlib.Path(root_raw)
    if str(root) != '/workspace/ejepa_rgb_handoff' or pathlib.PurePosixPath(name).name != name:
        raise ValueError('Pod2 handoff target escaped fixed root')
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError('Pod2 handoff root is a symlink')
    final, partial = root / name, root / (name + '.partial')
    if final.is_symlink() or partial.is_symlink():
        raise ValueError('Pod2 handoff file is a symlink')
    total, offset, reserve = int(total_raw), int(offset_raw), int(reserve_raw)
    if action == 'probe':
        target = final if final.is_file() else partial
        actual = hash_drop(target) if final.is_file() else None
        emit(status='PRESENT' if target.exists() else 'MISSING', final=final.is_file(), bytes=target.stat().st_size if target.exists() else 0, sha256=actual)
        return
    quota = json.loads(pathlib.Path(quota_raw).read_text(encoding='utf-8'))
    observed = quota.get('observed_timestamp')
    if quota.get('schema') != 'ejepa_event_volume_snapshot_v1' or quota.get('run_root') != '/workspace/ejepa_event_runs' or quota.get('volume_root') != '/workspace':
        raise ValueError('Pod2 quota snapshot authority differs')
    if not isinstance(observed, (int, float)) or observed > now.timestamp() or now.timestamp() - observed > int(age_raw):
        raise ValueError('Pod2 quota snapshot is stale, future, or malformed')
    quota_bytes, used_bytes = quota.get('quota_bytes'), quota.get('used_bytes')
    if not isinstance(quota_bytes, int) or not isinstance(used_bytes, int) or used_bytes < 0 or used_bytes > quota_bytes:
        raise ValueError('Pod2 quota byte counts differ')
    free_after_reservation = quota_bytes - used_bytes - 53687091200 - 4294967296
    remaining = total - offset
    if free_after_reservation < remaining + reserve:
        raise OSError('Pod2 quota cannot cover remaining bytes plus 2GiB')
    if action == 'write':
        payload = sys.stdin.buffer.read()
        if len(payload) > 1048576 or hashlib.sha256(payload).hexdigest() != chunk_sha:
            raise ValueError('Pod2 chunk bytes/hash differ')
        mode = 'r+b' if partial.exists() else 'xb'
        with partial.open(mode) as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() != offset:
                raise ValueError('Pod2 partial offset differs')
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), offset, len(payload), os.POSIX_FADV_DONTNEED)
        emit(status='WROTE', bytes=partial.stat().st_size)
        return
    if action != 'finalize' or not partial.is_file() or partial.stat().st_size != total:
        raise ValueError('Pod2 finalize state differs')
    actual = hash_drop(partial)
    if actual != file_sha:
        raise ValueError('Pod2 completed SHA-256 differs')
    os.replace(partial, final)
    descriptor = os.open(root, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    receipt = {'schema':'ejepa_rgb_pod2_file_receipt_v1','status':'PASS','name':name,'bytes':total,'sha256':actual,'final_path':str(final),'extracted':False}
    receipt_path = root / (name + '.json')
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError('Existing Pod2 receipt differs')
    temporary = root / ('.' + name + '.receipt.pending')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, receipt_path)
    emit(status='COMPLETE', bytes=total, sha256=actual, receipt=str(receipt_path))
'''


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quota_free_after_reserve(snapshot: Mapping[str, Any], now_epoch: float) -> int:
    """Validate the exact Pod2 publisher contract and derive usable free bytes."""
    observed = snapshot.get("observed_timestamp")
    quota = snapshot.get("quota_bytes")
    used = snapshot.get("used_bytes")
    if (
        snapshot.get("schema") != QUOTA_SCHEMA
        or snapshot.get("run_root") != QUOTA_RUN_ROOT
        or snapshot.get("volume_root") != QUOTA_VOLUME_ROOT
        or not isinstance(observed, (int, float))
        or observed > now_epoch
        or now_epoch - observed > 120
        or not isinstance(quota, int)
        or not isinstance(used, int)
        or used < 0
        or used > quota
    ):
        raise ValueError("Pod2 quota snapshot authority/freshness differs")
    return quota - used - EVENT_RESERVED_BYTES - SYSTEM_RESERVED_BYTES


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _identity(item: Mapping[str, Any]) -> tuple[Path, int, str]:
    path = Path(str(item.get("local_path"))).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Handoff source is not a regular file: {path}")
    receipt_raw = item.get("identity_receipt")
    if receipt_raw is None:
        size, sha = path.stat().st_size, digest(path)
    else:
        receipt = read(Path(str(receipt_raw)).resolve(strict=True))
        size, sha = receipt.get("archive_bytes"), receipt.get("archive_sha256")
        if receipt.get("status") not in {"READY", "PASS"}:
            raise ValueError(f"Handoff identity receipt did not pass: {receipt_raw}")
    if (
        not isinstance(size, int)
        or size < 0
        or not isinstance(sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sha)
        or path.stat().st_size != size
        or digest(path) != sha
    ):
        raise ValueError(f"Handoff local identity differs: {path}")
    return path, size, sha


class Pod2Handoff:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        if config.get("schema") != SCHEMA:
            raise ValueError("Pod2 handoff config schema differs")
        self.config = dict(config)
        self.runner = runner
        self.local = config["local"]
        self.remote = config["remote"]
        files = config.get("files")
        if not isinstance(files, list) or not 3 <= len(files) <= 8:
            raise ValueError("Pod2 handoff requires 3-8 explicit files")
        self.files = files
        if (
            self.remote.get("root") != REMOTE_ROOT
            or self.remote.get("host") != "213.173.98.75"
            or self.remote.get("user") != "root"
            or self.remote.get("port") != 24247
            or self.remote.get("deadline_utc") != "2026-10-27T14:09:00+00:00"
            or self.remote.get("quota_snapshot")
            != "/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json"
            or self.remote.get("quota_max_age_seconds") != 120
        ):
            raise ValueError("Pod2 handoff remote authority differs")

    def _ssh(self) -> list[str]:
        key = Path(os.path.expanduser(str(self.remote["identity_file"]))).resolve(strict=True)
        known = Path(os.path.expanduser(str(self.remote["known_hosts_file"]))).resolve(
            strict=True
        )
        return [
            "ssh", "-T", "-p", str(self.remote["port"]), "-i", str(key),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o",
            "StrictHostKeyChecking=yes", "-o", f'UserKnownHostsFile="{known.as_posix()}"',
            f"{self.remote['user']}@{self.remote['host']}",
        ]

    def _remote_call(
        self,
        action: str,
        name: str,
        offset: int,
        total: int,
        file_sha: str,
        chunk_sha: str = "-",
        payload: bytes = b"",
    ) -> dict[str, Any]:
        bootstrap = "import sys;n=int(sys.stdin.buffer.readline());exec(sys.stdin.buffer.read(n));main()"
        arguments = [
            "python3", "-c", bootstrap, action, REMOTE_ROOT, name, str(offset), str(total),
            file_sha, chunk_sha, str(self.remote["quota_snapshot"]),
            str(self.remote["quota_max_age_seconds"]), str(self.remote["deadline_utc"]),
            str(RESERVE_BYTES),
        ]
        framed = str(len(REMOTE_HELPER.encode())).encode() + b"\n" + REMOTE_HELPER.encode() + payload
        process = self.runner(
            self._ssh() + [shlex.join(arguments)],
            input=framed,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if process.returncode != 0:
            raise ConnectionError(process.stderr.decode(errors="replace").strip())
        value = json.loads(process.stdout.decode())
        if not isinstance(value, dict):
            raise ValueError("Pod2 helper returned a non-object")
        return value

    def gates(self) -> None:
        try:
            coordinator = read(Path(str(self.local["coordinator_status"])).resolve(strict=True))
            proof = read(Path(str(self.local["scientific_proof"])).resolve(strict=True))
            supplement = read(Path(str(self.local["supplement_receipt"])).resolve(strict=True))
            supplement_verification = read(
                Path(str(self.local["supplement_verification"])).resolve(strict=True)
            )
        except FileNotFoundError as error:
            raise LocalPendingError("Verified local handoff inputs are not present yet") from error
        if coordinator.get("stage") != "COMPLETE":
            raise LocalPendingError("Global RGB coordinator is not COMPLETE yet")
        archives = supplement_verification.get("archives")
        if not isinstance(archives, list):
            raise ValueError("Supplement LOCAL_VERIFICATION lacks archives evidence")
        required_supplements = [
            item
            for item in self.files
            if item.get("remote_name")
            in {
                "POD1_OPERATIONAL_SUPPLEMENT.tar",
                "POD1_EXACT_CODE_CLOSURE.tar",
                "LOCAL_PORTABILITY_FINAL.tar",
            }
        ]
        verified_identities = {
            (
                record.get("sha256", record.get("archive_sha256")),
                record.get("bytes", record.get("archive_bytes")),
            )
            for record in archives
            if isinstance(record, dict)
            and record.get("all_member_sha256_verified") is True
        }
        for item in required_supplements:
            path, size, sha = _identity(item)
            if (sha, size) not in verified_identities:
                raise ValueError(f"Supplement verification omits exact archive: {path.name}")
        if (
            proof.get("schema") != "ejepa_local_rgb_scientific_reusability_v1"
            or proof.get("status") != "PASS"
            or proof.get("archive_sha256") != coordinator.get("archive_sha256")
            or supplement.get("schema") != "ejepa_pod1_operational_supplement_v1"
            or supplement.get("status") != "READY"
            or supplement.get("primary_archive_sha256") != coordinator.get("archive_sha256")
            or supplement_verification.get("schema")
            != "ejepa_pod1_supplement_local_verification_v1"
            or supplement_verification.get("status") != "PASS"
        ):
            raise ValueError("Pod2 handoff local completion/scientific/supplement gates failed")

    def run(self) -> dict[str, Any]:
        deadline = datetime.fromisoformat(str(self.remote["deadline_utc"]))
        if datetime.now(UTC) >= deadline:
            raise TimeoutError("Pod2 handoff deadline reached before transfer")
        self.gates()
        completed = []
        status_path = Path(str(self.local["status_path"]))
        for item in self.files:
            path, total, sha = _identity(item)
            name = str(item.get("remote_name"))
            if PurePosixPath(name).name != name:
                raise ValueError("Pod2 remote file name must be one basename")
            probe = self._remote_call("probe", name, 0, total, sha)
            offset = int(probe.get("bytes", 0))
            if probe.get("final") is True:
                if offset != total or probe.get("sha256") != sha:
                    raise ValueError("Pod2 final file size/SHA differs")
                completed.append(name)
                continue
            if offset < 0 or offset > total:
                raise ValueError("Pod2 partial offset is invalid")
            with path.open("rb") as stream:
                stream.seek(offset)
                while offset < total:
                    chunk = stream.read(min(CHUNK_BYTES, total - offset))
                    if not chunk:
                        raise EOFError("Local handoff source ended early")
                    chunk_sha = hashlib.sha256(chunk).hexdigest()
                    for attempt in range(int(self.config.get("max_retries", 5))):
                        try:
                            result = self._remote_call(
                                "write", name, offset, total, sha, chunk_sha, chunk
                            )
                            break
                        except ConnectionError as error:
                            if attempt + 1 >= int(self.config.get("max_retries", 5)):
                                raise
                            resumed = self._remote_call("probe", name, 0, total, sha)
                            observed = int(resumed.get("bytes", -1))
                            if observed == offset + len(chunk):
                                result = {"bytes": observed}
                                break
                            if observed != offset:
                                raise ValueError(
                                    "Pod2 partial advanced by an unknown range"
                                ) from error
                            time.sleep(float(self.config.get("retry_seconds", 10)))
                    offset = int(result["bytes"])
                    atomic_json(
                        status_path,
                        {
                            "schema": "ejepa_rgb_pod2_handoff_status_v1",
                            "stage": "TRANSFERRING",
                            "file": name,
                            "bytes": offset,
                            "total": total,
                            "updated_utc": datetime.now(UTC).isoformat(),
                            "remote_deletion_performed": False,
                        },
                    )
            final = self._remote_call("finalize", name, total, total, sha)
            if final.get("status") != "COMPLETE" or final.get("sha256") != sha:
                raise ValueError("Pod2 final receipt differs")
            completed.append(name)
        result = {
            "schema": "ejepa_rgb_pod2_handoff_status_v1",
            "stage": "COMPLETE",
            "files": completed,
            "destination": REMOTE_ROOT,
            "extracted": False,
            "remote_deletion_performed": False,
            "updated_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(status_path, result)
        return result


__all__ = [
    "CHUNK_BYTES",
    "LocalPendingError",
    "Pod2Handoff",
    "REMOTE_ROOT",
    "SCHEMA",
    "quota_free_after_reserve",
]
