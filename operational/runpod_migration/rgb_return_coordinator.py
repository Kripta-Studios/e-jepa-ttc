"""Recoverable Windows coordinator for the final complete RGB cloud return."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, Self

from .rgb_return_archive import admit_local_rgb_return, sha256

CONFIG_SCHEMA = "ejepa_rgb_return_coordinator_config_v1"
RESULT_SCHEMA = "ejepa_cloud_rgb_downstream_queue_v1"
EXPORT_SCHEMA = "ejepa_cloud_rgb_complete_export_v1"
POLL_SECONDS = 60


class RetryableRemoteError(RuntimeError):
    """An SSH/SCP connection failed and may be retried without changing state."""


class ScientificVerificationError(RuntimeError):
    """The local original scientific verifier failed; never retry as a network issue."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _expand_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Configured path must be a nonempty string")
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _remote_path(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise ValueError(f"Unsafe remote path: {value!r}")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError(f"Remote path traversal is forbidden: {value}")
    return value


def _normalize_flat_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Build the closed remote argv from the operator-facing path-only contract."""
    if "ssh" in raw:
        return dict(raw)
    endpoint = raw.get("ssh_host")
    if not isinstance(endpoint, str) or endpoint.count("@") != 1:
        raise ValueError("ssh_host must use user@host")
    user, host = endpoint.split("@", 1)
    roots = raw.get("remote_run_roots")
    rotations_raw = raw.get("rotations", [2, 3, 4])
    rotations = tuple(rotations_raw) if isinstance(rotations_raw, list) else ()
    if rotations not in {(2, 3, 4), (4,)}:
        raise ValueError("rotations must be exactly [2,3,4] or individual [4]")
    if not isinstance(roots, dict) or set(roots) != set(map(str, rotations)):
        raise ValueError("remote_run_roots must equal the selected rotations exactly")
    python = _remote_path(raw.get("remote_python"))
    entrypoint = _remote_path(raw.get("remote_export_entrypoint"))
    coverage = _remote_path(raw.get("remote_coverage"))
    authorization = _remote_path(raw.get("remote_authorization_root"))
    archive = _remote_path(raw.get("remote_archive"))
    sources = raw.get("remote_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("remote_sources must be a nonempty allowlist")
    source_paths = [_remote_path(value) for value in sources]
    exporter = ["env", "PYTHONPATH=", python, entrypoint]
    result_paths: list[str] = []
    for rotation in rotations:
        item = roots[str(rotation)]
        if not isinstance(item, dict) or set(item) != {"a5", "c2f", "downstream"}:
            raise ValueError(f"R{rotation} requires exactly a5, c2f and downstream roots")
        a5 = _remote_path(item["a5"])
        c2f = _remote_path(item["c2f"])
        downstream = _remote_path(item["downstream"])
        exporter.extend(("--a5-run", f"{rotation}={a5}"))
        exporter.extend(("--c2f-run", f"{rotation}={c2f}"))
        exporter.extend(("--downstream-run", f"{rotation}={downstream}"))
        result_paths.append(f"{downstream}/RGB_DOWNSTREAM_RESULT.json")
    if rotations == (4,):
        exporter.extend(("--rotation", "4"))
    exporter.extend(("--coverage-receipt", coverage))
    exporter.extend(("--authorization-root", authorization))
    for source in source_paths:
        exporter.extend(("--operational-source", source))
    exporter.extend(("--output", archive))
    return {
        "schema": raw.get("schema"),
        "rotations": list(rotations),
        "poll_seconds": raw.get("poll_seconds"),
        "ssh": {
            "host": host,
            "user": user,
            "port": raw.get("ssh_port"),
            "identity_file": raw.get("ssh_key"),
            "known_hosts_file": raw.get("ssh_known_hosts"),
            "connect_timeout_seconds": raw.get("ssh_connect_timeout_seconds", 15),
        },
        "remote": {
            "result_paths": result_paths,
            "exporter_argv": exporter,
            "archive_path": archive,
            "receipt_path": archive + ".json",
        },
        "local": {
            "incoming_dir": raw.get("local_incoming_dir"),
            "final_rgb_dir": raw.get("local_final_rgb_dir"),
            "status_path": raw.get("local_status_path"),
            "history_path": raw.get("local_history_path"),
            "lock_path": raw.get("local_lock_path"),
            "scientific_verifier": raw.get("scientific_verifier"),
            "scientific_proof_output": raw.get("scientific_proof_output"),
            "python_executable": raw.get("local_python_executable"),
        },
    }


def _validate_config(raw: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    raw = _normalize_flat_config(raw)
    if raw.get("schema") != CONFIG_SCHEMA or raw.get("poll_seconds") != POLL_SECONDS:
        raise ValueError("RGB coordinator config schema/poll interval is invalid")
    ssh = raw.get("ssh")
    remote = raw.get("remote")
    local = raw.get("local")
    if not all(isinstance(value, dict) for value in (ssh, remote, local)):
        raise ValueError("Coordinator requires ssh, remote and local objects")
    assert isinstance(ssh, dict) and isinstance(remote, dict) and isinstance(local, dict)
    rotations_raw = raw.get("rotations", [2, 3, 4])
    rotations = tuple(rotations_raw) if isinstance(rotations_raw, list) else ()
    if rotations not in {(2, 3, 4), (4,)}:
        raise ValueError("Coordinator rotation closure is invalid")
    host, user, port = ssh.get("host"), ssh.get("user"), ssh.get("port")
    if (
        not isinstance(host, str)
        or not re.fullmatch(r"[A-Za-z0-9_.:-]+", host)
        or host.startswith("-")
        or not isinstance(user, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", user)
        or user.startswith("-")
        or not isinstance(port, int)
        or not 1 <= port <= 65535
    ):
        raise ValueError("SSH endpoint is invalid")
    identity = _expand_path(ssh.get("identity_file")).resolve(strict=True)
    known_hosts = _expand_path(ssh.get("known_hosts_file")).resolve(strict=True)
    timeout = ssh.get("connect_timeout_seconds", 15)
    if not isinstance(timeout, int) or not 1 <= timeout <= 120:
        raise ValueError("SSH connect timeout is invalid")
    results = remote.get("result_paths")
    if not isinstance(results, list) or len(results) != len(rotations):
        raise ValueError("Remote RGB result paths must equal the selected rotations")
    result_paths = [_remote_path(value) for value in results]
    if len(set(result_paths)) != len(rotations):
        raise ValueError("Remote RGB result paths must be unique")
    exporter = remote.get("exporter_argv")
    if (
        not isinstance(exporter, list)
        or not exporter
        or any(not isinstance(value, str) or not value or "\x00" in value for value in exporter)
    ):
        raise ValueError("Remote exporter argv must be a nonempty string list")
    archive_path = _remote_path(remote.get("archive_path"))
    receipt_path = _remote_path(remote.get("receipt_path"))
    if "--output" not in exporter or exporter[exporter.index("--output") + 1] != archive_path:
        raise ValueError("Remote exporter output differs from configured archive path")
    repository = repository.resolve(strict=True)
    leaf = "final_rgb" if rotations == (2, 3, 4) else "individual_r4"
    allowed = repository / f"artifacts/runpod_returns/irtz8y3sigd621/{leaf}"
    final_rgb = _expand_path(local.get("final_rgb_dir")).resolve()
    incoming = _expand_path(local.get("incoming_dir")).resolve()
    if final_rgb != allowed.resolve() or incoming != (allowed / "incoming").resolve():
        raise ValueError("Local RGB destination/incoming directories are not isolated")
    verifier = _expand_path(local.get("scientific_verifier")).resolve(strict=True)
    expected_verifier = repository / "scripts/verify_runpod_rgb_scientific_return_v1.py"
    if verifier != expected_verifier.resolve(strict=True):
        raise ValueError("Unexpected RGB scientific verifier")
    python = _expand_path(local.get("python_executable")).resolve(strict=True)
    status = _expand_path(local.get("status_path")).resolve()
    history = _expand_path(local.get("history_path")).resolve()
    lock = _expand_path(local.get("lock_path")).resolve()
    proof = _expand_path(local.get("scientific_proof_output")).resolve()
    for path in (status, history, lock):
        if not path.is_relative_to(incoming):
            raise ValueError("Coordinator state and lock must remain in final_rgb/incoming")
    if proof.is_relative_to(incoming) or proof == final_rgb or not proof.is_relative_to(final_rgb):
        raise ValueError("Scientific proof must be outside incoming and inside isolated final_rgb")
    return {
        "ssh": {
            "host": host,
            "user": user,
            "port": port,
            "identity_file": identity,
            "known_hosts_file": known_hosts,
            "connect_timeout_seconds": timeout,
        },
        "remote": {
            "result_paths": result_paths,
            "exporter_argv": list(exporter),
            "archive_path": archive_path,
            "receipt_path": receipt_path,
        },
        "local": {
            "final_rgb_dir": final_rgb,
            "incoming_dir": incoming,
            "scientific_verifier": verifier,
            "python_executable": python,
            "status_path": status,
            "history_path": history,
            "lock_path": lock,
            "scientific_proof_output": proof,
        },
        "poll_seconds": POLL_SECONDS,
        "rotations": rotations,
    }


class SingleInstanceLock(AbstractContextManager["SingleInstanceLock"]):
    """One-byte nonblocking process lock, using Windows locking on the target host."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream: Any = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            self.stream.seek(0)
            if self.stream.read(1) == b"":
                self.stream.write(b"0")
                self.stream.flush()
                os.fsync(self.stream.fileno())
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError("Another RGB return coordinator owns the lock") from error
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.stream is None:
            return
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None


RESULTS_PROBE = r"""
# EJEPART_RGB_RESULTS_PROBE_V1
import json, pathlib, sys
expected = {int(item) for item in sys.argv[1].split(",")}
values = []
for raw in sys.argv[2:]:
    path = pathlib.Path(raw)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        values.append({"path": raw, "ready": False, "reason": type(error).__name__})
        continue
    ready = (
        value.get("schema") == "ejepa_cloud_rgb_downstream_queue_v1"
        and value.get("stage") == "RGB_OOF"
        and value.get("status") == "COMPLETE"
        and value.get("rotation") in (2, 3, 4)
    )
    values.append({"path": raw, "ready": ready, "rotation": value.get("rotation")})
complete = (
    len(values) == len(expected)
    and all(item["ready"] for item in values)
    and {item["rotation"] for item in values} == expected
)
print(json.dumps({"status": "COMPLETE" if complete else "WAITING", "results": values}))
"""

EXPORT_RECEIPT_PROBE = r"""
# EJEPART_RGB_EXPORT_RECEIPT_PROBE_V1
import json, pathlib, re, sys
receipt_path, archive_path = map(pathlib.Path, sys.argv[1:])
if not receipt_path.is_file():
    print(json.dumps({"status": "MISSING"}))
    raise SystemExit(0)
try:
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    print(json.dumps({"status": "INVALID", "reason": "JSON"}))
    raise SystemExit(0)
valid = (
    value.get("schema") == "ejepa_cloud_rgb_complete_export_v1"
    and value.get("status") == "READY"
    and value.get("archive") == str(archive_path)
    and value.get("receipt") == str(receipt_path)
    and isinstance(value.get("archive_bytes"), int)
    and value["archive_bytes"] > 0
    and re.fullmatch(r"[0-9a-f]{64}", str(value.get("archive_sha256", "")))
    and archive_path.is_file()
    and archive_path.stat().st_size == value["archive_bytes"]
)
print(json.dumps(value if valid else {"status": "INVALID", "reason": "IDENTITY"}))
"""


class Coordinator:
    """Advance the return workflow through durable, idempotent boundaries."""

    def __init__(
        self,
        config: Mapping[str, Any],
        repository: Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        admit: Callable[..., dict[str, Any]] = admit_local_rgb_return,
    ) -> None:
        self.repository = repository.resolve(strict=True)
        self.config = _validate_config(config, self.repository)
        self.runner = runner
        self.admit = admit
        self.local = self.config["local"]
        self.local["incoming_dir"].mkdir(parents=True, exist_ok=True)

    def _known_hosts_option(self) -> str:
        """Quote a slash-normalized Windows path for OpenSSH's own option parser."""
        path = self.config["ssh"]["known_hosts_file"]
        value = path.as_posix()
        if '"' in value or "\n" in value or "\r" in value:
            raise ValueError("Known-hosts path contains an unsafe OpenSSH option character")
        return f'UserKnownHostsFile="{value}"'

    def _ssh_base(self) -> list[str]:
        ssh = self.config["ssh"]
        return [
            "ssh",
            "-T",
            "-p",
            str(ssh["port"]),
            "-i",
            str(ssh["identity_file"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            self._known_hosts_option(),
            "-o",
            f"ConnectTimeout={ssh['connect_timeout_seconds']}",
            f"{ssh['user']}@{ssh['host']}",
        ]

    def _scp_base(self) -> list[str]:
        ssh = self.config["ssh"]
        return [
            "scp",
            "-P",
            str(ssh["port"]),
            "-i",
            str(ssh["identity_file"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            self._known_hosts_option(),
            "-o",
            f"ConnectTimeout={ssh['connect_timeout_seconds']}",
        ]

    def _run(self, command: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner(
                list(command), capture_output=True, text=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as error:
            if Path(command[0]).name.lower() in {"ssh", "ssh.exe", "scp", "scp.exe"}:
                message = f"Remote transport timed out after {timeout}s"
                raise RetryableRemoteError(message) from error
            raise

    def _status(self, stage: str, **details: object) -> dict[str, Any]:
        value = {
            "schema": "ejepa_rgb_return_coordinator_status_v1",
            "stage": stage,
            "updated_utc": datetime.now(UTC).isoformat(),
            "remote_deletion_performed": False,
            "local_deletion_performed": False,
            "active_fits_modified": False,
            **details,
        }
        _atomic_json(self.local["status_path"], value)
        history = self.local["history_path"]
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return value

    @staticmethod
    def _stdout_json(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        lines = [line for line in process.stdout.splitlines() if line.strip()]
        for start in range(len(lines)):
            try:
                value = json.loads("\n".join(lines[start:]))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("Remote command did not return a JSON object")

    def remote_results(self) -> dict[str, Any]:
        encoded = base64.b64encode(RESULTS_PROBE.encode()).decode()
        argv = [
            "python3",
            "-c",
            f"import base64;exec(base64.b64decode('{encoded}'))",
            ",".join(map(str, self.config["rotations"])),
            *self.config["remote"]["result_paths"],
        ]
        process = self._run(self._ssh_base() + [shlex.join(argv)], timeout=45)
        if process.returncode == 255:
            raise RetryableRemoteError(process.stderr.strip() or "SSH result probe failed")
        if process.returncode != 0:
            raise RuntimeError(f"Remote result probe failed: {process.stderr.strip()}")
        value = self._stdout_json(process)
        if value.get("status") not in {"WAITING", "COMPLETE"}:
            raise ValueError("Remote result probe returned an invalid state")
        return value

    def remote_export(self) -> dict[str, Any]:
        command = self._ssh_base() + [shlex.join(self.config["remote"]["exporter_argv"])]
        process = self._run(command, timeout=3600)
        if process.returncode == 255:
            raise RetryableRemoteError(process.stderr.strip() or "SSH exporter connection failed")
        value = self._stdout_json(process)
        if value.get("status") == "NOT_READY" and process.returncode in {0, 3}:
            return value
        if process.returncode != 0 or value.get("schema") != EXPORT_SCHEMA:
            raise RuntimeError(f"Remote RGB exporter failed closed: {value}")
        if (
            value.get("status") != "READY"
            or value.get("archive") != self.config["remote"]["archive_path"]
            or value.get("receipt") != self.config["remote"]["receipt_path"]
        ):
            raise ValueError("Remote exporter READY paths differ from configuration")
        return value

    def existing_remote_export(self) -> dict[str, Any] | None:
        """Reuse a sealed remote receipt after a local coordinator interruption."""
        encoded = base64.b64encode(EXPORT_RECEIPT_PROBE.encode()).decode()
        argv = [
            "python3",
            "-c",
            f"import base64;exec(base64.b64decode('{encoded}'))",
            self.config["remote"]["receipt_path"],
            self.config["remote"]["archive_path"],
        ]
        process = self._run(self._ssh_base() + [shlex.join(argv)], timeout=45)
        if process.returncode == 255:
            raise RetryableRemoteError(process.stderr.strip() or "SSH export receipt probe failed")
        if process.returncode != 0:
            raise RuntimeError(f"Remote export receipt probe failed: {process.stderr.strip()}")
        value = self._stdout_json(process)
        if value.get("status") == "MISSING":
            return None
        if (
            value.get("schema") != EXPORT_SCHEMA
            or value.get("status") != "READY"
            or value.get("archive") != self.config["remote"]["archive_path"]
            or value.get("receipt") != self.config["remote"]["receipt_path"]
        ):
            raise ValueError("Existing remote RGB export receipt is invalid")
        return value

    def _scp(self, remote_path: str, destination: Path, timeout: int) -> None:
        ssh = self.config["ssh"]
        source = f"{ssh['user']}@{ssh['host']}:{remote_path}"
        process = self._run(self._scp_base() + [source, str(destination)], timeout=timeout)
        if process.returncode != 0:
            raise RetryableRemoteError(process.stderr.strip() or f"SCP failed: {remote_path}")

    def download_receipt(self, export: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
        incoming = self.local["incoming_dir"]
        final = incoming / "RGB_COMPLETE.tar.json"
        partial = incoming / "RGB_COMPLETE.tar.json.partial"
        expected_sha = export.get("archive_sha256")
        if final.is_file():
            receipt = _json(final)
        else:
            self._scp(self.config["remote"]["receipt_path"], partial, timeout=120)
            receipt = _json(partial)
            os.replace(partial, final)
        if (
            receipt.get("schema") != EXPORT_SCHEMA
            or receipt.get("status") != "READY"
            or receipt.get("archive") != self.config["remote"]["archive_path"]
            or receipt.get("archive_sha256") != expected_sha
            or receipt.get("archive_bytes") != export.get("archive_bytes")
        ):
            raise ValueError("Downloaded RGB archive receipt differs from exporter READY")
        return final, receipt

    def download_archive(self, receipt: Mapping[str, Any]) -> Path:
        incoming = self.local["incoming_dir"]
        final = incoming / "RGB_COMPLETE.tar"
        partial = incoming / "RGB_COMPLETE.tar.partial"
        expected_size = receipt.get("archive_bytes")
        expected_sha = receipt.get("archive_sha256")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError("RGB archive receipt has invalid byte count")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("RGB archive receipt has invalid SHA-256")
        if final.is_file():
            if final.stat().st_size != expected_size or sha256(final) != expected_sha:
                raise ValueError("Existing final RGB tar differs from immutable receipt")
            return final
        if not (
            partial.is_file()
            and partial.stat().st_size == expected_size
            and sha256(partial) == expected_sha
        ):
            self._scp(self.config["remote"]["archive_path"], partial, timeout=7200)
        if partial.stat().st_size != expected_size or sha256(partial) != expected_sha:
            raise RetryableRemoteError("Partial RGB tar differs; next attempt will replace it")
        os.replace(partial, final)
        return final

    def _valid_proof(self, path: Path, archive_sha256: str) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        value = _json(path)
        if (
            value.get("schema") != "ejepa_local_rgb_scientific_reusability_v1"
            or value.get("status") != "PASS"
            or value.get("archive_sha256") != archive_sha256
            or value.get("scientific_reusability_verified") is not True
            or value.get("selected_root_path_maps_verified") is not True
            or (
                self.config["rotations"] == (2, 3, 4)
                and value.get("three_root_path_maps_verified") is not True
            )
            or value.get("active_fits_modified") is not False
        ):
            raise ScientificVerificationError("Existing RGB scientific proof is invalid")
        return value

    def scientific_verify(self, archive_root: Path, archive_sha256: str) -> Path:
        proof = self.local["scientific_proof_output"]
        if self._valid_proof(proof, archive_sha256) is not None:
            return proof
        proof.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.local["python_executable"]),
            str(self.local["scientific_verifier"]),
            "--archive-root",
            str(archive_root),
            "--archive-sha256",
            archive_sha256,
            "--output",
            str(proof),
        ]
        for rotation in self.config["rotations"]:
            command.extend(("--rotation", str(rotation)))
        process = self._run(command, timeout=3600)
        if process.returncode != 0:
            raise ScientificVerificationError(
                f"Original RGB scientific verifier failed: {process.stderr.strip()}"
            )
        if self._valid_proof(proof, archive_sha256) is None:
            raise ScientificVerificationError("Scientific verifier emitted no valid proof")
        return proof

    def step(self) -> dict[str, Any]:
        results = self.remote_results()
        if results["status"] != "COMPLETE":
            return self._status("WAITING_REMOTE_RESULTS", remote=results)
        export = self.existing_remote_export()
        if export is None:
            export = self.remote_export()
        if export.get("status") != "READY":
            return self._status("WAITING_REMOTE_EXPORT", remote=export)
        self._status("DOWNLOADING_RECEIPT", archive_sha256=export["archive_sha256"])
        _, receipt = self.download_receipt(export)
        self._status("DOWNLOADING_ARCHIVE", archive_sha256=receipt["archive_sha256"])
        archive = self.download_archive(receipt)
        self._status("VERIFYING_LOCAL_BYTES", archive_sha256=receipt["archive_sha256"])
        initial = self.admit(
            str(archive),
            expected_sha256=receipt["archive_sha256"],
            destination_root=self.local["final_rgb_dir"],
            repository=self.repository,
            expected_rotations=self.config["rotations"],
        )
        archive_root = Path(initial["snapshot_root"])
        self._status(
            "VERIFYING_SCIENTIFIC_REUSE",
            archive_sha256=receipt["archive_sha256"],
            archive_root=str(archive_root),
        )
        proof = self.scientific_verify(archive_root, receipt["archive_sha256"])
        try:
            final = self.admit(
                str(archive),
                expected_sha256=receipt["archive_sha256"],
                destination_root=self.local["final_rgb_dir"],
                repository=self.repository,
                reuse_verification=proof,
                expected_rotations=self.config["rotations"],
            )
        except Exception as error:
            raise ScientificVerificationError(
                f"Scientific reuse proof was rejected by final ingestion: {error}"
            ) from error
        if final.get("cleanup", {}).get("status") != "ELIGIBLE":
            raise ScientificVerificationError("Verified RGB return was not cleanup-eligible")
        eviction = final.get("eviction")
        if self.config["rotations"] == (4,) and (
            not isinstance(eviction, dict) or eviction.get("status") != "PASS"
        ):
            raise ScientificVerificationError("Individual R4 return lacks eviction eligibility")
        return self._status(
            "COMPLETE",
            archive_sha256=receipt["archive_sha256"],
            archive_root=str(archive_root),
            scientific_proof=str(proof),
            cleanup_eligibility="ELIGIBLE_BUT_NO_DELETION_AUTHORIZED",
            raw_r4_eviction_eligibility=(
                "PASS_EXTERNAL_ACTION_ONLY"
                if self.config["rotations"] == (4,)
                else "NOT_APPLICABLE"
            ),
        )

    def run(self, *, once: bool = False) -> int:
        with SingleInstanceLock(self.local["lock_path"]):
            while True:
                try:
                    value = self.step()
                except ScientificVerificationError as error:
                    self._status("SCIENTIFIC_VERIFICATION_FAILED", error=str(error))
                    return 5
                except RetryableRemoteError as error:
                    self._status("RETRYABLE_CONNECTION_ERROR", error=str(error))
                    if once:
                        return 4
                    time.sleep(POLL_SECONDS)
                    continue
                except Exception as error:
                    self._status("FAILED_CLOSED", error=f"{type(error).__name__}: {error}")
                    return 6
                if value["stage"] == "COMPLETE":
                    return 0
                if once:
                    return 3
                time.sleep(POLL_SECONDS)


def load_coordinator(config_path: Path, repository: Path) -> Coordinator:
    """Load a coordinator from a JSON file; no endpoint is embedded in code."""
    return Coordinator(_json(config_path.resolve(strict=True)), repository)


def validate_coordinator_config(config_path: Path, repository: Path) -> dict[str, Any]:
    """Validate the complete contract without creating files or contacting the host."""
    config_path = config_path.resolve(strict=True)
    value = _validate_config(_json(config_path), repository.resolve(strict=True))
    return {
        "schema": "ejepa_rgb_return_coordinator_config_validation_v1",
        "status": "PASS",
        "config_sha256": sha256(config_path),
        "poll_seconds": value["poll_seconds"],
        "remote_results": len(value["remote"]["result_paths"]),
        "remote_sources": value["remote"]["exporter_argv"].count("--operational-source"),
        "remote_archive": value["remote"]["archive_path"],
        "local_final_rgb": str(value["local"]["final_rgb_dir"]),
        "strict_host_key_checking": True,
        "network_contacted": False,
        "filesystem_modified": False,
    }


__all__ = [
    "CONFIG_SCHEMA",
    "Coordinator",
    "POLL_SECONDS",
    "RetryableRemoteError",
    "ScientificVerificationError",
    "SingleInstanceLock",
    "load_coordinator",
    "validate_coordinator_config",
]
