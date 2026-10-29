"""Fail-closed CPU primitives for campaign checkpoint migration.

Nothing in this module opens CUDA, stops a producer, or mutates a source
checkpoint.  A caller must provide an already reviewed target runtime contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import torch

from e_jepa_ttc.detection.campaign_preregistration import (
    base_runtime_relative_paths,
    build_runtime_provenance,
    optimized_runtime_relative_paths,
)
from e_jepa_ttc.detection.campaign_training import canonical_sha256, sha256_file

CHECKPOINT_SCHEMA = "eap_detection_campaign_checkpoint_v1"
CHECKPOINT_PRODUCER = "scripts/train_detection_campaign.py"
CHECKPOINT_INFERENCE_CONTRACT = "campaign_v1_strict_loader_required"
COMPONENT_KEYS = (
    "model",
    "optimizer",
    "scheduler",
    "sampler",
    "rng",
    "progress",
    "data_member_sha256",
    "update",
    "best_metric",
)
_ALLOWED_CHANGED_RUNTIME_PATHS = frozenset(
    {
        "scripts/train_detection_campaign.py",
        "scripts/smoke_monodgp_portable.py",
        "src/e_jepa_ttc/detection/development_metrics.py",
        "src/e_jepa_ttc/detection/evaluation_v2.py",
    }
)
_PREREG_INVARIANTS = (
    "schema",
    "config",
    "config_sha256",
    "protocol_sha256",
    "source_receipt",
    "source_receipt_sha256",
    "effective_model",
    "effective_model_sha256",
    "portable_source_sha256",
    "portable_source_sha256_manifest",
    "dataset_media_sha256",
    "media_admission_receipt_sha256",
    "media_admission_population_sha256",
    "matmul_precision",
)


class _Digest(Protocol):
    def update(self, value: bytes, /) -> None: ...


class _TensorLike(Protocol):
    dtype: object
    shape: Sequence[int]
    requires_grad: bool

    def stride(self) -> Sequence[int]: ...
    def detach(self) -> _TensorLike: ...
    def cpu(self) -> _TensorLike: ...
    def contiguous(self) -> _TensorLike: ...
    def reshape(self, *shape: int) -> _TensorLike: ...
    def view(self, dtype: object) -> _TensorLike: ...
    def numpy(self) -> _ByteArrayLike: ...


class _ByteArrayLike(Protocol):
    def tobytes(self) -> bytes: ...


_TORCH_TENSOR = cast(type[object], vars(torch)["Tensor"])
_TORCH_UINT8 = vars(torch)["uint8"]
_TORCH_EQUAL = cast(Callable[[object, object], bool], vars(torch)["equal"])
_TORCH_LOAD = cast(Callable[..., object], vars(torch)["load"])
_TORCH_SAVE = cast(Callable[..., None], vars(torch)["save"])


def windows_process_creation_time_100ns(pid: int) -> int | None:
    """Return the exact Windows process creation FILETIME, or ``None`` if absent."""
    if os.name != "nt":
        raise RuntimeError("Exact process-creation verification requires Windows")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("Process PID must be a positive integer")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    handle = open_process(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: no such PID
            return None
        raise ctypes.WinError(error)
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        get_times = kernel32.GetProcessTimes
        get_times.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        )
        get_times.restype = wintypes.BOOL
        if not get_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    finally:
        kernel32.CloseHandle(handle)


def _live_windows_processes() -> list[dict[str, object]]:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("PowerShell is required for live command ownership verification")
    command = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )
    value = json.loads(result.stdout)
    rows = value if isinstance(value, list) else [value]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Windows process inventory is malformed")
    return cast(list[dict[str, object]], rows)


def _windows_command_line_to_argv(command_line: str) -> list[str]:
    """Parse one Windows command line with the same API used by native programs."""
    if os.name != "nt":
        raise RuntimeError("Windows command-line parsing requires Windows")
    if not command_line:
        raise ValueError("Process command line is empty")
    import ctypes
    from ctypes import wintypes

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    parse = shell32.CommandLineToArgvW
    parse.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
    parse.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    pointer = parse(command_line, ctypes.byref(count))
    if not pointer or count.value < 1:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [pointer[index] for index in range(count.value)]
    finally:
        local_free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        local_free.argtypes = (ctypes.c_void_p,)
        local_free.restype = ctypes.c_void_p
        local_free(ctypes.cast(pointer, ctypes.c_void_p))


def _resolved_argument(value: str) -> Path:
    return Path(value).resolve()


def _verify_no_live_segment_owner(command: Sequence[str], output: Path) -> None:
    producer = _resolved_argument(str(command[1] if len(command) > 1 else command[0]))
    expected_output = output.resolve()
    for row in _live_windows_processes():
        command_line = row.get("CommandLine")
        if command_line is None:
            continue
        if not isinstance(command_line, str):
            raise RuntimeError("Cannot prove ownership from malformed process inventory")
        try:
            argv = _windows_command_line_to_argv(command_line)
        except (OSError, ValueError) as error:
            raise RuntimeError("Cannot prove ownership from malformed command line") from error
        if len(argv) < 2 or _resolved_argument(argv[1]) != producer:
            continue
        output_indices = [index for index, value in enumerate(argv) if value == "--output"]
        if len(output_indices) != 1 or output_indices[0] + 1 >= len(argv):
            raise RuntimeError("Cannot prove ownership: producer --output is malformed")
        observed_output = _resolved_argument(argv[output_indices[0] + 1])
        if observed_output == expected_output:
            raise RuntimeError("A producer still owns the completed segment output")


def verify_source_state_receipt(path: Path, *, source_run: Path) -> dict[str, object]:
    """Verify either a completed segment or an intentionally stopped producer."""
    receipt = _load_json(path)
    if receipt.get("schema") == "eap_detection_completed_segment_receipt_v1":
        expected = {
            "schema",
            "status",
            "queue_child_path",
            "queue_child_sha256",
            "summary_path",
            "summary_sha256",
            "final_update",
        }
        if set(receipt) != expected or receipt["status"] != "COMPLETED_SEGMENT":
            raise ValueError("Completed-segment receipt fields/status differ")
        queue_path = Path(str(receipt["queue_child_path"])).resolve(strict=True)
        summary_path = Path(str(receipt["summary_path"])).resolve(strict=True)
        if (
            sha256_file(queue_path) != receipt["queue_child_sha256"]
            or sha256_file(summary_path) != receipt["summary_sha256"]
        ):
            raise ValueError("Completed-segment evidence bytes differ")
        queue = _load_json(queue_path)
        summary = _load_json(summary_path)
        command = queue.get("command")
        if (
            queue.get("schema") != "phase_ttc_detection_campaign_queue_child_v1"
            or queue.get("status") != "EXITED_ZERO"
            or queue.get("returncode") != 0
            or queue.get("timed_out") is not False
            or not isinstance(command, list)
            or any(not isinstance(item, str) or not item for item in command)
            or Path(str(queue.get("output"))).resolve() != source_run.resolve()
            or summary_path != (source_run / "summary.json").resolve()
            or summary.get("status") not in {"BUDGET_STOPPED", "COMPLETED"}
            or isinstance(receipt["final_update"], bool)
            or not isinstance(receipt["final_update"], int)
            or summary.get("final_update") != receipt["final_update"]
        ):
            raise ValueError("Completed-segment queue/summary contract differs")
        _verify_no_live_segment_owner(cast(list[str], command), source_run)
        return {
            "receipt_path": str(path.resolve()),
            "receipt_sha256": sha256_file(path),
            "queue_child_path": str(queue_path),
            "queue_child_sha256": receipt["queue_child_sha256"],
            "summary_path": str(summary_path),
            "summary_sha256": receipt["summary_sha256"],
            "command": command,
            "exit_code": 0,
            "status": "PASS_COMPLETED_SEGMENT_NO_LIVE_OWNER",
        }
    expected = {
        "schema",
        "status",
        "pid",
        "creation_time_filetime_100ns",
        "command",
        "exit_code",
        "ended_at_utc",
        "exit_evidence_path",
        "exit_evidence_sha256",
        "source_run",
        "producer_path",
        "config_path",
        "config_sha256",
    }
    if set(receipt) != expected:
        raise ValueError("Stopped-process receipt fields differ")
    if (
        receipt["schema"] != "eap_detection_stopped_process_receipt_v1"
        or receipt["status"] != "INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION"
    ):
        raise ValueError("Stopped-process receipt status differs")
    pid = receipt["pid"]
    creation = receipt["creation_time_filetime_100ns"]
    command = receipt["command"]
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(creation, bool)
        or not isinstance(creation, int)
        or creation <= 0
        or not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("Stopped-process identity/command is malformed")
    exit_code = receipt["exit_code"]
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ValueError("Stopped-process exit code must be the observed integer")
    declared_source = Path(str(receipt["source_run"])).resolve()
    producer_path = Path(str(receipt["producer_path"])).resolve(strict=True)
    config_path = Path(str(receipt["config_path"])).resolve(strict=True)
    if (
        declared_source != source_run.resolve()
        or sha256_file(config_path) != receipt["config_sha256"]
    ):
        raise ValueError("Stopped-process source run/config binding differs")
    try:
        output_index = command.index("--output")
        config_index = command.index("--config")
    except ValueError as error:
        raise ValueError("Stopped producer command lacks --config/--output") from error
    if (
        output_index + 1 >= len(command)
        or Path(command[output_index + 1]).resolve() != source_run.resolve()
        or config_index + 1 >= len(command)
        or Path(command[config_index + 1]).resolve(strict=True) != config_path
        or len(command) < 2
        or Path(command[1]).resolve(strict=True) != producer_path
    ):
        raise ValueError("Stopped producer argv does not own the source run")
    evidence_path_raw = receipt["exit_evidence_path"]
    evidence_sha = receipt["exit_evidence_sha256"]
    ended_at = receipt["ended_at_utc"]
    if not isinstance(ended_at, str) or not ended_at or not ended_at.endswith(("Z", "+00:00")):
        raise ValueError("Stopped-process end timestamp must be explicit UTC")
    if not isinstance(evidence_path_raw, str) or not evidence_path_raw:
        raise ValueError("Stopped-process exit evidence path is malformed")
    evidence_path = Path(evidence_path_raw).resolve(strict=True)
    if not isinstance(evidence_sha, str) or sha256_file(evidence_path) != evidence_sha:
        raise ValueError("Stopped-process exit evidence bytes differ")
    exit_evidence = _load_json(evidence_path)
    evidence_expected = {
        "schema": "eap_detection_process_exit_evidence_v1",
        "status": receipt["status"],
        "pid": pid,
        "creation_time_filetime_100ns": creation,
        "command": command,
        "returncode": exit_code,
        "ended_at_utc": ended_at,
        "source_run": str(source_run.resolve()),
    }
    if exit_evidence != evidence_expected:
        raise ValueError("Stopped-process exit evidence fields/values differ")
    observed = windows_process_creation_time_100ns(pid)
    if observed == creation:
        raise RuntimeError("The receipt-bound producer process is still alive")
    _verify_no_live_segment_owner(cast(list[str], command), source_run)
    return {
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha256_file(path),
        "pid": pid,
        "creation_time_filetime_100ns": creation,
        "observed_pid_creation_time_filetime_100ns": observed,
        "command": command,
        "exit_code": exit_code,
        "exit_evidence_path": str(evidence_path),
        "exit_evidence_sha256": evidence_sha,
        "config_path": str(config_path),
        "config_sha256": receipt["config_sha256"],
        "source_stop_status": receipt["status"],
        "training_completion_claimed": receipt["status"] == "EXITED_ZERO",
        "status": "PASS_RECEIPT_PROCESS_IDENTITY_NOT_LIVE",
    }


def _length_prefixed(digest: _Digest, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _typename(value: object) -> bytes:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}".encode()


def _feed(digest: _Digest, value: object) -> None:
    if isinstance(value, _TORCH_TENSOR):
        tensor = cast(_TensorLike, value)
        digest.update(b"T")
        _length_prefixed(digest, str(tensor.dtype).encode())
        _feed(digest, tuple(tensor.shape))
        _feed(digest, tuple(tensor.stride()))
        _feed(digest, tensor.requires_grad)
        logical = (
            tensor.detach().cpu().contiguous().reshape(-1).view(_TORCH_UINT8).numpy()
        )
        _length_prefixed(digest, logical.tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        _length_prefixed(digest, _typename(value))
        digest.update(struct.pack(">Q", len(value)))
        for key, nested in value.items():
            _feed(digest, key)
            _feed(digest, nested)
        metadata = getattr(value, "_metadata", None)
        digest.update(b"m1" if metadata is not None else b"m0")
        if metadata is not None:
            _feed(digest, metadata)
        return
    if isinstance(value, tuple):
        digest.update(b"Q")
        _length_prefixed(digest, _typename(value))
        digest.update(struct.pack(">Q", len(value)))
        for nested in value:
            _feed(digest, nested)
        return
    if isinstance(value, list):
        digest.update(b"L")
        digest.update(struct.pack(">Q", len(value)))
        for nested in value:
            _feed(digest, nested)
        return
    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        digest.update(b"I")
        _length_prefixed(digest, str(value).encode())
        return
    if isinstance(value, float):
        digest.update(b"F")
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        digest.update(b"S")
        _length_prefixed(digest, value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        digest.update(b"Y")
        _length_prefixed(digest, value)
        return
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")


def typed_state_sha256(value: object) -> str:
    """Hash nested state with unambiguous type and length framing."""
    digest = hashlib.sha256()
    digest.update(b"phase-ttc-campaign-state-v1\0")
    _feed(digest, value)
    return digest.hexdigest()


def component_sha256(payload: Mapping[str, object]) -> dict[str, str]:
    """Hash every state component that migration must preserve exactly."""
    missing = set(COMPONENT_KEYS) - set(payload)
    if missing:
        raise ValueError(f"Checkpoint state components are missing: {sorted(missing)}")
    return {key: typed_state_sha256(payload[key]) for key in COMPONENT_KEYS}


def assert_deep_exact(left: object, right: object, path: str = "state") -> None:
    """Require exact nested types, order, tensor layout/bytes and mapping metadata."""
    if isinstance(left, _TORCH_TENSOR) or isinstance(right, _TORCH_TENSOR):
        if not isinstance(left, _TORCH_TENSOR) or not isinstance(right, _TORCH_TENSOR):
            raise ValueError(f"Tensor type differs at {path}")
        left_tensor = cast(_TensorLike, left)
        right_tensor = cast(_TensorLike, right)
        if (
            left_tensor.dtype != right_tensor.dtype
            or tuple(left_tensor.shape) != tuple(right_tensor.shape)
            or tuple(left_tensor.stride()) != tuple(right_tensor.stride())
            or left_tensor.requires_grad != right_tensor.requires_grad
            or not _TORCH_EQUAL(left, right)
        ):
            raise ValueError(f"Tensor state differs at {path}")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"Mapping type differs at {path}")
        if type(left) is not type(right) or list(left) != list(right):
            raise ValueError(f"Mapping type/order differs at {path}")
        left_metadata = getattr(left, "_metadata", None)
        right_metadata = getattr(right, "_metadata", None)
        assert_deep_exact(left_metadata, right_metadata, f"{path}._metadata")
        for key in left:
            assert_deep_exact(left[key], right[key], f"{path}[{key!r}]")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or type(left) is not type(right):
            raise ValueError(f"Sequence state differs at {path}")
        if len(left) != len(right):
            raise ValueError(f"Sequence state differs at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            assert_deep_exact(left_item, right_item, f"{path}[{index}]")
        return
    if isinstance(right, (list, tuple)):
        raise ValueError(f"Sequence state differs at {path}")
    if type(left) is not type(right):
        raise ValueError(f"Scalar type differs at {path}")
    if isinstance(left, float):
        if struct.pack(">d", left) != struct.pack(">d", right):
            raise ValueError(f"Float bits differ at {path}")
        return
    if left != right:
        raise ValueError(f"Scalar state differs at {path}")


def validate_checkpoint_envelope(payload: Mapping[str, object]) -> None:
    """Validate the immutable checkpoint envelope before any migration."""
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "producer": CHECKPOINT_PRODUCER,
        "inference_contract": CHECKPOINT_INFERENCE_CONTRACT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint {key} differs from the campaign contract")
    provenance = payload.get("runtime_provenance")
    if not isinstance(provenance, Mapping) or payload.get(
        "runtime_provenance_sha256"
    ) != canonical_sha256(provenance):
        raise ValueError("Checkpoint runtime provenance is internally inconsistent")
    component_sha256(payload)


def checkpoint_selection_identity(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate one checkpoint fully, then retain only selection-boundary fields."""
    validate_checkpoint_envelope(payload)
    return {
        key: payload[key]
        for key in (
            "config_sha256",
            "protocol_sha256",
            "runtime_provenance_sha256",
            "update",
            "best_metric",
        )
    }


def stable_copy_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_source_sha256: str,
) -> dict[str, object]:
    """Copy a receipt-proven stopped source with before/copy/after agreement."""
    if len(expected_source_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_source_sha256
    ):
        raise ValueError("Expected checkpoint SHA-256 must be lowercase hexadecimal")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Stable checkpoint destination exists: {destination}")
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if partial.exists():
        raise FileExistsError(f"Stable checkpoint partial exists: {partial}")
    before_stat = source.stat()
    with source.open("rb") as reader:
        digest = hashlib.sha256()
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
        before_sha = digest.hexdigest()
        if before_sha != expected_source_sha256:
            raise ValueError("Source checkpoint differs from the explicitly pinned SHA-256")
        reader.seek(0)
        with partial.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    copied_sha = sha256_file(partial)
    after_sha = sha256_file(source)
    after_stat = source.stat()
    if before_sha != copied_sha or before_sha != after_sha:
        raise RuntimeError("Source checkpoint changed while the stable copy was created")
    if (
        before_stat.st_size != after_stat.st_size
        or before_stat.st_mtime_ns != after_stat.st_mtime_ns
    ):
        raise RuntimeError("Source checkpoint identity changed during stable copy")
    os.replace(partial, destination)
    return {
        "source": str(source.resolve()),
        "source_sha256_before": before_sha,
        "source_sha256_after": after_sha,
        "source_size_bytes": before_stat.st_size,
        "source_mtime_ns_before": before_stat.st_mtime_ns,
        "source_mtime_ns_after": after_stat.st_mtime_ns,
        "destination": destination.name,
        "destination_sha256": copied_sha,
        "status": "PASS_STABLE_ATOMIC_COPY",
    }


def _experimental_detached_hardlink_pin(
    source: Path, destination: Path, *, expected_source_sha256: str
) -> dict[str, object]:
    """Test a non-admitted live-source concept without exposing it in the CLI."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise RuntimeError("Experimental hardlink has not detached")
    first_sha = sha256_file(destination)
    second_sha = sha256_file(destination)
    if first_sha != expected_source_sha256 or second_sha != first_sha:
        raise RuntimeError("Detached experimental pin changed or has the wrong SHA-256")
    return {
        "destination_sha256": first_sha,
        "status": "EXPERIMENTAL_NOT_ADMITTED_DETACHED_HARDLINK_PIN",
    }


def migrate_checkpoint_provenance(
    source_copy: Path,
    destination: Path,
    *,
    old_provenance: Mapping[str, object],
    new_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Change only strict runtime provenance in an immutable checkpoint copy."""
    if destination.exists():
        raise FileExistsError(f"Migrated checkpoint destination exists: {destination}")
    payload = cast(
        Mapping[str, object],
        _TORCH_LOAD(source_copy, map_location="cpu", weights_only=True),
    )
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint root must be a mapping")
    validate_checkpoint_envelope(payload)
    if payload["runtime_provenance"] != dict(old_provenance):
        raise ValueError("Checkpoint is not bound to the admitted old provenance")
    before_components = component_sha256(payload)
    before = dict(payload)
    migrated = dict(payload)
    migrated["runtime_provenance"] = dict(new_provenance)
    migrated["runtime_provenance_sha256"] = canonical_sha256(new_provenance)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if partial.exists():
        raise FileExistsError(f"Migrated checkpoint partial exists: {partial}")
    with partial.open("xb") as stream:
        _TORCH_SAVE(migrated, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, destination)
    restored = cast(
        Mapping[str, object],
        _TORCH_LOAD(destination, map_location="cpu", weights_only=True),
    )
    validate_checkpoint_envelope(restored)
    for key in before:
        if key in {"runtime_provenance", "runtime_provenance_sha256"}:
            continue
        assert_deep_exact(before[key], restored[key], f"checkpoint.{key}")
    if restored["runtime_provenance"] != dict(new_provenance):
        raise ValueError("Migrated checkpoint did not retain target provenance")
    after_components = component_sha256(restored)
    if before_components != after_components:
        raise ValueError("Migrated checkpoint component fingerprints changed")
    return {
        "parent_copy": str(source_copy.resolve()),
        "parent_full_sha256": sha256_file(source_copy),
        "destination": destination.name,
        "destination_full_sha256": sha256_file(destination),
        "component_sha256": before_components,
        "old_runtime_provenance_sha256": canonical_sha256(old_provenance),
        "new_runtime_provenance_sha256": canonical_sha256(new_provenance),
        "update": restored["update"],
        "best_metric": restored["best_metric"],
        "status": "PASS_EXACT_STATE_PROVENANCE_ONLY",
    }


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _hash_mapping(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a nonempty mapping")
    result: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
            raise ValueError(f"{field} contains an unsafe path")
        if not isinstance(raw_digest, str) or len(raw_digest) != 64 or any(
            char not in "0123456789abcdef" for char in raw_digest
        ):
            raise ValueError(f"{field}.{raw_path} is not a SHA-256")
        result[raw_path] = raw_digest
    return result


def verify_snapshot(root: Path, manifest: Mapping[str, str]) -> dict[str, object]:
    """Verify every manifest path beneath one immutable source snapshot."""
    resolved_root = root.resolve(strict=True)
    for relative, expected in manifest.items():
        path = (resolved_root / relative).resolve(strict=True)
        if not path.is_relative_to(resolved_root) or not path.is_file():
            raise ValueError(f"Snapshot path escapes its root: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"Snapshot bytes differ: {relative}")
    return {
        "root": str(resolved_root),
        "file_count": len(manifest),
        "manifest_sha256": canonical_sha256(dict(manifest)),
        "status": "PASS_EXACT_SOURCE_SNAPSHOT",
    }


def verify_runtime_transition(
    old_prereg: Mapping[str, object],
    target_prereg: Mapping[str, object],
    *,
    performance_config_relative: str,
) -> dict[str, object]:
    """Admit only the indexed-reader/AP runtime transition, never a recipe change."""
    for field in _PREREG_INVARIANTS:
        if old_prereg.get(field) != target_prereg.get(field):
            raise ValueError(f"Scientific/runtime invariant changed: {field}")
    old_effective = old_prereg.get("effective_model")
    if not isinstance(old_effective, Mapping):
        raise ValueError("Effective-model contract must be a mapping")
    if old_prereg.get("effective_model_sha256") != canonical_sha256(old_effective):
        raise ValueError("Parent effective-model hash is inconsistent")
    if target_prereg.get("effective_model_sha256") != canonical_sha256(
        target_prereg.get("effective_model")
    ):
        raise ValueError("Target effective-model hash is inconsistent")

    old_code = _hash_mapping(old_prereg.get("code_sha256"), "old.code_sha256")
    target_code = _hash_mapping(target_prereg.get("code_sha256"), "target.code_sha256")
    if old_prereg.get("code_sha256_manifest") != canonical_sha256(old_code):
        raise ValueError("Parent code manifest hash is inconsistent")
    if target_prereg.get("code_sha256_manifest") != canonical_sha256(target_code):
        raise ValueError("Target code manifest hash is inconsistent")
    base = set(base_runtime_relative_paths())
    optimized = set(optimized_runtime_relative_paths(performance_config_relative))
    historical_base = base - {
        "src/e_jepa_ttc/detection/campaign_preregistration.py",
        "src/e_jepa_ttc/detection/evaluation_ap_fast.py",
    }
    parent_extras = {"data/admission/media_admission.json"}
    if set(old_code) != historical_base | parent_extras:
        raise ValueError("Parent code population differs from its exact historical closure")
    if set(target_code) != optimized | parent_extras:
        raise ValueError("Target code population differs from the shared optimized authority")
    changed = {
        path
        for path in set(old_code) & set(target_code)
        if old_code[path] != target_code[path]
    }
    if changed != _ALLOWED_CHANGED_RUNTIME_PATHS:
        raise ValueError(f"Changed runtime paths leave the allowlist: {sorted(changed)}")
    added = set(target_code) - set(old_code)
    expected_added = optimized - historical_base
    if added != expected_added or set(old_code) - set(target_code):
        raise ValueError("Runtime additions/removals differ from the indexed/AP transition")
    extension = target_prereg.get("runtime_extension")
    if not isinstance(extension, Mapping) or extension.get("schema") != (
        "eap_detection_indexed_runtime_v1"
    ):
        raise ValueError("Target indexed runtime extension is missing")
    if old_prereg.get("runtime_extension") not in (None, {}):
        raise ValueError("Parent unexpectedly contains a runtime extension")
    return {
        "unchanged_invariants": list(_PREREG_INVARIANTS),
        "effective_model_exact": True,
        "changed_paths": sorted(changed),
        "added_paths": sorted(added),
        "removed_paths": [],
        "parent_code_manifest_sha256": old_prereg["code_sha256_manifest"],
        "target_code_manifest_sha256": target_prereg["code_sha256_manifest"],
        "status": "PASS_INDEXED_AP_RUNTIME_ONLY",
    }


def runtime_provenance_from_file(
    path: Path, prereg: Mapping[str, object]
) -> dict[str, object]:
    """Derive checkpoint provenance from the exact preregistration bytes."""
    return build_runtime_provenance(sha256_file(path), prereg)


def verify_evaluation_boundary(
    old_run: Path,
    *,
    checkpoint: Mapping[str, object],
    selected: Mapping[str, object] | None,
    evaluation_updates: Sequence[int],
    full_selection_updates: Sequence[int],
    selection_metric: str,
    evaluation_root_by_update: Mapping[int, Path] | None = None,
) -> dict[str, object]:
    """Require every scheduled evaluation through the durable checkpoint update."""
    validate_checkpoint_envelope(checkpoint)
    update = checkpoint["update"]
    if isinstance(update, bool) or not isinstance(update, int):
        raise ValueError("Checkpoint update must be an integer")
    required = tuple(item for item in evaluation_updates if item <= update)
    files: dict[str, dict[str, object]] = {}
    best_value: float | None = None
    best_update: int | None = None
    protocol_sha256 = checkpoint["protocol_sha256"]
    roots = (
        {scheduled: old_run for scheduled in required}
        if evaluation_root_by_update is None
        else dict(evaluation_root_by_update)
    )
    if set(roots) != set(required):
        raise ValueError("Evaluation-root assignment differs from required updates")
    known_roots = {path.resolve(strict=True) for path in roots.values()}
    for scheduled in required:
        panels = ["selection", "pilot_parity"]
        if scheduled in full_selection_updates:
            panels.append("full_selection")
        for panel in panels:
            name = f"{panel}_update_{scheduled:06d}.json"
            expected_root = roots[scheduled].resolve(strict=True)
            path = expected_root / "evaluation" / name
            present = [
                root / "evaluation" / name
                for root in known_roots
                if (root / "evaluation" / name).is_file()
            ]
            if len(present) != 1 or present[0] != path:
                raise ValueError(
                    f"Required evaluation ownership is missing or duplicated: {name}"
                )
            payload = _load_json(path)
            metrics = payload.get("metrics_v2")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"Evaluation metrics_v2 is missing: {path.name}")
            if (
                metrics.get("protocol_sha256") != protocol_sha256
                or metrics.get("official_eap_score") is not False
                or metrics.get("sample_count", 0) <= 0
                or metrics.get("population", {}).get("identity_check") is not True
                or metrics.get("predictions", {}).get("identity_check") is not True
            ):
                raise ValueError(f"Evaluation completion identity failed: {path.name}")
            files[path.name] = {
                "sha256": sha256_file(path),
                "sample_count": metrics["sample_count"],
                "root": str(expected_root),
            }
            if panel == "selection":
                value = metrics.get(selection_metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Selection metric is invalid: {path.name}")
                numeric = float(value)
                if best_value is None or numeric > best_value:
                    best_value, best_update = numeric, scheduled
    if checkpoint["best_metric"] != best_value:
        raise ValueError("Checkpoint best metric differs from completed selection evaluations")
    if best_value is None:
        if selected is not None:
            raise ValueError("Selected checkpoint exists before any completed selection")
    else:
        if selected is None:
            raise ValueError("Selected checkpoint is missing for a non-null best metric")
        required_selected = {
            "config_sha256",
            "protocol_sha256",
            "runtime_provenance_sha256",
            "update",
            "best_metric",
        }
        if set(selected) != required_selected:
            raise ValueError("Selected checkpoint identity fields differ")
        for key in ("config_sha256", "protocol_sha256", "runtime_provenance_sha256"):
            if selected[key] != checkpoint[key]:
                raise ValueError(f"Selected checkpoint {key} differs from last checkpoint")
        if selected["update"] != best_update or selected["best_metric"] != best_value:
            raise ValueError("Selected checkpoint update/metric differs from strict-max history")
    return {
        "durable_update": update,
        "required_evaluation_updates": list(required),
        "required_full_selection_updates": [
            item for item in full_selection_updates if item <= update
        ],
        "best_metric": best_value,
        "selected_update": best_update,
        "files": files,
        "status": "PASS_COMPLETE_EVALUATION_BOUNDARY",
    }


__all__ = [
    "COMPONENT_KEYS",
    "assert_deep_exact",
    "checkpoint_selection_identity",
    "component_sha256",
    "migrate_checkpoint_provenance",
    "runtime_provenance_from_file",
    "stable_copy_checkpoint",
    "typed_state_sha256",
    "validate_checkpoint_envelope",
    "verify_evaluation_boundary",
    "verify_runtime_transition",
    "verify_snapshot",
    "verify_source_state_receipt",
    "windows_process_creation_time_100ns",
]
