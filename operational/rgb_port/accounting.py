"""Durable I/O, leases, and optimizer-update accounting for RGB-PORT."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "rgb_port_accounting_v1"
CAPS = {"scientific": 228_408, "technical": 500, "recovery": 11_092}
PHYSICAL_CAP = 240_000
TRANSIENT_WINERRORS = {5, 32, 33}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_read_windows(path: Path) -> bytes:
    """Read while allowing concurrent atomic replacement on Windows."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    read_file = kernel32.ReadFile
    read_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    read_file.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    get_size = kernel32.GetFileSizeEx
    get_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
    get_size.restype = ctypes.c_int
    handle = create_file(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = ctypes.c_int64()
        if not get_size(handle, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if size.value < 0 or size.value > 64 * 1024 * 1024:
            raise ValueError(f"Live JSON has an invalid size: {path}")
        result = bytearray()
        remaining = int(size.value)
        while remaining:
            chunk_size = min(remaining, 1024 * 1024)
            data = ctypes.create_string_buffer(chunk_size)
            read = ctypes.c_uint32()
            if not read_file(handle, data, chunk_size, ctypes.byref(read), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if read.value == 0:
                break
            result.extend(data.raw[: read.value])
            remaining -= read.value
        return bytes(result)
    finally:
        close_handle(handle)


def read_bytes_shared(path: Path, retries: int = 8) -> bytes:
    for attempt in range(retries):
        try:
            if os.name == "nt":
                return _shared_read_windows(path)
            return path.read_bytes()
        except OSError as error:
            if (
                getattr(error, "winerror", None) not in TRANSIENT_WINERRORS
                or attempt + 1 == retries
            ):
                raise
            time.sleep(min(0.025 * 2**attempt, 0.5))
    raise AssertionError("unreachable")


def read_json_shared(path: Path) -> dict[str, Any]:
    value = json.loads(read_bytes_shared(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_write_bytes(path: Path, payload: bytes, retries: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(retries):
            try:
                os.replace(temporary, path)
                break
            except OSError as error:
                if (
                    getattr(error, "winerror", None) not in TRANSIENT_WINERRORS
                    or attempt + 1 == retries
                ):
                    raise
                time.sleep(min(0.025 * 2**attempt, 0.75))
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def durable_replace_file(source: Path, target: Path, retries: int = 10) -> None:
    """Fsync and atomically move an already materialized large file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb+") as stream:
        os.fsync(stream.fileno())
    for attempt in range(retries):
        try:
            os.replace(source, target)
            return
        except OSError as error:
            if (
                getattr(error, "winerror", None) not in TRANSIENT_WINERRORS
                or attempt + 1 == retries
            ):
                raise
            time.sleep(min(0.025 * 2**attempt, 0.75))


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def process_create_time(pid: int) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except (psutil.AccessDenied, OSError) as error:
        raise RuntimeError(f"Cannot verify PID {pid} creation identity") from error


def identity_is_live(identity: Mapping[str, Any]) -> bool:
    try:
        pid = int(identity["pid"])
        expected = float(identity["create_time"])
    except (KeyError, TypeError, ValueError):
        return False
    actual = process_create_time(pid)
    return actual is not None and abs(actual - expected) < 0.01


@dataclass(frozen=True)
class OwnerLease:
    path: Path

    def acquire(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            current = read_json_shared(self.path)
            if identity_is_live(current):
                if int(current["pid"]) == os.getpid():
                    return current
                raise RuntimeError(f"Live RGB-PORT owner already exists: {current}")
            self.path.unlink(missing_ok=True)
        created = process_create_time(os.getpid())
        if created is None:
            raise RuntimeError("Could not establish current PID creation time")
        owner = {"schema": "rgb_port_owner_v1", "pid": os.getpid(), "create_time": created}
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as error:
            raise RuntimeError("Another RGB-PORT supervisor acquired the owner lease") from error
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(owner, sort_keys=True) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        verified = read_json_shared(self.path)
        if verified != owner:
            raise RuntimeError("Owner lease lost during acquisition")
        return owner

    def release(self) -> None:
        if not self.path.exists():
            return
        current = read_json_shared(self.path)
        if int(current.get("pid", -1)) == os.getpid():
            self.path.unlink(missing_ok=True)


class UpdateLedger:
    """Append-only-by-id accounting with per-fit and campaign hard caps."""

    def __init__(self, path: Path, fit_limits: Mapping[str, int]) -> None:
        self.path = path
        self.fit_limits = {str(key): int(value) for key, value in fit_limits.items()}

    def initialize(self) -> dict[str, Any]:
        if self.path.exists():
            return self.validate(read_json_shared(self.path))
        ledger: dict[str, Any] = {
            "schema": SCHEMA,
            "caps": {**CAPS, "physical": PHYSICAL_CAP},
            "fit_limits": self.fit_limits,
            "events": {},
        }
        atomic_write_json(self.path, ledger)
        return ledger

    def totals(self, ledger: Mapping[str, Any]) -> dict[str, int]:
        totals = {**{name: 0 for name in CAPS}, "physical": 0}
        for event in ledger.get("events", {}).values():
            category = str(event["category"])
            updates = int(event["charged_updates"])
            totals[category] += updates
            totals["physical"] += updates
        return totals

    def validate(self, ledger: dict[str, Any]) -> dict[str, Any]:
        if ledger.get("schema") != SCHEMA:
            raise ValueError("Unknown RGB-PORT accounting schema")
        if ledger.get("fit_limits") != self.fit_limits:
            raise ValueError("Fit limits differ from the immutable ledger")
        totals = self.totals(ledger)
        for category, cap in CAPS.items():
            if totals[category] > cap:
                raise ValueError(f"{category} cap exceeded: {totals[category]} > {cap}")
        if totals["physical"] > PHYSICAL_CAP:
            raise ValueError(f"physical cap exceeded: {totals['physical']} > {PHYSICAL_CAP}")
        per_fit: dict[str, int] = {}
        for event in ledger.get("events", {}).values():
            if event["category"] != "scientific":
                continue
            fit_id = str(event["fit_id"])
            per_fit[fit_id] = per_fit.get(fit_id, 0) + int(event["charged_updates"])
        for fit_id, charged in per_fit.items():
            if fit_id not in self.fit_limits or charged > self.fit_limits[fit_id]:
                raise ValueError(f"Per-fit cap exceeded or unknown fit: {fit_id}={charged}")
        return ledger

    def charge(
        self,
        *,
        event_id: str,
        fit_id: str,
        category: str,
        charged_updates: int,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        if category not in CAPS or charged_updates < 0 or fit_id not in self.fit_limits:
            raise ValueError("Invalid accounting charge")
        ledger = self.initialize()
        event = {
            "fit_id": fit_id,
            "category": category,
            "charged_updates": int(charged_updates),
            "evidence": dict(evidence),
        }
        prior = ledger["events"].get(event_id)
        if prior is not None:
            if prior != event:
                raise ValueError(f"Accounting event {event_id!r} was redefined")
            return self.validate(ledger)
        ledger["events"][event_id] = event
        self.validate(ledger)
        atomic_write_json(self.path, ledger)
        return ledger

    def admit_fit_attempt(
        self, *, fit_id: str, scientific_upper: int, recovery_upper: int
    ) -> dict[str, int]:
        """Reject an optimizer attempt unless its prospective upper bounds fit every cap."""
        if fit_id not in self.fit_limits or scientific_upper < 0 or recovery_upper < 0:
            raise ValueError("Invalid prospective fit admission")
        ledger = self.initialize()
        totals = self.totals(ledger)
        charged_fit = sum(
            int(event["charged_updates"])
            for event in ledger["events"].values()
            if event["category"] == "scientific" and event["fit_id"] == fit_id
        )
        projected = {
            **totals,
            "scientific": totals["scientific"] + scientific_upper,
            "recovery": totals["recovery"] + recovery_upper,
            "physical": totals["physical"] + scientific_upper + recovery_upper,
        }
        if charged_fit + scientific_upper > self.fit_limits[fit_id]:
            raise RuntimeError(f"Prospective per-fit cap exceeded for {fit_id}")
        for category, cap in CAPS.items():
            if projected[category] > cap:
                raise RuntimeError(f"Prospective {category} cap exceeded")
        if projected["physical"] > PHYSICAL_CAP:
            raise RuntimeError("Prospective physical cap exceeded")
        return projected
