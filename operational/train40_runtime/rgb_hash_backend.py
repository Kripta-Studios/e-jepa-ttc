"""Optional pinned private BLAKE3 backend for immutable RGB arrival tokens.

Verification/import belongs to explicit service setup, before any sensor arrival.
This module neither installs packages nor imports BLAKE3 on the default SHA path.
It authenticates an implementation, never substitutes an opaque ID for a digest.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
import weakref
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import Distribution
from pathlib import Path
from typing import Protocol, cast

_VERSION = "1.0.11"
_WHEEL_NAME = "blake3-1.0.11-cp311-cp311-win_amd64.whl"
_WHEEL_SHA256 = "de3fbfeef38f68b32c23ae954a83bbfc0c69189c480b045f91ae55e0f0ef9007"
_METADATA_SHA256 = "f59074cbe1c763b4b9b99a9db0b8209cb2c733702680a09dcf765dac3efa008e"
_LICENSE = "CC0-1.0 OR Apache-2.0"
_EMPTY_VECTOR = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
_MODULE_FILES = {
    "blake3": "blake3/__init__.py",
    "blake3.blake3": "blake3/blake3.cp311-win_amd64.pyd",
}


class _HashOutput(Protocol):
    def hexdigest(self, *, length: int) -> str: ...


@dataclass(frozen=True, slots=True)
class _BackendRecord:
    constructor: Callable[..., _HashOutput]
    receipt_json: str
    protected_paths: tuple[Path, ...]


class VerifiedBlake3:
    """Factory-only verified backend; every payload receives a full content hash.

    Like FrozenRGBFrame, registry authentication is an in-process ownership
    contract, not isolation from code that can overwrite Python module globals.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> VerifiedBlake3:
        raise TypeError("Use load_verified_blake3 to construct this backend")

    def __init_subclass__(cls) -> None:
        raise TypeError("VerifiedBlake3 cannot be subclassed")

    def digest(self, payload: bytes) -> str:
        """Hash owned bytes as BLAKE3-256 with exactly one worker thread."""
        record = _record(self)
        if type(payload) is not bytes:
            raise TypeError("BLAKE3 arrival hashing requires immutable bytes")
        result = record.constructor(payload, max_threads=1).hexdigest(length=32)
        if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
            raise ValueError("Verified BLAKE3 returned an invalid 256-bit digest")
        return result

    @property
    def receipt(self) -> dict[str, object]:
        """Return a fresh dependency receipt, including paths for run snapshots."""
        return json.loads(_record(self).receipt_json)

    @property
    def protected_paths(self) -> tuple[Path, ...]:
        """Return immutable dependency paths that a complete run must snapshot."""
        return _record(self).protected_paths


_RECORDS: weakref.WeakKeyDictionary[VerifiedBlake3, _BackendRecord] = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()


def _record(backend: VerifiedBlake3) -> _BackendRecord:
    if type(backend) is not VerifiedBlake3:
        raise ValueError("RGB V2 requires a factory-verified BLAKE3 backend")
    with _LOCK:
        record = _RECORDS.get(backend)
    if record is None:
        raise ValueError("Unregistered or forged BLAKE3 backend")
    return record


def validate_blake3_backend(backend: VerifiedBlake3) -> None:
    """Authenticate a backend without loading dependencies or scanning pixels."""
    _record(backend)


def load_verified_blake3(directory: Path) -> VerifiedBlake3:
    """Load the pinned CPython 3.11 Windows wheel from an existing private target.

    Validate independent pins, installed bytes, distribution version/license,
    module origins and an official vector. No fallback, install or global path
    change survives setup. A previously imported foreign module is rejected.
    Files must remain unchanged through service use and final snapshot checks.
    """
    started = time.perf_counter()
    directory = directory.resolve()
    receipt_path = directory / "download_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Private BLAKE3 dependency is missing; supply its verified target")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    pins = {
        "package": "blake3",
        "version": _VERSION,
        "filename": _WHEEL_NAME,
        "wheel_sha256": _WHEEL_SHA256,
        "release_metadata_sha256": _METADATA_SHA256,
        "license": _LICENSE,
        "bytes": 220_544,
    }
    if any(receipt.get(key) != value for key, value in pins.items()):
        raise ValueError("Private BLAKE3 receipt differs from the pinned dependency")
    wheel = directory / "wheels" / _WHEEL_NAME
    metadata = directory / "pypi_release_metadata.json"
    for path, checksum in ((wheel, _WHEEL_SHA256), (metadata, _METADATA_SHA256)):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Pinned BLAKE3 dependency is missing or altered: {path.name}")
    site = (directory / "site").resolve()
    if not site.is_dir() or not site.is_relative_to(directory):
        raise ValueError("Private BLAKE3 site must exist inside its dependency directory")
    paths = [receipt_path, wheel, metadata]
    installed = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            path = (site / name).resolve()
            if not path.is_relative_to(site) or not path.is_file():
                raise ValueError("Installed BLAKE3 file is missing or outside private target")
            data = path.read_bytes()
            # Installers rewrite RECORD; it is snapshotted, never trusted for verification.
            if not name.endswith(".dist-info/RECORD") and data != archive.read(name):
                raise ValueError(f"Installed BLAKE3 bytes differ from pinned wheel: {name}")
            paths.append(path)
            installed.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
    distribution = Distribution.at(site / f"blake3-{_VERSION}.dist-info")
    if distribution.version != _VERSION or distribution.metadata["License"] != _LICENSE:
        raise ValueError("Installed BLAKE3 version/license differs from the pinned wheel")
    expected_modules = {name: (site / path).resolve() for name, path in _MODULE_FILES.items()}
    with _LOCK:
        for name, expected in expected_modules.items():
            if name in sys.modules:
                filename = getattr(sys.modules[name], "__file__", None)
                if filename is None or Path(filename).resolve() != expected:
                    raise ValueError(
                        f"BLAKE3 module already loaded from an unverified origin: {name}"
                    )
        original_path = sys.path[:]
        original_modules = {name: sys.modules.get(name) for name in expected_modules}
        try:
            sys.path.insert(0, str(site))
            module = importlib.import_module("blake3")
            for name, expected in expected_modules.items():
                imported = importlib.import_module(name)
                filename = imported.__file__
                if (
                    filename is None
                    or Path(filename).resolve() != expected
                    or expected not in paths
                ):
                    raise ValueError(f"BLAKE3 imported from an unverified origin: {name}")
            constructor = cast(Callable[..., _HashOutput], module.blake3)
            if constructor(b"", max_threads=1).hexdigest(length=32) != _EMPTY_VECTOR:
                raise ValueError("BLAKE3 failed its official empty-input vector")
        except BaseException:
            for name, previous in original_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
            raise
        finally:
            sys.path[:] = original_path
        protected = tuple(sorted(set(paths)))
        binding = {
            **receipt,
            "content_algorithm": "blake3-256",
            "digest_bytes": 32,
            "max_threads": 1,
            "installed_files": installed,
            "imported_modules": {name: str(path) for name, path in expected_modules.items()},
            "installed_version": distribution.version,
            "protected_paths": [str(path) for path in protected],
            "setup_verification_seconds": time.perf_counter() - started,
            "shared_environment_modified": False,
            "verification_scope": "SETUP_ONLY_NO_SENSOR_PIXELS",
        }
        backend = object.__new__(VerifiedBlake3)
        _RECORDS[backend] = _BackendRecord(
            constructor, json.dumps(binding, sort_keys=True), protected
        )
    return backend
