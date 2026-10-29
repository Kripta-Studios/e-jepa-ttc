"""Strict contracts for the isolated indexed-RGB campaign runner.

This module deliberately contains no CUDA initialization.  It binds the
performance-only choices separately from the scientific campaign config and
provides deterministic phase accounting used by the future producer.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical_sha256(value: object) -> str:
    """Hash one finite JSON-compatible contract."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PerformanceConfig:
    """Non-scientific implementation choices for the isolated runner."""

    schema: str
    reader: str
    max_open_handles: int
    shared_store: bool
    prefetch_batches: int
    instrument_phases: tuple[str, ...]
    expected_indexed_reader_sha256: str
    expected_indexed_store_sha256: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PerformanceConfig:
        """Parse an exact sidecar, rejecting silent fallback fields."""
        expected = {
            "schema",
            "reader",
            "max_open_handles",
            "shared_store",
            "prefetch_batches",
            "instrument_phases",
            "expected_indexed_reader_sha256",
            "expected_indexed_store_sha256",
        }
        if set(raw) != expected:
            raise ValueError(
                f"Performance config fields differ: missing={sorted(expected - set(raw))}, "
                f"extra={sorted(set(raw) - expected)}"
            )
        phases = raw["instrument_phases"]
        if not isinstance(phases, list) or any(not isinstance(item, str) for item in phases):
            raise TypeError("instrument_phases must be a list of strings")
        result = cls(
            schema=str(raw["schema"]),
            reader=str(raw["reader"]),
            max_open_handles=raw["max_open_handles"],
            shared_store=raw["shared_store"],
            prefetch_batches=raw["prefetch_batches"],
            instrument_phases=tuple(phases),
            expected_indexed_reader_sha256=str(raw["expected_indexed_reader_sha256"]),
            expected_indexed_store_sha256=str(raw["expected_indexed_store_sha256"]),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path: Path) -> tuple[PerformanceConfig, dict[str, str]]:
        """Load and bind exact sidecar bytes plus canonical semantics."""
        resolved = path.resolve(strict=True)
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("Performance config must be a JSON object")
        result = cls.from_dict(raw)
        return result, {
            "file_sha256": sha256_file(resolved),
            "canonical_sha256": result.sha256,
        }

    def validate(self) -> None:
        """Enforce the first-pass exact-reader-only optimization scope."""
        required_phases = {
            "tar_index",
            "payload_read",
            "png_decode",
            "prepare",
            "h2d",
            "compute",
            "evaluation",
        }
        if self.schema != "eap_detection_performance_config_v1":
            raise ValueError("Unknown performance config schema")
        if self.reader != "shared_indexed_tar_v1" or self.shared_store is not True:
            raise ValueError("First optimized runner requires one shared indexed TAR store")
        if (
            isinstance(self.max_open_handles, bool)
            or not isinstance(self.max_open_handles, int)
            or self.max_open_handles < 1
        ):
            raise ValueError("max_open_handles must be a positive integer")
        if self.prefetch_batches != 0:
            raise ValueError("First-pass reader attribution requires prefetch_batches=0")
        if set(self.instrument_phases) != required_phases or len(self.instrument_phases) != len(
            required_phases
        ):
            raise ValueError("instrument_phases must contain the exact required phase set")
        for field, digest in (
            ("expected_indexed_reader_sha256", self.expected_indexed_reader_sha256),
            ("expected_indexed_store_sha256", self.expected_indexed_store_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{field} must be lowercase SHA-256")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


def exact_union(*populations: Sequence[str]) -> tuple[str, ...]:
    """Return stable first-occurrence union while rejecting malformed tokens."""
    result: list[str] = []
    seen: set[str] = set()
    for population in populations:
        if len(set(population)) != len(population):
            raise ValueError("Each role population must contain unique tokens")
        for token in population:
            if not isinstance(token, str) or not token:
                raise ValueError("Dataset tokens must be nonempty strings")
            if token not in seen:
                seen.add(token)
                result.append(token)
    return tuple(result)


@dataclass
class PhaseSeries:
    """Bounded scalar timing series with explicit observation counts."""

    values: list[float]

    def add(self, seconds: float) -> None:
        if not isinstance(seconds, float) or not 0 <= seconds < float("inf"):
            raise ValueError("Phase duration must be finite and nonnegative")
        self.values.append(seconds)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "count": len(self.values),
            "total_seconds": sum(self.values),
            "median_seconds": statistics.median(self.values) if self.values else None,
            "p90_seconds": (
                sorted(self.values)[int(0.9 * (len(self.values) - 1))]
                if self.values
                else None
            ),
        }


class PhaseMetrics:
    """Fixed-schema timing registry; unknown phases fail closed."""

    def __init__(self, phases: Iterable[str]) -> None:
        names = tuple(phases)
        if len(set(names)) != len(names):
            raise ValueError("Metric phase names must be unique")
        self._series = {name: PhaseSeries([]) for name in names}

    def add(self, phase: str, seconds: float) -> None:
        try:
            series = self._series[phase]
        except KeyError as error:
            raise KeyError(f"Undeclared performance phase: {phase}") from error
        series.add(seconds)

    def to_dict(self) -> dict[str, dict[str, float | int | None]]:
        return {name: series.to_dict() for name, series in self._series.items()}


def transactional_stop_due(
    update: int,
    stop_after_update: int | None,
    *,
    required_evaluation_complete: bool,
    checkpoint_durable: bool,
) -> bool:
    """Commit an external stop only after due evaluation and checkpointing."""
    if stop_after_update is None or update != stop_after_update:
        return False
    if not required_evaluation_complete:
        raise RuntimeError("Requested stop reached before its required evaluation completed")
    if not checkpoint_durable:
        raise RuntimeError("Requested stop reached before its checkpoint was durable")
    return True


__all__ = [
    "PerformanceConfig",
    "PhaseMetrics",
    "canonical_sha256",
    "exact_union",
    "sha256_file",
    "transactional_stop_due",
]
