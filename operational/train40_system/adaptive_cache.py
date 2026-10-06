"""Pressure-aware sealed TRAIN40 shard cache without repeated availability polling."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from operational.train40_system.engine_overlap import SealedInputs

GIB = 1024**3

POLICY: dict[str, int | float] = {
    "absolute_cache_bytes": 8 * GIB,
    "host_reserve_bytes": 3 * GIB,
    "host_trim_target_bytes": int(3.25 * GIB),
    "tree_rss_soft_cap_bytes": 15_000_000_000,
}


@dataclass(frozen=True)
class CachePolicy:
    """Soft cache limits below the unchanged trainer resource guard."""

    absolute_cache_bytes: int = int(POLICY["absolute_cache_bytes"])
    host_reserve_bytes: int = int(POLICY["host_reserve_bytes"])
    host_trim_target_bytes: int = int(POLICY["host_trim_target_bytes"])
    tree_rss_soft_cap_bytes: int = int(POLICY["tree_rss_soft_cap_bytes"])

    def __post_init__(self) -> None:
        if min(
            self.absolute_cache_bytes,
            self.host_reserve_bytes,
            self.host_trim_target_bytes,
            self.tree_rss_soft_cap_bytes,
        ) <= 0:
            raise ValueError("Adaptive cache limits must be positive")
        if self.host_trim_target_bytes < self.host_reserve_bytes:
            raise ValueError("Trim target must not be below the host reserve")


@dataclass(frozen=True)
class MemorySnapshot:
    """One process-tree and host-memory observation for a decoded admission."""

    available_bytes: int
    tree_rss_bytes: int


def _system_memory_snapshot() -> MemorySnapshot:
    import psutil

    process = psutil.Process()
    tree = [process, *process.children(recursive=True)]
    rss = sum(candidate.memory_info().rss for candidate in tree if candidate.is_running())
    return MemorySnapshot(
        available_bytes=int(psutil.virtual_memory().available),
        tree_rss_bytes=int(rss),
    )


class AdaptiveSealedInputs(SealedInputs):
    """Retain exact sealed shards under byte limits and coordinated memory pressure."""

    def __init__(
        self,
        output: Path,
        *,
        policy: CachePolicy | None = None,
        memory_probe: Callable[[], MemorySnapshot] | None = None,
    ) -> None:
        super().__init__(output)
        self.policy = policy or CachePolicy()
        self.memory_probe = memory_probe or _system_memory_snapshot
        self.inflight: dict[int, Future[dict[str, np.ndarray]]] = {}
        self.decisions = 0
        self.evictions = 0
        self.evicted_bytes = 0
        self.admission_skips = 0
        self.coalesced_loads = 0
        self.load_failures = 0
        self.pressure_episodes = 0
        self._release_credit_bytes = 0
        self._last_raw_pressure_bytes = 0
        self._pressure_active = False
        self._last_memory: MemorySnapshot | None = None

    @staticmethod
    def _value_bytes(value: dict[str, np.ndarray]) -> int:
        return sum(array.nbytes for array in value.values())

    def _verify_identity(self, number: int) -> tuple[Path, Path]:
        name = f"shard_{number:05d}.npz"
        paths = (self.directory / name, self.output / "teacher" / name)
        for path in paths:
            entry, stat = self.identities[path], path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
                raise ValueError(f"Sealed TRAIN40 shard changed: {path}")
        return paths

    @staticmethod
    def _decode(paths: tuple[Path, Path]) -> dict[str, np.ndarray]:
        event_path, teacher_path = paths
        with np.load(event_path, allow_pickle=False) as stored:
            value = {key: stored[key] for key in stored.files}
        with np.load(teacher_path, allow_pickle=False) as stored:
            if not np.array_equal(value["ordinals"], stored["ordinals"]):
                raise ValueError("Teacher/event ordinals differ")
            if not np.array_equal(value["tokens"], stored["tokens"]):
                raise ValueError("Teacher/event object tokens differ")
            value["relation_targets"] = stored["relation_targets"]
            value["relation_valid"] = stored["relation_valid"]
        return value

    def _pressure_deficit(self, snapshot: MemorySnapshot) -> int:
        newly_active = (
            snapshot.available_bytes < self.policy.host_reserve_bytes
            or snapshot.tree_rss_bytes > self.policy.tree_rss_soft_cap_bytes
        )
        if newly_active and not self._pressure_active:
            self._pressure_active = True
            self.pressure_episodes += 1
        if self._pressure_active and (
            snapshot.available_bytes >= self.policy.host_trim_target_bytes
            and snapshot.tree_rss_bytes <= self.policy.tree_rss_soft_cap_bytes
        ):
            self._pressure_active = False
        host_deficit = (
            max(0, self.policy.host_trim_target_bytes - snapshot.available_bytes)
            if self._pressure_active
            else 0
        )
        rss_deficit = max(0, snapshot.tree_rss_bytes - self.policy.tree_rss_soft_cap_bytes)
        raw = max(host_deficit, rss_deficit)
        if raw < self._last_raw_pressure_bytes:
            observed_release = self._last_raw_pressure_bytes - raw
            self._release_credit_bytes = max(
                0, self._release_credit_bytes - observed_release
            )
        self._last_raw_pressure_bytes = raw
        return max(0, raw - self._release_credit_bytes)

    def _admit(self, number: int, value: dict[str, np.ndarray]) -> None:
        size = self._value_bytes(value)
        with self.lock:
            # Keep pressure observations and their credit reconciliation in the same
            # order as admissions; concurrent decodes still happen outside this lock.
            snapshot = self.memory_probe()
            self.decisions += 1
            self.reads += 1
            self._last_memory = snapshot
            pressure_deficit = self._pressure_deficit(snapshot)
            capacity_excess = (
                0
                if self._pressure_active
                else max(0, self.cache_bytes + size - self.policy.absolute_cache_bytes)
            )
            eviction_target = max(capacity_excess, pressure_deficit)
            released = 0
            while self.cache and released < eviction_target:
                _, old = self.cache.popitem(last=False)
                old_size = self._value_bytes(old)
                self.cache_bytes -= old_size
                released += old_size
                self.evictions += 1
                self.evicted_bytes += old_size
            if pressure_deficit:
                credited = min(released, pressure_deficit)
                self._release_credit_bytes += credited
            if (
                not self._pressure_active
                and self.cache_bytes + size <= self.policy.absolute_cache_bytes
                and released >= eviction_target
            ):
                self.cache[number] = value
                self.cache_bytes += size
            else:
                self.admission_skips += 1

    def shard(self, number: int) -> dict[str, np.ndarray]:
        """Verify every access, coalesce duplicate decodes, and admit with one snapshot."""
        paths = self._verify_identity(number)
        with self.lock:
            if number in self.cache:
                self.cache.move_to_end(number)
                self.hits += 1
                return self.cache[number]
            future = self.inflight.get(number)
            if future is None:
                future = Future()
                self.inflight[number] = future
                owner = True
            else:
                self.coalesced_loads += 1
                owner = False
        if not owner:
            return future.result()
        try:
            value = self._decode(paths)
            self._admit(number, value)
        except BaseException as error:
            with self.lock:
                self.load_failures += 1
                self.inflight.pop(number, None)
                future.set_exception(error)
            raise
        with self.lock:
            self.inflight.pop(number, None)
            future.set_result(value)
        return value

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe cache and policy evidence for durable sidecars."""
        with self.lock:
            memory = asdict(self._last_memory) if self._last_memory is not None else None
            return {
                "schema": "train40_adaptive_cache_runtime_v1",
                "policy": asdict(self.policy),
                "cache_entries": len(self.cache),
                "cache_bytes": self.cache_bytes,
                "cache_reads": self.reads,
                "cache_hits": self.hits,
                "admission_decisions": self.decisions,
                "admission_skips": self.admission_skips,
                "evictions": self.evictions,
                "evicted_bytes": self.evicted_bytes,
                "coalesced_loads": self.coalesced_loads,
                "load_failures": self.load_failures,
                "pressure_episodes": self.pressure_episodes,
                "pressure_active": self._pressure_active,
                "projected_release_credit_bytes": self._release_credit_bytes,
                "last_memory_snapshot": memory,
            }


__all__ = ["POLICY", "AdaptiveSealedInputs", "CachePolicy", "MemorySnapshot"]
