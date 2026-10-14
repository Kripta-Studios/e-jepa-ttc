"""Optional lossless RGB input residency, preserving native decode and collation."""

from __future__ import annotations

from pathlib import Path

import psutil

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.rgb_port.data import PreparedRGBCache

from .rgb_data import RGBPairDataset


class GuardedRGBCache(PreparedRGBCache):
    """Retain native uint8 shards, and detect source changes even on memory hits."""

    def __init__(self, base: PreparedRGBCache, *, limit_bytes: int) -> None:
        if not 256 * 1024**2 <= limit_bytes <= 8 * 1024**3:
            raise ValueError("RGB cache must remain within the admitted 256 MiB–8 GiB range")
        self.base = base
        self.base._max_loaded_bytes = limit_bytes
        self.limit_bytes = limit_bytes
        self.fingerprints: dict[str, tuple[int, int, int]] = {}
        self.hits = self.misses = 0
        self.peak_bytes = 0
        self.binding_stats = {}
        for path in (base.binding_path, base.manifest_path):
            stat = Path(path).stat()
            self.binding_stats[str(path)] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def validate_binding(self) -> None:
        """Detect manifest replacement across scopes without rehashing per sample."""
        for path, expected in self.binding_stats.items():
            stat = Path(path).stat()
            if expected != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
                raise ValueError("RGB cache manifest or binding changed while resident")

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)

    def get(self, index: int, token: str) -> dict:
        """Delegate exact native values; bound memory and keep first-use hash admission."""
        shard, _ = self.base._lookup[index]
        key = str(shard["path"])
        stat = Path(key).stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key in self.fingerprints and self.fingerprints[key] != fingerprint:
            raise ValueError("Previously admitted RGB shard changed")
        while (
            self.base._loaded_bytes > 256 * 1024**2
            and psutil.virtual_memory().available < 4 * 1024**3
        ):
            _, arrays = self.base._loaded.popitem(last=False)
            self.base._loaded_bytes -= sum(array.nbytes for array in arrays.values())
        hit = key in self.base._loaded
        result = self.base.get(index, token)
        after = Path(key).stat()
        if fingerprint != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("RGB shard changed during native decoding")
        self.fingerprints[key] = fingerprint
        self.hits += int(hit)
        self.misses += int(not hit)
        self.peak_bytes = max(self.peak_bytes, self.base._loaded_bytes)
        return result

    def counters(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "retained_bytes": self.base._loaded_bytes,
            "peak_bytes": self.peak_bytes,
            "limit_bytes": self.limit_bytes,
            "loaded_shards": len(self.base._loaded),
        }


class CachedRGBPairDataset(RGBPairDataset):
    """Use unchanged RGBPairDataset.batch and PreparedRGBCache.get numerical operations."""

    def __init__(self, role: str = "P", *, limit_bytes: int = 6 * 1024**3) -> None:
        super().__init__(role)
        if self.source.prepared_cache is None:
            raise ValueError("Admitted native RGB cache required")
        self.source.prepared_cache = GuardedRGBCache(
            self.source.prepared_cache, limit_bytes=limit_bytes
        )

    def close(self) -> None:
        """Keep read-only resident shards through a trainer's scope boundary."""
        if self.source.reader is not None:
            self.source.reader.close()

    def batch(self, indices: list[int]) -> ObjectEventV4Batch:
        """Keep native collation exact while validating persistent input identity."""
        self.source.prepared_cache.validate_binding()
        return super().batch(indices)

    def shutdown(self) -> None:
        """Release private memory only when the owning process is done."""
        self.close()
        cache = self.source.prepared_cache
        cache._loaded.clear()
        cache.base._loaded_bytes = 0

    def counters(self) -> dict:
        return self.source.prepared_cache.counters()
