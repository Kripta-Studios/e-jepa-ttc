"""Lossless P2-only residency of the native RGB training inputs."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.rgb_port.data import PreparedRGBCache

from .rgb_cached_data import CachedRGBPairDataset, GuardedRGBCache

PAIR_FIELDS = (
    "rgb_uint8",
    "sensor_frame_times_us",
    "annotation_frame_times_us",
    "boxes_in_crop_xyxy",
    "visible_heights_px",
)
STATIC_FIELDS = ("indices", "tokens", "roi_xyxy")


def compact_shard(stored: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep exact final-two native observations and fields consumed by RGBPairDataset."""
    if stored["rgb_uint8"].dtype != np.uint8 or stored["rgb_uint8"].shape[1] not in (2, 3):
        raise ValueError("P2 compaction requires admitted native T2/T3 uint8 frames")
    count = len(stored["indices"])
    if any(len(stored[name]) != count for name in (*PAIR_FIELDS, *STATIC_FIELDS, "delta_t_s")):
        raise ValueError("P2 shard field populations differ")
    # Copy even contiguous single-row slices: a view could keep the discarded T3 bytes alive.
    result = {name: stored[name][:, -2:].copy(order="C") for name in PAIR_FIELDS}
    result.update({name: stored[name] for name in STATIC_FIELDS})
    result["delta_t_s"] = stored["delta_t_s"][:, -1:].copy(order="C")
    return result


class P2RGBCache(GuardedRGBCache):
    """Hash/decode through the native implementation, then retain only lossless P2 fields."""

    def __init__(self, base: PreparedRGBCache, *, limit_bytes: int) -> None:
        super().__init__(base, limit_bytes=limit_bytes)
        self.compactions = 0
        self.discarded_bytes = 0

    def get(self, index: int, token: str) -> dict:
        """Preserve the original first-use decode and every consumed sample value."""
        shard, _ = self.base._lookup[index]
        key = str(shard["path"])
        was_loaded = key in self.base._loaded
        result = super().get(index, token)
        # Memory pressure in the parent may evict a hit before it is decoded again.
        stored = self.base._loaded[key]
        if (
            not was_loaded
            or set(stored) != {*PAIR_FIELDS, *STATIC_FIELDS, "delta_t_s"}
            or stored["rgb_uint8"].shape[1] != 2
        ):
            before = sum(array.nbytes for array in stored.values())
            compact = compact_shard(stored)
            after = sum(array.nbytes for array in compact.values())
            if after > before:
                raise ValueError("P2 compaction unexpectedly expanded a native shard")
            self.base._loaded[key] = compact
            self.base._loaded_bytes += after - before
            self.compactions += 1
            self.discarded_bytes += before - after
        return result

    def counters(self) -> dict:
        return {
            **super().counters(),
            "compactions": self.compactions,
            "discarded_bytes_over_loads": self.discarded_bytes,
            "stored_scope": "Native final-two frames and metadata consumed by P2 only",
        }


class P2CachedRGBPairDataset(CachedRGBPairDataset):
    """Keep the registered collation and uint8-to-FP32 conversion unchanged."""

    def __init__(self, role: str = "P", *, limit_bytes: int = 6 * 1024**3) -> None:
        if role != "P":
            raise ValueError("This runtime optimization is admitted only for P training inputs")
        super().__init__(role, limit_bytes=limit_bytes)
        previous = self.source.prepared_cache
        self.source.prepared_cache = P2RGBCache(previous.base, limit_bytes=limit_bytes)
