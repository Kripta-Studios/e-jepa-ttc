"""All-or-none RGB feature reuse for a shared multiobject sensor update.

Only producer outputs are cached. Callers must rebuild RGB metadata and TTC
contexts on every update, including hits, and include this bookkeeping and raw
frame fingerprinting inside their complete-update timing boundary.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import Tensor

from operational.train40_runtime.feature_cache import (
    CachedRGBFeatures,
    ExactRGBFeatureCache,
    RGBFeatureKey,
    RGBHistoryIdentity,
    bind_rgb_history,
    rgb_feature_key,
)
from operational.train40_runtime.rgb_frozen_identity import bind_frozen_rgb_history
from operational.train40_runtime.rgb_immutable_history import FrozenRGBHistory

if TYPE_CHECKING:
    from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory


def _validate_sha(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class RGBBatchCacheTicket:
    """One update's lookup receipt; cached tensors follow present_indices order.

    ``cached_features is None`` means all present objects must be recomputed.
    An empty list means there are no RGB-present objects. Neither metadata nor
    context is retained. Only the exact latest ticket may be stored once.
    """

    keys: tuple[RGBFeatureKey, ...]
    present_indices: tuple[int, ...]
    cached_features: list[CachedRGBFeatures] | None
    identity: RGBHistoryIdentity | None
    diagnostics: dict[str, Any]


class RGBBatchCache:
    """Bounded exact features with the producer's object chunk shape in each key.

    The startup digests must bind all RGB weights, preprocessing/producer code,
    normalization, library versions, precision/layout flags and chunk options.
    That state must remain frozen for this instance's lifetime. Raw history is
    content-hashed once per update, shared by all objects. Factory-authenticated
    FrozenRGBHistory packets bind arrival-time hashes without rescanning RAW.
    No source-stat trust is inferred from ordinary RawRGBHistory packets.

    Instances belong to one thread and CUDA stream, like ExactRGBFeatureCache.
    Store only after all producer guards and stream waits have completed.
    """

    def __init__(
        self,
        *,
        model_sha256: str,
        runtime_sha256: str,
        rgb_object_chunk_size: int | None = None,
        max_entries: int = 20,
        max_bytes: int = 1024**2,
    ) -> None:
        self._model_sha256 = _validate_sha(model_sha256, "model_sha256")
        self._runtime_sha256 = _validate_sha(runtime_sha256, "runtime_sha256")
        if rgb_object_chunk_size is not None and (
            not isinstance(rgb_object_chunk_size, int)
            or isinstance(rgb_object_chunk_size, bool)
            or not 1 <= rgb_object_chunk_size <= 20
        ):
            raise ValueError("rgb_object_chunk_size must be None or an integer from 1 to 20")
        self._chunk_size = rgb_object_chunk_size
        self._cache = ExactRGBFeatureCache(max_entries=max_entries, max_bytes=max_bytes)
        self._active_ticket: RGBBatchCacheTicket | None = None

    @property
    def rgb_object_chunk_size(self) -> int | None:
        """Expose the frozen chunk ABI so a service can reject mismatched setup."""
        return self._chunk_size

    @property
    def stats(self) -> dict[str, int]:
        """Return cumulative hit/miss/invalidation and resident-byte receipts."""
        return self._cache.stats

    def begin(
        self, objects: Sequence[ObjectRegion], raw_history: RawRGBHistory
    ) -> RGBBatchCacheTicket:
        """Identify current inputs and serve features only if the whole lot hits.

        RGB absence removes that object's entry. A missing shared history clears
        all RGB entries. Writable/unattested weak frame identity is rejected by
        the content binding; source bytes must remain frozen until consumption.
        """
        started = time.perf_counter()
        self._active_ticket = None
        if not 1 <= len(objects) <= 20:
            raise ValueError("RGB batch cache supports one to twenty objects")
        object_ids = [item.object_id for item in objects]
        if any(not isinstance(value, str) or not value for value in object_ids) or len(
            set(object_ids)
        ) != len(object_ids):
            raise ValueError("Object IDs must be unique nonempty strings")
        if raw_history.unavailable_reason and raw_history.triplets:
            raise ValueError("Unavailable RGB history cannot contain producer observations")
        present = tuple(
            index
            for index, item in enumerate(objects)
            if item.rgb_square is not None and raw_history.triplets
        )
        self._cache.retain_objects([object_ids[index] for index in present])
        identity = None
        keys: tuple[RGBFeatureKey, ...] = ()
        cached: list[CachedRGBFeatures] | None = []
        if present:
            identity = (
                bind_frozen_rgb_history(
                    raw_history.tokens, raw_history.triplets, query_us=raw_history.query_us
                )
                if isinstance(raw_history, FrozenRGBHistory)
                else bind_rgb_history(
                    raw_history.frames,
                    raw_history.timestamps_us,
                    raw_history.triplets,
                    query_us=raw_history.query_us,
                )
            )
            group_size = len(identity.triplets)
            steps = len(identity.triplets[0])
            chunk_size = self._chunk_size or len(present)
            built_keys = []
            for position, index in enumerate(present):
                chunk_start = position // chunk_size * chunk_size
                chunk_objects = min(chunk_size, len(present) - chunk_start)
                square = objects[index].rgb_square
                assert square is not None
                built_keys.append(
                    rgb_feature_key(
                        object_ids[index],
                        square,
                        identity,
                        model_sha256=self._model_sha256,
                        runtime_sha256=self._runtime_sha256,
                        batch_shape=(chunk_objects * group_size, steps, 3, 128, 128),
                        object_slot=position - chunk_start,
                    )
                )
            keys = tuple(built_keys)
            cached = self._cache.lookup_many(keys)
        ticket = RGBBatchCacheTicket(
            keys=keys,
            present_indices=present,
            cached_features=cached,
            identity=identity,
            diagnostics={
                "status": "NO_RGB" if not present else "HIT" if cached is not None else "MISS",
                "present_objects": len(present),
                "history_observations": len(identity.triplets) if identity else 0,
                "raw_history_bytes": identity.raw_bytes if identity else 0,
                "hashed_bytes": identity.hashed_bytes if identity else 0,
                "fingerprint_seconds": identity.fingerprint_seconds if identity else 0.0,
                "history_digest": identity.digest if identity else None,
                "key_digests": [key.digest for key in keys],
                "begin_host_seconds": time.perf_counter() - started,
                "cache_stats": self._cache.stats,
            },
        )
        self._active_ticket = ticket
        return ticket

    def store(
        self,
        ticket: RGBBatchCacheTicket,
        rawflat: Tensor,
        expertsflat: Tensor,
        *,
        guard_passed: bool,
    ) -> bool:
        """Own a guarded flat RGB lot in present-object/history-major order.

        Flat shapes must be [present_objects * history, 17] and [..., 3].
        Guard or insertion failure consumes this ticket; a new begin is needed.
        No values derived from context, targets or TTC enter this interface.
        """
        if ticket is not self._active_ticket:
            raise ValueError("Only this cache's latest unused ticket may be stored")
        self._active_ticket = None
        if not ticket.keys or ticket.cached_features is not None or ticket.identity is None:
            raise ValueError("Only a missed nonempty RGB lot may be stored")
        rows = len(ticket.keys) * len(ticket.identity.triplets)
        if (
            not isinstance(rawflat, Tensor)
            or not isinstance(expertsflat, Tensor)
            or rawflat.shape != (rows, 17)
            or expertsflat.shape != (rows, 3)
        ):
            raise ValueError("Flat RGB producer features do not match the ticket lot")
        return self._cache.store_many(
            ticket.keys,
            rawflat.reshape(len(ticket.keys), -1, 17),
            expertsflat.reshape(len(ticket.keys), -1, 3),
            guard_passed=guard_passed,
        )

    def clear(self) -> None:
        """Invalidate all RGB entries and any pending insertion ticket."""
        self._active_ticket = None
        self._cache.clear()
