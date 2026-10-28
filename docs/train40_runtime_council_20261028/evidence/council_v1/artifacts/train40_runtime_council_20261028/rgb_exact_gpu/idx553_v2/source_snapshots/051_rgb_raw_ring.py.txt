"""Arrival-time immutable RGB tokens and a bounded shared resident frame ring.

Copying and hashing belong to frame arrival, inside the chosen sensor-to-TTC
measurement boundary. A resident hit does not rehash RAW history. This module
provides neither cropping nor a complete-update latency guarantee.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True, slots=True)
class RGBSourceIdentity:
    """Caller-provided source revision; content hashes still bind every frame."""

    name: str
    revision_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Source name must be a nonempty string")
        if (
            not isinstance(self.revision_sha256, str)
            or len(self.revision_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.revision_sha256)
        ):
            raise ValueError("Source revision must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class _FrameRecord:
    source_name: str
    source_revision: str
    frame_id: int
    timestamp_us: int
    shape: tuple[int, int, int]
    payload: bytes
    sha256: str
    identity_sha256: str
    freeze_seconds: float


class FrozenRGBFrame:
    """Factory-only token with immutable bytes and no writable object fields.

    ``array`` creates a readonly NumPy view backed by bytes, not merely an owner
    array with a reversible WRITEABLE flag. Its shape/view metadata may be
    changed by the caller without affecting the token or subsequent views.
    The private weak registry verifies factory provenance without rehashing.
    This is an ownership contract, not isolation from hostile Python code that
    can modify module globals or native memory.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> FrozenRGBFrame:
        raise TypeError("Use freeze_rgb_frame to construct immutable RGB tokens")

    def __init_subclass__(cls) -> None:
        raise TypeError("FrozenRGBFrame cannot be subclassed")

    @property
    def source(self) -> RGBSourceIdentity:
        """Return a value copy of the source revision captured at arrival."""
        record = _record(self)
        return RGBSourceIdentity(record.source_name, record.source_revision)

    @property
    def frame_id(self) -> int:
        """Return the source frame identifier captured at arrival."""
        return _record(self).frame_id

    @property
    def timestamp_us(self) -> int:
        """Return the source observation timestamp in microseconds."""
        return _record(self).timestamp_us

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the immutable uint8 HWC dimensions."""
        return _record(self).shape

    @property
    def payload(self) -> bytes:
        """Return owned immutable C-order RGB bytes."""
        return _record(self).payload

    @property
    def array(self) -> np.ndarray:
        """Return a fresh zero-copy readonly view backed by immutable bytes."""
        record = _record(self)
        return np.frombuffer(record.payload, dtype=np.uint8).reshape(record.shape)

    @property
    def sha256(self) -> str:
        """Return the RAW content checksum computed once during freezing."""
        return _record(self).sha256

    @property
    def content_sha256(self) -> str:
        """Alias for the RAW content checksum."""
        return self.sha256

    @property
    def identity_sha256(self) -> str:
        """Bind source, frame ID, timestamp, shape, dtype and content checksum."""
        return _record(self).identity_sha256

    @property
    def hashed_bytes(self) -> int:
        """Number of RAW bytes hashed once when the token was constructed."""
        return len(_record(self).payload)

    @property
    def freeze_seconds(self) -> float:
        """Measured arrival-time validation, copying and checksum host duration."""
        return _record(self).freeze_seconds


_RECORDS: weakref.WeakKeyDictionary[FrozenRGBFrame, _FrameRecord] = weakref.WeakKeyDictionary()
_RECORD_LOCK = threading.RLock()


def _record(frame: FrozenRGBFrame) -> _FrameRecord:
    if type(frame) is not FrozenRGBFrame:
        raise ValueError("Ring frames must be factory-created FrozenRGBFrame tokens")
    with _RECORD_LOCK:
        record = _RECORDS.get(frame)
    if record is None:
        raise ValueError("Unregistered or forged FrozenRGBFrame token")
    return record


def validate_frozen_rgb_frame(frame: FrozenRGBFrame) -> None:
    """Verify immutable factory provenance without hashing the payload again.

    Authentic tokens expose only properties of the factory's private immutable
    record. Thus provenance validation also binds their source, ID, timestamp,
    shape and recorded content identity; caller-owned readonly arrays are not
    accepted as equivalent evidence. No RAW byte scanning occurs here.
    """
    _record(frame)


def freeze_rgb_frame(
    frame: np.ndarray,
    frame_id: int,
    timestamp_us: int,
    source: RGBSourceIdentity,
    *,
    max_bytes: int = 128 * 1024**2,
) -> FrozenRGBFrame:
    """Copy a newly arrived frame to bytes and hash those bytes exactly once.

    The caller must own the source array until copying completes. Subsequent
    mutations of that array cannot affect this token. A source revision is
    descriptive provenance, not permission to skip hashing changed content.
    Invoke once per sensor arrival and share the resulting token across ROIs
    and updates; repeated factory calls intentionally repeat the arrival cost.
    """
    started = time.perf_counter()
    frame_id = _integer(frame_id, "frame_id")
    timestamp_us = _integer(timestamp_us, "timestamp_us")
    max_bytes = _integer(max_bytes, "max_bytes", minimum=1)
    if not isinstance(source, RGBSourceIdentity):
        raise ValueError("A validated RGBSourceIdentity is required")
    source_copy = RGBSourceIdentity(source.name, source.revision_sha256)
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape) <= 0
    ):
        raise ValueError("RGB frames must be nonempty uint8 HWC arrays with three channels")
    if frame.nbytes > max_bytes:
        raise ValueError("RGB frame exceeds the arrival byte bound")
    shape = (int(frame.shape[0]), int(frame.shape[1]), 3)
    payload = frame.tobytes(order="C")
    checksum = hashlib.sha256(payload).hexdigest()
    descriptor = (
        source_copy.name,
        source_copy.revision_sha256,
        frame_id,
        timestamp_us,
        shape,
        "uint8_HWC_RGB",
        checksum,
    )
    identity = hashlib.sha256(json.dumps(descriptor, separators=(",", ":")).encode()).hexdigest()
    record = _FrameRecord(
        source_copy.name,
        source_copy.revision_sha256,
        frame_id,
        timestamp_us,
        shape,
        payload,
        checksum,
        identity,
        time.perf_counter() - started,
    )
    token = object.__new__(FrozenRGBFrame)
    with _RECORD_LOCK:
        _RECORDS[token] = record
    return token


@dataclass
class _ResidentEntry:
    identity: str
    slot: int
    last_used: int


@dataclass(frozen=True)
class RGBRingBinding:
    """Borrowed resident slots in input order, valid until mark_consumed()."""

    slots: tuple[int, ...]
    source: RGBSourceIdentity
    frame_ids: tuple[int, ...]
    timestamps_us: tuple[int, ...]
    frame_identities: tuple[str, ...]
    receipt: dict[str, Any]
    _owner: ResidentRGBFrameRing = field(repr=False, compare=False)

    @property
    def tensor(self) -> Tensor:
        """Return the complete resident tensor, indexed using this binding's slots."""
        self.assert_valid()
        return self._owner.tensor

    def assert_valid(self) -> None:
        """Reject expired or foreign leases before reading resident frames."""
        self._owner.assert_valid(self)


class ResidentRGBFrameRing:
    """Bounded RAW storage with one explicit same-thread/same-stream lease.

    ``max_bytes`` limits each resident/staging pool, not their combined size.
    CUDA allocates one equally sized pinned host pool; CPU copies directly into
    resident storage. Allocation receipts report both pools and their sum.
    Historical capacity defaults to 16 so both alternating seven-frame chains
    from a 20 Hz camera can remain resident when native observations are spaced
    by 100 ms. Each query still requests at most ten frames. Initial use of
    each chain requires a causal warmup/upload; one upload per arrival is not
    promised for the first query or for other sensor/history patterns.
    Consumers must not modify resident pixels and must run on the owner stream.
    After all consumers have been
    enqueued, mark_consumed records a fence. The next bind waits before reusing
    storage, including pinned host slots. No RAW checksum occurs in bind.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        *,
        frame_shape: Sequence[int],
        capacity: int = 16,
        max_bytes: int = 128 * 1024**2,
    ) -> None:
        self._capacity = _integer(capacity, "capacity", minimum=1)
        if self._capacity > 32:
            raise ValueError("Historical RAW RGB ring capacity cannot exceed 32 frames")
        if len(frame_shape) != 3:
            raise ValueError("frame_shape must be nonempty HWC with three channels")
        shape = tuple(_integer(value, "frame dimension", minimum=1) for value in frame_shape)
        if shape[2] != 3:
            raise ValueError("frame_shape must be nonempty HWC with three channels")
        self._frame_shape = (shape[0], shape[1], shape[2])
        self._frame_bytes = shape[0] * shape[1] * shape[2]
        self._pool_bytes = self._capacity * self._frame_bytes
        self._max_bytes = _integer(max_bytes, "max_bytes", minimum=1)
        if self._pool_bytes > self._max_bytes:
            raise ValueError("Resident RAW pool exceeds max_bytes before allocation")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("RAW RGB ring supports only CPU and CUDA")
        self._thread = threading.get_ident()
        self._stream: int | None = None
        self._consumed_event: torch.cuda.Event | None = None
        self._pending = False
        self._closed = self._poisoned = False
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            self._stream = torch.cuda.current_stream(self.device).cuda_stream
            self._consumed_event = torch.cuda.Event()
        self._resident: Tensor | None = torch.empty(
            (self._capacity, *self._frame_shape), dtype=torch.uint8, device=self.device
        )
        self._host = (
            torch.empty((self._capacity, *self._frame_shape), dtype=torch.uint8, pin_memory=True)
            if self.device.type == "cuda"
            else None
        )
        self._host_array: np.ndarray | None = (
            self._host if self._host is not None else self._resident
        ).numpy()
        self._entries: dict[str, _ResidentEntry] = {}
        self._slots: list[_ResidentEntry | None] = [None] * self._capacity
        self._source: tuple[str, str] | None = None
        self._query_us: int | None = None
        self._active: RGBRingBinding | None = None
        self._tick = 0
        self._stats: dict[str, int] = dict.fromkeys(
            (
                "binds",
                "hits",
                "uploads",
                "uploaded_bytes",
                "h2d_bytes",
                "evictions",
                "source_resets",
                "rollback_resets",
                "explicit_resets",
                "invalidated_frames",
                "completion_waits",
                "failed_binds",
            ),
            0,
        )

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Fixed HWC dimensions of every resident frame."""
        return self._frame_shape

    @property
    def capacity(self) -> int:
        """Maximum number of simultaneously resident frames."""
        return self._capacity

    @property
    def tensor(self) -> Tensor:
        """Borrow the full allocation; valid frame contents require a live binding."""
        self._owner()
        self._usable()
        assert self._resident is not None
        return self._resident

    @property
    def allocation_receipt(self) -> dict[str, Any]:
        """Report fixed logical storage, excluding allocator overhead and input tokens."""
        staging = self._pool_bytes if self.device.type == "cuda" else 0
        return {
            "device": str(self.device),
            "capacity": self._capacity,
            "max_frames_per_bind": min(self._capacity, 10),
            "frame_shape": list(self._frame_shape),
            "frame_bytes": self._frame_bytes,
            "resident_buffer_bytes": self._pool_bytes,
            "pinned_host_buffer_bytes": staging,
            "total_preallocated_tensor_bytes": self._pool_bytes + staging,
            "max_bytes_per_pool": self._max_bytes,
            "host_pinned": self.device.type == "cuda",
            "closed": self._closed,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Return cumulative frame/upload/reset counters and current occupancy."""
        return {
            **self._stats,
            "resident_frames": len(self._entries),
            "resident_payload_bytes": len(self._entries) * self._frame_bytes,
            "frame_bytes_hashed_inside_bind": 0,
        }

    def _owner(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("RAW RGB ring requires its owning thread")
        if self.device.type == "cuda":
            if torch.cuda.current_stream(self.device).cuda_stream != self._stream:
                raise RuntimeError("RAW RGB ring requires its owning CUDA stream")

    def _usable(self) -> None:
        if self._closed or self._poisoned:
            raise RuntimeError("RAW RGB ring is closed or unusable after a failed upload")

    def _wait(self) -> None:
        if self._pending:
            assert self._consumed_event is not None
            self._consumed_event.synchronize()
            self._pending = False
            self._stats["completion_waits"] += 1

    def _invalidate(self) -> None:
        self._stats["invalidated_frames"] += len(self._entries)
        self._entries.clear()
        self._slots = [None] * self._capacity

    def bind(self, frames: Sequence[FrozenRGBFrame], query_us: int) -> RGBRingBinding:
        """Upload only missing immutable frames; preserve all currently required slots.

        A source revision change or query-clock rollback resets occupancy and is
        explicit in the receipt. Current frames must have distinct frame IDs,
        match the configured shape, share a source and be available by query_us.
        Their ordering is preserved in slots; no GPU index tensor is allocated.
        """
        started = time.perf_counter()
        self._owner()
        self._usable()
        if self._active is not None:
            raise RuntimeError("mark_consumed(binding) is required before another bind")
        query_us = _integer(query_us, "query_us")
        if not 1 <= len(frames) <= min(self._capacity, 10):
            raise ValueError("Current RAW history requires 1..min(capacity, 10) frames")
        for frame in frames:
            validate_frozen_rgb_frame(frame)
        records = tuple(_record(frame) for frame in frames)
        sources = {(record.source_name, record.source_revision) for record in records}
        if len(sources) != 1:
            raise ValueError("One RAW binding cannot mix source identities")
        if len({record.frame_id for record in records}) != len(records):
            raise ValueError("Current RAW frame IDs must be distinct")
        if any(record.shape != self._frame_shape for record in records):
            raise ValueError("A RAW frame differs from the configured resident shape")
        if any(record.timestamp_us > query_us for record in records):
            raise ValueError("A RAW frame is not available at this query timestamp")
        source = next(iter(sources))
        source_changed = self._source is not None and self._source != source
        rollback = self._query_us is not None and query_us < self._query_us
        waited = time.perf_counter()
        self._wait()
        wait_seconds = time.perf_counter() - waited
        if source_changed or rollback:
            self._invalidate()
            self._stats["source_resets"] += int(source_changed)
            self._stats["rollback_resets"] += int(rollback)
        required = {record.identity_sha256 for record in records}
        slots = []
        hits = uploads = evictions = 0
        self._tick += 1
        assert self._host_array is not None and self._resident is not None
        try:
            for record in records:
                entry = self._entries.get(record.identity_sha256)
                if entry is not None:
                    hits += 1
                    entry.last_used = self._tick
                    slots.append(entry.slot)
                    continue
                free = next((i for i, entry in enumerate(self._slots) if entry is None), None)
                if free is None:
                    victim = min(
                        (
                            entry
                            for entry in self._entries.values()
                            if entry.identity not in required
                        ),
                        key=lambda entry: (entry.last_used, entry.slot),
                    )
                    free = victim.slot
                    self._entries.pop(victim.identity)
                    evictions += 1
                pixels = np.frombuffer(record.payload, dtype=np.uint8).reshape(record.shape)
                np.copyto(self._host_array[free], pixels)
                if self._host is not None:
                    self._resident[free].copy_(self._host[free], non_blocking=True)
                entry = _ResidentEntry(record.identity_sha256, free, self._tick)
                self._entries[entry.identity] = entry
                self._slots[free] = entry
                slots.append(free)
                uploads += 1
        except BaseException:
            self._poisoned = True
            self._stats["failed_binds"] += 1
            self._invalidate()
            raise
        self._source, self._query_us = source, query_us
        uploaded = uploads * self._frame_bytes
        h2d = uploaded if self.device.type == "cuda" else 0
        for key, value in (
            ("binds", 1),
            ("hits", hits),
            ("uploads", uploads),
            ("uploaded_bytes", uploaded),
            ("h2d_bytes", h2d),
            ("evictions", evictions),
        ):
            self._stats[key] += value
        binding = RGBRingBinding(
            tuple(slots),
            RGBSourceIdentity(*source),
            tuple(record.frame_id for record in records),
            tuple(record.timestamp_us for record in records),
            tuple(record.identity_sha256 for record in records),
            {
                "hits": hits,
                "uploads": uploads,
                "evictions": evictions,
                "uploaded_bytes": uploaded,
                "h2d_bytes": h2d,
                "source_reset": source_changed,
                "query_rollback_reset": rollback,
                "query_us": query_us,
                "source_name": source[0],
                "source_revision_sha256": source[1],
                "frame_bytes_hashed_inside_bind": 0,
                "wait_previous_host_seconds": wait_seconds,
                "bind_enqueue_host_seconds": time.perf_counter() - started,
                "stats": self.stats,
            },
            self,
        )
        self._active = binding
        return binding

    def assert_valid(self, binding: RGBRingBinding) -> None:
        """Require this ring's currently active lease on the owner thread/stream."""
        self._owner()
        self._usable()
        if binding is not self._active:
            raise ValueError("RAW RGB binding is foreign, released or superseded")

    def mark_consumed(self, binding: RGBRingBinding) -> None:
        """End a lease after every owner-stream consumer has been enqueued."""
        self.assert_valid(binding)
        if self._consumed_event is not None:
            self._consumed_event.record(torch.cuda.current_stream(self.device))
            self._pending = True
        self._active = None

    def reset(self) -> None:
        """Explicitly forget a consumed source/history without resetting counters."""
        self._owner()
        self._usable()
        if self._active is not None:
            raise RuntimeError("mark_consumed(binding) is required before reset")
        self._wait()
        self._invalidate()
        self._source = self._query_us = None
        self._stats["explicit_resets"] += 1

    def close(self) -> None:
        """Wait for consumed work and invalidate storage; active leases must finish first."""
        self._owner()
        if self._closed:
            return
        if self._active is not None:
            raise RuntimeError("mark_consumed(binding) is required before close")
        if self._poisoned and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        self._wait()
        self._invalidate()
        self._host_array = None
        self._host = None
        self._resident = None
        self._closed = True
