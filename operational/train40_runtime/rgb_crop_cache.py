"""Bounded exact PIL crop reuse from authenticated causal immutable RAW frames.

Only CPU pixels are retained. Producer batch shapes, current metadata, contexts
and TTC are outside this cache. The caller still uses the original PIL crop on
every miss and stacks all original history rows before invoking the producer.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import __version__ as pillow_version

from e_jepa_ttc.rgb_port.data import RGB_SIZE
from operational.train40_runtime.rgb_raw_ring import FrozenRGBFrame, validate_frozen_rgb_frame

CROP_SHAPE = (3, RGB_SIZE, RGB_SIZE)
CROP_BYTES = 3 * RGB_SIZE * RGB_SIZE * np.dtype(np.float32).itemsize
CROP_ABI = (
    "PIL_ROI_FIRST_EFFECTIVE_SQUARE_BILINEAR_CHW_FP32_DIV255_V1",
    pillow_version,
    RGB_SIZE,
)
_ABI_SHA256 = hashlib.sha256(json.dumps(CROP_ABI, separators=(",", ":")).encode()).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class _CropKey:
    object_id: str
    frame_identity_sha256: str
    roi_hex: tuple[str, ...]
    abi_sha256: str


@dataclass(frozen=True)
class _CropValue:
    payload: bytes

    def view(self) -> np.ndarray:
        """Return fresh shape metadata backed by immutable owned FP32 bytes."""
        return np.frombuffer(self.payload, dtype=np.float32).reshape(CROP_SHAPE)


def _key(
    object_id: str,
    frame: FrozenRGBFrame,
    roi: Sequence[float],
    query_us: int,
) -> _CropKey:
    validate_frozen_rgb_frame(frame)
    if not isinstance(object_id, str) or not 1 <= len(object_id) <= 256:
        raise ValueError("Crop object ID must contain one to 256 characters")
    if isinstance(query_us, bool) or not isinstance(query_us, int) or query_us < 0:
        raise ValueError("Crop query timestamp must be a nonnegative integer")
    if not 0 <= query_us - frame.timestamp_us <= 650_000:
        raise ValueError("Crop frame must be available within the causal 650 ms history")
    if len(roi) != 4 or any(isinstance(value, (bool, str, bytes)) for value in roi):
        raise ValueError("Crop ROI must contain four finite numeric coordinates")
    values = tuple(float(value) for value in roi)
    if not all(math.isfinite(value) for value in values) or (
        values[2] <= values[0] or values[3] <= values[1]
    ):
        raise ValueError("Crop ROI must be finite with positive width and height")
    return _CropKey(
        object_id, frame.identity_sha256, tuple(value.hex() for value in values), _ABI_SHA256
    )


class ExactRGBCropCache:
    """Single-worker LRU of authenticated original PIL crops, bounded in CPU bytes.

    Construct on the service thread; ownership is claimed by the first get/put,
    normally the RGB preprocessing coordinator. Original crop-pool workers never
    touch this cache: their returned arrays are inserted by that coordinator.
    All returned views own a strong reference to immutable bytes and survive
    eviction/clear/close. Stats snapshots may be read from another thread.

    The byte bound covers crop payloads, not bounded Python metadata, caller-owned
    arrays or temporary copies returned by stacking. A put momentarily holds one
    extra crop while copying to bytes; stats expose that explicit scratch bound.
    The caller must bind the executed PIL/normalization source in its run receipt.
    """

    def __init__(
        self,
        *,
        max_objects: int = 2,
        max_crops_per_object: int = 16,
        max_bytes: int | None = None,
    ) -> None:
        self.max_objects = _positive_integer(max_objects, "max_objects")
        self.max_crops_per_object = _positive_integer(max_crops_per_object, "max_crops_per_object")
        if self.max_objects > 20 or self.max_crops_per_object > 16:
            raise ValueError("Crop cache supports at most twenty objects and sixteen crops each")
        default_bytes = self.max_objects * self.max_crops_per_object * CROP_BYTES
        self.max_bytes = _positive_integer(
            default_bytes if max_bytes is None else max_bytes, "max_bytes"
        )
        self._entries: OrderedDict[_CropKey, _CropValue] = OrderedDict()
        self._objects: Counter[str] = Counter()
        self._counters: Counter[str] = Counter()
        self._bytes = 0
        self._owner_thread: threading.Thread | None = None
        self._closed = False
        self._lock = threading.RLock()

    def _owner(self) -> None:
        if self._closed:
            raise RuntimeError("RGB crop cache is closed")
        current = threading.current_thread()
        if self._owner_thread is None:
            self._owner_thread = current
        elif current is not self._owner_thread:
            raise RuntimeError("RGB crop cache requires its owning preprocessing coordinator")

    @property
    def stats(self) -> dict[str, int | bool | str | list[str | int]]:
        """Snapshot counters/bounds without transferring or hashing RAW/crop bytes."""
        with self._lock:
            return {
                **self._counters,
                "entries": len(self._entries),
                "objects": len(self._objects),
                "crop_bytes": self._bytes,
                "max_objects": self.max_objects,
                "max_crops_per_object": self.max_crops_per_object,
                "max_crop_bytes": self.max_bytes,
                "copy_scratch_upper_bound_bytes": CROP_BYTES,
                "raw_payload_bytes_hashed": 0,
                "crop_abi": list(CROP_ABI),
                "crop_abi_sha256": _ABI_SHA256,
                "closed": self._closed,
            }

    def get(
        self,
        object_id: str,
        frame: FrozenRGBFrame,
        roi: Sequence[float],
        *,
        query_us: int,
    ) -> np.ndarray | None:
        """Return an immutable original crop or a miss; authenticate every lookup."""
        with self._lock:
            self._owner()
            key = _key(object_id, frame, roi, query_us)
            value = self._entries.get(key)
            if value is None:
                self._counters["misses"] += 1
                return None
            self._entries.move_to_end(key)
            self._counters["hits"] += 1
            return value.view()

    def _evict(self, key: _CropKey, reason: str) -> None:
        value = self._entries.pop(key)
        self._bytes -= len(value.payload)
        self._objects[key.object_id] -= 1
        if not self._objects[key.object_id]:
            del self._objects[key.object_id]
        self._counters[reason] += 1

    def put(
        self,
        object_id: str,
        frame: FrozenRGBFrame,
        roi: Sequence[float],
        crop: np.ndarray,
        *,
        query_us: int,
    ) -> bool:
        """Copy one verified-shape original PIL result to owned immutable bytes.

        The service computes crop using the authenticated frame's array and the
        original _normalized_crop function. This method validates its shape,
        dtype/range, not the provenance of an arbitrary caller-supplied crop.
        Rejected oversized insertions never evict existing useful entries.
        """
        with self._lock:
            self._owner()
            key = _key(object_id, frame, roi, query_us)
            if (
                not isinstance(crop, np.ndarray)
                or crop.shape != CROP_SHAPE
                or crop.dtype != np.float32
            ):
                raise ValueError("Cached PIL crops must be float32 CHW [3,128,128]")
            if not np.isfinite(crop).all() or np.any(crop < 0) or np.any(crop > 1):
                raise ValueError("Cached PIL crops must contain finite normalized RGB in [0,1]")
            if crop.nbytes > self.max_bytes:
                self._counters["oversized_rejections"] += 1
                return False
            value = _CropValue(crop.tobytes(order="C"))
            if key in self._entries:
                self._evict(key, "replacements")
            if key.object_id not in self._objects and len(self._objects) >= self.max_objects:
                victim = next(iter(self._entries)).object_id
                for old in tuple(self._entries):
                    if old.object_id == victim:
                        self._evict(old, "object_capacity_evictions")
            while self._objects.get(key.object_id, 0) >= self.max_crops_per_object:
                old = next(old for old in self._entries if old.object_id == key.object_id)
                self._evict(old, "per_object_capacity_evictions")
            while self._bytes + len(value.payload) > self.max_bytes:
                self._evict(next(iter(self._entries)), "byte_capacity_evictions")
            self._entries[key] = value
            self._objects[key.object_id] += 1
            self._bytes += len(value.payload)
            self._counters["stored_crops"] += 1
            self._counters["copied_crop_bytes"] += len(value.payload)
            return True

    def clear(self) -> None:
        """Forget all crops on the owner, or after its thread has terminated."""
        with self._lock:
            if self._owner_thread is not None and (
                self._owner_thread is not threading.current_thread()
                and self._owner_thread.is_alive()
            ):
                raise RuntimeError("Join the preprocessing owner before clearing its crop cache")
            self._entries.clear()
            self._objects.clear()
            self._bytes = 0
            self._counters["clears"] += 1

    def close(self) -> None:
        """Release storage after preprocessing shutdown; borrowed views remain valid."""
        with self._lock:
            if self._closed:
                return
            self.clear()
            self._closed = True
