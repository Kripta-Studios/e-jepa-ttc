"""Bounded exact memoization of RGB producer features, never context or TTC.

Frame identity is established once for the shared raw history, before making
per-object ROI keys. Unattested frames are content-hashed. A faster file/stat
binding is explicitly restricted to read-only frames from the already validated
source reader; it is not protection against adversarial restoration of mtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RGBHistoryIdentity:
    """Shared source/frame identity excluding the changing query-context clock."""

    digest: str
    source_binding: tuple[str, ...]
    triplets: tuple[tuple[int, ...], ...]
    frame_clocks_us: tuple[tuple[int, int], ...]
    delta_seconds_hex: tuple[tuple[str, ...], ...]
    raw_bytes: int
    hashed_bytes: int
    fingerprint_seconds: float


def bind_rgb_history(
    frames: Mapping[int, np.ndarray],
    timestamps_us: Mapping[int, int],
    triplets: Sequence[Sequence[int]],
    *,
    query_us: int,
    source_path: Path | None = None,
    expected_source_stat: Mapping[str, int] | None = None,
    dataset: str = "blackflys/left/data",
    max_raw_bytes: int = 128 * 1024**2,
) -> RGBHistoryIdentity:
    """Validate and identify raw history once, shared by all object keys.

    With no source path, SHA-256 covers every referenced frame's uint8 bytes,
    shape, ID and timestamp. Rebuild this identity each update; do not cache it
    by NumPy object address or timestamp when the source is mutable/unattested.
    All frames must be read-only and remain unchanged through consumption.
    A sensor that reuses writable buffers must first supply owned frozen frames.

    The optional stat-bound mode requires both a path and expected size/mtime,
    checks the source before and after binding, and accepts only read-only C
    contiguous frames. The caller attests these are the unmodified frames read
    from that source/dataset. It must invalidate its own frame cache on a source
    change. A stat binding is deliberately labeled differently from a content
    hash and does not claim a full HDF5 checksum.

    Query availability is checked but excluded from the digest: producer inputs
    do not depend on the current query clock. Context ages must be recomputed.
    """
    begin = time.perf_counter()
    _integer(query_us, "query_us")
    _integer(max_raw_bytes, "max_raw_bytes", minimum=1)
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be a nonempty source dataset name")
    groups = tuple(tuple(_integer(i, "frame index") for i in group) for group in triplets)
    if not 1 <= len(groups) <= 8 or len({len(group) for group in groups}) != 1:
        raise ValueError("RGB history requires one to eight equally sized observations")
    if any(len(group) not in (2, 3) for group in groups):
        raise ValueError("Each RGB observation must contain two or three frames")
    frame_ids = sorted({index for group in groups for index in group})
    if len(frame_ids) > 10:
        raise ValueError("RGB history exceeds ten distinct frames")
    clocks = {}
    for index in frame_ids:
        if index not in frames or index not in timestamps_us:
            raise ValueError("RGB history lacks a referenced frame or timestamp")
        clocks[index] = _integer(timestamps_us[index], "frame timestamp")
    for group in groups:
        times = [clocks[index] for index in group]
        if any(a >= b for a, b in zip(times[:-1], times[1:], strict=True)):
            raise ValueError("RGB frame timestamps must be strictly chronological")
        if times[-1] > query_us or query_us - times[0] > 650_000:
            raise ValueError("RGB history must be available and inside the causal support")
    anchors = [clocks[group[-1]] for group in groups]
    if any(a >= b for a, b in zip(anchors[:-1], anchors[1:], strict=True)):
        raise ValueError("RGB observation anchors must be strictly chronological")
    if (source_path is None) != (expected_source_stat is None):
        raise ValueError("Stat binding requires both source_path and expected_source_stat")
    path = None
    source_binding = ("RAW_FRAME_SHA256", dataset)
    expected_stat = None
    if source_path is not None:
        path = source_path.resolve(strict=True)
        assert expected_source_stat is not None
        expected_stat = (
            _integer(expected_source_stat.get("size_bytes"), "source size"),
            _integer(expected_source_stat.get("mtime_ns"), "source mtime"),
        )
        stat = path.stat()
        if not path.is_file() or (stat.st_size, stat.st_mtime_ns) != expected_stat:
            raise ValueError("RGB source stat differs from the validated reader binding")
        source_binding = ("READONLY_READER_STAT", str(path), *map(str, expected_stat), dataset)
    descriptors = []
    raw_bytes = hashed_bytes = 0
    for index in frame_ids:
        frame = frames[index]
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3:
            raise ValueError("RGB raw frames must be uint8 HWC arrays")
        if frame.shape[-1] != 3 or min(frame.shape) <= 0:
            raise ValueError("RGB raw frames must contain three nonempty channels")
        if frame.flags.writeable:
            raise ValueError("RGB frame identity requires read-only arrays until consumption")
        raw_bytes += frame.nbytes
        if raw_bytes > max_raw_bytes:
            raise ValueError("Shared RGB frame history exceeds the configured byte bound")
        checksum = None
        if path is None:
            checksum = hashlib.sha256(memoryview(np.ascontiguousarray(frame)).cast("B")).hexdigest()
            hashed_bytes += frame.nbytes
        elif not frame.flags.c_contiguous:
            raise ValueError("Stat-bound RGB frames must be read-only contiguous reader arrays")
        descriptors.append((index, clocks[index], tuple(frame.shape), checksum))
    if path is not None:
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != expected_stat:
            raise ValueError("RGB source changed while binding frame history")
    deltas = tuple(
        tuple(
            ((clocks[b] - clocks[a]) / 1e6).hex()
            for a, b in zip(group[:-1], group[1:], strict=True)
        )
        for group in groups
    )
    digest = _digest((source_binding, groups, descriptors, deltas))
    return RGBHistoryIdentity(
        digest,
        source_binding,
        groups,
        tuple((index, clocks[index]) for index in frame_ids),
        deltas,
        raw_bytes,
        hashed_bytes,
        time.perf_counter() - begin,
    )


@dataclass(frozen=True)
class RGBFeatureKey:
    """Complete per-object producer call identity; query-context time is absent."""

    object_id: str
    model_sha256: str
    runtime_sha256: str
    precision: str
    roi_hex: tuple[str, ...]
    history: RGBHistoryIdentity
    batch_shape: tuple[int, ...]
    object_slot: int

    @property
    def digest(self) -> str:
        """Stable receipt digest without timing diagnostics in the shared identity."""
        return _digest(
            (
                self.object_id,
                self.model_sha256,
                self.runtime_sha256,
                self.precision,
                self.roi_hex,
                self.history.digest,
                self.batch_shape,
                self.object_slot,
            )
        )


def rgb_feature_key(
    object_id: str,
    roi: Sequence[float],
    history: RGBHistoryIdentity,
    *,
    model_sha256: str,
    runtime_sha256: str,
    batch_shape: Sequence[int],
    object_slot: int,
    precision: str = "FP32_TF32_DISABLED",
) -> RGBFeatureKey:
    """Bind an exact ROI to source history, model, runtime and actual batch ABI.

    ``runtime_sha256`` must identify executed producer/preprocessing code,
    compiler/layout flags and relevant library versions. ``batch_shape`` and
    ``object_slot`` describe the producer invocation that created the features.
    A caller must not silently reuse a batch-20 entry as batch-1 reference.
    """
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("object_id must be a nonempty string")
    _sha(model_sha256, "model_sha256")
    _sha(runtime_sha256, "runtime_sha256")
    _sha(history.digest, "history digest")
    if precision != "FP32_TF32_DISABLED":
        raise ValueError("This cache is admitted only for the explicit FP32/TF32-off contract")
    if len(roi) != 4 or any(isinstance(value, bool) for value in roi):
        raise ValueError("ROI requires four finite coordinates")
    square = tuple(float(value) for value in roi)
    if not all(math.isfinite(value) for value in square):
        raise ValueError("ROI requires four finite coordinates")
    if square[2] <= square[0] or square[3] <= square[1]:
        raise ValueError("ROI must have positive width and height")
    shape = tuple(_integer(value, "batch dimension", minimum=1) for value in batch_shape)
    count = len(history.triplets)
    steps = len(history.triplets[0])
    if len(shape) != 5 or shape[1:] != (steps, 3, 128, 128) or shape[0] % count:
        raise ValueError("RGB producer batch must be [objects*history, T, 3, 128, 128]")
    _integer(object_slot, "object_slot")
    if object_slot >= shape[0] // count:
        raise ValueError("object_slot is outside the actual RGB producer batch")
    return RGBFeatureKey(
        object_id,
        model_sha256,
        runtime_sha256,
        precision,
        tuple(value.hex() for value in square),
        history,
        shape,
        object_slot,
    )


@dataclass
class CachedRGBFeatures:
    """Owned feature copies returned to a caller; never a cached context/TTC."""

    raw: Tensor
    experts: Tensor


@dataclass
class _Entry:
    key: RGBFeatureKey
    value: CachedRGBFeatures
    bytes: int


class ExactRGBFeatureCache:
    """One latest entry per object, bounded by tensor bytes and entry count.

    Instances have one owning thread and CUDA stream. Store only after validating
    the producer guards and waiting for its streams. Returned tensors are clones,
    so caller mutation cannot poison future hits. This cache does not authorize
    a numerical variant: the complete original-prediction gate is still required.
    """

    def __init__(self, *, max_entries: int = 20, max_bytes: int = 1024**2) -> None:
        self.max_entries = _integer(max_entries, "max_entries", minimum=1)
        self.max_bytes = _integer(max_bytes, "max_bytes", minimum=1)
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._thread = threading.get_ident()
        self._device: torch.device | None = None
        self._stream: int | None = None
        self._stats: Counter[str] = Counter()

    def _owner(self, device: torch.device | None = None) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("RGB feature cache has a single owning thread")
        selected = device or self._device
        if self._device is not None and selected != self._device:
            raise ValueError("RGB feature cache cannot mix tensor devices")
        if selected is not None and selected.type == "cuda":
            stream = torch.cuda.current_stream(selected).cuda_stream
            if self._stream is not None and self._stream != stream:
                raise RuntimeError("RGB feature cache has a single owning CUDA stream")
            self._stream = stream
        if device is not None:
            self._device = device

    @property
    def stats(self) -> dict[str, int]:
        """Report served hits separately from matching keys in an incomplete lot."""
        return {
            **dict(self._stats),
            "entries": len(self._entries),
            "tensor_bytes": self._bytes,
            "max_entries": self.max_entries,
            "max_tensor_bytes": self.max_bytes,
        }

    def lookup_many(self, keys: Sequence[RGBFeatureKey]) -> list[CachedRGBFeatures] | None:
        """Return cloned features only when every requested object's key matches.

        The all-hit policy supports skipping the entire RGB producer invocation.
        If any key misses, recompute the complete same-shape batch first. Running
        only misses changes batch arithmetic and needs separate admission/keys.
        """
        self._owner()
        if not keys or len({key.object_id for key in keys}) != len(keys):
            raise ValueError("Lookup requires distinct nonempty object keys")
        found = []
        for key in keys:
            entry = self._entries.get(key.object_id)
            # fingerprint_seconds is diagnostic, not part of call identity.
            if entry is None or entry.key.digest != key.digest:
                self._stats["key_misses"] += 1
                if entry is not None:
                    self._bytes -= self._entries.pop(key.object_id).bytes
                    self._stats["invalidations"] += 1
                continue
            self._stats["key_matches"] += 1
            found.append(entry)
        if len(found) != len(keys):
            self._stats["batch_misses"] += 1
            return None
        self._stats["batch_hits"] += 1
        self._stats["served_object_hits"] += len(keys)
        for key in keys:
            self._entries.move_to_end(key.object_id)
        return [
            CachedRGBFeatures(entry.value.raw.clone(), entry.value.experts.clone())
            for entry in found
        ]

    def store_many(
        self,
        keys: Sequence[RGBFeatureKey],
        raw: Tensor,
        experts: Tensor,
        *,
        guard_passed: bool,
    ) -> bool:
        """Own validated [objects,history,17/3] features on the caller's stream.

        The additional finite check is one scalar synchronization for the whole
        lot. Tensor clones and cache bookkeeping belong inside update latency.
        An entry larger than the configured budget is rejected without evicting
        previously valid entries. Returns whether the whole lot was retained.
        """
        self._owner(raw.device)
        if not keys or len({key.object_id for key in keys}) != len(keys):
            raise ValueError("Store requires distinct nonempty object keys")
        lengths = {len(key.history.triplets) for key in keys}
        if len(lengths) != 1:
            raise ValueError("A stored RGB lot must have a uniform history length")
        count = next(iter(lengths))
        if (
            guard_passed is not True
            or raw.shape != (len(keys), count, 17)
            or experts.shape != (len(keys), count, 3)
            or raw.dtype != torch.float32
            or experts.dtype != torch.float32
            or experts.device != raw.device
            or raw.requires_grad
            or experts.requires_grad
        ):
            raise ValueError("Cache insertion requires guarded resident frozen FP32 features")
        if not bool(torch.isfinite(raw).all() & torch.isfinite(experts).all()):
            raise ValueError("Cache insertion rejects nonfinite producer features")
        entry_bytes = count * 20 * raw.element_size()
        if entry_bytes > self.max_bytes or len(keys) > self.max_entries:
            self._stats["rejected_lots"] += 1
            return False
        if entry_bytes * len(keys) > self.max_bytes:
            self._stats["rejected_lots"] += 1
            return False
        for index, key in enumerate(keys):
            old = self._entries.pop(key.object_id, None)
            if old is not None:
                self._bytes -= old.bytes
                self._stats["replacements"] += 1
            value = CachedRGBFeatures(raw[index].detach().clone(), experts[index].detach().clone())
            self._entries[key.object_id] = _Entry(key, value, entry_bytes)
            self._bytes += entry_bytes
            self._stats["stored_entries"] += 1
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, stale = self._entries.popitem(last=False)
                self._bytes -= stale.bytes
                self._stats["evictions"] += 1
        return True

    def retain_objects(self, object_ids: Sequence[str]) -> None:
        """Invalidate departed tracks without retaining their feature storage."""
        self._owner()
        active = set(object_ids)
        if len(active) != len(object_ids) or any(not isinstance(i, str) or not i for i in active):
            raise ValueError("Active object IDs must be distinct nonempty strings")
        for identity in list(self._entries):
            if identity not in active:
                self._bytes -= self._entries.pop(identity).bytes
                self._stats["invalidations"] += 1

    def clear(self) -> None:
        """Release all owned feature tensors without clearing evidence counters."""
        self._owner()
        self._stats["invalidations"] += len(self._entries)
        self._entries.clear()
        self._bytes = 0
