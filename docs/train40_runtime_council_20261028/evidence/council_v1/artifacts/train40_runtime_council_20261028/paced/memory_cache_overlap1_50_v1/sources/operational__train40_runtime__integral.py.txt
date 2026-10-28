"""Bounded causal event moments for native TRAIN40 streaming inputs.

The ring stores integer counts and within-bucket timestamp sums before any ROI
is known. Query-time rectangle sums apply the current ROI to the complete
600 ms history. This changes floating-point reduction order relative to the
frozen encoder, so deployment requires explicit tensor and prediction parity.
It does not alter weights, round query clocks, or use future observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE

BUCKET_US = 5_000
UPDATE_US = 20_000
HISTORY_US = 600_000
BUCKET_COUNT = HISTORY_US // BUCKET_US


def projection_bounds(length: int, low: float, high: float, size: int) -> np.ndarray:
    """Return source-pixel boundaries of the unchanged float64 ROI projection."""
    if length < 1 or size < 1 or not np.isfinite((low, high)).all() or low >= high:
        raise ValueError("finite ordered ROI and positive dimensions required")
    axis = np.arange(length, dtype=np.float64)
    positions = np.flatnonzero((axis >= low) & (axis < high))
    if not len(positions):
        return np.zeros(size + 1, dtype=np.int64)
    projected = np.clip(
        np.floor((axis[positions] - low) * size / (high - low)).astype(np.int32), 0, size - 1
    )
    return np.searchsorted(projected, np.arange(size + 1), side="left") + positions[0]


def _validate_raw(
    raw: Mapping[str, np.ndarray], start: int, end: int, width: int, height: int
) -> dict[str, np.ndarray]:
    if (
        set(raw) != {"x", "y", "t", "p"}
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= 2**53
    ):
        raise ValueError("x/y/t/p columns and positive half-open time interval required")
    values = {key: np.asarray(value) for key, value in raw.items()}
    size = values["t"].shape
    if any(value.ndim != 1 or value.shape != size for value in values.values()):
        raise ValueError("event columns must be aligned one-dimensional arrays")
    if any(value.dtype.kind not in "iu" for value in values.values()):
        raise ValueError("event columns must use integer dtypes")
    t = values["t"]
    if len(t):
        if int(t[0]) < start or int(t[-1]) >= end or np.any(t[1:] < t[:-1]):
            raise ValueError("timestamps must be monotonic and within the half-open packet")
        if (
            int(values["x"].min()) < 0
            or int(values["x"].max()) >= width
            or int(values["y"].min()) < 0
            or int(values["y"].max()) >= height
        ):
            raise ValueError("event coordinates exceed the declared sensor canvas")
    return values


class IntegralEventHistory:
    """Maintain a fixed-phase 600 ms ring and ingest complete 20 ms CPU packets.

    ``initialize`` establishes any integer microsecond phase. Subsequent packets
    must follow its exact 20 ms cadence. An irregular clock requires an explicit
    reset/reinitialization; timestamps are never rounded. ``query`` returns a
    borrowed device tensor, overwritten by the next query. The caller must copy
    it if retaining results or dispatching concurrent work on another stream.

    Ring, prefix, packet, gather and sort storage are preallocated. Small eager
    arithmetic temporaries remain and are disclosed in ``stats``. No CUDA graph
    is captured by this class. The timing boundary starts when the CPU raw packet
    is available and includes validation, copies, ingestion and query encoding;
    acquisition, file replay and model inference are separate operations.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        width: int = EAP_IMAGE_SIZE[0],
        height: int = EAP_IMAGE_SIZE[1],
        roi_size: int = 128,
        max_packet_events: int = 2_000_000,
        max_memory_bytes: int = 4 * 1024**3,
        async_transfers: bool = False,
        max_objects: int = 20,
        roi_batch_size: int = 4,
    ) -> None:
        if (
            min(
                width,
                height,
                roi_size,
                max_packet_events,
                max_memory_bytes,
                max_objects,
                roi_batch_size,
            )
            <= 0
        ):
            raise ValueError("dimensions, packet capacity and memory budget must be positive")
        if max_packet_events >= 2**31:
            raise ValueError("packet capacity exceeds the int32 count-prefix guarantee")
        self.device = torch.device(device)
        self.width, self.height, self.roi_size = width, height, roi_size
        self.max_packet_events = max_packet_events
        self.max_objects = max_objects
        self.roi_batch_size = min(roi_batch_size, max_objects)
        batch = self.roi_batch_size
        self.async_transfers = bool(async_transfers)
        if self.async_transfers and self.device.type != "cuda":
            raise ValueError("asynchronous pinned transfers require a CUDA device")
        self.end_us: int | None = None
        self._oldest = 0
        self._filled = 0
        self._ingested_packets = 0
        self._queries = 0
        self._buffers: list[Tensor] = []
        # Exact retained-buffer accounting, checked before allocating a ring.
        # Eager operation temporaries and allocator caching remain separate.
        estimated = (
            BUCKET_COUNT * 2 * (height + 1) * (width + 1) * 12
            + width * height * 2 * 36
            + max_packet_events * 46
            + roi_size * roi_size * (14_460 * batch + max_objects * 576)
            + 1_000
            + 96 * batch
        )
        if estimated > max_memory_bytes:
            raise ValueError(f"integral workspace needs up to {estimated} bytes, above budget")

        def allocate(shape: Sequence[int], dtype: torch.dtype, *, zero: bool = False) -> Tensor:
            value = (
                torch.zeros(tuple(shape), dtype=dtype, device=self.device)
                if zero
                else (torch.empty(tuple(shape), dtype=dtype, device=self.device))
            )
            self._buffers.append(value)
            return value

        sensor = (2, height, width)
        ring = (BUCKET_COUNT, 2, height + 1, width + 1)
        gathered = (BUCKET_COUNT, 2, batch * roi_size * roi_size)
        self._counts = allocate(ring, torch.int32, zero=True)
        self._moments = allocate(ring, torch.int64, zero=True)
        self._count_work = [allocate(sensor, torch.int32) for _ in range(3)]
        self._moment_work = [allocate(sensor, torch.int64) for _ in range(3)]
        self._raw = {
            key: allocate((max_packet_events,), dtype)
            for key, dtype in (
                ("x", torch.int32),
                ("y", torch.int32),
                ("t", torch.int64),
                ("p", torch.int8),
            )
        }
        self._flat = allocate((max_packet_events,), torch.int64)
        self._relative = allocate((max_packet_events,), torch.int64)
        self._negative = allocate((max_packet_events,), torch.bool)
        self._polarity = allocate((max_packet_events,), torch.int64)
        self._ones = allocate((max_packet_events,), torch.int32)
        self._ones.fill_(1)
        self._count_gather = [allocate(gathered, torch.int32) for _ in range(3)]
        self._moment_gather = [allocate(gathered, torch.int64) for _ in range(3)]
        self._weighted_counts = allocate((batch, 24, 5, 2, roi_size, roi_size), torch.int64)
        quarter = (batch, 24, 2, roi_size, roi_size)
        self._quarter_count = allocate(quarter, torch.int64)
        self._quarter_moment = allocate(quarter, torch.int64)
        self._upper = allocate(quarter, torch.float64)
        self._lower = allocate(quarter, torch.float64)
        self._voxel = allocate((batch, 6, 2, 5, roi_size, roi_size), torch.float64)
        flattened = (batch * 6, 10 * roi_size * roi_size)
        self._sort_values = allocate(flattened, torch.float64)
        self._sort_indices = allocate(flattened, torch.int64)
        self._occupied = allocate(flattened, torch.bool)
        self._six = allocate((batch * 6, 12, roi_size, roi_size), torch.float32)
        self._output = allocate((batch * 12, 12, roi_size, roi_size), torch.float32)
        self._many_output = allocate((max_objects, 4, 3, 12, roi_size, roi_size), torch.float32)
        self._offsets = (
            torch.arange(5, device=self.device, dtype=torch.int64).reshape(1, 1, 5, 1, 1, 1)
            * BUCKET_US
        )
        triplets = torch.tensor([0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5], device=self.device)
        self._triplets = (torch.arange(batch, device=self.device)[:, None] * 6 + triplets).flatten()
        self._slots = torch.arange(BUCKET_COUNT, device=self.device)
        self._buffers.extend((self._offsets, self._triplets, self._slots))
        self._pinned: dict[str, Tensor] = {}
        self._copy_event: torch.cuda.Event | None = None
        self._roi_copy_event: torch.cuda.Event | None = None
        self._copy_pending = False
        self._roi_copy_pending = False
        self._staging_waits = 0
        if self.async_transfers:
            self._pinned = {
                key: torch.empty_like(value, device="cpu", pin_memory=True)
                for key, value in self._raw.items()
            }
            self._pinned["xb"] = torch.empty(
                (batch, roi_size + 1), dtype=torch.int64, pin_memory=True
            )
            self._pinned["yb"] = torch.empty(
                (batch, roi_size + 1), dtype=torch.int64, pin_memory=True
            )
            self._copy_event = torch.cuda.Event()
            self._roi_copy_event = torch.cuda.Event()
        self.allocated_bytes = sum(value.numel() * value.element_size() for value in self._buffers)
        if self.allocated_bytes > max_memory_bytes:
            raise RuntimeError("allocated integral storage exceeds the explicit budget")

    @property
    def ready(self) -> bool:
        """Whether a complete causal history has been ingested."""
        return self._filled == BUCKET_COUNT

    @property
    def stats(self) -> dict[str, int | bool | str | None]:
        """Report retained storage and the actual runtime/capture scope."""
        return {
            "allocated_bytes": self.allocated_bytes,
            "max_packet_events": self.max_packet_events,
            "max_objects": self.max_objects,
            "roi_batch_size": self.roi_batch_size,
            "history_us": HISTORY_US,
            "bucket_us": BUCKET_US,
            "update_us": UPDATE_US,
            "end_us": self.end_us,
            "ready": self.ready,
            "ingested_packets": self._ingested_packets,
            "queries": self._queries,
            "cuda_graph_captures": 0,
            "small_eager_temporaries": True,
            "async_transfers": self.async_transfers,
            "pinned_staging_bytes": sum(
                value.numel() * value.element_size() for value in self._pinned.values()
            ),
            "staging_waits": self._staging_waits,
            "arithmetic": "INTEGER_MOMENTS_REQUIRES_EXPLICIT_PARITY_ADMISSION",
        }

    def reset(self, start_us: int) -> None:
        """Invalidate history and establish an explicit new integer clock phase."""
        if not isinstance(start_us, int) or not 0 <= start_us <= 2**53:
            raise ValueError("reset timestamp must be a nonnegative integer microsecond")
        self.end_us = start_us
        self._oldest = 0
        self._filled = 0

    def initialize(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None:
        """Fill exactly 600 ms in bounded packets; record this as startup cost."""
        if end_us - start_us != HISTORY_US:
            raise ValueError("initialization requires exactly 600000 microseconds")
        values = _validate_raw(raw, start_us, end_us, self.width, self.height)
        edges = np.searchsorted(values["t"], np.arange(start_us, end_us + 1, UPDATE_US))
        if np.any(np.diff(edges) > self.max_packet_events):
            raise ValueError("initialization contains a packet above event capacity")
        self.reset(start_us)
        for index, left in enumerate(range(start_us, end_us, UPDATE_US)):
            packet = {key: value[edges[index] : edges[index + 1]] for key, value in values.items()}
            self.ingest(packet, left, left + UPDATE_US)

    def ingest(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None:
        """Validate and ingest one complete new 20 ms CPU raw observation packet."""
        if self.end_us is None or start_us != self.end_us or end_us - start_us != UPDATE_US:
            raise ValueError("packet clock rollback/gap: exact contiguous 20000us updates required")
        values = _validate_raw(raw, start_us, end_us, self.width, self.height)
        count = len(values["t"])
        if count > self.max_packet_events:
            raise ValueError("raw packet exceeds the declared event capacity; no events truncated")
        if self._copy_event is not None and self._copy_pending and not self._copy_event.query():
            # A pinned buffer may only be overwritten after its previous DMA.
            # Normal paced updates have already consumed the preceding TTC.
            self._copy_event.synchronize()
            self._staging_waits += 1
        for key, value in values.items():
            # Polarity only encodes a sign, so widening input integer dtypes
            # must not wrap a positive value before it is normalized.
            cpu = np.where(value > 0, 1, -1).astype(np.int8) if key == "p" else value
            host = torch.from_numpy(cpu)
            if self.async_transfers:
                self._pinned[key][:count].copy_(host)
                host = self._pinned[key][:count]
            self._raw[key][:count].copy_(host, non_blocking=self.async_transfers)
        if self._copy_event is not None:
            self._copy_event.record(torch.cuda.current_stream(self.device))
            self._copy_pending = True
        flat = self._flat[:count]
        torch.mul(self._raw["y"][:count], self.width, out=flat)
        flat.add_(self._raw["x"][:count])
        torch.le(self._raw["p"][:count], 0, out=self._negative[:count])
        torch.mul(self._negative[:count], self.width * self.height, out=self._polarity[:count])
        flat.add_(self._polarity[:count])
        edges = np.searchsorted(values["t"], np.arange(start_us, end_us + 1, BUCKET_US))
        for index, left in enumerate(range(start_us, end_us, BUCKET_US)):
            lo, hi = int(edges[index]), int(edges[index + 1])
            slot = self._oldest
            self._oldest = (self._oldest + 1) % BUCKET_COUNT
            c0, c1, c2 = self._count_work
            m0, m1, m2 = self._moment_work
            c0.zero_()
            m0.zero_()
            c0.view(-1).scatter_add_(0, flat[lo:hi], self._ones[: hi - lo])
            torch.sub(self._raw["t"][lo:hi], left, out=self._relative[: hi - lo])
            m0.view(-1).scatter_add_(0, flat[lo:hi], self._relative[: hi - lo])
            torch.cumsum(c0, 1, dtype=torch.int32, out=c1)
            torch.cumsum(c1, 2, dtype=torch.int32, out=c2)
            torch.cumsum(m0, 1, out=m1)
            torch.cumsum(m1, 2, out=m2)
            self._counts[slot, :, 1:, 1:].copy_(c2)
            self._moments[slot, :, 1:, 1:].copy_(m2)
            self._filled = min(BUCKET_COUNT, self._filled + 1)
        self.end_us = end_us
        self._ingested_packets += 1

    def query(self, roi: Sequence[float]) -> Tensor:
        """Return borrowed H4 producer inputs at the last complete packet end."""
        return self._query_chunk([roi])[0]

    def query_many(self, rois: Sequence[Sequence[float]]) -> Tensor:
        """Encode every requested ROI in order, sharing one sensor-history ring.

        Workspace batching bounds memory; no ROI is omitted or aliased to another
        ROI. The returned tensor is borrowed until the next ``query_many`` call.
        """
        count = len(rois)
        if not 1 <= count <= self.max_objects:
            raise ValueError(f"request must contain 1..{self.max_objects} object ROIs")
        for first in range(0, count, self.roi_batch_size):
            last = min(count, first + self.roi_batch_size)
            self._many_output[first:last].copy_(self._query_chunk(rois[first:last]))
        return self._many_output[:count]

    def _query_chunk(self, rois: Sequence[Sequence[float]]) -> Tensor:
        if not self.ready:
            raise ValueError("complete 600ms history required before querying")
        count = len(rois)
        if not 1 <= count <= self.roi_batch_size or any(len(roi) != 4 for roi in rois):
            raise ValueError("ROI must contain four coordinates")
        boxes = [tuple(map(float, roi)) for roi in rois]
        size = self.roi_size
        if (
            self._roi_copy_event is not None
            and self._roi_copy_pending
            and not self._roi_copy_event.query()
        ):
            self._roi_copy_event.synchronize()
            self._staging_waits += 1
        bounds = {}
        for key, values in (
            (
                "xb",
                np.stack([projection_bounds(self.width, box[0], box[2], size) for box in boxes]),
            ),
            (
                "yb",
                np.stack([projection_bounds(self.height, box[1], box[3], size) for box in boxes]),
            ),
        ):
            host = torch.from_numpy(values)
            if self.async_transfers:
                self._pinned[key][:count].copy_(host)
                host = self._pinned[key][:count]
            bounds[key] = host.to(self.device, non_blocking=self.async_transfers)
        if self._roi_copy_event is not None:
            self._roi_copy_event.record(torch.cuda.current_stream(self.device))
            self._roi_copy_pending = True
        xb, yb = bounds["xb"], bounds["yb"]
        corners = [
            (yb[:, yi : yi + size, None] * (self.width + 1) + xb[:, None, xi : xi + size]).reshape(
                -1
            )
            for yi, xi in ((1, 1), (0, 1), (1, 0), (0, 0))
        ]
        order = (self._slots + self._oldest) % BUCKET_COUNT
        for source, work in (
            (self._counts, self._count_gather),
            (self._moments, self._moment_gather),
        ):
            flat = source.reshape(BUCKET_COUNT, 2, -1)
            elements = BUCKET_COUNT * 2 * count * size * size
            rectangle, temporary, ordered = (
                value.view(-1)[:elements].view(BUCKET_COUNT, 2, count * size * size)
                for value in work
            )
            torch.index_select(flat, 2, corners[0], out=rectangle)
            torch.index_select(flat, 2, corners[1], out=temporary)
            rectangle.sub_(temporary)
            torch.index_select(flat, 2, corners[2], out=temporary)
            rectangle.sub_(temporary)
            torch.index_select(flat, 2, corners[3], out=temporary)
            rectangle.add_(temporary)
            torch.index_select(rectangle, 0, order, out=ordered)
        counts, moments = (
            value.view(-1)[:elements]
            .view(BUCKET_COUNT, 2, count, size, size)
            .permute(2, 0, 1, 3, 4)
            .reshape(count, 24, 5, 2, size, size)
            for value in (self._count_gather[2], self._moment_gather[2])
        )
        torch.sum(counts, 2, dtype=torch.int64, out=self._quarter_count[:count])
        torch.mul(counts, self._offsets, out=self._weighted_counts[:count])
        self._weighted_counts[:count].add_(moments)
        torch.sum(self._weighted_counts[:count], 2, out=self._quarter_moment[:count])
        self._upper[:count].copy_(self._quarter_moment[:count]).div_(25_000.0)
        self._lower[:count].copy_(self._quarter_count[:count]).sub_(self._upper[:count])
        upper = self._upper[:count].reshape(count, 6, 4, 2, size, size).transpose(2, 3)
        lower = self._lower[:count].reshape(count, 6, 4, 2, size, size).transpose(2, 3)
        self._voxel[:count].zero_()
        self._voxel[:count, :, :, :4].copy_(lower)
        self._voxel[:count, :, :, 1:].add_(upper)
        self._normalize(count)
        torch.index_select(
            self._six[: count * 6], 0, self._triplets[: count * 12], out=self._output[: count * 12]
        )
        self._queries += count
        return self._output[: count * 12].view(count, 4, 3, 12, size, size)

    def _normalize(self, objects: int) -> None:
        size = self.roi_size
        windows = objects * 6
        voxel = self._voxel[:objects].reshape(windows, -1)
        occupied = self._occupied[:windows]
        values = self._sort_values[:windows]
        torch.gt(voxel, 1e-6, out=occupied)
        nonzero = occupied.sum(1)
        torch.sort(voxel, dim=1, out=(values, self._sort_indices[:windows]))
        rank = (nonzero - 1).clamp_min(0).double() * 0.95
        lower, upper = rank.floor().long(), rank.ceil().long()
        offset = voxel.shape[1] - nonzero
        a = values.gather(1, (offset + lower).clamp_max(voxel.shape[1] - 1)[:, None])[:, 0]
        b = values.gather(1, (offset + upper).clamp_max(voxel.shape[1] - 1)[:, None])[:, 0]
        fraction = rank - lower
        scale = torch.where(fraction >= 0.5, b - (b - a) * (1.0 - fraction), a + (b - a) * fraction)
        voxel.div_(scale.clamp_min(1e-6)[:, None])
        voxel.masked_fill_(~occupied, 0.0)
        self._six[:windows, :10].copy_(voxel.reshape(windows, 10, size, size))
        count = (
            self._quarter_count[:objects]
            .reshape(windows, 4, 2, size, size)
            .sum((1, 2, 3, 4))
            .double()
        )
        scalar = torch.stack((torch.log1p(count), torch.log1p(count / (100_000 * 1e-6))), 1)
        self._six[:windows, 10:].copy_(scalar[:, :, None, None].expand(-1, -1, size, size))


__all__ = ["IntegralEventHistory", "projection_bounds"]
