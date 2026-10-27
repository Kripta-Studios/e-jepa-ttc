"""Bounded temporal prefixes for the unchanged TRAIN40 event representation.

The 600 ms history is queried at 25 quarter-window boundaries rather than at
120 separate 5 ms buckets. Prefixes restart every 600 ms; one previous-epoch
total reconstructs quarters that cross that restart. Counts and timestamp
moments therefore remain bounded throughout a long service, without rebasing
the entire ring. Integer moment reordering still requires explicit numerical
admission against the original floating-point encoder and final predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from operational.train40_runtime.integral import (
    BUCKET_COUNT,
    BUCKET_US,
    HISTORY_US,
    UPDATE_US,
    IntegralEventHistory,
    _validate_raw,
    projection_bounds,
)
from operational.train40_runtime.normalization import ExactTopKNormalizer, percentile95_tail_size

PREFIX_COUNT = BUCKET_COUNT + 1
QUARTER_COUNT = 24
BOUNDARY_COUNT = QUARTER_COUNT + 1


class TemporalPrefixEventHistory(IntegralEventHistory):
    """Maintain exact bounded integer prefixes with the integral-history API.

    ``initialize``, ``ingest``, ``query`` and ``query_many`` have the same clock,
    ROI and borrowed-output contracts as :class:`IntegralEventHistory`. All
    methods must execute on one owner CUDA stream. Initialization is a declared
    startup operation; each subsequent update processes only its new packet.

    Spatial prefixes are stored for 121 temporal boundaries. A boundary at an
    epoch restart contains zero; the preceding epoch total is kept separately.
    At most two epochs intersect a query's 600 ms history. This avoids unbounded
    cumulative counts or timestamp sums, including when absolute clocks are
    large. The capacity guard proves that even an epoch filled with maximum
    packets fits int32 counts and int64 moments before allocating anything.
    """

    _cumulative_count: Tensor
    _cumulative_moment: Tensor
    _previous_count: Tensor
    _previous_moment: Tensor
    _quarter_work: Tensor
    _address: Tensor
    _slot_indices: Tensor
    _left_relative: Tensor
    _cross: Tensor
    _boundary_offsets: Tensor
    _query_state: Tensor

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
        normalization: str = "sort",
        graphs: bool = False,
    ) -> None:
        dimensions = (
            width,
            height,
            roi_size,
            max_packet_events,
            max_memory_bytes,
            max_objects,
            roi_batch_size,
        )
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("dimensions, packet capacity and memory budget must be positive ints")
        max_epoch_events = max_packet_events * (HISTORY_US // UPDATE_US)
        if max_epoch_events > np.iinfo(np.int32).max:
            raise ValueError("epoch event capacity exceeds the int32 count-prefix guarantee")
        if max_epoch_events * HISTORY_US > np.iinfo(np.int64).max:
            raise ValueError("epoch timestamp moment exceeds the int64 prefix guarantee")
        self.device = torch.device(device)
        self.width, self.height, self.roi_size = width, height, roi_size
        self.max_packet_events = max_packet_events
        self.max_objects = max_objects
        self.roi_batch_size = min(roi_batch_size, max_objects)
        batch = self.roi_batch_size
        self.async_transfers = bool(async_transfers)
        if self.async_transfers and self.device.type != "cuda":
            raise ValueError("asynchronous pinned transfers require a CUDA device")
        if graphs and self.device.type != "cuda":
            raise ValueError("query CUDA graphs require a CUDA device")
        self.graphs = bool(graphs)
        self._query_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._query_graph_allocated_delta = 0
        self._query_graph_reserved_delta = 0
        self._query_graph_allocated_before: int | None = None
        self._query_graph_allocated_after: int | None = None
        self._query_graph_reserved_before: int | None = None
        self._query_graph_reserved_after: int | None = None
        self._synthetic_query_priming_calls = 0
        if normalization not in ("sort", "topk"):
            raise ValueError("normalization must be the explicit sort or topk option")
        self.normalization = normalization
        self.end_us: int | None = None
        self._oldest = 0
        self._filled = 0
        self._ingested_packets = 0
        self._queries = 0
        self._total_buckets = 0
        self._epoch_start_us = 0
        self._buffers: list[Tensor] = []
        self._specs: list[tuple[str, tuple[int, ...], torch.dtype, bool]] = []

        def reserve(
            name: str, shape: Sequence[int], dtype: torch.dtype, zero: bool = False
        ) -> None:
            self._specs.append((name, tuple(shape), dtype, zero))

        sensor = (2, height, width)
        plane = (2, height + 1, width + 1)
        ring = (PREFIX_COUNT, *plane)
        # Two gather tensors per dtype: rectangle and temporary. Direct boundary
        # addressing avoids the old reordered scratch. The extra gathered plane
        # holds the previous epoch total.
        gathered = (BOUNDARY_COUNT + 1, 2, batch * roi_size * roi_size)
        reserve("_counts", ring, torch.int32, True)
        reserve("_moments", ring, torch.int64, True)
        for name in ("_cumulative_count", "_previous_count"):
            reserve(name, plane, torch.int32, True)
        for name in ("_cumulative_moment", "_previous_moment"):
            reserve(name, plane, torch.int64, True)
        for index in range(3):
            reserve(f"_count_work_{index}", sensor, torch.int32)
            reserve(f"_moment_work_{index}", sensor, torch.int64)
        for key, dtype in (
            ("x", torch.int32),
            ("y", torch.int32),
            ("t", torch.int64),
            ("p", torch.int8),
        ):
            reserve(f"_raw_{key}", (max_packet_events,), dtype)
        for name, dtype in (
            ("_flat", torch.int64),
            ("_relative", torch.int64),
            ("_negative", torch.bool),
            ("_polarity", torch.int64),
            ("_ones", torch.int32),
        ):
            reserve(name, (max_packet_events,), dtype)
        for index in range(2):
            reserve(f"_count_gather_{index}", gathered, torch.int32)
            reserve(f"_moment_gather_{index}", gathered, torch.int64)
        quarter = (batch, QUARTER_COUNT, 2, roi_size, roi_size)
        for name, dtype in (
            ("_quarter_count", torch.int64),
            ("_quarter_moment", torch.int64),
            ("_quarter_work", torch.int64),
            ("_upper", torch.float64),
            ("_lower", torch.float64),
        ):
            reserve(name, quarter, dtype)
        reserve("_voxel", (batch, 6, 2, 5, roi_size, roi_size), torch.float64)
        flattened = (batch * 6, 10 * roi_size * roi_size)
        if normalization == "sort":
            reserve("_sort_values", flattened, torch.float64)
            reserve("_sort_indices", flattened, torch.int64)
            reserve("_occupied", flattened, torch.bool)
        reserve("_six", (batch * 6, 12, roi_size, roi_size), torch.float32)
        reserve("_output", (batch * 12, 12, roi_size, roi_size), torch.float32)
        reserve("_many_output", (max_objects, 4, 3, 12, roi_size, roi_size), torch.float32)
        reserve("_triplets", (batch * 12,), torch.int64)
        reserve("_boundary_offsets", (BOUNDARY_COUNT,), torch.int64)
        reserve("_cross", (1, QUARTER_COUNT, 1, 1, 1), torch.int64)
        reserve("_left_relative", (1, QUARTER_COUNT, 1, 1, 1), torch.int64)
        reserve("_slot_indices", (BOUNDARY_COUNT,), torch.int64)
        # Index arrays are built per query; account for their maximum retained
        # working extent, even though allocator temporaries are not retained.
        reserve("_address", (BOUNDARY_COUNT, 2, batch * roi_size * roi_size), torch.int64)
        # One host-to-device transfer carries all ROI boundaries plus the current
        # temporal phase. Graph replay must not capture a Python phase constant.
        reserve("_query_state", (2 * max_objects * (roi_size + 1) + 1,), torch.int64, True)
        self.allocated_bytes = sum(
            int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
            for _, shape, dtype, _ in self._specs
        )
        if normalization == "topk":
            self.allocated_bytes += flattened[0] * (
                flattened[1] + 16 * percentile95_tail_size(flattened[1])
            )
        if self.allocated_bytes > max_memory_bytes:
            raise ValueError(
                f"temporal prefix workspace needs {self.allocated_bytes} bytes, above budget"
            )
        for name, shape, dtype, zero in self._specs:
            value = (
                torch.zeros(shape, dtype=dtype, device=self.device)
                if zero
                else torch.empty(shape, dtype=dtype, device=self.device)
            )
            setattr(self, name, value)
            self._buffers.append(value)
        self._count_work = [getattr(self, f"_count_work_{i}") for i in range(3)]
        self._moment_work = [getattr(self, f"_moment_work_{i}") for i in range(3)]
        self._raw = {key: getattr(self, f"_raw_{key}") for key in ("x", "y", "t", "p")}
        self._count_gather = [getattr(self, f"_count_gather_{i}") for i in range(2)]
        self._moment_gather = [getattr(self, f"_moment_gather_{i}") for i in range(2)]
        self._normalizer = (
            ExactTopKNormalizer(flattened[0], flattened[1], device=self.device)
            if normalization == "topk"
            else None
        )
        if self._normalizer is not None:
            self._buffers.extend(
                (self._normalizer.occupied, self._normalizer.values, self._normalizer.indices)
            )
        self._ones.fill_(1)
        triplets = np.arange(batch)[:, None] * 6 + np.array([0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5])
        self._triplets.copy_(torch.from_numpy(triplets.flatten()))
        self._boundary_offsets.copy_(torch.arange(0, PREFIX_COUNT, 5))
        self._pinned: dict[str, Tensor] = {}
        self._copy_event: torch.cuda.Event | None = None
        self._roi_copy_event: torch.cuda.Event | None = None
        self._copy_pending = False
        self._roi_copy_pending = False
        self._staging_waits = 0
        self._query_host = torch.zeros(
            self._query_state.shape,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.async_transfers,
        )
        self._query_host_array = self._query_host.numpy()
        self._query_bounds = self._query_state[:-1].view(2, max_objects, roi_size + 1)
        self._query_first_bucket = self._query_state[-1:]
        if self.async_transfers:
            self._pinned = {
                key: torch.empty_like(value, device="cpu", pin_memory=True)
                for key, value in self._raw.items()
            }
            for key in ("xb", "yb"):
                self._pinned[key] = torch.empty(
                    (batch, roi_size + 1), dtype=torch.int64, pin_memory=True
                )
            self._copy_event = torch.cuda.Event()
            self._roi_copy_event = torch.cuda.Event()
            self._pinned["query_state"] = self._query_host

    @property
    def stats(self) -> dict[str, int | bool | str | None]:
        """Describe the actual bounded representation and retained memory."""
        result = super().stats
        result.update(
            backend="TEMPORAL_PREFIX_600MS_EPOCHS",
            prefix_boundaries=PREFIX_COUNT,
            queried_boundaries=BOUNDARY_COUNT,
            epoch_total_planes=1,
            epoch_start_us=self._epoch_start_us,
            total_buckets=self._total_buckets,
            moment_origin="600MS_EPOCH_RELATIVE_BOUNDED",
            normalization=(
                "EXACT_FLOAT64_UPPER_TAIL_TOPK"
                if self.normalization == "topk"
                else "UNCHANGED_FULL_FLOAT64_SORT"
            ),
            small_eager_temporaries=False,
            max_take_temporary_bytes=BOUNDARY_COUNT
            * 2
            * self.roi_batch_size
            * self.roi_size**2
            * 8,
            cuda_graph_captures=len(self._query_graphs),
            query_graphs=self.graphs,
            query_graph_object_counts=",".join(map(str, sorted(self._query_graphs))),
            query_graph_allocated_delta_bytes=self._query_graph_allocated_delta,
            query_graph_reserved_delta_bytes=self._query_graph_reserved_delta,
            query_graph_allocated_before_bytes=self._query_graph_allocated_before,
            query_graph_allocated_after_bytes=self._query_graph_allocated_after,
            query_graph_reserved_before_bytes=self._query_graph_reserved_before,
            query_graph_reserved_after_bytes=self._query_graph_reserved_after,
            query_graph_memory_scope="PYTORCH_PROCESS_ALLOCATOR_CHANGE_DURING_STARTUP",
            query_graph_scope="ROI_QUERY_ONLY_EXCLUDES_INGESTION_AND_CPU_ROI_PACKING",
            synthetic_query_priming_calls=self._synthetic_query_priming_calls,
            query_host_staging_bytes=self._query_host.numel() * self._query_host.element_size(),
        )
        return result

    @torch.inference_mode()
    def prime_query_graphs(self, counts: Sequence[int]) -> None:
        """Capture requested object counts using only zero startup buffers.

        Call this explicitly after construction and before ``reset``,
        ``initialize`` or ``ingest``. Graph priming is forbidden after any sensor
        history has been installed; query data never becomes excluded warmup.
        Each captured shape has its own private graph pool. Allocated and reserved
        process-allocator changes are reported separately from the constructor's
        buffer budget. Allocated bytes alone do not include the graph's reserved
        scratch blocks. These deltas are observations, not exclusive ownership
        accounting when another thread allocates during startup.
        """
        if not self.graphs or self.device.type != "cuda":
            raise ValueError("enable graphs on CUDA before explicit startup priming")
        if self.end_us is not None or self._ingested_packets or self._total_buckets:
            raise ValueError("query graphs must be primed before any sensor history or reset")
        if (
            not counts
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= self.max_objects
                for count in counts
            )
            or len(set(counts)) != len(counts)
        ):
            raise ValueError(f"startup counts must be unique integers in 1..{self.max_objects}")
        current = torch.cuda.current_stream(self.device)
        for count in counts:
            if count in self._query_graphs:
                continue
            # Bounds and phase remain zero. All prefix planes were zeroed by
            # construction and no sensor-dependent method has run.
            self._query_state.zero_()
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(current)
            before = torch.cuda.memory_allocated(self.device)
            reserved_before = torch.cuda.memory_reserved(self.device)
            if self._query_graph_allocated_before is None:
                self._query_graph_allocated_before = before
                self._query_graph_reserved_before = reserved_before
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._execute_query_many_device(count)
                    self._synthetic_query_priming_calls += 1
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                self._execute_query_many_device(count)
            self._synthetic_query_priming_calls += 1
            capture_stream.synchronize()
            current.wait_stream(capture_stream)
            self._query_graphs[count] = graph
            after = torch.cuda.memory_allocated(self.device)
            reserved_after = torch.cuda.memory_reserved(self.device)
            self._query_graph_allocated_after = after
            self._query_graph_reserved_after = reserved_after
            self._query_graph_allocated_delta += max(0, after - before)
            self._query_graph_reserved_delta += max(0, reserved_after - reserved_before)

    def query(self, roi: Sequence[float]) -> Tensor:
        """Return one borrowed ROI tensor, using its preprimed graph if enabled."""
        return self.query_many([roi])[0] if self.graphs else super().query(roi)

    def query_many(self, rois: Sequence[Sequence[float]]) -> Tensor:
        """Encode all ROIs, including current CPU packing and transfer in the call."""
        if not self.graphs:
            return super().query_many(rois)
        if not self.ready:
            raise ValueError("complete 600ms history required before querying")
        count = len(rois)
        if not 1 <= count <= self.max_objects:
            raise ValueError(f"request must contain 1..{self.max_objects} object ROIs")
        if count not in self._query_graphs:
            raise ValueError("object count was not explicitly primed at synthetic startup")
        self._prepare_query_state(rois)
        self._query_graphs[count].replay()
        self._queries += count
        return self._many_output[:count]

    def _prepare_query_state(self, rois: Sequence[Sequence[float]]) -> None:
        """Pack canonical ROI bounds and current phase before a device execution."""
        count = len(rois)
        if not 1 <= count <= self.max_objects or any(len(roi) != 4 for roi in rois):
            raise ValueError("ROI count is outside capacity or ROI lacks four coordinates")
        boxes = [tuple(map(float, roi)) for roi in rois]
        # Validate every ROI before mutating the shared staging state.
        packed = np.stack(
            [
                np.stack([projection_bounds(self.width, b[0], b[2], self.roi_size) for b in boxes]),
                np.stack(
                    [projection_bounds(self.height, b[1], b[3], self.roi_size) for b in boxes]
                ),
            ]
        )
        if (
            self._roi_copy_event is not None
            and self._roi_copy_pending
            and not self._roi_copy_event.query()
        ):
            self._roi_copy_event.synchronize()
            self._staging_waits += 1
        self._query_host_array[:-1].reshape(2, self.max_objects, self.roi_size + 1)[:, :count] = (
            packed
        )
        self._query_host_array[-1] = self._total_buckets - BUCKET_COUNT
        self._query_state.copy_(self._query_host, non_blocking=self.async_transfers)
        if self._roi_copy_event is not None:
            self._roi_copy_event.record(torch.cuda.current_stream(self.device))
            self._roi_copy_pending = True

    def _execute_query_many_device(self, count: int) -> Tensor:
        """Static-shape device core; it never reads sensor state into Python."""
        self._set_query_metadata(self._query_first_bucket)
        for first in range(0, count, self.roi_batch_size):
            last = min(count, first + self.roi_batch_size)
            result = self._query_chunk_device(
                self._query_bounds[0, first:last], self._query_bounds[1, first:last], last - first
            )
            self._many_output[first:last].copy_(result)
        return self._many_output[:count]

    def reset(self, start_us: int) -> None:
        """Invalidate history and clear only the new epoch's accumulators."""
        super().reset(start_us)
        self._total_buckets = 0
        self._epoch_start_us = start_us
        for value in (
            self._cumulative_count,
            self._cumulative_moment,
            self._previous_count,
            self._previous_moment,
            self._counts[0],
            self._moments[0],
        ):
            value.zero_()

    def ingest(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None:
        """Ingest exactly one contiguous packet and append four spatial prefixes."""
        if self.end_us is None or start_us != self.end_us or end_us - start_us != UPDATE_US:
            raise ValueError("packet clock rollback/gap: exact contiguous 20000us updates required")
        values = _validate_raw(raw, start_us, end_us, self.width, self.height)
        count = len(values["t"])
        if count > self.max_packet_events:
            raise ValueError("raw packet exceeds the declared event capacity; no events truncated")
        if self._copy_event is not None and self._copy_pending and not self._copy_event.query():
            self._copy_event.synchronize()
            self._staging_waits += 1
        for key, value in values.items():
            host = torch.from_numpy(
                np.where(value > 0, 1, -1).astype(np.int8) if key == "p" else value
            )
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
        for index in range(UPDATE_US // BUCKET_US):
            lo, hi = int(edges[index]), int(edges[index + 1])
            c0, c1, c2 = self._count_work
            m0, m1, m2 = self._moment_work
            c0.zero_()
            m0.zero_()
            c0.view(-1).scatter_add_(0, flat[lo:hi], self._ones[: hi - lo])
            torch.sub(self._raw["t"][lo:hi], self._epoch_start_us, out=self._relative[: hi - lo])
            m0.view(-1).scatter_add_(0, flat[lo:hi], self._relative[: hi - lo])
            torch.cumsum(c0, 1, dtype=torch.int32, out=c1)
            torch.cumsum(c1, 2, dtype=torch.int32, out=c2)
            torch.cumsum(m0, 1, out=m1)
            torch.cumsum(m1, 2, out=m2)
            self._cumulative_count[:, 1:, 1:].add_(c2)
            self._cumulative_moment[:, 1:, 1:].add_(m2)
            self._total_buckets += 1
            slot = self._total_buckets % PREFIX_COUNT
            if self._total_buckets % BUCKET_COUNT == 0:
                self._previous_count.copy_(self._cumulative_count)
                self._previous_moment.copy_(self._cumulative_moment)
                self._cumulative_count.zero_()
                self._cumulative_moment.zero_()
                self._epoch_start_us += HISTORY_US
            self._counts[slot].copy_(self._cumulative_count)
            self._moments[slot].copy_(self._cumulative_moment)
            self._filled = min(BUCKET_COUNT, self._filled + 1)
        self.end_us = end_us
        self._ingested_packets += 1

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
        bounds: dict[str, Tensor] = {}
        for key, array in (
            ("xb", np.stack([projection_bounds(self.width, b[0], b[2], size) for b in boxes])),
            ("yb", np.stack([projection_bounds(self.height, b[1], b[3], size) for b in boxes])),
        ):
            host = torch.from_numpy(array)
            if self.async_transfers:
                self._pinned[key][:count].copy_(host)
                host = self._pinned[key][:count]
            bounds[key] = host.to(self.device, non_blocking=self.async_transfers)
        if self._roi_copy_event is not None:
            self._roi_copy_event.record(torch.cuda.current_stream(self.device))
            self._roi_copy_pending = True
        self._set_query_metadata(self._total_buckets - BUCKET_COUNT)
        result = self._query_chunk_device(bounds["xb"], bounds["yb"], count)
        self._queries += count
        return result

    def _set_query_metadata(self, first_bucket: int | Tensor) -> None:
        """Keep phase-dependent values dynamic even inside a captured graph."""
        torch.add(self._boundary_offsets, first_bucket, out=self._slot_indices)
        self._left_relative.copy_(
            (self._slot_indices[:-1] % BUCKET_COUNT * BUCKET_US).view(1, QUARTER_COUNT, 1, 1, 1)
        )
        self._cross.copy_(
            (
                self._slot_indices[:-1].div(BUCKET_COUNT, rounding_mode="floor")
                != self._slot_indices[1:].div(BUCKET_COUNT, rounding_mode="floor")
            ).view(1, QUARTER_COUNT, 1, 1, 1)
        )
        self._slot_indices.remainder_(PREFIX_COUNT)

    def _query_chunk_device(self, xb: Tensor, yb: Tensor, count: int) -> Tensor:
        """Compute one workspace chunk with prepared resident bounds and phase."""
        size = self.roi_size
        corners = [
            (yb[:, yi : yi + size, None] * (self.width + 1) + xb[:, None, xi : xi + size]).reshape(
                -1
            )
            for yi, xi in ((1, 1), (0, 1), (1, 0), (0, 0))
        ]
        plane_size = (self.height + 1) * (self.width + 1)
        elements = count * size * size
        address = self._address[..., :elements]
        for source, previous, work in (
            (self._counts, self._previous_count, self._count_gather),
            (self._moments, self._previous_moment, self._moment_gather),
        ):
            rectangle, temporary = (value[..., :elements] for value in work)
            for corner_index, corner in enumerate(corners):
                address[:, 0].copy_(self._slot_indices[:, None]).mul_(2 * plane_size)
                address[:, 0].add_(corner)
                address[:, 1].copy_(address[:, 0]).add_(plane_size)
                target = rectangle if corner_index == 0 else temporary
                # Address is contiguous only for a full workspace batch; flatten
                # creates a bounded temporary for a partially filled last batch.
                gathered = torch.take(source, address)
                target[:BOUNDARY_COUNT].copy_(gathered)
                torch.index_select(previous.view(2, -1), 1, corner, out=target[BOUNDARY_COUNT])
                if corner_index in (1, 2):
                    rectangle.sub_(temporary)
                elif corner_index == 3:
                    rectangle.add_(temporary)
        counts, moments = (
            value[0][..., :elements]
            .view(BOUNDARY_COUNT + 1, 2, count, size, size)
            .permute(2, 0, 1, 3, 4)
            for value in (self._count_gather, self._moment_gather)
        )
        quarter_count = self._quarter_count[:count]
        quarter_moment = self._quarter_moment[:count]
        scratch = self._quarter_work[:count]
        torch.sub(counts[:, 1:BOUNDARY_COUNT], counts[:, :QUARTER_COUNT], out=quarter_count)
        torch.mul(counts[:, BOUNDARY_COUNT : BOUNDARY_COUNT + 1], self._cross, out=scratch)
        quarter_count.add_(scratch)
        torch.sub(moments[:, 1:BOUNDARY_COUNT], moments[:, :QUARTER_COUNT], out=quarter_moment)
        # ``out=int64`` does not promote int32 input arithmetic in torch.mul.
        # Widen before multiplying by 600000 or dense windows would overflow.
        scratch.copy_(counts[:, 1:BOUNDARY_COUNT]).mul_(HISTORY_US)
        scratch.add_(moments[:, BOUNDARY_COUNT : BOUNDARY_COUNT + 1]).mul_(self._cross)
        quarter_moment.add_(scratch)
        torch.mul(quarter_count, self._left_relative, out=scratch)
        quarter_moment.sub_(scratch)
        self._upper[:count].copy_(quarter_moment).div_(25_000.0)
        self._lower[:count].copy_(quarter_count).sub_(self._upper[:count])
        upper = self._upper[:count].reshape(count, 6, 4, 2, size, size).transpose(2, 3)
        lower = self._lower[:count].reshape(count, 6, 4, 2, size, size).transpose(2, 3)
        self._voxel[:count].zero_()
        self._voxel[:count, :, :, :4].copy_(lower)
        self._voxel[:count, :, :, 1:].add_(upper)
        self._normalize(count)
        torch.index_select(
            self._six[: count * 6], 0, self._triplets[: count * 12], out=self._output[: count * 12]
        )
        return self._output[: count * 12].view(count, 4, 3, 12, size, size)

    def _normalize(self, objects: int) -> None:
        if self._normalizer is None:
            super()._normalize(objects)
            return
        size = self.roi_size
        windows = objects * 6
        voxel = self._voxel[:objects].reshape(windows, -1)
        self._normalizer.normalize_(voxel)
        self._six[:windows, :10].copy_(voxel.reshape(windows, 10, size, size))
        count = (
            self._quarter_count[:objects]
            .reshape(windows, 4, 2, size, size)
            .sum((1, 2, 3, 4))
            .double()
        )
        scalar = torch.stack((torch.log1p(count), torch.log1p(count / (100_000 * 1e-6))), 1)
        self._six[:windows, 10:].copy_(scalar[:, :, None, None].expand(-1, -1, size, size))


__all__ = ["TemporalPrefixEventHistory"]
