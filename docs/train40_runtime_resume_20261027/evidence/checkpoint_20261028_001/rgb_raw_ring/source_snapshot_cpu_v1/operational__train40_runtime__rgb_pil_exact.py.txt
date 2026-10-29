"""Pillow-equivalent RGB bilinear arithmetic expressed as compilable Torch ops.

The frozen adapter clips integer crop bounds before resize. Coefficients use
Pillow 12.3.0's 22-bit quantization; both axes round and clamp to uint8 before
the next stage. No custom CUDA kernel or approximate float interpolation is
used. CPU parity is necessary but does not admit a compiled GPU implementation.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from operational.train40_runtime.rgb_raw_ring import (
    FrozenRGBFrame,
    ResidentRGBFrameRing,
    RGBRingBinding,
)

if TYPE_CHECKING:
    from operational.train40_runtime.multiobject import ObjectRegion

PRECISION_BITS = 22
ROUNDING_OFFSET = 1 << (PRECISION_BITS - 1)


@dataclass(frozen=True)
class PILBilinearCoefficients:
    """Crop-relative integer sample indexes and nonnegative fixed coefficients."""

    input_size: int
    output_size: int
    indices: np.ndarray
    weights: np.ndarray
    bound_max_accumulator: int


def pil_bilinear_coefficients(input_size: int, output_size: int = 128) -> PILBilinearCoefficients:
    """Build the exact 8-bit PIL bilinear coefficient table on CPU.

    The sequential binary64 weight sum and quantization order are deliberate.
    Zero coefficient padding uses valid clamped indexes. Coefficients are
    recalculated for a current ROI inside preparation, rather than inferred
    from approximate ROI equality or prepared using future tracker metadata.
    """
    for name, value in (("input_size", input_size), ("output_size", output_size)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 16_384:
            raise ValueError(f"{name} must be an integer from 1 to 16384")
    scale = float(input_size) / output_size
    filter_scale = max(scale, 1.0)
    reciprocal = 1.0 / filter_scale
    kernel_size = int(math.ceil(filter_scale)) * 2 + 1
    if output_size * kernel_size > 4_194_304:
        raise ValueError("PIL coefficient table exceeds the bounded CPU allocation")
    indices = np.empty((output_size, kernel_size), dtype=np.int64)
    weights = np.zeros((output_size, kernel_size), dtype=np.int32)
    maximum = 0
    for output_index in range(output_size):
        center = (output_index + 0.5) * scale
        start = max(0, int(center - filter_scale + 0.5))
        end = min(input_size, int(center + filter_scale + 0.5))
        values = []
        total = 0.0
        for sample in range(start, end):
            distance = abs((sample - center + 0.5) * reciprocal)
            weight = 1.0 - distance if distance < 1.0 else 0.0
            values.append(weight)
            total += weight
        if not total > 0.0:
            raise ValueError("PIL coefficient normalization must have positive support")
        fixed = [int(0.5 + weight / total * (1 << PRECISION_BITS)) for weight in values]
        weights[output_index, : len(fixed)] = fixed
        indices[output_index] = np.minimum(
            np.arange(start, start + kernel_size, dtype=np.int64), input_size - 1
        )
        maximum = max(maximum, ROUNDING_OFFSET + sum(fixed) * 255)
    if maximum > np.iinfo(np.int32).max:
        raise ValueError("PIL bilinear accumulator does not fit the int32 kernel contract")
    indices.flags.writeable = weights.flags.writeable = False
    return PILBilinearCoefficients(input_size, output_size, indices, weights, maximum)


def effective_rgb_square(
    height: int, width: int, square: Sequence[float]
) -> tuple[int, int, int, int]:
    """Match the frozen adapter's truncation/clipping, including outside ROIs."""
    if len(square) != 4 or any(isinstance(value, bool) for value in square):
        raise ValueError("RGB square must have four finite coordinates")
    values = tuple(float(value) for value in square)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("RGB square must have four finite coordinates")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError("RGB square must have positive dimensions")
    bounds = (max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1)))
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("RGB square is empty after clipping to the source frame")
    return bounds


def pil_horizontal_uint8(
    raw: Tensor, slots: Tensor, rows: Tensor, x_indices: Tensor, coefficients: Tensor
) -> Tensor:
    """Pure Torch horizontal integer pass, suitable for fullgraph compilation.

    Raw is [capacity,H,W,3] uint8. Output is [frames,rows,output_width,3]
    uint8. Index/shape/overflow validation belongs outside this compiled body.
    """
    height, width = raw.shape[1:3]
    locations = (
        slots[:, None, None, None] * height + rows[None, :, None, None]
    ) * width + x_indices[None, None, :, :]
    pixels = raw.reshape(-1, 3)[locations]
    accumulator = (pixels.to(torch.int32) * coefficients[None, None, :, :, None]).sum(
        dim=3, dtype=torch.int32
    )
    return ((accumulator + ROUNDING_OFFSET) >> PRECISION_BITS).clamp(0, 255).to(torch.uint8)


def pil_vertical_uint8(horizontal: Tensor, y_indices: Tensor, coefficients: Tensor) -> Tensor:
    """Pure Torch vertical pass with the second required uint8 quantization."""
    pixels = horizontal[:, y_indices, :, :]
    accumulator = (pixels.to(torch.int32) * coefficients[None, :, :, None, None]).sum(
        dim=2, dtype=torch.int32
    )
    values = ((accumulator + ROUNDING_OFFSET) >> PRECISION_BITS).clamp(0, 255).to(torch.uint8)
    return values.permute(0, 3, 1, 2).contiguous()


def pil_vertical_normalized(
    horizontal: Tensor, y_indices: Tensor, coefficients: Tensor, lut: Tensor
) -> Tensor:
    """Fuse-capable vertical integer pass followed by exact FP32 LUT lookup."""
    pixels = pil_vertical_uint8(horizontal, y_indices, coefficients)
    return lut[pixels.to(torch.int64)]


def resize_uint8_exact(frame: Tensor, square: Sequence[float] | None = None) -> Tensor:
    """Direct arithmetic reference for one frame, returning uint8 CHW 128x128.

    This convenience path allocates its tables and intermediates. The service
    primitive below uses a bounded resident RAW ring and reusable workspaces.
    """
    if frame.dtype != torch.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected one uint8 HWC RGB frame")
    height, width = frame.shape[:2]
    if min(height, width) < 1:
        raise ValueError("RGB frame must be nonempty")
    x0, y0, x1, y1 = effective_rgb_square(
        height, width, square if square is not None else (0, 0, width, height)
    )
    horizontal = pil_bilinear_coefficients(x1 - x0)
    vertical = pil_bilinear_coefficients(y1 - y0)
    work = pil_horizontal_uint8(
        frame.unsqueeze(0),
        torch.zeros(1, dtype=torch.int64, device=frame.device),
        torch.arange(y0, y1, dtype=torch.int64, device=frame.device),
        torch.tensor(horizontal.indices + x0, device=frame.device),
        torch.tensor(horizontal.weights.copy(), device=frame.device),
    )
    return pil_vertical_uint8(
        work,
        torch.tensor(vertical.indices.copy(), device=frame.device),
        torch.tensor(vertical.weights.copy(), device=frame.device),
    )[0]


def _horizontal_into(
    raw: Tensor, slots: Tensor, rows: Tensor, indices: Tensor, weights: Tensor, output: Tensor
) -> Tensor:
    return output.copy_(pil_horizontal_uint8(raw, slots, rows, indices, weights))


def _vertical_into(
    horizontal: Tensor, indices: Tensor, weights: Tensor, lut: Tensor, output: Tensor
) -> Tensor:
    return output.copy_(pil_vertical_normalized(horizontal, indices, weights, lut))


@dataclass(frozen=True)
class FrozenRGBHistory:
    """Causal history referencing immutable arrival-time RGB frame packets.

    Seal each newly arrived frame once with freeze_rgb_frame, then retain those
    immutable packets in a bounded sensor-side history. Rebuilding or hashing
    every old frame at every query defeats this interface's transport purpose.
    Arrival sealing and current ROI preparation remain measured service work.
    """

    query_us: int
    frames: Mapping[int, FrozenRGBFrame]
    triplets: Sequence[tuple[int, ...]]
    unavailable_reason: str = ""


@dataclass(frozen=True)
class PreparedExactRGB:
    """Borrowed original-order RGB inputs, valid through an explicit release."""

    pixels: Tensor
    delta: Tensor
    metadata: list[list[dict[str, Any]]]
    present_indices: tuple[int, ...]
    receipt: dict[str, Any]
    ready_event: torch.cuda.Event | None
    _owner: ExactPILRGBRuntime = field(repr=False, compare=False)
    _binding: RGBRingBinding | None = field(repr=False, compare=False)

    @property
    def sensor(self) -> Tensor:
        """Alias for the FP32 [present*history,T,3,128,128] device tensor."""
        return self.pixels

    def assert_valid(self) -> None:
        """Reject stale or foreign borrowed storage before consuming it."""
        self._owner.assert_valid(self)

    def release(self) -> None:
        """Release after all consuming streams have joined the owner stream."""
        self._owner.release(self)


class ExactPILRGBRuntime:
    """Current-ROI integer resize over a shared bounded resident RAW ring.

    Coefficients and crop offsets are computed from this update's ROIs inside
    prepare. CUDA compilation is explicit and lazy: cold compilation belongs
    to startup or the measured call in which it occurs. Torch's eager fallback
    materializes intermediates; a conservative byte guard bounds that work.
    No GPU speed, memory peak or numerical admission is implied by CPU tests.
    """

    def __init__(
        self,
        ring: ResidentRGBFrameRing,
        *,
        max_objects: int = 20,
        max_history: int = 8,
        frame_chunk_size: int = 1,
        compile_kernels: bool = False,
        compile_backend: str | None = None,
        max_eager_bytes: int = 256 * 1024**2,
    ) -> None:
        for name, value, maximum in (
            ("max_objects", max_objects, 20),
            ("max_history", max_history, 8),
            ("frame_chunk_size", frame_chunk_size, min(10, ring.capacity)),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}")
        if (
            not isinstance(max_eager_bytes, int)
            or isinstance(max_eager_bytes, bool)
            or max_eager_bytes < 1
        ):
            raise ValueError("max_eager_bytes must be a positive integer")
        if not isinstance(compile_kernels, bool) or (compile_backend and not compile_kernels):
            raise ValueError("A compiler backend requires explicit compile_kernels=True")
        self.ring = ring
        self._history_frame_capacity = min(10, ring.capacity)
        self.device = ring.tensor.device
        self.max_objects, self.max_history = max_objects, max_history
        self.frame_chunk_size, self.max_eager_bytes = frame_chunk_size, max_eager_bytes
        height, width, channels = ring.frame_shape
        if channels != 3 or max(height, width) > 16_384:
            raise ValueError("Exact PIL runtime requires RGB source axes at most 16384")
        self._height, self._width = height, width
        self._kx = int(math.ceil(max(width / 128, 1.0))) * 2 + 1
        self._ky = int(math.ceil(max(height / 128, 1.0))) * 2 + 1
        self._thread = threading.get_ident()
        self._stream = (
            torch.cuda.current_stream(self.device).cuda_stream
            if self.device.type == "cuda"
            else None
        )
        self._ready_event = torch.cuda.Event() if self.device.type == "cuda" else None
        self._release_event = torch.cuda.Event() if self.device.type == "cuda" else None
        self._release_pending = False
        self._active: PreparedExactRGB | None = None
        self._closed = False
        rows = max_objects * max_history
        layouts = {
            "x_indices": ((max_objects, 128, self._kx), torch.int64),
            "x_weights": ((max_objects, 128, self._kx), torch.int32),
            "y_indices": ((max_objects, 128, self._ky), torch.int64),
            "y_weights": ((max_objects, 128, self._ky), torch.int32),
            "rows": ((max_objects, height), torch.int64),
            "slots": ((self._history_frame_capacity + frame_chunk_size,), torch.int64),
            "mapping": ((rows * 3,), torch.int64),
            "delta": ((rows * 2,), torch.float32),
        }
        self._host = {
            name: torch.empty(
                shape, dtype=dtype, device="cpu", pin_memory=self.device.type == "cuda"
            )
            for name, (shape, dtype) in layouts.items()
        }
        self._arrays = {name: tensor.numpy() for name, tensor in self._host.items()}
        self._buffers = {
            name: torch.empty(shape, dtype=dtype, device=self.device)
            for name, (shape, dtype) in layouts.items()
        }
        self._buffers.update(
            {
                "horizontal": torch.empty(
                    (frame_chunk_size, height, 128, 3), dtype=torch.uint8, device=self.device
                ),
                # A padded final frame chunk can overlap the next not-yet-written object.
                "unique": torch.empty(
                    (max_objects * self._history_frame_capacity + frame_chunk_size, 3, 128, 128),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "expanded": torch.empty(
                    (rows * 3, 3, 128, 128), dtype=torch.float32, device=self.device
                ),
                "lut": torch.empty(256, dtype=torch.float32, device=self.device),
            }
        )
        self._buffers["lut"].copy_(
            torch.from_numpy(np.arange(256, dtype=np.float32) / np.float32(255))
        )
        self._horizontal: Callable[..., Tensor] = _horizontal_into
        self._vertical: Callable[..., Tensor] = _vertical_into
        self.compile_kernels = compile_kernels
        self.compile_backend = compile_backend or "inductor"
        if compile_kernels:
            self._horizontal = torch.compile(
                _horizontal_into, backend=self.compile_backend, fullgraph=True, dynamic=False
            )
            self._vertical = torch.compile(
                _vertical_into, backend=self.compile_backend, fullgraph=True, dynamic=False
            )

    @property
    def allocation_receipt(self) -> dict[str, Any]:
        """Report persistent logical bytes separately from eager/compiler temporaries."""
        return {
            "ring": self.ring.allocation_receipt,
            "host_table_bytes": sum(t.numel() * t.element_size() for t in self._host.values()),
            "device_workspace_bytes": sum(
                t.numel() * t.element_size() for t in self._buffers.values()
            ),
            "max_eager_bytes": self.max_eager_bytes,
            "max_objects": self.max_objects,
            "max_history": self.max_history,
            "frame_chunk_size": self.frame_chunk_size,
            "compile_kernels": self.compile_kernels,
            "compile_backend": self.compile_backend,
        }

    def _owner(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Exact PIL runtime has one owning thread")
        if self.device.type == "cuda":
            if torch.cuda.current_stream(self.device).cuda_stream != self._stream:
                raise RuntimeError("Exact PIL runtime has one owning CUDA stream")

    def _validate(
        self, objects: Sequence[ObjectRegion], history: FrozenRGBHistory
    ) -> tuple[tuple[int, ...], list[int], int]:
        if not 1 <= len(objects) <= self.max_objects:
            raise ValueError("Object count exceeds exact RGB capacity")
        ids = [item.object_id for item in objects]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(
            ids
        ):
            raise ValueError("Object IDs must be distinct nonempty strings")
        if (
            not isinstance(history.query_us, int)
            or isinstance(history.query_us, bool)
            or history.query_us < 0
        ):
            raise ValueError("query_us must be a nonnegative integer")
        if history.unavailable_reason and history.triplets:
            raise ValueError("Unavailable RGB history cannot contain observations")
        if len(history.triplets) > self.max_history:
            raise ValueError("RGB history exceeds exact observation capacity")
        lengths = {len(group) for group in history.triplets}
        if lengths and (len(lengths) != 1 or next(iter(lengths)) not in (2, 3)):
            raise ValueError("RGB history requires consistent T2 or T3 observations")
        frame_ids = sorted({index for group in history.triplets for index in group})
        if len(frame_ids) > self._history_frame_capacity:
            raise ValueError("RGB history exceeds resident frame capacity")
        for index in frame_ids:
            if index not in history.frames or history.frames[index].frame_id != index:
                raise ValueError("RGB history frame mapping is incomplete or inconsistent")
        anchors = []
        for group in history.triplets:
            times = [history.frames[index].timestamp_us for index in group]
            if any(a >= b for a, b in zip(times[:-1], times[1:], strict=True)):
                raise ValueError("RGB timestamps must increase strictly")
            if times[-1] > history.query_us or history.query_us - times[0] > 650_000:
                raise ValueError("RGB history must remain inside causal support")
            anchors.append(times[-1])
        if any(a >= b for a, b in zip(anchors[:-1], anchors[1:], strict=True)):
            raise ValueError("RGB observation anchors must increase strictly")
        present = []
        for index, item in enumerate(objects):
            if item.rgb_square is not None:
                effective_rgb_square(self._height, self._width, item.rgb_square)
                if history.triplets:
                    present.append(index)
        return tuple(present), frame_ids, next(iter(lengths)) if lengths else 0

    @staticmethod
    def _bucket(value: int, maximum: int) -> int:
        return min(maximum, 1 << (value - 1).bit_length())

    @torch.inference_mode()
    def prepare(
        self, objects: Sequence[ObjectRegion], history: FrozenRGBHistory
    ) -> PreparedExactRGB:
        """Upload new RAW frames, bind current ROI tables and run exact integer resize."""
        self._owner()
        if self._closed or self._active is not None:
            raise RuntimeError("Exact RGB runtime is closed or its previous lease is unreleased")
        started = time.perf_counter()
        if self._release_pending:
            assert self._release_event is not None
            self._release_event.synchronize()
            self._release_pending = False
        present, frame_ids, steps = self._validate(objects, history)
        group_size = len(history.triplets)
        count, unique_count = len(present), len(frame_ids)
        output_rows = count * group_size
        expanded_count, delta_count = output_rows * steps, output_rows * max(steps - 1, 0)
        metadata: list[list[dict[str, Any]]] = [[] for _ in objects]
        binding = None
        table_bytes = 0
        shapes = []
        try:
            if present:
                binding = self.ring.bind(
                    [history.frames[index] for index in frame_ids], query_us=history.query_us
                )
                self._arrays["slots"][:unique_count] = binding.slots
                self._arrays["slots"][unique_count : unique_count + self.frame_chunk_size] = (
                    binding.slots[-1]
                )
                for name in ("x_indices", "x_weights", "y_indices", "y_weights", "rows"):
                    self._arrays[name][:count].fill(0)
                frame_positions = {index: position for position, index in enumerate(frame_ids)}
                for position, object_index in enumerate(present):
                    square = objects[object_index].rgb_square
                    assert square is not None
                    x0, y0, x1, y1 = effective_rgb_square(self._height, self._width, square)
                    x_table, y_table = (
                        pil_bilinear_coefficients(x1 - x0),
                        pil_bilinear_coefficients(y1 - y0),
                    )
                    kx, ky = x_table.weights.shape[1], y_table.weights.shape[1]
                    bucket_h = self._bucket(y1 - y0, self._height)
                    bucket_x, bucket_y = self._bucket(kx, self._kx), self._bucket(ky, self._ky)
                    horizontal_elements = self.frame_chunk_size * bucket_h * 128 * bucket_x * 3
                    vertical_elements = self.frame_chunk_size * 128 * bucket_y * 128 * 3
                    eager_bound = max(horizontal_elements * 12, vertical_elements * 12)
                    if eager_bound > self.max_eager_bytes:
                        raise ValueError(
                            "Integer resize exceeds the configured eager temporary bound"
                        )
                    shapes.append((bucket_h, bucket_x, bucket_y, eager_bound))
                    self._arrays["x_indices"][position, :, :kx] = x_table.indices + x0
                    self._arrays["x_weights"][position, :, :kx] = x_table.weights
                    self._arrays["y_indices"][position, :, :ky] = y_table.indices
                    self._arrays["y_weights"][position, :, :ky] = y_table.weights
                    self._arrays["rows"][position, :bucket_h] = (
                        np.minimum(np.arange(bucket_h, dtype=np.int64), y1 - y0 - 1) + y0
                    )
                    for group_index, group in enumerate(history.triplets):
                        row = position * group_size + group_index
                        times = [history.frames[index].timestamp_us for index in group]
                        differences = (np.diff(times) / 1e6).tolist()
                        self._arrays["mapping"][row * steps : (row + 1) * steps] = [
                            position * unique_count + frame_positions[index] for index in group
                        ]
                        self._arrays["delta"][row * (steps - 1) : (row + 1) * (steps - 1)] = (
                            differences
                        )
                        metadata[object_index].append(
                            {
                                "first_sensor_ts_us": times[0],
                                "anchor_us": times[-1],
                                "available_us": history.query_us,
                                "timestamps_us": times,
                                "frame_indices": list(group),
                                "delta_t_s": differences,
                                "square_xyxy": list(square),
                            }
                        )
            coefficients_ready = time.perf_counter()
            if present:
                lengths = {
                    "slots": unique_count + self.frame_chunk_size,
                    "mapping": expanded_count,
                    "delta": delta_count,
                }
                for name, host in self._host.items():
                    length = lengths.get(name, count)
                    source = host[:length]
                    self._buffers[name][:length].copy_(source, non_blocking=True)
                    table_bytes += source.numel() * source.element_size()
                for position, (bucket_h, bucket_x, bucket_y, _) in enumerate(shapes):
                    horizontal = self._buffers["horizontal"][:, :bucket_h]
                    for frame_start in range(0, unique_count, self.frame_chunk_size):
                        slots = self._buffers["slots"][
                            frame_start : frame_start + self.frame_chunk_size
                        ]
                        self._horizontal(
                            self.ring.tensor,
                            slots,
                            self._buffers["rows"][position, :bucket_h],
                            self._buffers["x_indices"][position, :, :bucket_x],
                            self._buffers["x_weights"][position, :, :bucket_x],
                            horizontal,
                        )
                        target_start = position * unique_count + frame_start
                        target = self._buffers["unique"][
                            target_start : target_start + self.frame_chunk_size
                        ]
                        self._vertical(
                            horizontal,
                            self._buffers["y_indices"][position, :, :bucket_y],
                            self._buffers["y_weights"][position, :, :bucket_y],
                            self._buffers["lut"],
                            target,
                        )
                torch.index_select(
                    self._buffers["unique"],
                    0,
                    self._buffers["mapping"][:expanded_count],
                    out=self._buffers["expanded"][:expanded_count],
                )
            pixels = self._buffers["expanded"][:expanded_count].view(
                output_rows, steps, 3, 128, 128
            )
            delta = self._buffers["delta"][:delta_count].view(output_rows, max(steps - 1, 0))
            if self._ready_event is not None:
                self._ready_event.record(torch.cuda.current_stream(self.device))
            prepared = PreparedExactRGB(
                pixels,
                delta,
                metadata,
                present,
                {
                    "ring": binding.receipt if binding is not None else {},
                    "roi_table_candidate_h2d_bytes": table_bytes,
                    "roi_table_h2d_bytes": table_bytes if self.device.type == "cuda" else 0,
                    "coefficient_and_ring_host_seconds": coefficients_ready - started,
                    "kernel_enqueue_host_seconds": time.perf_counter() - coefficients_ready,
                    "prepare_host_seconds": time.perf_counter() - started,
                    "unique_roi_frame_crops": count * unique_count,
                    "expanded_crops": expanded_count,
                    "shape_buckets": shapes,
                    "compile_kernels": self.compile_kernels,
                    "compile_backend": self.compile_backend,
                },
                self._ready_event,
                self,
                binding,
            )
            self._active = prepared
            return prepared
        except BaseException:
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            if binding is not None:
                self.ring.mark_consumed(binding)
            raise

    def assert_valid(self, prepared: PreparedExactRGB) -> None:
        """Validate the outstanding exclusive output-buffer lease."""
        if self._closed or prepared is not self._active:
            raise ValueError("Exact RGB output lease is stale or belongs to another runtime")

    def release(self, prepared: PreparedExactRGB) -> None:
        """Fence joined consumers before any host tables or RAW slots are reused."""
        self._owner()
        self.assert_valid(prepared)
        if prepared._binding is not None:
            self.ring.mark_consumed(prepared._binding)
        if self._release_event is not None:
            self._release_event.record(torch.cuda.current_stream(self.device))
            self._release_pending = True
        self._active = None

    def close(self) -> None:
        """Join local work without closing an externally owned RAW ring."""
        self._owner()
        if self._active is not None:
            self.release(self._active)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        self._closed = True

    def __enter__(self) -> ExactPILRGBRuntime:
        """Return this runtime for deterministic lease cleanup."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close only the local runtime; the caller owns the supplied RAW ring."""
        self.close()
