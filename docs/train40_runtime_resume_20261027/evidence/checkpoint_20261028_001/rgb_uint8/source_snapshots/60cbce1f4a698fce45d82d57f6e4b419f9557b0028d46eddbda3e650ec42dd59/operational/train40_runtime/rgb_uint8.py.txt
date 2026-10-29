"""Exact PIL crops transferred once as uint8, then normalized by an FP32 LUT.

This is a preprocessing primitive, not a claim about complete TTC latency.
There is one reusable buffer lease. CPU packing may run on a worker; device
assembly and release belong to the construction thread and CUDA stream.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from e_jepa_ttc.rgb_port.data import RGB_SIZE, _effective_square
from operational.train40_runtime.encoding import crop_uint8_roi_first

if TYPE_CHECKING:
    from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory


def _bounded_integer(value: int, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


@dataclass(frozen=True)
class PackedRGBUInt8:
    """CPU packing receipt, valid only for its owner's next assemble call."""

    present_indices: tuple[int, ...]
    metadata: list[list[dict[str, Any]]]
    group_size: int
    steps: int
    unique_crops: int
    receipt: dict[str, Any]


@dataclass(frozen=True)
class PreparedRGBUInt8:
    """Borrowed device inputs; release after all producer consumption completes.

    ``pixels`` and ``delta`` alias bounded reusable storage. They must not be
    retained after release or changed in place. Consumers on another CUDA
    stream wait on ready_event, then the owner stream waits for those consumers
    before release. Releasing records a completion fence; the next pack waits
    before reusing either pinned host storage or device outputs.
    """

    pixels: Tensor
    delta: Tensor
    metadata: list[list[dict[str, Any]]]
    present_indices: tuple[int, ...]
    receipt: dict[str, Any]
    ready_event: torch.cuda.Event | None
    _owner: RGBUInt8Runtime = field(repr=False, compare=False)

    @property
    def sensor(self) -> Tensor:
        """Alias for the contiguous [present*history,T,3,128,128] pixels."""
        return self.pixels

    def assert_valid(self) -> None:
        """Reject a released or superseded lease before using borrowed tensors."""
        self._owner.assert_valid(self)

    def release(self) -> None:
        """End this lease after the owner stream has joined every consumer."""
        self._owner.release(self)


class RGBUInt8Runtime:
    """Bounded uint8 transport with exact NumPy division values on CPU or CUDA.

    The 256-entry LUT is computed once as float32 NumPy division by 255. Each
    unique resized crop crosses H2D once. Device cast/index_select normalizes
    unique crops, then index_select assembles the original object-major batch.
    All tensor outputs/workspaces use preallocated out buffers. PIL task and
    metadata allocations remain bounded host work and are included in timing.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        *,
        max_objects: int = 20,
        max_history: int = 8,
        max_frames: int = 10,
        crop_workers: int = 1,
        shared_images: bool = True,
    ) -> None:
        self.max_objects = _bounded_integer(max_objects, "max_objects", 20)
        self.max_history = _bounded_integer(max_history, "max_history", 8)
        self.max_frames = _bounded_integer(max_frames, "max_frames", 10)
        _bounded_integer(crop_workers, "crop_workers", 20)
        if not isinstance(shared_images, bool):
            raise ValueError("shared_images must be boolean")
        self.shared_images = shared_images
        self.crop_workers = crop_workers
        self.max_raw_history_bytes = 128 * 1024**2
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("RGB uint8 transport supports only CPU and CUDA")
        self._thread = threading.get_ident()
        self._stream: int | None = None
        self._ready_event: torch.cuda.Event | None = None
        self._release_event: torch.cuda.Event | None = None
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            self._stream = torch.cuda.current_stream(self.device).cuda_stream
            self._ready_event = torch.cuda.Event()
            self._release_event = torch.cuda.Event()
        self._lock = threading.RLock()
        self._state = "IDLE"
        self._active_packed: PackedRGBUInt8 | None = None
        self._active: PreparedRGBUInt8 | None = None
        self._release_pending = False
        self._executor = ThreadPoolExecutor(max_workers=crop_workers) if crop_workers > 1 else None
        unique_capacity = max_objects * max_frames
        rows_capacity = max_objects * max_history
        expanded_capacity = rows_capacity * 3
        crop_shape = (3, RGB_SIZE, RGB_SIZE)
        pinned = self.device.type == "cuda"
        self._host = {
            "unique_uint8": torch.empty(
                (unique_capacity, *crop_shape), dtype=torch.uint8, device="cpu", pin_memory=pinned
            ),
            "mapping": torch.empty(
                expanded_capacity, dtype=torch.int64, device="cpu", pin_memory=pinned
            ),
            "delta": torch.empty(
                rows_capacity * 2, dtype=torch.float32, device="cpu", pin_memory=pinned
            ),
        }
        self._buffers = {
            "unique_uint8": torch.empty(
                (unique_capacity, *crop_shape), dtype=torch.uint8, device=self.device
            ),
            "pixel_indices": torch.empty(
                (unique_capacity, *crop_shape), dtype=torch.int32, device=self.device
            ),
            "unique_fp32": torch.empty(
                (unique_capacity, *crop_shape), dtype=torch.float32, device=self.device
            ),
            "mapping": torch.empty(expanded_capacity, dtype=torch.int64, device=self.device),
            "delta": torch.empty(rows_capacity * 2, dtype=torch.float32, device=self.device),
            "expanded_fp32": torch.empty(
                (expanded_capacity, *crop_shape), dtype=torch.float32, device=self.device
            ),
            "lut": torch.empty(256, dtype=torch.float32, device=self.device),
        }
        self._host_arrays = {key: value.numpy() for key, value in self._host.items()}
        lut = np.arange(256, dtype=np.float32) / np.float32(255)
        self._buffers["lut"].copy_(torch.from_numpy(lut))
        self.lut_sha256 = hashlib.sha256(lut.tobytes()).hexdigest()

    @property
    def allocation_receipt(self) -> dict[str, Any]:
        """Logical fixed tensor storage; CUDA allocator overhead is not included."""
        return {
            "host_buffer_bytes": sum(t.numel() * t.element_size() for t in self._host.values()),
            "device_buffer_bytes": sum(
                t.numel() * t.element_size() for t in self._buffers.values()
            ),
            "host_pinned": self.device.type == "cuda",
            "device": str(self.device),
            "max_objects": self.max_objects,
            "max_history": self.max_history,
            "max_frames": self.max_frames,
            "max_raw_history_bytes": self.max_raw_history_bytes,
            "crop_workers": self.crop_workers,
            "device_buffers": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
                for name, tensor in self._buffers.items()
            },
            "lut_sha256": self.lut_sha256,
        }

    def _owner(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Device assembly/release requires the owning thread")
        if self.device.type == "cuda":
            if torch.cuda.current_stream(self.device).cuda_stream != self._stream:
                raise RuntimeError("Device assembly/release requires the owning CUDA stream")

    def _validate(
        self, objects: Sequence[ObjectRegion], history: RawRGBHistory
    ) -> tuple[tuple[int, ...], list[int], int]:
        if not 1 <= len(objects) <= self.max_objects:
            raise ValueError("Object count exceeds the configured RGB bound")
        ids = [item.object_id for item in objects]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(
            ids
        ):
            raise ValueError("Object IDs must be distinct nonempty strings")
        if (
            not isinstance(history.query_us, Integral)
            or isinstance(history.query_us, bool)
            or history.query_us < 0
        ):
            raise ValueError("query_us must be a nonnegative integer")
        if history.unavailable_reason and history.triplets:
            raise ValueError("Unavailable RGB history cannot contain observations")
        if len(history.triplets) > self.max_history:
            raise ValueError("RGB history exceeds the configured observation bound")
        lengths = {len(group) for group in history.triplets}
        if lengths and (len(lengths) != 1 or next(iter(lengths)) not in (2, 3)):
            raise ValueError("RGB observations require uniform T2 or T3 histories")
        frame_set = {index for group in history.triplets for index in group}
        if any(
            not isinstance(index, Integral) or isinstance(index, bool) or index < 0
            for index in frame_set
        ):
            raise ValueError("Frame indices must be nonnegative integers")
        frame_ids = sorted(frame_set)
        if len(frame_ids) > self.max_frames:
            raise ValueError("RGB history exceeds the configured frame bound")
        raw_bytes = 0
        for index in frame_ids:
            if index not in history.frames or index not in history.timestamps_us:
                raise ValueError("RGB history lacks a referenced frame or timestamp")
            stamp = history.timestamps_us[index]
            if not isinstance(stamp, Integral) or isinstance(stamp, bool) or stamp < 0:
                raise ValueError("Frame timestamps must be nonnegative integers")
            frame = history.frames[index]
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[-1] != 3
                or min(frame.shape) < 1
                or frame.flags.writeable
            ):
                raise ValueError("Raw RGB frames must be nonempty read-only uint8 HWC arrays")
            raw_bytes += frame.nbytes
            if raw_bytes > self.max_raw_history_bytes:
                raise ValueError("RAW RGB history exceeds the 128 MiB host-source bound")
        anchors = []
        for group in history.triplets:
            clocks = [history.timestamps_us[index] for index in group]
            if any(a >= b for a, b in zip(clocks[:-1], clocks[1:], strict=True)):
                raise ValueError("RGB frame clocks must increase strictly")
            if clocks[-1] > history.query_us or history.query_us - clocks[0] > 650_000:
                raise ValueError("RGB history must be inside the available causal support")
            anchors.append(clocks[-1])
        if any(a >= b for a, b in zip(anchors[:-1], anchors[1:], strict=True)):
            raise ValueError("RGB observation anchors must increase strictly")
        present = []
        for index, item in enumerate(objects):
            if item.rgb_square is None:
                continue
            square = item.rgb_square
            if len(square) != 4 or not np.isfinite(square).all():
                raise ValueError("RGB ROI requires four finite coordinates")
            if square[0] >= square[2] or square[1] >= square[3]:
                raise ValueError("RGB ROI must have positive dimensions")
            for frame_id in frame_ids:
                _effective_square(history.frames[frame_id], square)
            if history.triplets:
                present.append(index)
        return tuple(present), frame_ids, next(iter(lengths)) if lengths else 0

    def _crop_into(
        self,
        destination: int,
        frame: np.ndarray,
        square: Sequence[float],
        image: Image.Image | None,
    ) -> None:
        if image is None:
            pixels = crop_uint8_roi_first(frame, square)
        else:
            with image.crop(_effective_square(frame, square)) as crop:
                with crop.resize((RGB_SIZE, RGB_SIZE), Image.Resampling.BILINEAR) as resized:
                    pixels = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
                    np.copyto(self._host_arrays["unique_uint8"][destination], pixels)
                    return
        np.copyto(self._host_arrays["unique_uint8"][destination], pixels)

    def pack(self, objects: Sequence[ObjectRegion], history: RawRGBHistory) -> PackedRGBUInt8:
        """Run bounded PIL work on CPU, optionally from a preprocessing worker.

        No CUDA operations are enqueued here. A preceding release fence may be
        synchronized before pinned host storage is overwritten. Source frames
        must stay immutable until pack returns.
        """
        started = time.perf_counter()
        with self._lock:
            if self._state != "IDLE":
                raise RuntimeError("Release the current RGB lease before packing another update")
            self._state = "PACKING"
        images: dict[int, Image.Image] = {}
        tasks = []
        try:
            if self._release_pending:
                assert self._release_event is not None
                self._release_event.synchronize()
                self._release_pending = False
            after_wait = time.perf_counter()
            present, frame_ids, steps = self._validate(objects, history)
            group_size = len(history.triplets)
            if present and self.shared_images:
                for index in frame_ids:
                    image = Image.fromarray(history.frames[index], mode="RGB")
                    image.load()
                    images[index] = image
            after_images = time.perf_counter()
            for position, index in enumerate(present):
                square = objects[index].rgb_square
                assert square is not None
                for frame_position, frame_id in enumerate(frame_ids):
                    arguments = (
                        position * len(frame_ids) + frame_position,
                        history.frames[frame_id],
                        square,
                        images.get(frame_id),
                    )
                    if self._executor is None:
                        self._crop_into(*arguments)
                    else:
                        tasks.append(self._executor.submit(self._crop_into, *arguments))
            for task in tasks:
                task.result()
            after_crops = time.perf_counter()
            metadata: list[list[dict[str, Any]]] = [[] for _ in objects]
            frame_position = {index: position for position, index in enumerate(frame_ids)}
            mapping = self._host_arrays["mapping"]
            delta = self._host_arrays["delta"]
            for position, index in enumerate(present):
                for group_index, group in enumerate(history.triplets):
                    row = position * group_size + group_index
                    clocks = [int(history.timestamps_us[frame_id]) for frame_id in group]
                    differences = (np.diff(clocks) / 1e6).tolist()
                    for step, frame_id in enumerate(group):
                        mapping[row * steps + step] = (
                            position * len(frame_ids) + frame_position[frame_id]
                        )
                    delta[row * (steps - 1) : (row + 1) * (steps - 1)] = differences
                    metadata[index].append(
                        {
                            "first_sensor_ts_us": clocks[0],
                            "anchor_us": clocks[-1],
                            "available_us": history.query_us,
                            "timestamps_us": clocks,
                            "frame_indices": list(group),
                            "delta_t_s": differences,
                            "square_xyxy": list(objects[index].rgb_square or ()),
                        }
                    )
            unique = len(present) * len(frame_ids)
            rows = len(present) * group_size
            crop_bytes = unique * 3 * RGB_SIZE**2
            mapping_bytes = rows * steps * 8
            delta_bytes = rows * max(steps - 1, 0) * 4
            packed = PackedRGBUInt8(
                present,
                metadata,
                group_size,
                steps,
                unique,
                {
                    "unique_crop_count": unique,
                    "expanded_crop_count": rows * steps,
                    "unique_crop_bytes": crop_bytes,
                    "mapping_bytes": mapping_bytes,
                    "delta_bytes": delta_bytes,
                    "candidate_h2d_bytes": crop_bytes + mapping_bytes + delta_bytes,
                    "h2d_bytes": crop_bytes + mapping_bytes + delta_bytes
                    if self.device.type == "cuda"
                    else 0,
                    "baseline_expanded_fp32_bytes": rows * steps * 3 * RGB_SIZE**2 * 4,
                    "wait_previous_host_seconds": after_wait - started,
                    "validation_and_pil_host_seconds": after_images - after_wait,
                    "crop_host_seconds": after_crops - after_images,
                    "mapping_host_seconds": time.perf_counter() - after_crops,
                    "pack_host_seconds": time.perf_counter() - started,
                    "shared_pil_images": len(images),
                    "raw_history_bytes": sum(history.frames[index].nbytes for index in frame_ids),
                    "lut_sha256": self.lut_sha256,
                },
            )
            with self._lock:
                self._active_packed = packed
                self._state = "PACKED"
            return packed
        except BaseException:
            for task in tasks:
                task.cancel()
            wait(tasks)
            with self._lock:
                self._state = "IDLE"
            raise
        finally:
            for image in images.values():
                image.close()

    @torch.inference_mode()
    def assemble(self, packed: PackedRGBUInt8) -> PreparedRGBUInt8:
        """Copy unique bytes, normalize by exact LUT, then expand original order."""
        self._owner()
        started = time.perf_counter()
        with self._lock:
            if self._state != "PACKED" or packed is not self._active_packed:
                raise ValueError("Only this runtime's latest packed receipt can be assembled")
            self._state = "ASSEMBLING"
        rows = len(packed.present_indices) * packed.group_size
        expanded = rows * packed.steps
        deltas = rows * max(packed.steps - 1, 0)
        try:
            if packed.unique_crops:
                unique = self._buffers["unique_uint8"][: packed.unique_crops]
                unique.copy_(self._host["unique_uint8"][: packed.unique_crops], non_blocking=True)
                indices = self._buffers["pixel_indices"][: packed.unique_crops]
                indices.copy_(unique)
                normalized = self._buffers["unique_fp32"][: packed.unique_crops]
                torch.index_select(
                    self._buffers["lut"], 0, indices.reshape(-1), out=normalized.reshape(-1)
                )
                mapping = self._buffers["mapping"][:expanded]
                mapping.copy_(self._host["mapping"][:expanded], non_blocking=True)
                torch.index_select(
                    normalized, 0, mapping, out=self._buffers["expanded_fp32"][:expanded]
                )
                self._buffers["delta"][:deltas].copy_(
                    self._host["delta"][:deltas], non_blocking=True
                )
            pixels = self._buffers["expanded_fp32"][:expanded].view(
                rows, packed.steps, 3, RGB_SIZE, RGB_SIZE
            )
            delta = self._buffers["delta"][:deltas].view(rows, max(packed.steps - 1, 0))
            if self._ready_event is not None:
                self._ready_event.record(torch.cuda.current_stream(self.device))
            prepared = PreparedRGBUInt8(
                pixels,
                delta,
                packed.metadata,
                packed.present_indices,
                {**packed.receipt, "assemble_enqueue_host_seconds": time.perf_counter() - started},
                self._ready_event,
                self,
            )
            with self._lock:
                self._active_packed = None
                self._active = prepared
                self._state = "LEASED"
            return prepared
        except BaseException:
            # Protect pinned buffers from partially enqueued uploads on failure.
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            with self._lock:
                self._active_packed = None
                self._state = "IDLE"
            raise

    def prepare(self, objects: Sequence[ObjectRegion], history: RawRGBHistory) -> PreparedRGBUInt8:
        """Pack then assemble on the owner thread; source loading is caller work."""
        self._owner()
        return self.assemble(self.pack(objects, history))

    def assert_valid(self, prepared: PreparedRGBUInt8) -> None:
        """Check that borrowed output tensors still have their exclusive lease."""
        with self._lock:
            if self._state != "LEASED" or prepared is not self._active:
                raise ValueError("RGB output lease is released or belongs to another runtime")

    def release(self, prepared: PreparedRGBUInt8) -> None:
        """Fence all joined consumers before allowing pinned/device storage reuse."""
        self._owner()
        self.assert_valid(prepared)
        with self._lock:
            if self._release_event is not None:
                self._release_event.record(torch.cuda.current_stream(self.device))
                self._release_pending = True
            self._active = None
            self._state = "IDLE"

    def close(self) -> None:
        """Join pending work and close CPU workers; invalidates outstanding leases."""
        self._owner()
        with self._lock:
            if self._state == "CLOSED":
                return
            if self._state in ("PACKING", "ASSEMBLING"):
                raise RuntimeError("Cannot close while RGB preparation is executing")
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._active = self._active_packed = None
            self._state = "CLOSED"

    def __enter__(self) -> RGBUInt8Runtime:
        """Return this runtime for deterministic worker cleanup."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close workers and invalidate any remaining lease."""
        self.close()
