"""Persistent shared-sensor TRAIN40 updates with one result per object.

Raw frames and packets are the input boundary. This module does not detect or
track objects: callers supply current event and RGB regions and stable IDs.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import h5py
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from e_jepa_ttc.rgb_port.data import RGB_SIZE, _effective_square
from operational.rgb_port import transfer as engine
from operational.rgb_port_revision.transfer_inputs import sensor_triplets
from operational.train40_runtime.batched_context import BatchedContextRuntime, ContextObservation
from operational.train40_runtime.inputs import PersistentSensorInputs, _normalized_crop
from operational.train40_runtime.producers import ProducerRuntime

if TYPE_CHECKING:
    from operational.train40_runtime.multiobject_cache import RGBBatchCache

Square = tuple[float, float, float, float]
RGBObservation = tuple[np.ndarray, dict[str, Any]]


def _normalized_pil_crop(image: Image.Image, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Apply the original crop/resize arithmetic to an already shared PIL image."""
    with image.crop(bounds) as cropped:
        with cropped.resize((RGB_SIZE, RGB_SIZE), resample=Image.Resampling.BILINEAR) as resized:
            pixels = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
            return np.ascontiguousarray(pixels.astype(np.float32) / 255)


class EventHistory(Protocol):
    """Shared causal raw-event backend, including temporal-prefix variants."""

    @property
    def ready(self) -> bool: ...

    @property
    def stats(self) -> dict[str, Any]: ...

    def initialize(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None: ...

    def ingest(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None: ...

    def query_many(self, rois: Sequence[Sequence[float]]) -> Tensor: ...


@dataclass(frozen=True)
class ObjectRegion:
    """Current regions supplied by an upstream tracker, not TTC targets."""

    object_id: str
    event_square: Square
    rgb_square: Square | None


@dataclass(frozen=True)
class RawRGBHistory:
    """Original uint8 frames available at one query, without ROI processing."""

    query_us: int
    frames: Mapping[int, np.ndarray]
    timestamps_us: Mapping[int, int]
    triplets: Sequence[tuple[int, ...]]
    unavailable_reason: str = ""


@dataclass(frozen=True)
class PreparedRGBBatch:
    """Aligned metadata with flattened pixels or previously guarded features.

    Pixel and feature rows use present-object/history order. The caller owns
    their lifetime until infer_prepared returns. Metadata is always current;
    producer features never substitute for recomputing the temporal context.
    """

    metadata: Sequence[Sequence[dict[str, Any]]]
    pixels: Tensor | None = None
    features: tuple[Tensor, Tensor] | None = None


def rgb_metadata(
    objects: Sequence[ObjectRegion], history: RawRGBHistory
) -> list[list[dict[str, Any]]]:
    """Rebuild causal clocks and availability without processing image pixels."""
    return [
        [
            {
                "first_sensor_ts_us": history.timestamps_us[group[0]],
                "anchor_us": history.timestamps_us[group[-1]],
                "available_us": history.query_us,
                "timestamps_us": [history.timestamps_us[index] for index in group],
                "frame_indices": list(group),
                "delta_t_s": (
                    np.diff([history.timestamps_us[index] for index in group]) / 1e6
                ).tolist(),
                "square_xyxy": list(item.rgb_square),
            }
            for group in history.triplets
        ]
        if item.rgb_square is not None
        else []
        for item in objects
    ]


def read_raw_rgb_history(inputs: PersistentSensorInputs, row: Mapping[str, Any]) -> RawRGBHistory:
    """Read/cache only causal raw frames; all region work remains in update()."""
    reader = inputs._source(row)
    handle = reader._require_handle()
    query = int(row["anchor_us"])
    data, timestamps = handle.get("blackflys/left/data"), handle.get("blackflys/left/ts")
    if not isinstance(data, h5py.Dataset) or not isinstance(timestamps, h5py.Dataset):
        return RawRGBHistory(query, {}, {}, (), "RGB_DATASET_MISSING")
    if inputs._clock is None:
        inputs._clock = np.asarray(timestamps, dtype=np.int64)
    clock = inputs._clock
    groups = sensor_triplets(clock, query)
    if not groups:
        return RawRGBHistory(query, {}, {}, (), "INSUFFICIENT_CAUSAL_RGB_HISTORY")
    ids = [int(index) for index in np.unique(np.concatenate(groups))]
    return RawRGBHistory(
        query,
        {index: inputs._frame(data, index) for index in ids},
        {index: int(clock[index]) for index in ids},
        [tuple(int(index) for index in group) for group in groups],
    )


def _validate_regions(objects: Sequence[ObjectRegion], max_objects: int) -> None:
    if not 1 <= len(objects) <= max_objects:
        raise ValueError("Object count is outside the configured one-to-twenty bound")
    ids = [item.object_id for item in objects]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Object IDs must be unique nonempty strings")
    for item in objects:
        for square in (item.event_square, item.rgb_square):
            if square is None:
                continue
            if len(square) != 4 or not np.isfinite(square).all():
                raise ValueError("ROI must have four finite coordinates")
            if square[2] <= square[0] or square[3] <= square[1]:
                raise ValueError("ROI must have positive width and height")


def _validate_rgb(history: RawRGBHistory, query_us: int) -> None:
    if history.query_us != query_us:
        raise ValueError("RGB raw packet belongs to a different update")
    if len(history.triplets) > 8:
        raise ValueError("Native TRAIN40 RGB history cannot exceed eight observations")
    ids = {index for group in history.triplets for index in group}
    if len({len(group) for group in history.triplets}) > 1:
        raise ValueError("A native RGB history must have a consistent observation length")
    if len(ids) > 10:
        raise ValueError("Native RGB history exceeds ten frames")
    for group in history.triplets:
        if len(group) not in (2, 3) or any(
            index not in history.frames or index not in history.timestamps_us for index in group
        ):
            raise ValueError(
                "RGB observations require two or three aligned raw frames and timestamps"
            )
        times = [history.timestamps_us[index] for index in group]
        if times[-1] > query_us or any(a >= b for a, b in zip(times[:-1], times[1:], strict=True)):
            raise ValueError("RGB triplets must be strictly chronological and causal")
        if query_us - times[0] > 650_000:
            raise ValueError("RGB triplet begins outside the frozen history")
    for index in ids:
        frame = history.frames[index]
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("RGB frames must be uint8 HWC arrays with three channels")


class MultiObjectTrain40Service:
    """Update up to twenty objects from one new packet and shared raw RGB.

    Models, graph capture and initial history construction are startup work.
    ``update`` includes current ROI processing, event ingestion, transfers, all
    producers and context heads, and the host availability of every result.
    This single-owner service does not cache embeddings across changing ROIs.
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        normalizers: Mapping[str, tuple[Tensor, Tensor]],
        history: EventHistory,
        device: str | torch.device = "cuda",
        *,
        graphs: bool = True,
        parallel_branches: bool = True,
        rgb_workers: int = 8,
        max_objects: int = 20,
        retain_input_diagnostics: bool = False,
        reuse_event_encodings: bool | None = None,
        shared_rgb_images: bool = False,
        rgb_object_chunk_size: int | None = None,
        parallel_preprocessing: bool = True,
        rgb_feature_cache: RGBBatchCache | None = None,
    ) -> None:
        if not isinstance(max_objects, int) or not 1 <= max_objects <= 20:
            raise ValueError("max_objects must be between one and twenty")
        if not 1 <= rgb_workers <= 8:
            raise ValueError("rgb_workers must be between one and eight")
        if rgb_object_chunk_size is not None and (
            isinstance(rgb_object_chunk_size, bool)
            or not isinstance(rgb_object_chunk_size, int)
            or not 1 <= rgb_object_chunk_size <= 20
        ):
            raise ValueError("RGB object chunk size must be one to twenty or None")
        if rgb_feature_cache is not None and (
            rgb_feature_cache.rgb_object_chunk_size != rgb_object_chunk_size
        ):
            raise ValueError("RGB feature cache chunk shape must match the producer configuration")
        self.device = torch.device(device)
        if parallel_branches and (not graphs or self.device.type != "cuda"):
            raise ValueError("Parallel branches require CUDA block graphs")
        self.history = history
        self.max_objects = max_objects
        self.shared_rgb_images = shared_rgb_images
        self.rgb_object_chunk_size = rgb_object_chunk_size
        self.parallel_preprocessing = parallel_preprocessing
        self.rgb_feature_cache = rgb_feature_cache
        self.last_rgb_profile: dict[str, float | int] = {}
        self.producers = ProducerRuntime(
            dict(models),
            self.device,
            graphs=graphs,
            block_graphs=graphs,
            reuse_encodings=graphs,
            reuse_event_encodings=reuse_event_encodings,
        )
        self.context = BatchedContextRuntime(
            models, normalizers, self.device, graphs=graphs, max_objects=max_objects
        )
        self.parallel_branches = parallel_branches
        self._streams = (
            (torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device))
            if parallel_branches
            else None
        )
        self._warm_signatures: set[tuple[tuple[int, ...], tuple[int, ...], int]] = set()
        self._crops = ThreadPoolExecutor(
            max_workers=rgb_workers, thread_name_prefix="train40-multi-crop"
        )
        self._preprocess = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train40-multi-rgb")
        self._closed = False
        self.retain_input_diagnostics = retain_input_diagnostics
        self.last_rgb_inputs: list[list[RGBObservation]] | None = None
        self.last_event_inputs: Tensor | None = None
        self.last_rgb_producer_features: tuple[Tensor, Tensor] | None = None
        self.last_rgb_cache_receipt: dict[str, Any] | None = None

    def close(self) -> None:
        """Wait for owned host workers; do not close the caller's sensor reader."""
        self._preprocess.shutdown(wait=True, cancel_futures=True)
        self._crops.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def _rgb(
        self, objects: Sequence[ObjectRegion], history: RawRGBHistory
    ) -> list[list[RGBObservation]]:
        rgb_started_ns = time.perf_counter_ns()
        begin = time.perf_counter()
        tasks = {}
        ids = sorted({index for group in history.triplets for index in group})
        images = {}
        try:
            if self.shared_rgb_images and any(item.rgb_square is not None for item in objects):
                for index in ids:
                    images[index] = Image.fromarray(history.frames[index], mode="RGB")
                    images[index].load()
            converted = time.perf_counter()
            for item in objects:
                if item.rgb_square is not None:
                    for index in ids:
                        if self.shared_rgb_images:
                            tasks[(item.object_id, index)] = self._crops.submit(
                                _normalized_pil_crop,
                                images[index],
                                _effective_square(history.frames[index], item.rgb_square),
                            )
                        else:
                            tasks[(item.object_id, index)] = self._crops.submit(
                                _normalized_crop, history.frames[index], item.rgb_square
                            )
            frames = {key: task.result() for key, task in tasks.items()}
        except BaseException:
            for task in tasks.values():
                task.cancel()
            wait(list(tasks.values()))
            raise
        finally:
            for image in images.values():
                image.close()
        cropped = time.perf_counter()
        result = []
        for item in objects:
            current: list[RGBObservation] = []
            if item.rgb_square is not None:
                for group in history.triplets:
                    times = [history.timestamps_us[index] for index in group]
                    current.append(
                        (
                            np.ascontiguousarray(
                                np.stack([frames[(item.object_id, index)] for index in group])
                            ),
                            {
                                "first_sensor_ts_us": times[0],
                                "anchor_us": times[-1],
                                "available_us": history.query_us,
                                "timestamps_us": times,
                                "frame_indices": list(group),
                                "delta_t_s": (np.diff(times) / 1e6).tolist(),
                                "square_xyxy": list(item.rgb_square),
                            },
                        )
                    )
            result.append(current)
        self.last_rgb_profile = {
            "pil_image_creation_seconds": converted - begin,
            "crop_seconds": cropped - converted,
            "stack_seconds": time.perf_counter() - cropped,
            "total_seconds": time.perf_counter() - begin,
            "pil_images_created": len(images),
            "roi_frame_tasks": len(tasks),
            "started_ns": rgb_started_ns,
            "completed_ns": time.perf_counter_ns(),
        }
        return result

    @torch.inference_mode()
    def infer_prepared(
        self,
        objects: Sequence[ObjectRegion],
        query_us: int,
        event: Tensor,
        rgb: Sequence[Sequence[RGBObservation]] | PreparedRGBBatch,
        unavailable_reason: str = "",
    ) -> list[dict[str, Any]]:
        """Run grouped producers and object-isolated contexts; no sensor timing claim."""
        _validate_regions(objects, self.max_objects)
        if tuple(event.shape) != (len(objects), 4, 3, 12, 128, 128):
            raise ValueError("Events must have shape [objects,4,3,12,128,128]")
        metadata = (
            rgb.metadata
            if isinstance(rgb, PreparedRGBBatch)
            else [[meta for _, meta in rows] for rows in rgb]
        )
        if len(metadata) != len(objects):
            raise ValueError("RGB histories must align with object order")
        sensor = event.reshape(-1, 3, 12, 128, 128).to(self.device)
        delta = torch.full((len(sensor), 2), 0.1, dtype=torch.float32, device=self.device)
        present = [index for index, observations in enumerate(metadata) if observations]
        counts = {len(metadata[index]) for index in present}
        if len(counts) > 1:
            raise ValueError(
                "Shared RGB packet must have the same triplet count per available object"
            )
        rraw = rexperts = None
        self.last_rgb_producer_features = None
        cached = rgb.features if isinstance(rgb, PreparedRGBBatch) else None
        if isinstance(rgb, PreparedRGBBatch) and rgb.pixels is not None and cached is not None:
            raise ValueError("RGB batch must supply pixels or features, not both")
        if cached is not None:
            rraw, rexperts = cached
            rows = sum(len(observations) for observations in metadata)
            if (
                not present
                or rraw.shape != (rows, 17)
                or rexperts.shape != (rows, 3)
                or rraw.device != sensor.device
                or rexperts.device != sensor.device
                or rraw.dtype != torch.float32
                or rexperts.dtype != torch.float32
            ):
                raise ValueError("Cached RGB features must match resident FP32 metadata rows")
            eraw, eexperts = self.producers.expert_block("E", sensor, delta, history_group_size=4)
        elif present:
            group_size = next(iter(counts))
            if isinstance(rgb, PreparedRGBBatch):
                if rgb.pixels is None:
                    raise ValueError("Available RGB requires pixels or guarded producer features")
                rsensor = rgb.pixels.to(self.device)
                steps = len(metadata[present[0]][0]["delta_t_s"]) + 1
                if tuple(rsensor.shape) != (len(present) * group_size, steps, 3, 128, 128):
                    raise ValueError("Prepared RGB pixels do not match aligned metadata")
            else:
                rsensor = torch.stack(
                    [torch.from_numpy(pixels) for index in present for pixels, _ in rgb[index]]
                ).to(self.device)
            rdelta = torch.tensor(
                [meta["delta_t_s"] for index in present for meta in metadata[index]],
                dtype=torch.float32,
                device=self.device,
            )
            chunk_objects = self.rgb_object_chunk_size or len(present)
            chunk_rows = chunk_objects * group_size
            chunks = [
                (rsensor[start : start + chunk_rows], rdelta[start : start + chunk_rows])
                for start in range(0, len(rsensor), chunk_rows)
            ]
            if self.parallel_branches:
                for chunk, clocks in chunks:
                    signature = (tuple(sensor.shape), tuple(chunk.shape), group_size)
                    if signature not in self._warm_signatures:
                        self.producers.expert_block("E", sensor, delta, history_group_size=4)
                        self.producers.expert_block(
                            "R", chunk, clocks, history_group_size=group_size
                        )
                        self._warm_signatures.add(signature)
                assert self._streams is not None
                current = torch.cuda.current_stream(self.device)
                event_stream, rgb_stream = self._streams
                event_stream.wait_stream(current)
                rgb_stream.wait_stream(current)
                with torch.cuda.stream(event_stream):
                    eticket = self.producers.expert_block_async(
                        "E", sensor, delta, history_group_size=4
                    )
                with torch.cuda.stream(rgb_stream):
                    rtickets = [
                        self.producers.expert_block_async(
                            "R", chunk, clocks, history_group_size=group_size
                        )
                        for chunk, clocks in chunks
                    ]
                current.wait_stream(event_stream)
                current.wait_stream(rgb_stream)
                self.producers.validate_async_blocks(eticket, *rtickets)
                eraw, eexperts = eticket.raw, eticket.experts
                rraw = torch.cat([ticket.raw for ticket in rtickets])
                rexperts = torch.cat([ticket.experts for ticket in rtickets])
            else:
                eraw, eexperts = self.producers.expert_block(
                    "E", sensor, delta, history_group_size=4
                )
                blocks = [
                    self.producers.expert_block("R", chunk, clocks, history_group_size=group_size)
                    for chunk, clocks in chunks
                ]
                rraw, rexperts = (torch.cat([block[index] for block in blocks]) for index in (0, 1))
        else:
            eraw, eexperts = self.producers.expert_block("E", sensor, delta, history_group_size=4)
        if rraw is not None and rexperts is not None:
            self.last_rgb_producer_features = rraw, rexperts
        rposition = {index: position for position, index in enumerate(present)}
        contexts = []
        for index, item in enumerate(objects):
            rr = re = None
            if index in rposition:
                assert rraw is not None and rexperts is not None
                start = rposition[index] * len(metadata[index])
                rr, re = (
                    rraw[start : start + len(metadata[index])],
                    rexperts[start : start + len(metadata[index])],
                )
            contexts.append(
                ContextObservation(
                    item.object_id,
                    eraw[index * 4 : (index + 1) * 4],
                    eexperts[index * 4 : (index + 1) * 4],
                    [query_us - int(lag) for lag in engine.EVENT_LAGS_US],
                    rr,
                    re,
                    list(metadata[index]),
                    query_us,
                )
            )
        outputs = self.context.predict_many(contexts)
        for index, output in enumerate(outputs):
            region = objects[index]
            output["rgb_unavailable_reason"] = (
                "" if metadata[index] else unavailable_reason or "RGB_ROI_UNAVAILABLE"
            )
            output["input_receipt"] = {
                "query_us": query_us,
                "event_square": list(region.event_square),
                "rgb_square": list(region.rgb_square) if region.rgb_square is not None else None,
                "event_windows_us": [
                    [
                        [query_us - int(lag) + offset, query_us - int(lag) + offset + 100_000]
                        for offset in (-300_000, -200_000, -100_000)
                    ]
                    for lag in engine.EVENT_LAGS_US
                ],
                "rgb_observations": [dict(meta) for meta in metadata[index]],
            }
        return outputs

    def update(
        self,
        raw_packet: Mapping[str, np.ndarray],
        start_us: int,
        end_us: int,
        objects: Sequence[ObjectRegion],
        rgb_history: RawRGBHistory,
    ) -> dict[str, Any]:
        """Consume a new raw packet and return every object's completed host TTC."""
        started_ns = time.perf_counter_ns()
        if self._closed:
            raise RuntimeError("Service is closed")
        self.last_rgb_inputs = None
        self.last_event_inputs = None
        self.last_rgb_cache_receipt = None
        self.last_rgb_producer_features = None
        _validate_regions(objects, self.max_objects)
        _validate_rgb(rgb_history, end_us)
        ticket = (
            self.rgb_feature_cache.begin(objects, rgb_history) if self.rgb_feature_cache else None
        )
        hit = ticket is not None and ticket.cached_features is not None
        if ticket is not None:
            self.last_rgb_cache_receipt = {
                **ticket.diagnostics,
                "present_indices": list(ticket.present_indices),
            }
        pending = (
            self._preprocess.submit(self._rgb, objects, rgb_history)
            if self.parallel_preprocessing and not hit
            else None
        )
        try:
            rgb: Sequence[Sequence[RGBObservation]] | PreparedRGBBatch | None
            if hit:
                assert ticket is not None and ticket.cached_features is not None
                features = (
                    (
                        torch.cat([value.raw for value in ticket.cached_features]),
                        torch.cat([value.experts for value in ticket.cached_features]),
                    )
                    if ticket.cached_features
                    else None
                )
                rgb = PreparedRGBBatch(rgb_metadata(objects, rgb_history), features=features)
                self.last_rgb_profile = {"cache_hit_skipped_pixels": 1, "roi_frame_tasks": 0}
            else:
                rgb = self._rgb(objects, rgb_history) if pending is None else None
            event_dispatch_start_ns = time.perf_counter_ns()
            self.history.ingest(raw_packet, start_us, end_us)
            event = self.history.query_many([item.event_square for item in objects])
            if self.retain_input_diagnostics:
                self.last_event_inputs = event
            event_enqueued_ns = time.perf_counter_ns()
            if pending is not None:
                rgb = pending.result()
            assert rgb is not None
            if self.retain_input_diagnostics and not isinstance(rgb, PreparedRGBBatch):
                self.last_rgb_inputs = [list(rows) for rows in rgb]
            prepared_ns = time.perf_counter_ns()
            outputs = self.infer_prepared(
                objects, end_us, event, rgb, rgb_history.unavailable_reason
            )
            if ticket is not None and not hit and ticket.present_indices:
                assert self.rgb_feature_cache is not None
                produced = cast(tuple[Tensor, Tensor] | None, self.last_rgb_producer_features)
                assert produced is not None
                stored = self.rgb_feature_cache.store(ticket, *produced, guard_passed=True)
                assert self.last_rgb_cache_receipt is not None
                self.last_rgb_cache_receipt.update(
                    stored=stored, cache_stats=self.rgb_feature_cache.stats
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except BaseException:
            if pending is not None:
                pending.cancel()
                wait([pending])
            raise
        completed_ns = time.perf_counter_ns()
        return {
            "objects": outputs,
            "object_count": len(outputs),
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "complete_wall_ms": (completed_ns - started_ns) / 1e6,
            "event_enqueue_host_ms": (event_enqueued_ns - started_ns) / 1e6,
            "event_dispatch_host_ms": (event_enqueued_ns - event_dispatch_start_ns) / 1e6,
            "preprocessing_ready_host_ms": (prepared_ns - started_ns) / 1e6,
            "rgb_preprocess": dict(self.last_rgb_profile),
            "rgb_feature_cache": self.last_rgb_cache_receipt,
            "parallel_preprocessing": self.parallel_preprocessing,
            "input_boundary": "last_required_raw_CPU_packet_and_RGB_frames_available",
            "output_boundary": "all_object_TTC_values_available_on_host",
            "targets_read": False,
            "optimizer_updates": 0,
        }
