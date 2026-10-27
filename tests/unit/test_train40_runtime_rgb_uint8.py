"""CPU-only exact RGB transport, bounded storage and lifetime contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import torch

from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.train40_runtime.inputs import _normalized_crop
from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory
from operational.train40_runtime.rgb_uint8 import RGBUInt8Runtime


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def history(groups=5, steps=3, shape=(37, 43)):
    generator = np.random.default_rng(725)
    frames = {
        index: generator.integers(0, 256, (*shape, 3), dtype=np.uint8)
        for index in range(groups + steps - 1)
    }
    for frame in frames.values():
        frame.flags.writeable = False
    clocks = {index: 100_000 + 50_001 * index + index**2 for index in frames}
    return RawRGBHistory(
        max(clocks.values()) + 12_345,
        frames,
        clocks,
        tuple(tuple(range(index, index + steps)) for index in range(groups)),
    )


def objects(count=2):
    return [
        ObjectRegion(
            f"object_{index}",
            (0, 0, 32, 32),
            (0.25 + (index % 5) * 1.25, 0.75 + (index // 5) * 1.5, 39.9, 35.2),
        )
        for index in range(count)
    ]


def runtime_for(regions, packet, **kwargs):
    return RGBUInt8Runtime(
        "cpu",
        max_objects=max(1, len(regions)),
        max_history=max(1, len(packet.triplets)),
        max_frames=max(1, len(packet.frames)),
        **kwargs,
    )


def assert_original_exact(batch, regions, packet):
    present = tuple(
        index
        for index, region in enumerate(regions)
        if region.rgb_square is not None and packet.triplets
    )
    assert batch.present_indices == present
    assert batch.sensor is batch.pixels
    assert batch.pixels.device.type == batch.delta.device.type == "cpu"
    assert batch.pixels.dtype == batch.delta.dtype == torch.float32
    assert batch.pixels.is_contiguous() and batch.delta.is_contiguous()
    assert batch.ready_event is None
    assert len(batch.metadata) == len(regions)
    expected_deltas = []
    for index, region in enumerate(regions):
        if index not in present:
            assert batch.metadata[index] == []
            continue
        crops = {}
        for frame_id, frame in packet.frames.items():
            original = crop_uint8(frame, region.rgb_square).astype(np.float32) / 255
            np.testing.assert_array_equal(original, _normalized_crop(frame, region.rgb_square))
            crops[frame_id] = original
        for group_index, group in enumerate(packet.triplets):
            row = present.index(index) * len(packet.triplets) + group_index
            expected = np.stack([crops[frame_id] for frame_id in group])
            np.testing.assert_array_equal(batch.pixels[row].numpy(), expected)
            clocks = [packet.timestamps_us[frame_id] for frame_id in group]
            differences = (np.diff(clocks) / 1e6).tolist()
            expected_deltas.append(differences)
            assert batch.metadata[index][group_index] == {
                "first_sensor_ts_us": clocks[0],
                "anchor_us": clocks[-1],
                "available_us": packet.query_us,
                "timestamps_us": clocks,
                "frame_indices": list(group),
                "delta_t_s": differences,
                "square_xyxy": list(region.rgb_square),
            }
    if present:
        np.testing.assert_array_equal(batch.delta.numpy(), np.asarray(expected_deltas, np.float32))


@pytest.mark.parametrize("count", [1, 2, 20])
@pytest.mark.parametrize("groups", range(1, 9))
@pytest.mark.parametrize("steps", [2, 3])
def test_all_supported_history_shapes_are_bit_exact_to_both_original_adapters(count, groups, steps):
    regions, packet = objects(count), history(groups, steps)
    with runtime_for(regions, packet) as runtime:
        batch = runtime.prepare(regions, packet)
        assert batch.pixels.shape == (count * groups, steps, 3, 128, 128)
        assert batch.delta.shape == (count * groups, steps - 1)
        assert_original_exact(batch, regions, packet)
        batch.release()


@pytest.mark.parametrize("shared_images", [False, True])
@pytest.mark.parametrize("crop_workers", [1, 3])
def test_clipped_fractional_single_pixel_and_identity_rois_match_pil(shared_images, crop_workers):
    packet = history(2, 3, (137, 173))
    squares = [
        (-3.9, -2.1, 200.8, 160.2),
        (0.1, 0.9, 128.9, 128.1),
        (11.25, 12.9, 12.7, 13.2),
        (5.9, 7.1, 151.8, 134.99),
    ]
    regions = [
        replace(region, rgb_square=square)
        for region, square in zip(objects(4), squares, strict=True)
    ]
    with runtime_for(
        regions, packet, shared_images=shared_images, crop_workers=crop_workers
    ) as runtime:
        batch = runtime.prepare(regions, packet)
        assert_original_exact(batch, regions, packet)
        batch.release()


def test_normalization_preserves_every_byte_value_and_differs_from_reciprocal_multiply():
    values = np.tile(np.arange(256, dtype=np.uint8), 64).reshape(128, 128)
    frame = np.repeat(values[:, :, None], 3, axis=2)
    frame.flags.writeable = False
    packet = RawRGBHistory(200_000, {0: frame, 1: frame}, {0: 100_000, 1: 200_000}, ((0, 1),))
    regions = [ObjectRegion("identity", (0, 0, 128, 128), (0, 0, 128, 128))]
    with runtime_for(regions, packet) as runtime:
        batch = runtime.prepare(regions, packet)
        expected = values.astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(batch.pixels[0, 0, 0].numpy(), expected)
        reciprocal = values.astype(np.float32) * np.float32(1 / 255)
        assert np.count_nonzero(expected != reciprocal) > 0
        assert (
            runtime.lut_sha256 == "010413efe9fc4438fee48de66c4d09f377b28af6a9fe2522201e8c1dbb831fc8"
        )
        batch.release()


def test_missing_regions_preserve_original_object_indices_and_metadata_alignment():
    packet, regions = history(), objects(4)
    regions[0], regions[2] = (
        replace(regions[0], rgb_square=None),
        replace(regions[2], rgb_square=None),
    )
    with runtime_for(regions, packet) as runtime:
        batch = runtime.prepare(regions, packet)
        assert batch.present_indices == (1, 3)
        assert_original_exact(batch, regions, packet)
        batch.release()


@pytest.mark.parametrize("kind", ["history", "regions"])
def test_absent_rgb_emits_empty_inputs_and_no_transfer(kind):
    packet, regions = history(), objects(2)
    if kind == "history":
        packet = RawRGBHistory(packet.query_us, {}, {}, (), "INSUFFICIENT_CAUSAL_RGB_HISTORY")
    else:
        regions = [replace(region, rgb_square=None) for region in regions]
    with runtime_for(regions, packet) as runtime:
        batch = runtime.prepare(regions, packet)
        assert batch.present_indices == ()
        assert batch.pixels.numel() == batch.delta.numel() == 0
        assert batch.metadata == [[], []]
        assert batch.receipt["candidate_h2d_bytes"] == batch.receipt["h2d_bytes"] == 0
        batch.release()


def test_twenty_object_byte_receipt_and_fixed_allocation_formula():
    packet, regions = history(5, 3), objects(20)
    with runtime_for(regions, packet) as runtime:
        allocation = runtime.allocation_receipt
        unique, rows, pixels_per_crop = 20 * 7, 20 * 5, 3 * 128 * 128
        expected_host = unique * pixels_per_crop + rows * 3 * 8 + rows * 2 * 4
        expected_device = unique * pixels_per_crop * (1 + 4 + 4) + rows * 3 * pixels_per_crop * 4
        expected_device += rows * 3 * 8 + rows * 2 * 4 + 256 * 4
        assert allocation["host_buffer_bytes"] == expected_host
        assert allocation["device_buffer_bytes"] == expected_device
        assert allocation["host_pinned"] is False
        batch = runtime.prepare(regions, packet)
        assert batch.receipt["unique_crop_count"] == 140
        assert batch.receipt["expanded_crop_count"] == 300
        assert batch.receipt["unique_crop_bytes"] == 6_881_280
        assert batch.receipt["baseline_expanded_fp32_bytes"] == 58_982_400
        assert batch.receipt["candidate_h2d_bytes"] == 6_881_280 + 2_400 + 800
        assert batch.receipt["h2d_bytes"] == 0
        batch.release()


def test_no_tensor_buffers_are_reallocated_when_history_shape_changes():
    regions, packet = objects(2), history(5, 3)
    with runtime_for(regions, packet) as runtime:
        before = {
            name: (tensor.data_ptr(), tensor.numel()) for name, tensor in runtime._buffers.items()
        }
        allocation = runtime.allocation_receipt
        for next_packet in (packet, history(1, 2), history(4, 3), packet):
            batch = runtime.prepare(regions, next_packet)
            assert batch.pixels.data_ptr() == before["expanded_fp32"][0]
            assert batch.delta.data_ptr() == before["delta"][0]
            batch.release()
            assert runtime.allocation_receipt == allocation
        assert before == {
            name: (tensor.data_ptr(), tensor.numel()) for name, tensor in runtime._buffers.items()
        }


def test_source_changes_between_updates_are_used_instead_of_cached_crops():
    regions, packet = objects(1), history(1, 2)
    with runtime_for(regions, packet) as runtime:
        first = runtime.prepare(regions, packet)
        previous = first.pixels.clone()
        first.release()
        replacement = np.full_like(packet.frames[0], 255)
        replacement.flags.writeable = False
        packet = replace(packet, frames={**packet.frames, 0: replacement})
        changed = runtime.prepare(regions, packet)
        assert not torch.equal(changed.pixels, previous)
        assert_original_exact(changed, regions, packet)
        changed.release()


def test_unreleased_foreign_stale_and_closed_leases_cannot_be_reused():
    packet, regions = history(1, 2), objects(1)
    with runtime_for(regions, packet) as first, runtime_for(regions, packet) as second:
        batch = first.prepare(regions, packet)
        batch.assert_valid()
        with pytest.raises(RuntimeError, match="Release"):
            first.pack(regions, packet)
        with pytest.raises(ValueError, match="lease"):
            second.release(batch)
        batch.release()
        with pytest.raises(ValueError, match="lease"):
            batch.assert_valid()
        with pytest.raises(ValueError, match="lease"):
            batch.release()
        new = first.prepare(regions, packet)
        with pytest.raises(ValueError, match="lease"):
            batch.assert_valid()
        first.close()
        with pytest.raises(ValueError, match="lease"):
            new.assert_valid()
        with pytest.raises(RuntimeError):
            first.prepare(regions, packet)


def test_packed_receipts_are_owner_bound_and_single_use():
    packet, regions = history(1, 2), objects(1)
    with runtime_for(regions, packet) as first, runtime_for(regions, packet) as second:
        packed = first.pack(regions, packet)
        with pytest.raises(ValueError, match="packed"):
            second.assemble(packed)
        batch = first.assemble(packed)
        with pytest.raises(ValueError, match="packed"):
            first.assemble(packed)
        batch.release()
        with pytest.raises(ValueError, match="packed"):
            first.assemble(packed)


def test_worker_can_pack_but_only_owner_can_assemble_or_release():
    packet, regions = history(2, 3), objects(2)
    with runtime_for(regions, packet, crop_workers=2) as runtime, ThreadPoolExecutor(1) as worker:
        packed = worker.submit(runtime.pack, regions, packet).result()
        with pytest.raises(RuntimeError, match="owning thread"):
            worker.submit(runtime.assemble, packed).result()
        batch = runtime.assemble(packed)
        assert_original_exact(batch, regions, packet)
        with pytest.raises(RuntimeError, match="owning thread"):
            worker.submit(batch.release).result()
        batch.release()


@pytest.mark.parametrize(
    "kind",
    [
        "writable",
        "dtype",
        "shape",
        "missing",
        "future",
        "stale",
        "mixed_steps",
        "anchors",
        "availability",
    ],
)
def test_invalid_sensor_history_fails_closed_and_runtime_recovers(kind):
    packet, regions = history(2, 3), objects(1)
    bad = packet
    if kind in ("writable", "dtype", "shape"):
        frame = packet.frames[0].copy()
        if kind == "dtype":
            frame = frame.astype(np.float32)
        elif kind == "shape":
            frame = frame[:, :, :2]
        if kind != "writable":
            frame.flags.writeable = False
        bad = replace(packet, frames={**packet.frames, 0: frame})
    elif kind == "missing":
        bad = replace(
            packet, frames={index: value for index, value in packet.frames.items() if index != 0}
        )
    elif kind == "future":
        bad = replace(packet, query_us=max(packet.timestamps_us.values()) - 1)
    elif kind == "stale":
        bad = replace(packet, query_us=min(packet.timestamps_us.values()) + 650_001)
    elif kind == "mixed_steps":
        bad = replace(packet, triplets=(packet.triplets[0][:2], packet.triplets[1]))
    elif kind == "anchors":
        bad = replace(packet, triplets=tuple(reversed(packet.triplets)))
    else:
        bad = replace(packet, unavailable_reason="MISSING")
    with runtime_for(regions, packet) as runtime:
        with pytest.raises(ValueError):
            runtime.prepare(regions, bad)
        batch = runtime.prepare(regions, packet)
        batch.release()


@pytest.mark.parametrize("square", [(0, 0, 0, 1), (0, 0, np.nan, 1), (100, 100, 110, 110)])
def test_invalid_or_empty_clipped_roi_is_rejected(square):
    packet, regions = history(1, 2), objects(1)
    with runtime_for(regions, packet) as runtime:
        with pytest.raises(ValueError):
            runtime.prepare([replace(regions[0], rgb_square=square)], packet)


@pytest.mark.parametrize("bound", ["objects", "history", "frames"])
def test_declared_capacity_limits_reject_oversized_inputs(bound):
    packet, regions = history(2, 3), objects(2)
    capacity = {"max_objects": 2, "max_history": 2, "max_frames": 4}
    capacity[f"max_{bound}"] = 1
    with RGBUInt8Runtime("cpu", **capacity) as runtime:
        with pytest.raises(ValueError, match="bound"):
            runtime.prepare(regions, packet)


@pytest.mark.parametrize(
    "kwargs", [{"max_objects": 21}, {"max_history": 0}, {"max_frames": True}, {"crop_workers": 0}]
)
def test_invalid_capacity_options_fail_before_allocation(kwargs):
    with pytest.raises(ValueError):
        RGBUInt8Runtime("cpu", **kwargs)


def test_raw_source_byte_bound_is_checked_before_large_pil_materialization():
    packet, regions = history(1, 2), objects(1)
    enormous = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (10_000, 10_000, 3))
    assert enormous.nbytes > 128 * 1024**2 and not enormous.flags.writeable
    oversized = replace(packet, frames={0: enormous, 1: packet.frames[1]})
    with runtime_for(regions, packet) as runtime:
        with pytest.raises(ValueError, match="host-source bound"):
            runtime.prepare(regions, oversized)


def test_cpu_path_never_queries_cuda_or_creates_cuda_events(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU RGB preparation must not access CUDA")

    for name in ("current_device", "current_stream", "synchronize", "Event", "is_available"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    packet, regions = history(1, 2), objects(1)
    with runtime_for(regions, packet) as runtime:
        batch = runtime.prepare(regions, packet)
        assert_original_exact(batch, regions, packet)
        batch.release()
