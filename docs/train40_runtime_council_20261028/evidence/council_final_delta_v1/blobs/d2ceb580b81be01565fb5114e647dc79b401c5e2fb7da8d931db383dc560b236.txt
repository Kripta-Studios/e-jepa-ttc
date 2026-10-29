"""Independent Pillow references for the proposed integer RGB resize path."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.train40_runtime.multiobject import ObjectRegion
from operational.train40_runtime.rgb_pil_exact import (
    ExactPILRGBRuntime,
    FrozenRGBHistory,
    pil_bilinear_coefficients,
    pil_horizontal_uint8,
    pil_vertical_normalized,
    pil_vertical_uint8,
    resize_uint8_exact,
)
from operational.train40_runtime.rgb_raw_ring import (
    ResidentRGBFrameRing,
    RGBSourceIdentity,
    freeze_rgb_frame,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pixels(shape, pattern):
    height, width = shape
    if pattern == "random":
        return np.random.default_rng(8146 + height * width).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
    if pattern in ("zero", "maximum"):
        return np.full((height, width, 3), 255 if pattern == "maximum" else 0, dtype=np.uint8)
    rows, columns, channels = np.indices((height, width, 3))
    if pattern == "checker":
        return (((rows + columns + channels) % 2) * 255).astype(np.uint8)
    if pattern == "rounding":
        return np.choose((rows * 3 + columns + channels) % 6, [0, 1, 127, 128, 254, 255]).astype(
            np.uint8
        )
    return ((rows * 17 + columns * 31 + channels * 73) % 256).astype(np.uint8)


def pillow_resize(frame):
    with Image.fromarray(frame, mode="RGB") as image:
        with image.resize((128, 128), Image.Resampling.BILINEAR) as resized:
            return np.array(resized, dtype=np.uint8).transpose(2, 0, 1)


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (1, 19), (19, 1), (7, 11), (63, 81), (127, 129), (128, 128), (129, 127), (199, 301)],
)
@pytest.mark.parametrize("pattern", ["random", "ramp", "zero", "maximum", "checker", "rounding"])
def test_full_frame_integer_resize_matches_pillow_exactly_and_preserves_source(shape, pattern):
    source = pixels(shape, pattern)
    before = source.copy()
    tensor = torch.from_numpy(source)
    actual = resize_uint8_exact(tensor)
    assert actual.dtype == torch.uint8 and actual.device.type == "cpu"
    assert actual.shape == (3, 128, 128)
    np.testing.assert_array_equal(actual.numpy(), pillow_resize(source))
    np.testing.assert_array_equal(source, before)
    # In particular the identity-size path must not expose a mutable source alias.
    actual.zero_()
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("shape", [(480, 641), (720, 1280), (1080, 1920), (1200, 1920)])
def test_large_downsample_and_actual_raw_dimensions_match_pillow(shape):
    source = pixels(shape, "random")
    actual = resize_uint8_exact(torch.from_numpy(source))
    np.testing.assert_array_equal(actual.numpy(), pillow_resize(source))


@pytest.mark.parametrize(
    "square",
    [
        (-20.9, -11.5, 89.9, 100.5),
        (100.99, 50.1, 270.8, 180.7),
        (250.2, 180.9, 400.8, 300.1),
        (0.0, 0.0, 301.0, 199.0),
        (55.1, 67.2, 183.9, 195.99),
        (55.0, 67.0, 56.0, 68.0),
        (-0.9, -0.9, 1.9, 1.9),
        (0.0, 0.0, 301.0, 1.0),
        (0.0, 0.0, 1.0, 199.0),
    ],
)
def test_roi_clipping_uses_original_truncation_and_crop_boundary_support(square):
    source = pixels((199, 301), "rounding")
    actual = resize_uint8_exact(torch.from_numpy(source), square)
    np.testing.assert_array_equal(actual.numpy(), crop_uint8(source, square))


@pytest.mark.parametrize(
    "square",
    [(400, 300, 500, 400), (-10, -10, -2, -2), (0, 0, 0, 1), (0.1, 0.1, 0.9, 0.9)],
)
def test_empty_or_wholly_external_roi_is_rejected_without_black_padding(square):
    source = pixels((199, 301), "random")
    with pytest.raises(ValueError):
        crop_uint8(source, square)
    with pytest.raises(ValueError):
        resize_uint8_exact(torch.from_numpy(source), square)


@pytest.mark.parametrize("input_size", [1, 2, 3, 7, 63, 127, 128, 129, 255, 1024, 1920])
@pytest.mark.parametrize("output_size", [1, 17, 128])
def test_coefficient_tables_have_valid_indices_nonnegative_weights_and_int32_bound(
    input_size, output_size
):
    table = pil_bilinear_coefficients(input_size, output_size)
    assert table.input_size == input_size
    assert table.output_size == output_size
    assert table.indices.dtype == np.int64
    assert table.weights.dtype == np.int32
    assert not table.indices.flags.writeable and not table.weights.flags.writeable
    assert table.indices.shape == table.weights.shape
    assert table.indices.shape[0] == output_size
    assert (table.indices >= 0).all() and (table.indices < input_size).all()
    assert (table.weights >= 0).all()
    sums = table.weights.astype(np.int64).sum(axis=1)
    # Every coefficient has at most half an LSB of independent quantization error.
    assert np.max(np.abs(sums - (1 << 22))) <= table.weights.shape[1] / 2 + 1
    independently_bounded = (1 << 21) + 255 * int(sums.max())
    assert table.bound_max_accumulator >= independently_bounded
    assert table.bound_max_accumulator <= np.iinfo(np.int32).max


@pytest.mark.parametrize(
    "input_size,output_size", [(0, 128), (-1, 128), (True, 128), (1.5, 128), (128, 0), (128, False)]
)
def test_invalid_coefficient_dimensions_fail_explicitly(input_size, output_size):
    with pytest.raises(ValueError):
        pil_bilinear_coefficients(input_size, output_size)


@pytest.mark.parametrize("kind", ["float", "channels", "dimensions", "empty"])
def test_invalid_raw_tensor_contract_is_rejected(kind):
    source = torch.zeros((13, 17, 3), dtype=torch.uint8)
    if kind == "float":
        source = source.float()
    elif kind == "channels":
        source = source[:, :, :2]
    elif kind == "dimensions":
        source = source[:, :, 0]
    else:
        source = source[:0]
    with pytest.raises(ValueError):
        resize_uint8_exact(source)


def test_noncontiguous_frame_view_matches_original_pixels_without_mutation():
    original = pixels((73, 99), "random")
    view = torch.from_numpy(original)[::2, ::2]
    assert not view.is_contiguous()
    actual = resize_uint8_exact(view)
    np.testing.assert_array_equal(actual.numpy(), pillow_resize(original[::2, ::2]))


def test_cpu_resize_never_queries_cuda_or_creates_cuda_events(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU exact resize must not access CUDA")

    for name in ("current_device", "current_stream", "synchronize", "Event", "is_available"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    source = pixels((19, 23), "checker")
    actual = resize_uint8_exact(torch.from_numpy(source), (0.1, 0.9, 20.7, 18.2))
    np.testing.assert_array_equal(actual.numpy(), crop_uint8(source, (0.1, 0.9, 20.7, 18.2)))


@pytest.mark.parametrize(
    "compiled",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.environ.get("TRAIN40_TEST_CPU_COMPILE") != "1",
                reason="Opt-in CPU eager fullgraph capture: TRAIN40_TEST_CPU_COMPILE=1",
            ),
        ),
    ],
)
def test_batched_passes_preserve_frame_slots_crop_offsets_and_exact_lut(compiled):
    frames = np.stack([pixels((13, 19), pattern) for pattern in ("random", "ramp", "checker")])
    raw = torch.from_numpy(frames)
    square = (3, 2, 17, 11)
    x0, y0, x1, y1 = square
    xtable = pil_bilinear_coefficients(x1 - x0)
    ytable = pil_bilinear_coefficients(y1 - y0)
    slots = torch.tensor([2, 0], dtype=torch.int64)
    rows = torch.arange(y0, y1, dtype=torch.int64)
    xindices = torch.from_numpy(xtable.indices.copy() + x0)
    yindices = torch.from_numpy(ytable.indices.copy())
    xweights = torch.from_numpy(xtable.weights.copy())
    yweights = torch.from_numpy(ytable.weights.copy())
    lut = torch.from_numpy(np.arange(256, dtype=np.float32) / np.float32(255))
    horizontal, vertical, normalized = (
        pil_horizontal_uint8,
        pil_vertical_uint8,
        pil_vertical_normalized,
    )
    if compiled:
        horizontal = torch.compile(horizontal, backend="eager", fullgraph=True)
        vertical = torch.compile(vertical, backend="eager", fullgraph=True)
        normalized = torch.compile(normalized, backend="eager", fullgraph=True)
    middle = horizontal(raw, slots, rows, xindices, xweights)
    assert middle.dtype == torch.uint8 and middle.shape == (2, y1 - y0, 128, 3)
    actual = vertical(middle, yindices, yweights)
    expected = np.stack([crop_uint8(frames[index], square) for index in (2, 0)])
    np.testing.assert_array_equal(actual.numpy(), expected)
    actual_normalized = normalized(middle, yindices, yweights, lut)
    np.testing.assert_array_equal(actual_normalized.numpy(), expected.astype(np.float32) / 255)
    np.testing.assert_array_equal(raw.numpy(), frames)


def frozen_history(groups=3, steps=3, shape=(11, 17)):
    source = RGBSourceIdentity("synthetic_rgb", "a" * 64)
    frames = {}
    for index in range(groups + steps - 1):
        values = pixels(shape, "random")
        values = (values.astype(np.int16) + index * 19).astype(np.uint8)
        frames[index] = freeze_rgb_frame(
            values, frame_id=index, timestamp_us=100_000 + 50_001 * index + index**2, source=source
        )
    return FrozenRGBHistory(
        max(frame.timestamp_us for frame in frames.values()) + 12_345,
        frames,
        tuple(tuple(range(index, index + steps)) for index in range(groups)),
    )


def runtime_regions(count=2):
    return [
        ObjectRegion(
            f"roi_{index}",
            (0, 0, 10, 10),
            (0.1 + (index % 5) * 0.8, 0.2 + (index // 5) * 0.6, 16.9, 10.8),
        )
        for index in range(count)
    ]


@contextmanager
def exact_runtime(regions, history, **options):
    first = next(iter(history.frames.values()))
    ring = ResidentRGBFrameRing("cpu", frame_shape=first.shape, capacity=len(history.frames))
    try:
        with ExactPILRGBRuntime(
            ring, max_objects=len(regions), max_history=max(1, len(history.triplets)), **options
        ) as runtime:
            yield runtime, ring
    finally:
        ring.close()


def assert_runtime_original(batch, regions, history):
    present = tuple(
        index
        for index, region in enumerate(regions)
        if region.rgb_square is not None and history.triplets
    )
    assert batch.present_indices == present
    assert batch.pixels is batch.sensor
    assert batch.pixels.dtype == batch.delta.dtype == torch.float32
    assert batch.pixels.is_contiguous() and batch.delta.is_contiguous()
    assert batch.pixels.device.type == "cpu" and batch.ready_event is None
    assert len(batch.metadata) == len(regions)
    expected_deltas = []
    for object_index, region in enumerate(regions):
        if object_index not in present:
            assert batch.metadata[object_index] == []
            continue
        originals = {
            index: crop_uint8(frame.array, region.rgb_square).astype(np.float32) / 255
            for index, frame in history.frames.items()
        }
        for group_index, group in enumerate(history.triplets):
            row = present.index(object_index) * len(history.triplets) + group_index
            np.testing.assert_array_equal(
                batch.pixels[row].numpy(), np.stack([originals[index] for index in group])
            )
            times = [history.frames[index].timestamp_us for index in group]
            delta = (np.diff(times) / 1e6).tolist()
            expected_deltas.append(delta)
            assert batch.metadata[object_index][group_index] == {
                "first_sensor_ts_us": times[0],
                "anchor_us": times[-1],
                "available_us": history.query_us,
                "timestamps_us": times,
                "frame_indices": list(group),
                "delta_t_s": delta,
                "square_xyxy": list(region.rgb_square),
            }
    if present:
        np.testing.assert_array_equal(batch.delta.numpy(), np.asarray(expected_deltas, np.float32))


@pytest.mark.parametrize("objects_count", [1, 2])
@pytest.mark.parametrize("groups,steps", [(1, 2), (1, 3), (5, 3), (8, 2), (8, 3)])
@pytest.mark.parametrize("chunk", [1, 2])
def test_runtime_one_two_objects_exact_original_for_native_history_and_frame_chunks(
    objects_count, groups, steps, chunk
):
    history, regions = frozen_history(groups, steps), runtime_regions(objects_count)
    with exact_runtime(regions, history, frame_chunk_size=chunk) as (runtime, ring):
        batch = runtime.prepare(regions, history)
        assert_runtime_original(batch, regions, history)
        assert batch.receipt["unique_roi_frame_crops"] == objects_count * len(history.frames)
        assert batch.receipt["roi_table_h2d_bytes"] == 0
        assert batch.receipt["ring"]["h2d_bytes"] == 0
        assert batch.receipt["ring"]["uploads"] == len(history.frames)
        assert ring.stats["frame_bytes_hashed_inside_bind"] == 0
        batch.release()


@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_runtime_padded_last_frame_chunk_cannot_corrupt_next_object(chunk):
    history, regions = frozen_history(4, 2), runtime_regions(4)
    regions[1] = replace(regions[1], rgb_square=None)
    regions[2] = replace(regions[2], rgb_square=(3.1, 1.2, 15.9, 9.7))
    regions[3] = replace(regions[3], rgb_square=(7.1, 3.1, 8.9, 4.9))
    with exact_runtime(regions, history, frame_chunk_size=chunk) as (runtime, _):
        batch = runtime.prepare(regions, history)
        assert batch.present_indices == (0, 2, 3)
        assert_runtime_original(batch, regions, history)
        batch.release()


def test_runtime_twenty_objects_small_input_exact_and_shared_raw_upload_count():
    history, regions = frozen_history(5, 3), runtime_regions(20)
    with exact_runtime(regions, history, frame_chunk_size=3) as (runtime, ring):
        batch = runtime.prepare(regions, history)
        assert batch.pixels.shape == (100, 3, 3, 128, 128)
        assert_runtime_original(batch, regions, history)
        assert batch.receipt["unique_roi_frame_crops"] == 140
        assert batch.receipt["ring"]["uploads"] == 7
        assert ring.stats["uploaded_bytes"] == 7 * 11 * 17 * 3
        batch.release()


def test_runtime_changed_roi_recomputes_crops_while_reusing_every_resident_raw_frame():
    history, regions = frozen_history(2, 3), runtime_regions(2)
    with exact_runtime(regions, history, frame_chunk_size=3) as (runtime, ring):
        initial = runtime.prepare(regions, history)
        previous = initial.pixels.clone()
        pointers = {name: tensor.data_ptr() for name, tensor in runtime._buffers.items()}
        allocation = runtime.allocation_receipt
        initial.release()
        regions[0] = replace(regions[0], rgb_square=(4.1, 2.1, 14.8, 9.9))
        later = replace(history, query_us=history.query_us + 20_000)
        changed = runtime.prepare(regions, later)
        assert changed.receipt["ring"]["uploads"] == 0
        assert changed.receipt["ring"]["hits"] == len(history.frames)
        assert changed.receipt["ring"]["uploaded_bytes"] == 0
        assert not torch.equal(changed.pixels, previous)
        assert_runtime_original(changed, regions, later)
        assert pointers == {name: tensor.data_ptr() for name, tensor in runtime._buffers.items()}
        assert runtime.allocation_receipt == allocation
        assert ring.stats["frame_bytes_hashed_inside_bind"] == 0
        changed.release()


def test_runtime_new_arrival_preserves_frame_order_after_ring_slot_reuse():
    history, regions = frozen_history(2, 3), runtime_regions(2)
    with exact_runtime(regions, history, frame_chunk_size=3) as (runtime, _):
        first = runtime.prepare(regions, history)
        first.release()
        last = history.frames[3]
        arriving = freeze_rgb_frame(
            pixels((11, 17), "checker"),
            frame_id=4,
            timestamp_us=last.timestamp_us + 50_013,
            source=last.source,
        )
        updated = FrozenRGBHistory(
            arriving.timestamp_us + 12_345,
            {1: history.frames[1], 2: history.frames[2], 3: last, 4: arriving},
            ((1, 2, 3), (2, 3, 4)),
        )
        batch = runtime.prepare(regions, updated)
        assert batch.receipt["ring"]["uploads"] == 1
        assert batch.receipt["ring"]["hits"] == 3
        assert batch.receipt["ring"]["evictions"] == 1
        assert_runtime_original(batch, regions, updated)
        batch.release()


@pytest.mark.parametrize("kind", ["history", "regions"])
def test_runtime_missing_rgb_does_not_bind_raw_ring(kind):
    history, regions = frozen_history(2, 3), runtime_regions(2)
    with exact_runtime(regions, history) as (runtime, ring):
        if kind == "history":
            history = FrozenRGBHistory(history.query_us, {}, (), "UNAVAILABLE")
        else:
            regions = [replace(region, rgb_square=None) for region in regions]
        batch = runtime.prepare(regions, history)
        assert batch.pixels.numel() == batch.delta.numel() == 0
        assert batch.metadata == [[], []]
        assert batch.present_indices == ()
        assert batch.receipt["roi_table_candidate_h2d_bytes"] == 0
        assert batch.receipt["ring"] == {}
        assert ring.stats["binds"] == 0
        batch.release()


def test_runtime_release_ends_both_output_and_raw_leases_and_close_keeps_external_ring():
    history, regions = frozen_history(1, 2), runtime_regions(1)
    with exact_runtime(regions, history) as (runtime, ring):
        batch = runtime.prepare(regions, history)
        batch.assert_valid()
        with pytest.raises(RuntimeError, match="lease"):
            runtime.prepare(regions, history)
        with pytest.raises(RuntimeError, match="mark_consumed"):
            ring.bind(list(history.frames.values()), history.query_us)
        batch.release()
        with pytest.raises(ValueError, match="lease"):
            batch.assert_valid()
        with pytest.raises(ValueError, match="lease"):
            batch.release()
        runtime.close()
        binding = ring.bind(list(history.frames.values()), history.query_us)
        assert binding.receipt["uploads"] == 0
        ring.mark_consumed(binding)
        with pytest.raises(RuntimeError, match="closed"):
            runtime.prepare(regions, history)


def test_runtime_foreign_output_lease_is_rejected():
    history, regions = frozen_history(1, 2), runtime_regions(1)
    with (
        exact_runtime(regions, history) as (first, _),
        exact_runtime(regions, history) as (second, _),
    ):
        batch = first.prepare(regions, history)
        with pytest.raises(ValueError, match="lease"):
            second.release(batch)
        batch.release()


def test_runtime_preparation_and_release_require_the_owning_thread():
    history, regions = frozen_history(1, 2), runtime_regions(1)
    with exact_runtime(regions, history) as (runtime, _), ThreadPoolExecutor(1) as worker:
        with pytest.raises(RuntimeError, match="owning thread"):
            worker.submit(runtime.prepare, regions, history).result()
        batch = runtime.prepare(regions, history)
        with pytest.raises(RuntimeError, match="owning thread"):
            worker.submit(batch.release).result()
        batch.release()


def test_runtime_eager_memory_bound_releases_ring_after_failure():
    history, regions = frozen_history(1, 2), runtime_regions(1)
    with exact_runtime(regions, history, max_eager_bytes=1) as (limited, ring):
        with pytest.raises(ValueError, match="temporary bound"):
            limited.prepare(regions, history)
        with ExactPILRGBRuntime(ring, max_objects=1, max_history=1) as healthy:
            batch = healthy.prepare(regions, history)
            assert_runtime_original(batch, regions, history)
            batch.release()


@pytest.mark.parametrize("kind", ["query", "missing", "mixed", "future", "stale", "availability"])
def test_runtime_bad_history_fails_before_acquiring_ring_and_next_call_recovers(kind):
    history, regions = frozen_history(2, 3), runtime_regions(1)
    bad = history
    if kind == "query":
        bad = replace(history, query_us=-1)
    elif kind == "missing":
        bad = replace(
            history, frames={index: frame for index, frame in history.frames.items() if index}
        )
    elif kind == "mixed":
        bad = replace(history, triplets=(history.triplets[0][:2], history.triplets[1]))
    elif kind == "future":
        bad = replace(
            history, query_us=max(frame.timestamp_us for frame in history.frames.values()) - 1
        )
    elif kind == "stale":
        bad = replace(
            history, query_us=min(frame.timestamp_us for frame in history.frames.values()) + 650_001
        )
    else:
        bad = replace(history, unavailable_reason="MISSING")
    with exact_runtime(regions, history) as (runtime, ring):
        with pytest.raises(ValueError):
            runtime.prepare(regions, bad)
        assert ring.stats["binds"] == 0
        batch = runtime.prepare(regions, history)
        batch.release()


@pytest.mark.skipif(
    os.environ.get("TRAIN40_TEST_CPU_COMPILE") != "1",
    reason="Opt-in CPU eager fullgraph capture: TRAIN40_TEST_CPU_COMPILE=1",
)
def test_runtime_compiled_eager_cpu_exact_for_two_objects_and_padded_chunk():
    history, regions = frozen_history(2, 3), runtime_regions(2)
    with exact_runtime(
        regions, history, frame_chunk_size=3, compile_kernels=True, compile_backend="eager"
    ) as (runtime, _):
        batch = runtime.prepare(regions, history)
        assert_runtime_original(batch, regions, history)
        batch.release()
