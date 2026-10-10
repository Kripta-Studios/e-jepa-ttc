"""Regression gates for approximate streaming and numerical execution variants."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.garl_input import native_feature
from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.models.local_transport import local_correlation_match
from operational.streaming_revision.distill import FeatureStudent
from operational.streaming_revision.kernels import (
    EncoderPrecision,
    correlation,
    owned_output,
    pad_head_inputs,
)
from operational.streaming_revision.packets import PacketRing, native_live_window
from operational.streaming_revision.preparation import IncrementalPreparer, Query, warp_voxel
from operational.streaming_revision.state import FeatureState, Observation, Policy, roi_distance


def query(anchor=1000000, roi=(0.0, 0.0, 128.0, 128.0)):
    return Query(
        "s",
        "object-1",
        anchor,
        anchor,
        tuple((anchor - x - 100000, anchor - x) for x in (200000, 100000, 0)),
        roi,
    )


def observation(anchor=950000):
    q = query(anchor)
    return Observation(anchor, q.windows, q.roi, 0.1, anchor, anchor, torch.ones(17))


def lookup(state, q, anchor=950000, used=None):
    return state.find(
        anchor=anchor,
        now=q.anchor,
        windows=q.at(anchor),
        roi=q.roi,
        delta=q.delta,
        used=set() if used is None else used,
    )


def test_reuse_is_causal_bounded_and_object_specific():
    state = FeatureState(Policy())
    state.begin("s", "object-1", 950000)
    state.commit([observation()])
    assert not state.begin("s", "object-1", 1000000)
    hit = lookup(state, query())
    assert hit is not None
    assert lookup(state, query(), 949999) is None
    assert lookup(state, query(), used={id(hit)}) is None
    assert lookup(state, query(), 1000000) is None
    assert lookup(state, query(2000000)) is None
    assert state.begin("s", "object-2", 1000000)
    assert not state.entries
    assert state.begin("s", "object-2", 900000)


def test_spatial_change_and_metadata_availability_invalidate():
    state = FeatureState(Policy())
    state.commit([observation()])
    assert lookup(state, query(roi=(1.0, 1.0, 129.0, 129.0))) is not None
    assert lookup(state, query(roi=(200.0, 0.0, 328.0, 128.0))) is None
    state.entries[0].available = 1000001
    assert lookup(state, query()) is None


def test_exact_policy_does_not_reuse_changed_inputs():
    state = FeatureState(Policy(time_tolerance_us=0, min_roi_iou=1, max_log_scale=0))
    state.commit([observation()])
    assert lookup(state, query()) is not None
    assert lookup(state, query(roi=(0.01, 0.0, 128.01, 128.0))) is None
    assert lookup(state, query(), 950001) is None


def test_retention_never_exceeds_limit():
    state = FeatureState(Policy(max_entries=8))
    state.commit([observation(i * 50000) for i in range(20)])
    assert len(state.entries) == 8
    with pytest.raises(ValueError, match="finite"):
        state.commit([replace(observation(), feature=torch.full((17,), float("nan")))])


@pytest.mark.parametrize("radius", [1, 2])
def test_transport_values_boundaries_and_gradients(radius):
    torch.manual_seed(7)
    a = torch.randn(2, 4, 9, 11, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    old = local_correlation_match(a, b, radius=radius, temperature=0.2, return_probability=True)
    new = correlation(a, b, radius=radius, temperature=0.2, return_probability=True)
    for field in ("dx", "dy", "entropy", "confidence_margin", "probability"):
        torch.testing.assert_close(getattr(old, field), getattr(new, field), atol=3e-6, rtol=2e-5)
    assert torch.equal(old.valid, new.valid)
    g0 = torch.autograd.grad(old.dx.sum() + old.dy.sum(), (a, b), retain_graph=True)
    g1 = torch.autograd.grad(new.dx.sum() + new.dy.sum(), (a, b))
    for left, right in zip(g0, g1, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)


def test_incremental_preparation_matches_reference_and_bounds_memory():
    raw = {
        "t": np.arange(300000, 1100000, 1000, dtype=np.int64),
        "x": np.full(800, 32, dtype=np.int32),
        "y": np.full(800, 64, dtype=np.int32),
        "p": np.ones(800, dtype=np.int8),
    }
    reads = []

    def read(a, b):
        reads.append((a, b))
        mask = (raw["t"] >= a) & (raw["t"] < b)
        return {k: v[mask] for k, v in raw.items()}

    prep = IncrementalPreparer(read, voxel_bytes=3 * 12 * 128 * 128 * 4)
    q = query()
    first = prep.prepare(q, [q.anchor])
    mapped = map_roi(raw, q.roi, 128, 0.0)
    expected = torch.stack([encode_mapped(mapped, a, b, 128, "s") for a, b in q.windows])
    assert torch.equal(first[0], expected)
    again = prep.prepare(q, [q.anchor])
    assert torch.equal(first, again) and len(reads) == 1
    assert prep.last_diagnostics["voxel_hits"] == 3
    prep.prepare(query(1050000), [1050000])
    assert reads[-1] == (1000000, 1050000)
    assert prep.retained_voxel_bytes <= prep.voxel_bytes


def test_student_never_consumes_removed_producers():
    model = FeatureStudent(torch.zeros(17), torch.ones(17)).eval()
    x = torch.randn(3, 17)
    changed = x.clone()
    changed[:, [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]] += 100
    torch.testing.assert_close(model(x), model(changed))
    torch.testing.assert_close(model(x)[:, 11], model(x)[:, 10] - x[:, 8])
    model(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_encoder_precision_restores_geometry_dtype():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3)

        def forward(self, x, **kwargs):
            z = self.conv(x)
            return z, z.mean((-2, -1)), None

    wrapped = EncoderPrecision(Encoder(), "bf16")
    outputs = wrapped(torch.randn(1, 3, 8, 8))
    assert outputs[0].dtype == outputs[1].dtype == torch.float32
    assert outputs[2] is None


def test_invalid_policies_and_query_fail_explicitly():
    with pytest.raises(ValueError):
        Policy(history=3)
    with pytest.raises(ValueError):
        Policy(time_tolerance_us=50000)
    with pytest.raises(ValueError):
        roi_distance((0, 0, 0, 2), (0, 0, 2, 2))
    with pytest.raises(ValueError):
        replace(query(), valid=(False, True, False, True, True, True, True, True))


def test_warp_identity_translation_and_scalars():
    value = torch.zeros(12, 128, 128)
    value[:10, 64, 32] = 1
    value[10:] = 7
    assert torch.equal(warp_voxel(value, (0, 0, 128, 128), (0, 0, 128, 128)), value)
    shifted = warp_voxel(value, (0, 0, 128, 128), (1, 0, 129, 128))
    torch.testing.assert_close(shifted[:10, 64, 31], torch.ones(10), atol=1e-5, rtol=0)
    assert torch.equal(shifted[10:], value[10:])
    assert shifted[:10, :, -1].abs().sum() == 0


def test_warp_reuses_two_old_windows_without_caching_resampled_outputs():
    reads = []

    def read(a, b):
        reads.append((a, b))
        t = np.arange(a, b, 1000, dtype=np.int64)
        return {
            "t": t,
            "x": np.full(len(t), 32, np.int32),
            "y": np.full(len(t), 64, np.int32),
            "p": np.ones(len(t), np.int8),
        }

    prep = IncrementalPreparer(read, approximate_voxels=True)
    prep.prepare(query(), [1000000])
    prep.prepare(query(1100000, roi=(1.0, 0.0, 129.0, 128.0)), [1100000])
    assert prep.last_diagnostics["approximate_voxel_hits"] == 2
    assert prep.last_diagnostics["voxel_misses"] == 1
    assert len(prep.voxels) == 4
    assert reads[-1] == (1000000, 1100000)


def test_packet_ring_coverage_empty_packets_ownership_and_gaps():
    ring = PacketRing()
    raw = {
        "t": np.array([10, 20], np.int64),
        "x": np.array([0, 1279], np.int32),
        "y": np.array([0, 0], np.int32),
        "p": np.array([1, -1], np.int8),
    }
    ring.push(raw, 0, 1000)
    raw["x"][:] = 99
    assert ring.read_window(0, 1000)["x"].tolist() == [0, 1279]
    assert native_live_window(ring, 0, 1000)["x"].tolist() == [5]
    empty = {k: v[:0] for k, v in raw.items()}
    ring.push(empty, 1000, 2000)
    assert not len(ring.read_window(1000, 2000)["t"])
    with pytest.raises(ValueError, match="gap or rollback"):
        ring.push(empty, 3000, 4000)
    with pytest.raises(ValueError, match="covered"):
        ring.read_window(0, 3000)
    ring.push(empty, 2000, 2000000)
    with pytest.raises(ValueError, match="covered"):
        ring.read_window(0, 1000)


def test_packet_overflow_and_unsigned_rollback_fail():
    raw = {
        "t": np.array([20, 10], np.uint64),
        "x": np.array([1, 2], np.int32),
        "y": np.array([1, 2], np.int32),
        "p": np.ones(2, np.int8),
    }
    with pytest.raises(ValueError, match="timestamp"):
        PacketRing().push(raw, 0, 100)
    raw["t"] = np.array([10, 20], np.int64)
    small = PacketRing(max_bytes=1)
    with pytest.raises(MemoryError):
        small.push(raw, 0, 100)
    assert small.end is None and not small.packets


def test_graph_output_has_independent_storage_recursively():
    a = torch.ones(2, 3, 5, 5)
    match = local_correlation_match(a, a, radius=1, temperature=0.1)
    original = {"nested": [match, (a, None)]}
    cloned = owned_output(original)
    a.fill_(7)
    match.dx.fill_(99)
    assert torch.equal(cloned["nested"][1][0], torch.ones_like(a))
    assert not (cloned["nested"][0].dx == 99).any()


def test_compact_packets_are_lossless_and_trim_partial_packets():
    base = 2**40
    t = base + np.arange(0, 900000, 1000, dtype=np.int64)
    raw = {
        "t": t,
        "x": np.arange(len(t), dtype=np.int32),
        "y": np.full(len(t), 719, np.int32),
        "p": np.where(np.arange(len(t)) % 2, -1, 1).astype(np.int8),
    }
    ring = PacketRing(horizon_us=650000)
    first = {k: v[:600] for k, v in raw.items()}
    ring.push(first, base, base + 600000)
    assert ring.retained_bytes == 600 * 9
    for key, value in first.items():
        assert np.array_equal(ring.read_window(base, base + 600000)[key], value)
    ring.push({k: v[600:] for k, v in raw.items()}, base + 600000, base + 900000)
    assert ring.start == base + 250000
    assert ring.packets[0][0] == ring.start
    assert ring.retained_bytes == 650 * 9
    for key, value in raw.items():
        restored = ring.read_window(base + 250000, base + 900000)[key]
        assert np.array_equal(restored, value[250:])
        assert restored.dtype == value.dtype
    with pytest.raises(ValueError, match="covered"):
        ring.read_window(base, base + 900000)


def test_packet_coordinate_overflow_uses_lossless_fallback():
    raw = {
        "t": np.array([1], np.int64),
        "x": np.array([40000], np.int32),
        "y": np.array([-40000], np.int32),
        "p": np.array([-1], np.int8),
    }
    ring = PacketRing()
    ring.push(raw, 0, 100)
    for key, value in raw.items():
        assert np.array_equal(ring.read_window(0, 100)[key], value)


def test_roi_first_preserves_voxels_and_garl_global_time_origin():
    rng = np.random.default_rng(5)
    times = np.arange(0, 300000, 30, dtype=np.int64)
    raw = {
        "t": times,
        "x": rng.integers(0, 1280, len(times), dtype=np.int32),
        "y": rng.integers(0, 720, len(times), dtype=np.int32),
        "p": np.ones(len(times), np.int8),
    }
    raw["x"][0], raw["y"][0] = 1, 1
    ring = PacketRing()
    ring.push({k: v[:5000] for k, v in raw.items()}, 0, 150000)
    ring.push({k: v[5000:] for k, v in raw.items()}, 150000, 300000)
    roi = (100, 200, 200, 300)
    cropped = ring.read_roi(0, 300000, roi, 5)
    old = map_roi(raw, roi, 128, 5)
    new = map_roi(cropped, roi, 128, 5)
    for key in old:
        assert np.array_equal(old[key], new[key])
    full = native_live_window(ring, 0, 300000)
    selected = native_live_window(ring, 0, 300000, roi)
    assert selected["t"][0] == full["t"][0] == 0
    assert len(selected["t"]) < len(full["t"])
    assert torch.equal(native_feature(full, roi), native_feature(selected, roi))


@pytest.mark.parametrize("backbone", ["gru", "transformer"])
def test_fixed_head_padding_preserves_all_predictions(backbone):
    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner

    torch.manual_seed(9)
    head = TemporalRefiner(
        TemporalConfig(hidden=64 if backbone == "gru" else 128, backbone=backbone)
    ).eval()
    with torch.no_grad():
        for layer in (head.location, head.width, head.cost):
            layer.weight.normal_(0, 0.01)
        for count in (1, 3, 5, 7, 8):
            xs = (
                torch.randn(1, count, 17),
                torch.randn(1, count, 4),
                torch.ones(1, count, dtype=torch.bool),
                torch.tensor([[0.01, 0.02, 0.03]]),
            )
            original = head(*xs)
            padded = head(*pad_head_inputs(xs, 8))
            for key in original:
                torch.testing.assert_close(original[key], padded[key], rtol=1e-4, atol=1e-6)


def test_packet_search_preserves_unsigned_maximum_and_half_open_end():
    raw = {
        "t": np.array([0, 2**32 - 1], np.int64),
        "x": np.array([1, 2], np.int32),
        "y": np.array([1, 2], np.int32),
        "p": np.ones(2, np.int8),
    }
    ring = PacketRing(horizon_us=2**32)
    ring.push(raw, 0, 2**32)
    assert ring.read_window(0, 2**32)["t"].tolist() == [0, 2**32 - 1]
    assert ring.read_window(1, 2**32)["t"].tolist() == [2**32 - 1]
    assert ring.read_window(0, 2**32 - 1)["t"].tolist() == [0]


def test_segmented_packets_cross_clock_wrap_and_keep_exact_coverage():
    base = 2**32 - 150000
    t = base + np.arange(0, 850000, 1000, dtype=np.int64)
    raw = {
        "t": t,
        "x": np.arange(len(t), dtype=np.int32),
        "y": np.full(len(t), 100, np.int32),
        "p": np.ones(len(t), np.int8),
    }
    ring = PacketRing(horizon_us=650000, packet_span_us=50000)
    ring.push({k: v[:650] for k, v in raw.items()}, base, base + 650000)
    assert len(ring.packets) == 13
    ring.push({k: v[650:] for k, v in raw.items()}, base + 650000, base + 850000)
    assert all(last - first <= 50000 for first, last, _ in ring.packets)
    assert ring.retained_bytes == 650 * 9
    recovered = ring.read_window(base + 200000, base + 850000)
    for key in raw:
        assert np.array_equal(recovered[key], raw[key][200:])
    assert ring.start == base + 200000
    with pytest.raises(ValueError, match="gap or rollback"):
        ring.push({k: v[:0] for k, v in raw.items()}, base + 850000, 2**63)


@pytest.mark.parametrize("timestamps", [[], [1, 2, 127], [255]])
def test_packet_coverage_can_exceed_timestamp_dtype(timestamps):
    raw = {
        "t": np.array(timestamps, np.uint8),
        "x": np.ones(len(timestamps), np.int16),
        "y": np.ones(len(timestamps), np.int16),
        "p": np.ones(len(timestamps), np.int8),
    }
    ring = PacketRing(horizon_us=650000, packet_span_us=50000)
    ring.push(raw, 0, 700000)
    assert ring.start == 50000 and ring.end == 700000
    assert not ring.read_window(50000, 700000)["t"].size
    assert ring.retained_bytes == 0
