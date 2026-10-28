"""Causal arrival accounting and specialization observability for the RGB probe."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory
from operational.train40_runtime.rgb_pil_exact import ExactPILRGBRuntime, FrozenRGBHistory
from operational.train40_runtime.rgb_raw_ring import ResidentRGBFrameRing, RGBSourceIdentity
from scripts import benchmark_train40_rgb_pil_exact as runner
from scripts.benchmark_train40_rgb_pil_exact import (
    arrival_partition,
    managed_runtime,
    measure_update,
    prime_synthetic,
    register_observed_backend,
    seal_frames,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_history(shape=(11, 17, 3)):
    rng = np.random.default_rng(28)
    return RawRGBHistory(
        260_000,
        {index: rng.integers(0, 256, shape, dtype=np.uint8) for index in range(4)},
        {index: 100_000 + 50_000 * index for index in range(4)},
        [(0, 1, 2), (1, 2, 3)],
    )


def test_arrival_cutoff_keeps_equal_old_and_counts_later_frames():
    assert arrival_partition({3: 200_000, 4: 240_000, 5: 250_000}, 260_000) == ([3, 4], [5])
    with pytest.raises(ValueError, match="causal"):
        arrival_partition({1: 260_001}, 260_000)
    with pytest.raises(ValueError, match="positive"):
        arrival_partition({1: 10}, 20, step_us=0)


def test_source_inventory_ignores_vendor_pseudo_files_but_binds_owned_source(tmp_path):
    source = tmp_path / "operational" / "sample.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    modules = {
        "torch.classes": SimpleNamespace(__file__="_classes.py"),
        "vendor.synthetic": SimpleNamespace(__file__=str(tmp_path / "missing.py")),
        "operational": SimpleNamespace(__path__=[str(source.parent)]),
        "operational.sample": SimpleNamespace(__file__=str(source)),
    }
    assert runner.project_source_paths(modules, tmp_path) == {source.resolve()}


@pytest.mark.parametrize("name", ["operational.missing", "e_jepa_ttc.missing", "scripts.missing"])
def test_source_inventory_fails_closed_for_missing_owned_sources(tmp_path, name):
    modules = {name: SimpleNamespace(__file__=str(tmp_path / "missing.py"))}
    with pytest.raises(ValueError, match="Owned runtime source is missing"):
        runner.project_source_paths(modules, tmp_path)


def test_source_inventory_rejects_owned_module_loaded_outside_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = tmp_path / "wrong_checkout.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside this checkout"):
        runner.project_source_paths(
            {"operational.sample": SimpleNamespace(__file__=str(source))}, checkout
        )


def test_failed_constructor_closes_ring_and_preserves_primary_failure(tmp_path):
    raw = tmp_path / "raw"
    raw.write_bytes(b"unchanged")
    stat = raw.stat()
    receipts, closed = [], []

    class Ring:
        def close(self):
            closed.append("ring")
            raise RuntimeError("close problem")

    def broken_factory(ring):
        raise ValueError("primary constructor failure")

    with pytest.raises(ValueError, match="primary constructor failure"):
        with managed_runtime(
            Ring,
            broken_factory,
            raw,
            {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            receipts,
        ):
            pytest.fail("Failed constructor cannot yield")
    assert closed == ["ring"]
    assert receipts[0]["raw_stat_check"]["passed"]
    assert receipts[0]["cleanup_errors"] == ["ring: RuntimeError: close problem"]


def test_cleanup_attempts_both_closes_and_raw_check_after_primary_failure(tmp_path):
    raw = tmp_path / "raw"
    raw.write_bytes(b"original")
    stat = raw.stat()
    receipts, closed = [], []

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)
            raise RuntimeError("close problem")

    with pytest.raises(ValueError, match="primary GPU failure"):
        with managed_runtime(
            lambda: Resource("ring"),
            lambda ring: Resource("runtime"),
            raw,
            {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            receipts,
        ):
            raw.write_bytes(b"changed length")
            raise ValueError("primary GPU failure")
    assert closed == ["runtime", "ring"]
    assert not receipts[0]["raw_stat_check"]["passed"]
    assert len(receipts[0]["cleanup_errors"]) == 2


def test_update_clock_accounts_new_arrivals_and_reports_both_miss_populations():
    raw = fixture_history()
    source = RGBSourceIdentity("fixture", "f" * 64)
    regions = [ObjectRegion("a", (0, 0, 1, 1), (1.2, 0.1, 16.9, 10.8))]
    ring = ResidentRGBFrameRing("cpu", frame_shape=(11, 17, 3), capacity=16)
    runtime = ExactPILRGBRuntime(ring, max_objects=1)
    try:
        full = measure_update(runtime, raw, regions, source, prefix=False, compilation_records=[])
        partial = measure_update(runtime, raw, regions, source, prefix=True, compilation_records=[])
        assert full["parity"]["passed"] and partial["parity"]["passed"]
        assert full["newly_sealed_frame_ids"] == [0, 1, 2, 3]
        assert partial["newly_sealed_frame_ids"] == [3]
        assert full["newly_sealed_bytes"] == 4 * 11 * 17 * 3
        assert partial["newly_sealed_bytes"] == 11 * 17 * 3
        assert full["prepare_receipt"]["ring"]["uploads"] == 4
        assert partial["prepare_receipt"]["ring"]["uploads"] == 1
        assert partial["prepare_receipt"]["ring"]["hits"] == 3
        assert partial["prefix_receipt"]["query_us"] == 240_000
        assert full["prefix_receipt"] is None
        assert full["wall_ms"] >= full["sealing_seconds_included"] * 1000
    finally:
        runtime.close()
        ring.close()


def test_synthetic_priming_cannot_leave_real_or_synthetic_frames_resident():
    raw = fixture_history()
    before = {index: frame.copy() for index, frame in raw.frames.items()}
    regions = [ObjectRegion("a", (0, 0, 1, 1), (0, 0, 17, 11))]
    ring = ResidentRGBFrameRing("cpu", frame_shape=(11, 17, 3), capacity=16)
    runtime = ExactPILRGBRuntime(ring, max_objects=1)
    try:
        primed = prime_synthetic(runtime, regions, raw)
        assert not primed["real_pixels_used"] and not primed["covers_arbitrary_future_roi"]
        source = RGBSourceIdentity("fixture", "f" * 64)
        real = measure_update(runtime, raw, regions, source, prefix=False, compilation_records=[])
        assert real["prepare_receipt"]["ring"]["hits"] == 0
        assert real["parity"]["passed"]
        for index, frame in raw.frames.items():
            np.testing.assert_array_equal(frame, before[index])
    finally:
        runtime.close()
        ring.close()


@pytest.mark.skipif(
    os.environ.get("TRAIN40_TEST_CPU_COMPILE") != "1",
    reason="Opt-in CPU graph specialization audit, not CUDA compilation",
)
def test_compiled_shapes_reuse_offsets_and_record_bucket_transition():
    raw = fixture_history((67, 99, 3))
    source = RGBSourceIdentity("fixture", "f" * 64)
    tokens = seal_frames(raw, sorted(raw.frames), source)
    history = FrozenRGBHistory(raw.query_us, tokens, raw.triplets)
    regions = [
        ObjectRegion("a", (0, 0, 1, 1), (0, 0, 31, 31)),
        ObjectRegion("b", (0, 0, 1, 1), (7, 5, 38, 36)),
    ]
    backend, records = register_observed_backend("eager")
    ring = ResidentRGBFrameRing("cpu", frame_shape=(67, 99, 3), capacity=16)
    runtime = ExactPILRGBRuntime(ring, max_objects=2, compile_kernels=True, compile_backend=backend)
    try:
        first = runtime.prepare(regions, history)
        first.release()
        first_count = len(records)
        assert first_count >= 2
        shifted = [replace(region, rgb_square=(11, 9, 42, 40)) for region in regions]
        same = runtime.prepare(shifted, history)
        same.release()
        assert len(records) == first_count
        larger = [replace(region, rgb_square=(2, 3, 53, 54)) for region in regions]
        changed = runtime.prepare(larger, history)
        changed.release()
        changed_count = len(records)
        assert changed_count > first_count
        returned = runtime.prepare(regions, history)
        returned.release()
        assert len(records) == changed_count
    finally:
        runtime.close()
        ring.close()
