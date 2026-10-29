"""CPU checks for complete 5ms backend admission and its bounded memory gate."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory
from operational.train40_runtime.temporal_prefix_subpacket import (
    SubpacketTemporalPrefixEventHistory,
)
from scripts import benchmark_train40_subpacket_backend as benchmark

STATE_TENSORS = (
    "_counts",
    "_moments",
    "_cumulative_count",
    "_cumulative_moment",
    "_previous_count",
    "_previous_moment",
)
STATE_SCALARS = ("end_us", "_total_buckets", "_epoch_start_us", "_filled", "_ingested_packets")
ROI = (-1.5, 1.25, 10.8, 11.0)
Raw = dict[str, np.ndarray[Any, Any]]


@pytest.fixture(autouse=True)
def limited_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _options(normalization: str = "sort") -> dict[str, Any]:
    return {
        "device": "cpu",
        "width": 12,
        "height": 10,
        "roi_size": 8,
        "max_packet_events": 1024,
        "max_objects": 1,
        "roi_batch_size": 1,
        "normalization": normalization,
    }


def _events(start: int) -> Raw:
    rng = np.random.default_rng(264)
    edges = np.arange(0, 1_200_000, 5000, dtype=np.int64)
    timestamps = np.sort(
        np.concatenate((edges, edges, edges + 4999, rng.integers(0, 1_200_000, 900)))
    )
    return {
        "x": rng.integers(0, 12, len(timestamps), dtype=np.int32),
        "y": rng.integers(0, 10, len(timestamps), dtype=np.int32),
        "t": timestamps + start,
        "p": rng.choice(np.array([-(2**31), 0, 2**31], dtype=np.int64), len(timestamps)),
    }


def _initialized_pair() -> tuple[TemporalPrefixEventHistory, SubpacketTemporalPrefixEventHistory]:
    original = TemporalPrefixEventHistory(**_options())
    candidate = SubpacketTemporalPrefixEventHistory(**_options())
    raw = benchmark.slice_raw(_events(738), 738, 600_738)
    for history in (original, candidate):
        history.initialize(raw, 738, 600_738)
    return original, candidate


def _forbid_cuda(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("This CPU diagnostic must not call a CUDA API")


@pytest.mark.parametrize("normalization", ["sort", "topk"])
@pytest.mark.parametrize("start", [738, 2**52 + 738])
def test_backend_checks_every_completed_boundary_and_partial_guards(
    monkeypatch: pytest.MonkeyPatch, normalization: str, start: int
) -> None:
    for name in ("synchronize", "current_stream", "mem_get_info", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, _forbid_cuda)
    record, (original, candidate) = benchmark.run_backend_case(
        _events(start), ROI, start, _options(normalization), query_graphs=False
    )
    assert record["passed"] is True
    checkpoints = record["checkpoints"]
    assert [row["bucket"] for row in checkpoints] == list(range(120, 241, 4))
    assert [row["bucket"] for row in checkpoints if row["explicit_checkpoint"]] == [120, 124, 240]
    assert all(row["state"]["passed"] and row["representation"]["passed"] for row in checkpoints)
    assert [(row["bucket"], row["phase"]) for row in record["partial_guards"]] == [
        (121, 1),
        (122, 2),
        (123, 3),
        (237, 1),
        (238, 2),
        (239, 3),
    ]
    assert all(row["passed"] for row in record["partial_guards"])
    assert original.device.type == candidate.device.type == "cpu"
    assert original.shape == candidate.shape == (1, 4, 3, 12, 8, 8)
    assert torch.equal(original, candidate)
    assert original.data_ptr() != candidate.data_ptr()
    assert torch.count_nonzero(original).item() > 0
    # Every receipt must remain a small serializable artifact, not retain tensors.
    json.dumps(record, allow_nan=False)


def test_transient_state_failure_at_124_cannot_be_hidden_by_correct_final_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_compare = benchmark.compare_history_state
    compared: list[int] = []

    def compare_with_failure(
        original: TemporalPrefixEventHistory, candidate: TemporalPrefixEventHistory
    ) -> dict[str, Any]:
        receipt = real_compare(original, candidate)
        compared.append(candidate._total_buckets)
        if candidate._total_buckets == 124:
            receipt["passed"] = False
            receipt["buffers"]["_counts"]["exact"] = False
        return receipt

    monkeypatch.setattr(benchmark, "compare_history_state", compare_with_failure)
    record, (original, candidate) = benchmark.run_backend_case(_events(738), ROI, 738, _options())
    assert compared == list(range(120, 241, 4))
    assert record["passed"] is False
    assert record["checkpoints"][1]["state"]["passed"] is False
    assert record["checkpoints"][-1]["state"]["passed"] is True
    assert torch.equal(original, candidate)


def test_partial_query_unexpectedly_succeeding_fails_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_query = SubpacketTemporalPrefixEventHistory.query

    def allow_partial_query(
        self: SubpacketTemporalPrefixEventHistory, roi: tuple[float, float, float, float]
    ) -> torch.Tensor:
        if self.subpacket_phase:
            return torch.zeros((4, 3, 12, self.roi_size, self.roi_size), device="cpu")
        return original_query(self, roi)

    monkeypatch.setattr(SubpacketTemporalPrefixEventHistory, "query", allow_partial_query)
    record, _ = benchmark.run_backend_case(_events(738), ROI, 738, _options())
    assert record["passed"] is False
    assert all(row["passed"] is False for row in record["partial_guards"])
    assert all(row["state"]["passed"] for row in record["checkpoints"])


def test_state_receipt_compares_all_six_buffers_and_all_five_scalars() -> None:
    original, candidate = _initialized_pair()
    receipt = benchmark.compare_history_state(original, candidate)
    assert tuple(benchmark.STATE_TENSORS) == STATE_TENSORS
    assert tuple(benchmark.STATE_SCALARS) == STATE_SCALARS
    assert receipt["passed"] is True
    assert set(receipt["buffers"]) == set(STATE_TENSORS)
    assert set(receipt["scalars"]) == set(STATE_SCALARS)
    for name in STATE_TENSORS:
        tensor = getattr(original, name)
        row = receipt["buffers"][name]
        assert row["exact"] is True
        assert row["bytes"] == tensor.numel() * tensor.element_size()
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("name", STATE_TENSORS)
def test_state_receipt_rejects_mutation_in_each_retained_buffer(name: str) -> None:
    original, candidate = _initialized_pair()
    getattr(candidate, name).reshape(-1)[0] += 1
    receipt = benchmark.compare_history_state(original, candidate)
    assert receipt["passed"] is False
    assert receipt["buffers"][name]["exact"] is False


@pytest.mark.parametrize("name", STATE_SCALARS)
def test_state_receipt_rejects_each_scalar_phase_or_counter_mismatch(name: str) -> None:
    original, candidate = _initialized_pair()
    setattr(candidate, name, getattr(candidate, name) + 1)
    assert benchmark.compare_history_state(original, candidate)["passed"] is False


@pytest.mark.parametrize(
    "free,headroom,required,expected",
    [
        (299, 100, 300, False),
        (300, 100, 300, True),
        (301, 100, 300, True),
        (0, 100, 300, False),
        (200, 0, 200, True),
        (199, 0, 200, False),
    ],
)
def test_memory_gate_requires_two_histories_and_headroom_at_exact_boundary(
    free: int, headroom: int, required: int, expected: bool
) -> None:
    receipt = benchmark.memory_preflight(100, free, 400, headroom)
    assert receipt["required_bytes"] == required
    assert receipt["passed"] is expected
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 300, 400, 100),
        (-1, 300, 400, 100),
        (True, 300, 400, 100),
        (100, -1, 400, 100),
        (100, 401, 400, 100),
        (100, 300, 0, 100),
        (100, 300, 400, -1),
        (100, 300.0, 400, 100),
        (100, 300, 400, None),
    ],
)
def test_memory_gate_rejects_invalid_or_unknown_measurements(arguments: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        benchmark.memory_preflight(*arguments)


@pytest.mark.parametrize("normalization", ["sort", "topk"])
def test_meta_memory_estimate_matches_cpu_constructor_without_cuda(
    monkeypatch: pytest.MonkeyPatch, normalization: str
) -> None:
    for name in ("synchronize", "current_stream", "mem_get_info", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, _forbid_cuda)
    options = _options(normalization)
    before = dict(options)
    actual = benchmark.estimate_history_bytes(options)
    expected = TemporalPrefixEventHistory(**options).allocated_bytes
    assert actual == expected > 0
    assert options == before


def test_raw_slices_preserve_half_open_boundaries_and_all_field_alignment() -> None:
    times = np.array([99, 100, 100, 104, 105, 105, 109, 110], dtype=np.int64)
    raw: Raw = {
        "x": np.arange(len(times), dtype=np.int32),
        "y": np.arange(len(times), dtype=np.int32) + 10,
        "t": times,
        "p": np.arange(len(times), dtype=np.int64) - 4,
    }
    first = benchmark.slice_raw(raw, 100, 105)
    second = benchmark.slice_raw(raw, 105, 110)
    for key in raw:
        np.testing.assert_array_equal(first[key], raw[key][1:4])
        np.testing.assert_array_equal(second[key], raw[key][4:7])
        np.testing.assert_array_equal(
            np.concatenate((first[key], second[key])), benchmark.slice_raw(raw, 100, 110)[key]
        )
    for key, value in benchmark.slice_raw(raw, 110, 115).items():
        np.testing.assert_array_equal(value, raw[key][7:])
    assert all(len(value) == 0 for value in benchmark.slice_raw(raw, 111, 115).values())
