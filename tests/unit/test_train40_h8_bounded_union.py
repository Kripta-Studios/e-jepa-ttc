"""CPU parity and capacity tests for bounded H8 mapped unions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_union as canonical_union
from operational.train40_system.h8_bounded_union import encode_union


class FakeReader:
    def __init__(self, raw: dict[str, np.ndarray], *, chunk_size: int = 7) -> None:
        self.raw = raw
        self.chunk_size = chunk_size
        self.calls: list[tuple[int, int]] = []

    def iter_window_chunks(
        self, start: int, end: int, *, chunk_events: int
    ) -> Iterator[dict[str, np.ndarray]]:
        del chunk_events
        self.calls.append((start, end))
        keep = (self.raw["t"] >= start) & (self.raw["t"] < end)
        indices = np.flatnonzero(keep)
        for offset in range(0, len(indices), self.chunk_size):
            selected = indices[offset : offset + self.chunk_size]
            yield {key: value[selected] for key, value in self.raw.items()}


def layout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = np.asarray([[400_000, 400_020], [400_020, 400_040], [400_040, 400_060]], np.int64)
    lags = np.arange(15, -1, -1, dtype=np.int64) * 50_000
    valid = np.concatenate((np.zeros(8, bool), np.ones(8, bool)))
    return windows, lags, valid


def raw_for_all_windows() -> dict[str, np.ndarray]:
    windows, lags, valid = layout()
    intervals = windows[None] - lags[:, None, None]
    timestamps = []
    for start, end in intervals[valid].reshape(-1, 2):
        timestamps.extend((int(start), int(start + 5), int(end - 1), int(end)))
    t = np.asarray(sorted(timestamps), np.int64)
    index = np.arange(len(t))
    return {
        # With +5 offset: 10 is ROI lower boundary, 29 is upper interior.
        "x": np.where(index % 3 == 0, 5, np.where(index % 3 == 1, 24, 25)).astype(np.int32),
        "y": np.where(index % 3 == 0, 20, np.where(index % 3 == 1, 39, 40)).astype(np.int32),
        "t": t,
        "p": np.asarray((-1, 0, 1), np.int8)[index % 3],
    }


def call(reader: FakeReader, cap: int, profile: dict | None = None) -> torch.Tensor:
    windows, lags, valid = layout()
    return encode_union(
        reader,
        windows,
        lags,
        valid,
        (10.0, 20.0, 30.0, 40.0),
        sequence_id="sequence-a",
        roi_size=16,
        event_pixel_diff=5.0,
        retained_bytes_max=cap,
        profile=profile,
    )


def test_recursive_groups_match_original_full_union_exactly() -> None:
    raw = raw_for_all_windows()
    windows, lags, valid = layout()
    reference = canonical_union(
        cast(Any, FakeReader(raw)),
        windows,
        lags,
        valid,
        (10.0, 20.0, 30.0, 40.0),
        sequence_id="sequence-a",
        roi_size=16,
        event_pixel_diff=5.0,
        retained_bytes_max=10_000_000,
    )
    reader = FakeReader(raw)
    profile: dict = {}
    actual = call(reader, 100, profile)
    assert torch.equal(actual, reference)
    assert actual.count_nonzero() > 0
    assert profile["split_count"] > 0
    assert profile["successful_groups"] > 1
    assert profile["read_attempts"] == len(reader.calls)
    assert profile["peak_group_retained_bytes"] <= 100
    assert sum(group["task_count"] for group in profile["groups"]) == 24


def test_exact_retained_byte_boundary_uses_one_group() -> None:
    windows, lags, valid = layout()
    intervals = windows[None] - lags[:, None, None]
    start = int(intervals[valid][0, 0, 0])
    raw = {
        "x": np.asarray([5, 24, 5, 24], np.int32),
        "y": np.asarray([20, 39, 20, 39], np.int32),
        "t": np.asarray([start, start + 1, start + 2, start + 3], np.int64),
        "p": np.asarray([-1, 0, 1, -1], np.int8),
    }
    reader = FakeReader(raw)
    profile: dict = {}
    result = call(reader, 4 * 17, profile)
    assert result.count_nonzero() > 0
    assert profile["read_attempts"] == 1
    assert profile["successful_groups"] == 1
    assert profile["peak_group_retained_bytes"] == 4 * 17


def test_single_oversized_window_preserves_resource_pause() -> None:
    windows, lags, valid = layout()
    intervals = windows[None] - lags[:, None, None]
    start, end = intervals[15, 2]
    t = np.arange(int(start), int(end), dtype=np.int64)
    raw = {
        "x": np.full(len(t), 5, np.int32),
        "y": np.full(len(t), 20, np.int32),
        "t": t,
        "p": np.ones(len(t), np.int8),
    }
    with pytest.raises(RuntimeError, match="RESOURCE_PAUSE:single window retained byte cap"):
        call(FakeReader(raw), 100)


@pytest.mark.parametrize(
    "valid",
    [
        np.ones(16, bool),
        np.asarray([False] * 8 + [True, False, True, True, True, True, True, True]),
        np.zeros(16, bool),
    ],
)
def test_invalid_h8_slot_layout_is_rejected(valid: np.ndarray) -> None:
    windows, lags, _ = layout()
    with pytest.raises(ValueError):
        encode_union(
            FakeReader(raw_for_all_windows()),
            windows,
            lags,
            valid,
            (10.0, 20.0, 30.0, 40.0),
            sequence_id="sequence-a",
            roi_size=16,
            event_pixel_diff=5.0,
        )
