"""Input contracts that protect causal competition inference."""

from __future__ import annotations

import numpy as np
import pytest

from operational.evttc_competition.geometry import Calibration
from operational.evttc_competition.prepare import (
    event_window_plan,
    prepare_query,
    rgb_frame_plan,
)


def test_startup_windows_partition_only_observed_prefix() -> None:
    plan = event_window_plan(20_266, 0)
    assert len(plan) == 1
    assert plan[0][0][0] == 0
    assert plan[0][-1][-1] == 20_266
    assert all(end > start >= 0 for start, end in plan[0])
    assert sum(end - start for start, end in plan[0]) == 20_266
    assert event_window_plan(1, 0) == []
    assert event_window_plan(20_000, 30_000) == []


def test_native_event_history_has_complete_support_within_budget() -> None:
    plan = event_window_plan(1_000_000, 0)
    assert len(plan) == 4
    assert min(start for windows in plan for start, _ in windows) == 400_000
    assert max(end for windows in plan for _, end in windows) == 1_000_000
    assert all(end - start == 100_000 for windows in plan for start, end in windows)
    assert len(event_window_plan(350_000, 0)) == 4


def test_startup_uses_fixed_history_anchors_with_positive_real_windows() -> None:
    plan = event_window_plan(120_245, 2_945)
    assert [windows[-1][1] for windows in plan] == [20_245, 120_245]
    assert all(windows[0][0] == 2_945 for windows in plan)
    for query in (2, 3, 100_002, 100_003, 299_999, 300_000, 599_999, 600_000):
        planned = event_window_plan(query, 0)
        assert [windows[-1][1] for windows in planned] == [
            query - lag for lag in (300_000, 200_000, 100_000, 0) if query - lag >= 3
        ]
        for windows in planned:
            assert len(windows) == 3
            assert all(0 <= start < end <= query for start, end in windows)
            assert all(
                left[1] == right[0] for left, right in zip(windows, windows[1:], strict=False)
            )
            assert query - windows[0][0] <= 650_000


def test_slider_plan_uses_real_measured_cadence_and_never_future_frames() -> None:
    clock = np.arange(50, dtype=np.int64) * 40_082
    groups = rgb_frame_plan(clock, 400_820)
    assert groups
    assert all(clock[index] <= 400_820 for group in groups for index in group)
    assert all(400_820 - clock[group[0]] <= 650_000 for group in groups)
    assert all(len(set(group)) == len(group) for group in groups)
    assert np.diff(clock[list(groups[-1])]).tolist() == [80_164, 80_164]
    assert rgb_frame_plan(clock, 0) == []
    assert rgb_frame_plan(clock, 40_082) == [(0, 1)]


class _Reader:
    sequence_id = "synthetic_no_targets"
    event_start_us = 0
    event_end_us = 1_000_001
    rgb_timestamps_us = np.arange(0, 1_000_001, 50_000, dtype=np.int64)

    def __init__(self, poison_future=False):
        timestamps = np.arange(0, 1_000_001, 100, dtype=np.int64)
        self.raw = np.column_stack(
            (
                50 + (timestamps // 100) % 20,
                50 + (timestamps // 700) % 20,
                timestamps,
                (timestamps // 100) % 2,
            )
        )
        self.poison_future = poison_future
        self.requests = []

    def read_events(self, start, end):
        self.requests.append((start, end))
        raw = self.raw.copy()
        if self.poison_future:
            raw[raw[:, 2] >= 350_000, :2] = 125
        return raw[(raw[:, 2] >= start) & (raw[:, 2] < end)]

    def read_rgb(self, index):
        if self.poison_future and self.rgb_timestamps_us[index] > 350_000:
            raise AssertionError("A future image was read")
        return np.full((128, 128, 3), index, np.uint8)


def _identity_calibration():
    return Calibration(
        np.array([100.0, 100.0, 64.0, 64.0]),
        np.zeros(4),
        (128, 128),
        np.array([100.0, 100.0, 64.0, 64.0]),
        np.zeros(4),
        (128, 128),
        np.eye(3),
        "synthetic",
        "identity",
    )


def test_preparation_is_invariant_to_future_events_frames_and_boxes() -> None:
    rows = [
        {"index": index, "anchor_us": index * 50_000, "box_xyxy": (40, 40, 80, 80)}
        for index in range(21)
    ]
    before = _Reader()
    after = _Reader(poison_future=True)
    poisoned_rows = [dict(row) for row in rows]
    for row in poisoned_rows:
        if row["anchor_us"] > 350_000:
            row["box_xyxy"] = (0, 0, 128, 128)
    a = prepare_query(before, rows, rows[7], _identity_calibration())
    b = prepare_query(after, poisoned_rows, poisoned_rows[7], _identity_calibration())
    assert all(end <= 350_000 for _, end in before.requests + after.requests)
    assert a.details == b.details
    assert len(a.events) == len(b.events) == 4
    for first, second in zip((*a.events, *a.rgb), (*b.events, *b.rgb), strict=True):
        np.testing.assert_array_equal(first.pixels, second.pixels)
    assert a.events[0].first_sensor_ts_us == 0
    assert a.events[-1].first_sensor_ts_us == 50_000
    assert all(item.available_us == 350_000 for item in a.events)


def test_real_empty_windows_are_counted_without_fabricating_history() -> None:
    reader = _Reader()
    reader.raw = np.empty((0, 4), dtype=np.int64)
    row = {"index": 0, "anchor_us": 20_266, "box_xyxy": (40, 40, 80, 80)}
    prepared = prepare_query(reader, [row], row, _identity_calibration())
    assert prepared.details["event_observations"][0]["roi_counts"] == [0, 0, 0]
    assert len(prepared.events) == 1
    assert prepared.details["event_observations"][0]["zero_support"] is True
    assert all(start >= 0 and end <= 20_266 for start, end in reader.requests)


def test_missing_recording_tail_is_not_treated_as_observed_empty_window() -> None:
    reader = _Reader()
    reader.event_end_us = 10_000
    row = {"index": 0, "anchor_us": 20_266, "box_xyxy": (40, 40, 80, 80)}
    prepared = prepare_query(reader, [row], row, _identity_calibration())
    assert not prepared.events
    assert not reader.requests
    assert prepared.details["event_coverage_reason"] == "query_after_recorded_event_coverage"


def test_crop_history_uses_chronology_not_csv_file_order() -> None:
    rows = [
        {"index": index, "anchor_us": index * 50_000, "box_xyxy": (40, 40, 80 + index, 80)}
        for index in range(8)
    ]
    original = prepare_query(_Reader(), rows, rows[-1], _identity_calibration())
    shuffled = [rows[index] for index in (7, 2, 6, 1, 5, 0, 4, 3)]
    reordered = prepare_query(_Reader(), shuffled, rows[-1], _identity_calibration())
    assert original.details == reordered.details
    for first, second in zip(original.events, reordered.events, strict=True):
        np.testing.assert_array_equal(first.pixels, second.pixels)


def test_duplicate_annotation_anchors_fail_without_track_identity() -> None:
    row = {"index": 0, "anchor_us": 20_266, "box_xyxy": (40, 40, 80, 80)}
    with pytest.raises(ValueError, match="Duplicate bbox anchors"):
        prepare_query(_Reader(), [row, dict(row)], row, _identity_calibration())
