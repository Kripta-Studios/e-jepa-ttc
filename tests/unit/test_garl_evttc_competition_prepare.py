from __future__ import annotations

import numpy as np

from operational.evttc_competition.geometry import Calibration
from operational.garl_evttc_competition.prepare import POLICY, prepare_query


class FakeReader:
    def __init__(
        self,
        *,
        event_start: int = 0,
        event_end: int = 1_000_001,
        rgb_clock: tuple[int, ...] = (0, 80_000, 160_000, 240_000),
        empty: bool = False,
    ) -> None:
        self.event_start_us = event_start
        self.event_end_us = event_end
        self.rgb_timestamps_us = np.asarray(rgb_clock, np.int64)
        self.empty = empty

    def read_events(self, start_us: int, end_us: int) -> np.ndarray:
        if self.empty:
            return np.empty((0, 4), np.int64)
        times = np.linspace(start_us, end_us - 1, 8, dtype=np.int64)
        return np.column_stack((np.full(8, 50), np.full(8, 50), times, np.arange(8) % 2)).astype(
            np.int64
        )

    def read_rgb(self, index: int) -> np.ndarray:
        return np.full((100, 100, 3), 20 + index, np.uint8)


def _calibration() -> Calibration:
    return Calibration(
        event_intrinsics=np.asarray((100.0, 100.0, 50.0, 50.0)),
        event_distortion=np.zeros(4),
        event_resolution=(100, 100),
        rgb_intrinsics=np.asarray((100.0, 100.0, 50.0, 50.0)),
        rgb_distortion=np.zeros(4),
        rgb_resolution=(100, 100),
        rotation_event_to_rgb=np.eye(3),
        source_kind="test",
        transform_source="identity",
    )


def _rows(query: int) -> list[dict[str, object]]:
    return [
        {
            "anchor_us": query,
            "row_index": 0,
            "index": 1,
            "box_xyxy": (30.0, 30.0, 70.0, 70.0),
        }
    ]


def test_event_mature_uses_two_native_windows() -> None:
    reader = FakeReader()
    result = prepare_query(reader, _rows(400_000), _rows(400_000)[0], _calibration(), "event")
    assert result.reason == "available" and result.sensor is not None
    assert result.sensor.shape == (40, 128, 128)
    assert result.delta_t_s == 0.1
    assert [item["window_us"] for item in result.details["event_windows"]] == [
        [200_000, 300_000],
        [300_000, 400_000],
    ]
    assert result.details["polarity_consumed"] is False
    assert POLICY["eap_x_plus_5_not_transferred"] is True
    assert POLICY["targets_used"] is False
    assert POLICY["missing_rgb_pair"] == "variant unavailable/null"
    assert POLICY["nonfinite_or_singular_height_ratio_output"] == "variant unavailable/null"
    assert "no accuracy" in str(POLICY["comparison_scope"])


def test_event_startup_partitions_only_observed_prefix() -> None:
    reader = FakeReader(event_start=10_000)
    row = _rows(70_000)[0]
    result = prepare_query(reader, [row], row, _calibration(), "event")
    assert result.reason == "available"
    assert result.delta_t_s == 0.03
    assert result.details["observed_prefix_adaptation"] is True
    assert [item["window_us"] for item in result.details["event_windows"]] == [
        [10_000, 40_000],
        [40_000, 70_000],
    ]


def test_empty_raw_event_window_is_unavailable_not_fabricated() -> None:
    reader = FakeReader(empty=True)
    row = _rows(400_000)[0]
    result = prepare_query(reader, [row], row, _calibration(), "event")
    assert result.sensor is None and result.delta_t_s is None
    assert result.reason == "raw_event_window_empty"


def test_visual_selects_two_distinct_causal_frames_and_actual_delta() -> None:
    reader = FakeReader(rgb_clock=(0, 80_000, 160_000, 260_000))
    row = _rows(250_000)[0]
    result = prepare_query(reader, [row], row, _calibration(), "visual")
    assert result.reason == "available" and result.sensor is not None
    assert result.sensor.shape == (6, 128, 128)
    assert result.details["rgb"]["frame_indices"] == [1, 2]
    assert result.details["rgb"]["timestamps_us"] == [80_000, 160_000]
    assert result.delta_t_s == 0.08
    assert result.details["variable_cadence_transfer"] is True


def test_full_aligns_adjacent_event_intervals_to_real_rgb_frames() -> None:
    reader = FakeReader(event_start=10_000, rgb_clock=(0, 80_000, 160_000))
    row = _rows(160_000)[0]
    result = prepare_query(reader, [row], row, _calibration(), "full")
    assert result.reason == "available" and result.sensor is not None
    assert result.sensor.shape == (46, 128, 128)
    assert result.delta_t_s == 0.08
    assert [item["window_us"] for item in result.details["event_windows"]] == [
        [10_000, 80_000],
        [80_000, 160_000],
    ]
    assert result.details["observed_prefix_adaptation"] is True


def test_full_caps_jittered_rgb_interval_support_at_native_100ms() -> None:
    reader = FakeReader(rgb_clock=(0, 100_024, 200_048))
    row = _rows(200_048)[0]
    result = prepare_query(reader, [row], row, _calibration(), "full")
    assert result.reason == "available" and result.sensor is not None
    assert result.delta_t_s == 0.100024
    assert [item["window_us"] for item in result.details["event_windows"]] == [
        [24, 100_024],
        [100_048, 200_048],
    ]
    assert result.details["event_support_us"] == 100_000
    assert result.details["actual_rgb_delta_minus_event_support_us"] == 24
    assert result.details["event_support_ends_at_each_rgb_endpoint"] is True


def test_rgb_never_duplicates_or_uses_future_frame() -> None:
    reader = FakeReader(rgb_clock=(100_000, 200_000))
    row = _rows(150_000)[0]
    result = prepare_query(reader, [row], row, _calibration(), "visual")
    assert result.sensor is None
    assert result.reason == "fewer_than_two_real_causal_rgb_frames"


def test_each_endpoint_uses_latest_box_available_at_that_endpoint() -> None:
    reader = FakeReader()
    rows = [
        {"anchor_us": anchor, "row_index": index, "index": index + 1, "box_xyxy": box}
        for index, (anchor, box) in enumerate(
            (
                (250_000, (25.0, 25.0, 65.0, 65.0)),
                (350_000, (30.0, 30.0, 70.0, 70.0)),
                (400_000, (35.0, 35.0, 75.0, 75.0)),
            )
        )
    ]
    result = prepare_query(reader, rows, rows[-1], _calibration(), "event")
    sources = result.details["bbox_sources"]
    assert [item["endpoint_us"] for item in sources] == [300_000, 400_000]
    assert [item["anchor_us"] for item in sources] == [250_000, 400_000]
    assert all(item["fallback"] is False for item in sources)


def test_past_endpoint_box_fallback_is_query_available_and_never_future_query() -> None:
    reader = FakeReader()
    row = _rows(400_000)[0]
    result = prepare_query(reader, [row], row, _calibration(), "event")
    first = result.details["bbox_sources"][0]
    assert first["endpoint_us"] == 300_000
    assert first["anchor_us"] == 400_000
    assert first["fallback"] is True
    assert first["available_us"] == 400_000
