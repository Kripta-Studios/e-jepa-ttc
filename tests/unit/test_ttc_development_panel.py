"""CPU contracts for the bounded TTC development panel."""

from __future__ import annotations

import pytest

from scripts.build_ttc_development_panel import (
    _unique_index,
    causal_prefix,
    evaluator_assignment,
    media_for_sequence,
    prediction_timing,
)


def media(token: str, start: int, end: int, sequence: str = "sequence-a") -> dict[str, object]:
    return {
        "sample_token": token,
        "sequence_id": sequence,
        "rgb_exposure_start_timestamp_us": start,
        "rgb_exposure_end_timestamp_us": end,
    }


def test_target_reference_is_exposure_start_and_ttc_is_not_shifted() -> None:
    query = {
        "timestamp_us": 10_100_000,
        "frame_timestamps_us": [10_000_000, 10_100_000],
        "event_windows_us": [[900_000, 1_000_000], [1_000_000, 1_100_000]],
    }
    result = prediction_timing(
        query,
        media("current", 1_100_000, 1_101_700),
        media("previous", 1_000_000, 1_001_700),
    )
    assert result["sensor_reference_us"] == 1_100_000
    assert result["prediction_available_cutoff_us"] == 1_101_700
    assert result["label_age_at_prediction_us"] == 1_700
    assert not result["current_rgb_fully_available_at_reference"]
    assert result["previous_fully_available_frame"] == "previous"
    assert result["ttc_target_shift_applied_us"] == 0


def test_timing_rejects_midpoint_or_end_as_unproved_reference() -> None:
    query = {
        "timestamp_us": 10_100_000,
        "frame_timestamps_us": [10_000_000, 10_100_000],
        "event_windows_us": [[900_000, 1_000_000], [1_000_000, 1_100_500]],
    }
    with pytest.raises(ValueError, match="exposure start"):
        prediction_timing(
            query,
            media("current", 1_100_000, 1_101_700),
            media("previous", 1_000_000, 1_001_700),
        )


def test_causal_prefix_is_shortest_suffix_with_no_future_frames() -> None:
    frames = [media(str(index), index * 100_000, index * 100_000 + 1_700) for index in range(9)]
    result = causal_prefix(frames, "8", 600_000)
    assert [row["sample_token"] for row in result] == ["2", "3", "4", "5", "6", "7", "8"]
    assert (
        int(result[-1]["rgb_exposure_end_timestamp_us"])
        - int(result[0]["rgb_exposure_end_timestamp_us"])
        == 600_000
    )


def test_causal_prefix_declares_missing_history_instead_of_shortening() -> None:
    frames = [media(str(index), index * 100_000, index * 100_000 + 1_700) for index in range(5)]
    with pytest.raises(ValueError, match="required causal history"):
        causal_prefix(frames, "4", 600_000)


def test_causal_history_and_previous_frame_are_sequence_local() -> None:
    first = [
        media(f"a-{index}", index * 100_000, index * 100_000 + 1_700, "a") for index in range(8)
    ]
    # These later-ending frames would contaminate both the prefix and previous-frame search
    # if a query used the combined release-wide table.
    second = [
        media(f"b-{index}", index * 100_000 + 50_000, index * 100_000 + 51_700, "b")
        for index in range(8)
    ]
    local = media_for_sequence([*first, *second], "a")
    prefix = causal_prefix(local, "a-7", 600_000)
    previous = max(
        (row for row in local if int(row["rgb_exposure_end_timestamp_us"]) <= 601_700),
        key=lambda row: int(row["rgb_exposure_end_timestamp_us"]),
    )
    assert {row["sequence_id"] for row in prefix} == {"a"}
    assert previous["sample_token"] == "a-6"


@pytest.mark.parametrize(
    "fields,name",
    [
        (("sequence_id", "rgb_member_path"), "sequence/member media"),
        (
            ("sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us"),
            "five-key TRAIN annotation",
        ),
    ],
)
def test_exact_metadata_indices_reject_duplicate_keys(fields, name) -> None:
    row = {
        "sequence_id": "sequence",
        "rgb_member_path": "rgb/frame.png",
        "sample_token": "query",
        "track_id": "track",
        "public_track_id": "public-track",
        "timestamp_us": 123,
    }
    with pytest.raises(ValueError, match=f"Duplicate {name}"):
        _unique_index([row, dict(row)], fields, name)


def test_evaluator_assignment_uses_geometry_only() -> None:
    result = evaluator_assignment(
        [(0, 0, 10, 10), (20, 0, 30, 10)],
        [(0.5, 0, 10.5, 10), (20.5, 0, 30.5, 10)],
    )
    assert [(row["reference_index"], row["predicted_index"]) for row in result["pairs"]] == [
        (0, 0),
        (1, 1),
    ]
    assert result["ambiguous_reference_indices"] == []
    assert result["ambiguous_predicted_indices"] == []


def test_evaluator_assignment_rejects_bidirectional_ambiguity() -> None:
    result = evaluator_assignment(
        [(0, 0, 8, 10), (2, 0, 10, 10)],
        [(0, 0, 10, 10)],
        ambiguity_margin=0.3,
    )
    assert result["pairs"] == []
    assert result["ambiguous_reference_indices"] == [0]
