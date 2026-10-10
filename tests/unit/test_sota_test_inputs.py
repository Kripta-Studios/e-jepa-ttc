"""Native official submission input contracts without test labels or raw files."""

from pathlib import Path

import numpy as np
import pytest

from operational.sota_evidence.test_inputs import describe_input, supported_job


def metadata() -> tuple[dict, dict]:
    """Two intentionally different clocks and slightly jittered raw intervals."""
    row = {
        "sequence_id": "seq",
        "sample_token": "token",
        "timestamp_us": 12_140_300_000,
        "frame_timestamps_us": [12_140_200_000, 12_140_300_000],
        "events_path": "data/test/seq/events.h5",
        "event_windows_us": [[720578076, 720678077], [720678077, 720778078]],
        "boxes_xyxy": [[10, 20, 30, 50], [9, 18, 31, 52]],
        "rgb_member_paths": ["rgb/a.png", "rgb/b.png"],
    }
    exposures = {
        ("seq", name): {
            "rgb_exposure_start_timestamp_us": end,
            "rgb_exposure_end_timestamp_us": end + 5000,
        }
        for name, end in zip(row["rgb_member_paths"], [720678077, 720778078], strict=True)
    }
    return row, exposures


def test_sensor_clock_and_roi_availability(tmp_path: Path) -> None:
    row, exposures = metadata()
    job = describe_input(row, exposures, tmp_path, split="test")
    assert job["anchor"] == 720778078
    assert job["available"] == 720783078
    assert job["delta"] == pytest.approx(0.1)
    assert job["windows"][0] == [720478075, 720578076]
    assert job["windows"][2] == row["event_windows_us"][1]


def test_missing_leading_history_is_masked(tmp_path: Path) -> None:
    row, exposures = metadata()
    job = describe_input(row, exposures, tmp_path, split="test")
    result = supported_job(job, 720378076, 721000000)
    np.testing.assert_array_equal(result["valid"], [False] * 6 + [True] * 2)
    with pytest.raises(ValueError, match="complete raw"):
        supported_job(job, 720500000, 721000000)
    with pytest.raises(ValueError, match="complete raw"):
        supported_job(job, 0, 720700000)


def test_input_never_reads_targets(tmp_path: Path) -> None:
    row, exposures = metadata()
    original = describe_input(row, exposures, tmp_path, split="test")
    row.update(ttc="POISON", frame_ttc=object(), track_id="irrelevant")
    assert describe_input(row, exposures, tmp_path, split="test") == original


def test_exposure_clock_mismatch_rejected(tmp_path: Path) -> None:
    row, exposures = metadata()
    exposures[("seq", "rgb/a.png")]["rgb_exposure_start_timestamp_us"] += 1
    with pytest.raises(ValueError, match="clock disagree"):
        describe_input(row, exposures, tmp_path, split="test")


@pytest.mark.parametrize("path", ["data/train/seq/events.h5", "data/test/../events.h5"])
def test_cross_split_path_rejected(tmp_path: Path, path: str) -> None:
    row, exposures = metadata()
    row["events_path"] = path
    with pytest.raises(ValueError, match="split and sequence"):
        describe_input(row, exposures, tmp_path, split="test")
