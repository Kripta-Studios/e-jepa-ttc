"""Parity against the frozen adapter and bounded streaming-cache invariants."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from operational.evttc_transfer.inputs import prepare as reference_prepare
from operational.ttc_revision.inputs import EventPreparer


def fixture_row(path: Path, anchor: int = 900_000) -> dict:
    if not path.exists():
        rng = np.random.default_rng(24)
        t = np.sort(rng.integers(0, 1_500_000, 4000, dtype=np.int64))
        with h5py.File(path, "w") as handle:
            group = handle.create_group("prophesee/event_cam_left")
            for name, values in {
                "x": rng.integers(0, 1280, len(t), dtype=np.int16),
                "y": rng.integers(0, 720, len(t), dtype=np.int16),
                "t": t,
                "p": rng.integers(0, 2, len(t), dtype=np.int8),
                "ms_map_idx": np.searchsorted(t, np.arange(1502) * 1000),
            }.items():
                group.create_dataset(name, data=values)
    stat = path.stat()
    return {
        "raw_path": str(path),
        "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "sequence_id": "fixture",
        "anchor_us": anchor,
        "windows_us": [
            [anchor - 300000, anchor - 200000],
            [anchor - 200000, anchor - 100000],
            [anchor - 100000, anchor],
        ],
        "boxes_xyxy3": [[100, 120, 450, 470]] * 3,
        "square_xyxy": [100, 120, 450, 470],
    }


def test_preparation_matches_frozen_for_shifted_and_changed_roi(tmp_path):
    with EventPreparer() as fast:
        for anchor, shift in [(900000, 0), (950000, 0), (1000000, 70), (900000, 0)]:
            row = fixture_row(tmp_path / "events.hdf5", anchor)
            row["square_xyxy"] = [100 + shift, 120, 450 + shift, 470]
            expected = reference_prepare(row)
            result = fast.prepare(row, system="both")
            for key in ("own_events", "garl_events", "valid", "delta_t_s"):
                np.testing.assert_array_equal(result[key], expected[key])
            assert result["diagnostics"]["unique_voxel_windows"] == 12
            assert fast.retained_bytes <= fast.cache_bytes


def test_system_specific_preparation_and_raw_cache(tmp_path):
    row = fixture_row(tmp_path / "events.hdf5")
    with EventPreparer() as fast:
        own = fast.prepare(row, system="h8")
        assert "garl_events" not in own
        assert own["diagnostics"]["requested_span_us"] == 650000
        garl = fast.prepare(row, system="garl")
        assert "own_events" not in garl
        assert garl["diagnostics"]["requested_span_us"] == 200000
        assert garl["diagnostics"]["raw_cache_hit"]
        fast.clear_raw_cache()
        assert fast.retained_bytes == 0


def test_uncached_streaming_crop_is_exact_at_fractional_sensor_edges(tmp_path):
    row = fixture_row(tmp_path / "events.hdf5")
    row["square_xyxy"] = [-0.5, 119.7, 450.2, 721.8]
    expected = reference_prepare(row)["own_events"]
    with EventPreparer(cache_bytes=0) as fast:
        result = fast.prepare(row)
        np.testing.assert_array_equal(result["own_events"], expected)
        assert result["diagnostics"]["unique_voxel_windows"] == 12
        assert fast.retained_bytes == 0


def test_invalid_windows_and_changed_raw_are_rejected(tmp_path):
    row = fixture_row(tmp_path / "events.hdf5")
    with EventPreparer() as fast:
        row["windows_us"][1][0] += 1
        with pytest.raises(ValueError, match="windows"):
            fast.prepare(row)
        row = fixture_row(tmp_path / "events.hdf5")
        row["raw_stat"]["size_bytes"] += 1
        with pytest.raises(ValueError, match="stat"):
            fast.prepare(row)


def test_small_cache_does_not_retain_oversized_payload(tmp_path):
    row = fixture_row(tmp_path / "events.hdf5")
    with EventPreparer(cache_bytes=32) as fast:
        actual = fast.prepare(row, system="garl")
        np.testing.assert_array_equal(actual["garl_events"], reference_prepare(row)["garl_events"])
        assert fast.retained_bytes == 0
