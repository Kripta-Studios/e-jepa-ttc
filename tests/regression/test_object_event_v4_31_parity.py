from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from e_jepa_ttc.data.event_v4_common_roi import CommonROIConfig, common_square, rasterize_common_roi
from scripts.build_object_event_v4_31_sanitized_cache import EventH5Reader, _intervals, _trajectory


def test_common_roi_half_open_and_scalar_invariance() -> None:
    events = {
        "x": np.array([0, 1, 2]),
        "y": np.array([0, 1, 2]),
        "t": np.array([0, 9, 10]),
        "p": np.array([1, -1, 1]),
    }
    value = rasterize_common_roi(events, (0, 0, 8, 8), start_us=0, end_us=10)
    assert (
        value.shape == (12, 128, 128)
        and torch.all(value[10] == value[10, 0, 0])
        and torch.all(value[11] == value[11, 0, 0])
    )
    assert value[10, 0, 0].item() == pytest.approx(np.log1p(2))


def test_common_square_v430_parameters() -> None:
    assert common_square(
        ((0, 0, 4, 4), (4, 4, 8, 8)), CommonROIConfig(minimum_edge=8, margin_fraction=0.25)
    ) == (-2.0, -2.0, 10.0, 10.0)


def test_h5_reader_uses_bounded_ms_index_and_exact_precontext(tmp_path: Path) -> None:
    path = tmp_path / "events.h5"
    index = np.zeros(402, dtype=np.int64)
    index[100:201] = 0
    index[201:] = 3
    with h5py.File(path, "w") as handle:
        group = handle.create_group("events")
        group.create_dataset("x", data=np.array([1, 2, 3], dtype=np.int16))
        group.create_dataset("y", data=np.array([1, 2, 3], dtype=np.int16))
        group.create_dataset("t", data=np.array([100000, 150000, 200000], dtype=np.int64))
        group.create_dataset("p", data=np.array([1, 0, 1], dtype=np.int8))
        handle.create_dataset("ms_to_idx", data=index)
    intervals = _intervals([[200000, 300000], [300000, 400000]])
    assert intervals[0] == (100000, 200000)
    reader = EventH5Reader(path)
    try:
        values = reader.read(intervals)
    finally:
        reader.close()
    assert values["t"].tolist() == [100000, 150000, 200000]
    with pytest.raises(ValueError, match="non-overlapping"):
        _intervals([[200000, 300000], [250000, 400000]])


def test_forbidden_evttc_alias_is_rejected_before_hdf5_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_open(*_: object, **__: object) -> dict[str, np.ndarray]:
        raise AssertionError("forbidden path reached HDF5 reader")

    monkeypatch.setattr(
        "scripts.build_object_event_v4_31_sanitized_cache._read_events", must_not_open
    )
    row = {
        "event_windows_us": [[200000, 300000], [300000, 400000]],
        "boxes_xyxy": [[0, 0, 8, 8], [0, 0, 8, 8]],
        "events_path": "EvTTC_dataset/forbidden.h5",
    }
    with pytest.raises(PermissionError):
        _trajectory(row, {}, tmp_path)


def test_development_alias_and_invalid_polarity_fail_before_voxelization(tmp_path: Path) -> None:
    from e_jepa_ttc.data.object_event_v4_31 import reject_forbidden_path

    with pytest.raises(PermissionError):
        reject_forbidden_path("development_alias/events.h5")
    path = tmp_path / "bad_polarity.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("events")
        for key, value in {
            "x": np.array([1]),
            "y": np.array([1]),
            "t": np.array([100000]),
            "p": np.array([2]),
        }.items():
            group.create_dataset(key, data=value)
        handle.create_dataset("ms_to_idx", data=np.array([0] * 101 + [1] * 101))
    reader = EventH5Reader(path)
    with pytest.raises(ValueError, match="polarity"):
        reader.read(((100000, 100001), (100000, 100001), (100000, 100001)))
    reader.close()
