"""TRAIN40 retains official targets and refuses media outside its training namespace."""

from pathlib import Path

import numpy as np
import pytest

from operational.train40_system.data_audit import describe_row, safe_media_path


def row():
    return {
        "sequence_id": "sequence",
        "sample_token": "token",
        "timestamp_us": 200000,
        "events_path": "data/train/sequence/events.h5",
        "frame_timestamps_us": [100000, 200000],
        "event_windows_us": [[100000, 200000], [200000, 300000]],
        "boxes_xyxy": [[10, 20, 30, 50], [12, 20, 35, 55]],
        "rgb_shard_paths": ["data/train/sequence/rgb_shards/rgb-00000.tar"] * 2,
        "rgb_member_paths": ["rgb/first.png", "rgb/second.png"],
        "ttc": 1.0,
        "frame_ttc": [1.1, 1.0],
        "box3d_h": 2.0,
        "box3d_Fcam": np.ones((2, 8, 3)) * 10,
    }


@pytest.mark.parametrize(
    "relative",
    [
        "data/test/sequence/events.h5",
        "data/train/other/events.h5",
        "data/train/sequence/../../../test/events.h5",
        "C:/data/train/sequence/events.h5",
    ],
)
def test_media_cannot_escape_train(relative, tmp_path):
    with pytest.raises(ValueError):
        safe_media_path(relative, "sequence", tmp_path)


def test_target_is_official_endpoint_and_not_input(tmp_path):
    first = describe_row(row(), tmp_path)
    changed = row()
    changed["ttc"] = -8.0
    changed["frame_ttc"] = [-3.1, -3.0]
    second = describe_row(changed, tmp_path)
    assert first["target"] == 1.0
    assert second["target"] == -3.0
    assert first["windows"] == second["windows"]
    np.testing.assert_array_equal(first["square"], second["square"])
    np.testing.assert_array_equal(first["motion"], second["motion"])
    assert first["proxy"]
    assert first["windows"][0] == (0, 100000)


@pytest.mark.parametrize("target", [0.0, 0.05, 0.1, float("nan"), float("inf")])
def test_invalid_phase_target_is_not_relabelled(target, tmp_path):
    value = row()
    value["frame_ttc"] = [1.1, target]
    with pytest.raises(ValueError):
        describe_row(value, tmp_path)


def test_unaligned_metadata_fails(tmp_path):
    value = row()
    value["rgb_member_paths"] = ["rgb/first.png"]
    with pytest.raises(ValueError, match="aligned"):
        describe_row(value, tmp_path)


def test_valid_media_is_relative_to_selected_sequence(tmp_path):
    assert (
        safe_media_path("data/train/sequence/events.h5", "sequence", tmp_path)
        == (Path(tmp_path) / "data/train/sequence/events.h5").resolve()
    )
