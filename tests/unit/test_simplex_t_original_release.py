"""Primitive-only source decoding and target-independent input projection."""

import pickle

import pytest

from e_jepa_ttc.simplex_t.original_release import input_observation, read_records


def test_target_and_depth_fields_do_not_enter_observation():
    row = dict(
        sequence_id="s",
        instance_id="i",
        file_name="a.png",
        bbox=[1, 2, 3, 4],
        rgb_exposure_start_timestamp_us=100,
        rgb_exposure_end_timestamp_us=110,
    )
    original = input_observation(row)
    row.update(ttc=float("nan"), velocity=object(), bbox_3d=object(), depth=-999)
    assert input_observation(row) == original
    assert original["boxes_xyxy"] == [1, 2, 4, 6]


def test_data_only_decoding(tmp_path):
    path = tmp_path / "records.pkl"
    path.write_bytes(pickle.dumps([{"bbox": [1, 2, 3, 4]}]))
    assert read_records(path) == [{"bbox": [1, 2, 3, 4]}]


def test_global_pickle_refused_without_execution(tmp_path):
    path = tmp_path / "records.pkl"
    path.write_bytes(pickle.dumps(eval))
    with pytest.raises(ValueError, match="non-data"):
        read_records(path)
