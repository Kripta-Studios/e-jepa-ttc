import numpy as np
import pandas as pd
import pytest

from operational.independent_refit.teacher_crosswalk import (
    _verify_physical_rows,
    validate_crosswalk_frame,
)


def _frames():
    population = pd.DataFrame(
        {
            "sample_token": ["a", "b"],
            "fit_ordinal": [0, 1],
            "public_source_ordinal": [7, 2],
            "group_id": ["g1", "g2"],
            "sequence_id": ["g1", "g2"],
            "track_id": ["t1", "t2"],
            "public_track_id": ["t1", "t2"],
            "timestamp_us": [1, 2],
        }
    )
    crosswalk = population.copy()
    crosswalk["teacher_ordinal"] = pd.Series([10, 11], dtype="int64")
    crosswalk["teacher_shard"] = ["shard_00000.npz", "shard_00000.npz"]
    crosswalk["teacher_shard_index"] = [0, 0]
    crosswalk["teacher_local_row"] = [0, 1]
    crosswalk["teacher_rgb_sha256_0"] = ["a" * 64, "b" * 64]
    crosswalk["teacher_rgb_sha256_1"] = ["c" * 64, "d" * 64]
    crosswalk["raw_rgb_member_0"] = ["m0", "m0"]
    crosswalk["raw_rgb_member_1"] = ["m1", "m1"]
    return population, crosswalk


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("sample_token", "alien"),
        ("public_source_ordinal", 99),
        ("teacher_ordinal", -1),
        ("teacher_shard", "shard_99999.npz"),
        ("teacher_local_row", 32),
        ("teacher_rgb_sha256_0", "bad"),
    ],
)
def test_crosswalk_rejects_identity_mutations(column, value):
    population, crosswalk = _frames()
    crosswalk.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_crosswalk_frame(crosswalk, population)


def test_crosswalk_rejects_duplicate_or_missing_rows():
    population, crosswalk = _frames()
    duplicate = pd.concat([crosswalk.iloc[[0]], crosswalk.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_crosswalk_frame(duplicate, population)
    with pytest.raises(ValueError):
        validate_crosswalk_frame(crosswalk.iloc[[0]], population)


def test_crosswalk_accepts_distinct_raw_and_teacher_ordinals():
    population, crosswalk = _frames()
    validate_crosswalk_frame(crosswalk, population)


def test_actual_known_ordinal_namespace_regression():
    frame = pd.read_parquet(
        "output/independent_phase/teacher_crosswalk_v3/teacher_crosswalk.parquet"
    )
    row = frame.loc[frame["public_source_ordinal"] == 2102].iloc[0]
    assert row["sample_token"] == "pzJcg1OCfc_000002_12321000000"
    assert int(row["teacher_ordinal"]) == 71589
    assert row["teacher_shard"] == "shard_02237.npz"
    assert int(row["teacher_local_row"]) == 5


def test_cross_shard_coordinate_permutation_is_rejected(tmp_path):
    population, crosswalk = _frames()
    crosswalk["teacher_ordinal"] = pd.Series([0, 1], dtype="int64")
    crosswalk["teacher_shard"] = ["shard_00000.npz", "shard_00001.npz"]
    crosswalk["teacher_shard_index"] = [0, 1]
    crosswalk["teacher_local_row"] = [0, 0]
    crosswalk["teacher_rgb_sha256_0"] = ["a" * 64, "b" * 64]
    crosswalk["teacher_rgb_sha256_1"] = ["c" * 64, "d" * 64]
    for index, token in enumerate(("a", "b")):
        np.savez_compressed(
            tmp_path / f"shard_{index:05d}.npz",
            tokens=np.asarray([token]),
            ordinals=np.asarray([index], dtype=np.int64),
            rgb_sha256=np.asarray(
                [
                    [
                        crosswalk.loc[index, "teacher_rgb_sha256_0"],
                        crosswalk.loc[index, "teacher_rgb_sha256_1"],
                    ]
                ]
            ),
        )
    _verify_physical_rows(crosswalk, tmp_path)
    coordinate_columns = [
        "teacher_ordinal",
        "teacher_shard",
        "teacher_shard_index",
        "teacher_local_row",
        "teacher_rgb_sha256_0",
        "teacher_rgb_sha256_1",
    ]
    permuted = crosswalk.copy()
    permuted[coordinate_columns] = permuted.loc[[1, 0], coordinate_columns].to_numpy()
    with pytest.raises(ValueError, match="shard/local row"):
        _verify_physical_rows(permuted, tmp_path)
