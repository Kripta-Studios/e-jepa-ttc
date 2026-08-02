from __future__ import annotations

import pandas as pd
import pytest

from scripts.plan_garlttc_rotating_cache import _shard_rows, identity_hash, row_identity


def _rows() -> pd.DataFrame:
    records = []
    for role, sequence in (("train", "train-a"), ("train", "train-b"), ("validation", "val-a")):
        for index in range(2 if sequence == "train-a" else 3):
            records.append(
                {
                    "sequence_id": sequence,
                    "sample_token": f"{sequence}-{index}",
                    "track_id": f"track-{index}",
                    "public_track_id": f"public-{index}",
                    "timestamp_us": index,
                    "role": role,
                }
            )
    return pd.DataFrame.from_records(records)


def test_rotating_plan_partitions_rows_without_overlap() -> None:
    frame = _rows()
    roles = {"train-a": "train", "train-b": "train", "val-a": "validation"}
    shards, counts = _shard_rows(
        frame,
        roles,
        shard_size=2,
        bytes_per_sample=100.0,
    )

    assert counts == {"train": 5, "validation": 3}
    assert sum(int(shard["row_count"]) for shard in shards) == len(frame)
    assert len({shard["row_identity_sha256"] for shard in shards}) == len(shards)
    assert all(shard["scratch_deletable_after_hash"] is True for shard in shards)
    assert identity_hash([row_identity(row) for row in frame.to_dict("records")])


def test_rotating_plan_rejects_duplicate_identity() -> None:
    frame = _rows()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _shard_rows(
            duplicate,
            {"train-a": "train", "train-b": "train", "val-a": "validation"},
            shard_size=4,
            bytes_per_sample=100.0,
        )
