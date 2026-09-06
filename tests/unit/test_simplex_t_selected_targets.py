"""Projection prevents unrelated labels from entering selected TRAIN supervision."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t.selected_targets import load_selected_train_targets


def fixture(tmp_path):
    path = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "sample_token": ["second", "closed", "first"],
            "sequence_id": ["train", "protected", "train"],
            "timestamp_us": [200, 300, 100],
            "ttc": [3.0, np.nan, 1.0],
            "box3d_h": [9.0, 9.0, 9.0],
        }
    ).to_parquet(path)
    return path, dict(
        expected_sha256=compute_file_hash(str(path)),
        tokens=np.array(["first", "second"]),
        sequences=np.array(["train", "train"]),
        timestamps_us=np.array([100, 200], np.int64),
        allowed_sequences={"train"},
        pool="DENSE_OLD",
    )


def test_selected_projection_and_original_order(tmp_path, monkeypatch):
    path, args = fixture(tmp_path)
    read, calls = pd.read_parquet, []

    def projected(*positional, **keywords):
        calls.append(keywords)
        return read(*positional, **keywords)

    monkeypatch.setattr(pd, "read_parquet", projected)
    result = load_selected_train_targets(path, **args)
    assert result.target_ttc.tolist() == [1.0, 3.0]
    assert calls == [
        {
            "columns": ["sample_token", "sequence_id", "timestamp_us", "ttc"],
            "filters": [
                ("sample_token", "in", ["first", "second"]),
                ("sequence_id", "in", ["train"]),
            ],
        }
    ]


@pytest.mark.parametrize("failure", ["role", "duplicate", "hash", "pool"])
def test_invalid_authority_rejected_before_parquet_read(tmp_path, monkeypatch, failure):
    path, args = fixture(tmp_path)
    if failure == "role":
        args["allowed_sequences"] = {"different"}
    elif failure == "duplicate":
        args["tokens"][1] = "first"
    elif failure == "hash":
        args["expected_sha256"] = "a" * 64
    else:
        args["pool"] = "invented"

    def forbidden(*positional, **keywords):
        raise AssertionError("target read before identity/authority validation")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError):
        load_selected_train_targets(path, **args)


def test_missing_selected_query_is_not_dropped(tmp_path):
    path, args = fixture(tmp_path)
    args["tokens"][1] = "absent"
    with pytest.raises(ValueError, match="missing or foreign"):
        load_selected_train_targets(path, **args)
