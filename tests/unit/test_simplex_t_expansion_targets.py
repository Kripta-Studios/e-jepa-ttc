"""Targets attach to frozen queries; invalid targets never select a replacement pool."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t.expansion_targets import align_expansion_targets, load_expansion_targets


def fixture():
    table = pd.DataFrame(
        {
            "sample_token": ["a", "b", "unused"],
            "sequence_id": ["one", "two", "one"],
            "timestamp_us": [10, 20, 30],
            "ttc": [1.0, 2.0, 3.0],
        }
    )
    kwargs = {
        "tokens": np.array(["b", "a"]),
        "sequences": np.array(["two", "one"]),
        "timestamps_us": np.array([20, 10], np.int64),
    }
    return table, kwargs


def test_pinned_parquet_loader_preserves_pool_order(tmp_path):
    table, kwargs = fixture()
    path = tmp_path / "train.parquet"
    table.to_parquet(path, index=False)
    result = load_expansion_targets(path, expected_sha256=compute_file_hash(str(path)), **kwargs)
    assert result.target_ttc.tolist() == [2.0, 1.0]
    assert result.target_phase.dtype == np.float32
    assert result.mass.tolist() == [0.5, 0.5]
    before = result.identity_sha256
    table.loc[0, "ttc"] = 4.0
    table.to_parquet(path, index=False)
    changed = load_expansion_targets(path, expected_sha256=compute_file_hash(str(path)), **kwargs)
    assert changed.identity_sha256 != before
    assert kwargs["tokens"].tolist() == ["b", "a"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "sequence", "time", "nan", "zero"])
def test_invalid_supervision_fails_without_dropping_queries(change):
    table, kwargs = fixture()
    if change == "missing":
        table = table.iloc[1:]
    elif change == "duplicate":
        table.loc[2, "sample_token"] = "a"
    elif change == "sequence":
        table.loc[0, "sequence_id"] = "protected"
    elif change == "time":
        table.loc[0, "timestamp_us"] = 11
    elif change == "nan":
        table.loc[0, "ttc"] = np.nan
    else:
        table.loc[0, "ttc"] = 0
    with pytest.raises(ValueError):
        align_expansion_targets(table, label_sha256="0" * 64, **kwargs)
    assert kwargs["tokens"].tolist() == ["b", "a"]


def test_wrong_pin_rejected_before_parquet_decode(tmp_path):
    path = tmp_path / "not_parquet"
    path.write_bytes(b"not a table")
    _, kwargs = fixture()
    with pytest.raises(ValueError, match="authoritative pin"):
        load_expansion_targets(path, expected_sha256="0" * 64, **kwargs)
