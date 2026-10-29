"""Type compatibility edits retain CSV selection semantics and numeric NPZ bytes."""

from io import BytesIO, StringIO
from typing import cast

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "columns", [["sequence_id"], ["track_id", "sequence_id"], ["timestamp_us", "track_id"]]
)
def test_index_usecols_preserves_values_dtypes_and_source_order(columns):
    csv = "sequence_id,track_id,timestamp_us,ignored\na,001,100,nonsense\nb,002,200,other\n"
    before = pd.read_csv(StringIO(csv), usecols=columns, dtype={"track_id": str})
    after = pd.read_csv(StringIO(csv), usecols=pd.Index(columns), dtype={"track_id": str})
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_index_usecols_still_rejects_missing_required_column():
    for columns in (["missing"], pd.Index(["missing"])):
        with pytest.raises(ValueError, match="Usecols do not match columns"):
            pd.read_csv(StringIO("present\n1\n"), usecols=columns)


def test_explicit_no_pickle_keeps_numeric_and_unicode_npz_bytes():
    arrays = {
        "features145": np.arange(290, dtype=np.float32).reshape(2, 145),
        "known": np.array([[True, False], [False, True]]),
        "tokens": np.array(["first", "second"]),
    }
    old, new = BytesIO(), BytesIO()
    np.savez_compressed(old, **arrays)
    np.savez_compressed(new, allow_pickle=False, **arrays)
    assert old.getvalue() == new.getvalue()


def test_fixed_tuple_annotations_do_not_transform_index_values():
    windows = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    bbox = np.array([1.0, 2.0, 3.0, 4.0])
    old_windows = tuple(tuple(int(v) for v in window) for window in windows)
    old_bbox = tuple(float(v) for v in bbox)
    assert (
        cast(tuple[tuple[int, int], tuple[int, int], tuple[int, int]], old_windows) is old_windows
    )
    assert cast(tuple[float, float, float, float], old_bbox) is old_bbox
