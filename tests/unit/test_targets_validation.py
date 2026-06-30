from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.data.targets import interpolate_ttc_seconds, load_ttc_csv
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.data.validation import normalize_polarity, validate_event_batch


def test_load_ttc_csv_and_interpolate(tmp_path: Path) -> None:
    csv_path = tmp_path / "ttc.csv"
    csv_path.write_text(
        "1 0.10 5.0 1.0 4.0\n2 0.20 4.8 1.0 3.0\n3 0.30 4.6 1.0 2.0\n",
        encoding="utf-8",
    )

    table = load_ttc_csv(csv_path)

    assert table["frame_id"].tolist() == [1, 2, 3]
    assert interpolate_ttc_seconds(table, 150_000) == pytest.approx(3.5)
    assert interpolate_ttc_seconds(table, 50_000) is None


def test_event_validation_rejects_nonmonotonic_timestamps() -> None:
    events = EventBatch(
        x=np.array([1, 2], dtype=np.int32),
        y=np.array([1, 2], dtype=np.int32),
        t_us=np.array([10, 9], dtype=np.int64),
        polarity=np.array([1, -1], dtype=np.int8),
        width=8,
        height=8,
        sequence_id="bad",
        t_start_us=0,
        t_end_us=10,
    )

    with pytest.raises(ValueError, match="monotonic"):
        validate_event_batch(events)


def test_normalize_polarity_accepts_bool_and_zero_one() -> None:
    assert normalize_polarity(np.array([False, True])).tolist() == [-1, 1]
    assert normalize_polarity(np.array([0, 1, 1])).tolist() == [-1, 1, 1]
