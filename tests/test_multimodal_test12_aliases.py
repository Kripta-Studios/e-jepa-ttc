"""Official tokens survive physical deduplication; conflicting ROIs cannot be merged."""

import pytest

from operational.multimodal_research.test12_identities import physical_histories


def test_identical_clip_aliases_are_one_history_observation() -> None:
    base = dict(
        sequence_id="A",
        track_id="object",
        anchor_us=1000000,
        input_span_us=300000,
        square_xyxy=[0, 0, 20, 20],
    )
    rows = [
        dict(base, sample_token="b"),
        dict(base, sample_token="a"),
        dict(base, sample_token="later", anchor_us=1100000),
    ]
    histories, aliases, unique = physical_histories(rows)
    assert aliases == {"a": "a", "b": "a", "later": "later"}
    assert set(histories) == {"a", "b", "later"}
    assert histories["a"] == histories["b"] == ["a"]
    assert histories["later"] == ["a", "later"]
    assert len(unique) == 2


def test_different_roi_at_same_object_timestamp_is_not_silently_merged() -> None:
    base = dict(sequence_id="A", track_id="object", anchor_us=1000000, input_span_us=300000)
    rows = [
        dict(base, sample_token="a", square_xyxy=[0, 0, 20, 20]),
        dict(base, sample_token="b", square_xyxy=[1, 0, 20, 20]),
    ]
    with pytest.raises(AssertionError, match="Conflicting sensor/ROI"):
        physical_histories(rows)
