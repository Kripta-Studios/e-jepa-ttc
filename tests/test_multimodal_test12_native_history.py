"""Test object identity and full sensor-support boundaries for official input histories."""

import pytest

from operational.multimodal_research.test12_native_inputs import native_histories


def job(token: str, anchor: int, *, sequence: str = "A", track: str = "object") -> dict:
    return dict(
        sample_token=token,
        anchor_us=anchor,
        sequence_id=sequence,
        track_id=track,
        input_span_us=300000,
    )


def test_histories_do_not_cross_sequence_or_object() -> None:
    rows = [
        job("old", 1000000),
        job("other_object", 1050000, track="second"),
        job("other_sequence", 1100000, sequence="B"),
        job("now", 1200000),
    ]
    actual = native_histories(rows)
    assert actual["now"] == ["old", "now"]
    assert actual["other_object"] == ["other_object"]
    assert actual["other_sequence"] == ["other_sequence"]


def test_history_bounds_entire_producer_input_and_excludes_future() -> None:
    rows = [
        job("too_old", 649999),
        job("boundary", 650000),
        job("now", 1000000),
        job("future", 1000001),
    ]
    assert native_histories(rows)["now"] == ["boundary", "now"]


def test_ambiguous_same_object_timestamp_is_rejected() -> None:
    with pytest.raises(AssertionError, match="Ambiguous"):
        native_histories([job("one", 1000000), job("two", 1000000)])
