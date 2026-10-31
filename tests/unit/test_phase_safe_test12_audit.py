"""Test12 histories respect physical ownership before using public ROI geometry."""

import numpy as np
import pytest

from scripts.build_phase_safe_test12_audit import query_history


def inventory():
    query = {
        "sample_token": "logical",
        "sequence_id": "sequence",
        "track_id": "object",
        "query_time_us": 1_000_000,
        "rgb_history": ["one", "two"],
    }
    observations = {
        "one": {
            "sample_token": "logical",
            "producer_times_us": [800_000, 900_000],
            "producer_available_us": [800_000, 900_000],
            "producer_boxes_xyxy": [[0, 0, 8, 8], [0, 0, 9, 9]],
        },
        "two": {
            "sample_token": "logical",
            "producer_times_us": [900_000, 1_000_000],
            "producer_available_us": [900_000, 1_000_000],
            "producer_boxes_xyxy": [[0, 0, 9, 9], [0, 0, 10, 10]],
        },
    }
    return query, observations, {"logical": query}


def test_test12_history_deduplicates_shared_frames():
    query, observations, owners = inventory()
    boxes, clocks = query_history(query, observations, owners)
    np.testing.assert_array_equal(clocks, [800_000, 900_000, 1_000_000])
    assert boxes.shape == (3, 4)


def test_test12_history_rejects_another_object():
    query, observations, owners = inventory()
    observations["one"]["sample_token"] = "other"
    owners["other"] = dict(query, track_id="different")
    with pytest.raises(ValueError, match="object or sequence"):
        query_history(query, observations, owners)


def test_test12_history_missing_is_explicit_empty():
    query, observations, owners = inventory()
    query["rgb_history"] = []
    boxes, clocks = query_history(query, observations, owners)
    assert boxes.shape == (0, 4)
    assert clocks.shape == (0,)
