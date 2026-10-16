"""Do not confuse storage fragment size with the producer's homogeneous forward batch."""

import pytest

from operational.joint_measurement.fusion_cost_batch_inventory import batch_shapes, rgb_frame_counts


def test_mixed_fragment_splits_by_time_shape() -> None:
    shapes = batch_shapes(["a", "b", "c"], {"a": 2, "b": 3, "c": 3})
    assert shapes["a"] == {"fragment_rows": 3, "input_frames": 2, "forward_batch": 1}
    assert shapes["c"] == {"fragment_rows": 3, "input_frames": 3, "forward_batch": 2}


@pytest.mark.parametrize("ids", [["a", "a"], ["unknown"]])
def test_ambiguous_fragment_membership_rejected(ids) -> None:
    with pytest.raises(ValueError):
        batch_shapes(ids, {"a": 3})


def test_repeated_observation_must_keep_frame_count() -> None:
    rows = [
        {"observation_ids": ["a"], "observation_members": [["x", "y", "z"]]},
        {"observation_ids": ["a"], "observation_members": [["x", "y"]]},
    ]
    with pytest.raises(ValueError):
        rgb_frame_counts(rows)


def test_cold_start_identity_is_stable() -> None:
    row = {
        "observation_ids": [],
        "producer_frame_count": 2,
        "track_id": "track",
        "rgb_anchor_us": 10,
    }
    counts = rgb_frame_counts([row, row])
    assert len(counts) == 1
    assert list(counts.values()) == [2]
