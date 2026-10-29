from __future__ import annotations

import pytest

from e_jepa_ttc.detection.causal_tracker import (
    CausalMultiObjectTracker,
    DetectionObservation,
    TrackState,
)


def detection(identity: str, box: tuple[float, float, float, float], cls: str = "car"):
    return DetectionObservation(identity, cls, 0.9, box, box)


def test_tracks_confirm_then_lost_and_close_without_reentry() -> None:
    tracker = CausalMultiObjectTracker(confirm_hits=2, lost_ttl_frames=1)
    first = tracker.update(100, [detection("d1", (0, 0, 10, 10))])
    assert first.assignments[0].track.state == TrackState.TENTATIVE
    confirmed = tracker.update(200, [detection("d2", (1, 0, 11, 10))])
    original = confirmed.assignments[0].track
    assert original.state == TrackState.CONFIRMED
    assert confirmed.assignments[0].phase_admissible
    lost = tracker.update(300, [])
    assert (
        next(item for item in lost.tracks if item.object_id == original.object_id).state == "lost"
    )
    replacement = tracker.update(400, [detection("d3", (1, 0, 11, 10))])
    assert replacement.assignments[0].association_status == "new"
    assert replacement.assignments[0].track.object_id != original.object_id
    assert (
        next(item for item in replacement.tracks if item.object_id == original.object_id).state
        == "closed"
    )


def test_inverse_ambiguity_rejects_both_competing_detections() -> None:
    tracker = CausalMultiObjectTracker(minimum_iou=0.3, ambiguity_margin=0.05, confirm_hits=1)
    original = tracker.update(100, [detection("old", (0, 0, 10, 10))]).assignments[0]
    update = tracker.update(
        200,
        [
            detection("left", (0, 0, 6, 10)),
            detection("right", (4, 0, 10, 10)),
        ],
    )
    assert all(item.track.object_id != original.track.object_id for item in update.assignments)
    ambiguous = [item for item in update.assignments if item.association_status == "ambiguous"]
    assert len(ambiguous) == 2
    assert {item.ambiguity_reason for item in ambiguous} == {"track_to_detections_margin"}
    assert all(not item.phase_admissible for item in ambiguous)


def test_global_assignment_is_class_gated_and_deterministic() -> None:
    tracker = CausalMultiObjectTracker(confirm_hits=1)
    first = tracker.update(
        100,
        [detection("car", (0, 0, 10, 10)), detection("truck", (20, 0, 30, 10), "truck")],
    )
    by_class = {item.detection.class_name: item.track.object_id for item in first.assignments}
    second = tracker.update(
        200,
        [detection("truck2", (20, 0, 30, 10), "truck"), detection("car2", (0, 0, 10, 10))],
    )
    assert {
        item.detection.class_name: item.track.object_id for item in second.assignments
    } == by_class


@pytest.mark.parametrize("timestamp", [100, 99])
def test_repeated_or_late_timestamp_fails(timestamp: int) -> None:
    tracker = CausalMultiObjectTracker()
    tracker.update(100, [])
    with pytest.raises(ValueError, match="increase strictly"):
        tracker.update(timestamp, [])


def test_extrapolated_roi_never_enters_phase() -> None:
    tracker = CausalMultiObjectTracker(confirm_hits=1)
    assignment = tracker.update(
        100,
        [DetectionObservation("x", "car", 0.8, (0, 0, 10, 10), None, "extrapolated")],
    ).assignments[0]
    assert assignment.track.state == TrackState.CONFIRMED
    assert not assignment.phase_admissible
