"""Protect authoritative GT assignment and target-independent clock provenance."""

import hashlib

import pytest

from scripts.audit_phase_safe_evaluator import (
    assign_public_road_labels,
    bounded_target,
    parse_verified_csv,
    verified_clock_map,
)


def test_public_sequence_assignment_repairs_swapped_labels_without_mutation() -> None:
    legacy = {"CCRs1-low": [[10, 8.0]], "CCRs1-medium": [[20, 4.0]], "other": [[1, 2.0]]}
    official = {"CCRs1-low": [[20, 4.0]], "CCRs1-medium": [[10, 8.0]]}
    corrected = assign_public_road_labels(legacy, official)
    assert corrected["CCRs1-low"] == [[20, 4.0]]
    assert corrected["CCRs1-medium"] == [[10, 8.0]]
    assert corrected["other"] == legacy["other"]
    assert legacy["CCRs1-low"] == [[10, 8.0]]


def test_source_hash_must_match_before_csv_is_accepted() -> None:
    data = b"1\t0.000010\t0\t0\t4.5\n2\t0.000020\t0\t0\t3.5\n"
    assert parse_verified_csv(data, hashlib.sha256(data).hexdigest()) == [[10, 4.5], [20, 3.5]]
    with pytest.raises(ValueError, match="SHA256"):
        parse_verified_csv(data, "0" * 64)


def test_clock_offsets_require_unique_label_blind_all_frame_identity() -> None:
    evidence = {
        "labels_read": False,
        "sequences": [
            {
                "competition_sequence": "road",
                "status": "unique",
                "matches": [
                    {
                        "timestamp_offset_us": 100,
                        "all_query_rgb_timestamps_match": True,
                        "all_rgb_frames": [
                            {"equal": True, "competition_sha256": "abc", "original_sha256": "abc"}
                        ],
                    }
                ],
            }
        ],
    }
    assert verified_clock_map({"road": 100}, evidence) == {"road": 100}
    with pytest.raises(ValueError, match="independent RGB"):
        verified_clock_map({"road": 101}, evidence)
    evidence["labels_read"] = True
    with pytest.raises(ValueError, match="label-blind"):
        verified_clock_map({"road": 100}, evidence)


def test_bounded_alignment_does_not_infer_offsets_or_extrapolate() -> None:
    assert bounded_target([[100, 5.0], [110, 3.0]], 105) == 4.0
    with pytest.raises(ValueError, match="outside"):
        bounded_target([[100, 5.0], [110, 3.0]], 99)
    with pytest.raises(ValueError, match="increasing"):
        bounded_target([[110, 5.0], [100, 3.0]], 105)
