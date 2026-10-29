from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from e_jepa_ttc.detection.temporal_data import (
    TemporalRGBIndex,
    TemporalRGBSample,
    audit_temporal_support,
)
from scripts.audit_detection_temporal_data import build_audit


def row(sequence: str, index: int, start: int, end: int) -> dict[str, object]:
    return {
        "sample_token": f"{sequence}:{index}",
        "sequence_id": sequence,
        "rgb_exposure_start_timestamp_us": start,
        "rgb_exposure_end_timestamp_us": end,
    }


def fixture_index() -> TemporalRGBIndex:
    return TemporalRGBIndex(
        [
            row("a", 0, 100, 110),
            row("a", 1, 110, 120),
            row("a", 2, 130, 140),
            row("a", 3, 150, 160),
            row("a", 4, 170, 180),
            row("b", 0, 100, 105),
            row("b", 1, 120, 125),
        ]
    )


def test_correct_past_accepts_end_equal_to_current_start_and_left_pads():
    index = fixture_index()
    first = index.resolve("a:0")
    second = index.resolve("a:1")
    third = index.resolve("a:2")
    assert first.history_tokens == (None, None)
    assert first.history_mask == (False, False)
    assert second.history_tokens == (None, "a:0")
    assert second.history_mask == (False, True)
    assert third.history_tokens == ("a:0", "a:1")
    assert third.history_age_us == (30, 20)


def test_controls_are_explicit_deterministic_and_causal():
    index = fixture_index()
    assert index.resolve("a:4", control="current_only").history_mask == (False, False)
    repeated = index.resolve("a:4", control="current_repeated")
    assert repeated.history_tokens == ("a:4", "a:4")
    assert repeated.history_age_us == (0, 0)
    assert index.resolve("a:4", control="correct_past").history_tokens == ("a:2", "a:3")
    assert index.resolve("a:4", control="delayed_past").history_tokens == ("a:1", "a:2")
    permuted = index.resolve("a:4", control="permuted_past", seed=7)
    assert permuted == index.resolve("a:4", control="permuted_past", seed=7)
    assert all(token is None or token.startswith("a:") for token in permuted.history_tokens)
    assert all(end is None or end <= 170 for end in permuted.history_exposure_end_us)


def test_overlapping_previous_exposure_is_not_history():
    index = TemporalRGBIndex([row("a", 0, 100, 125), row("a", 1, 120, 130)])
    assert index.resolve("a:1").history_mask == (False, False)


@pytest.mark.parametrize(
    "rows,match",
    [
        ([row("a", 0, 100, 110), row("a", 0, 120, 130)], "Duplicate sample"),
        ([row("a", 0, 100, 110), row("a", 1, 100, 120)], "Late or duplicate"),
        ([row("a", 0, 100, 110), row("a", 1, 90, 100)], "Late or duplicate"),
        ([row("a", 0, 100, 90)], "ordered"),
    ],
)
def test_invalid_duplicate_late_and_reversed_timestamps_fail(rows, match):
    with pytest.raises(ValueError, match=match):
        TemporalRGBIndex(rows)


def test_cross_sequence_and_future_manual_history_fail():
    index = fixture_index()
    cross = TemporalRGBSample(
        sequence_id="a",
        current_token="a:3",
        current_exposure_start_us=150,
        current_exposure_end_us=160,
        decision_sensor_us=160,
        history_tokens=(None, "b:1"),
        history_mask=(False, True),
        history_exposure_end_us=(None, 125),
        history_age_us=(None, 35),
        control="correct_past",
    )
    with pytest.raises(ValueError, match="crosses"):
        index.validate_sample(cross)
    with pytest.raises(ValueError, match="cutoff"):
        TemporalRGBSample(
            sequence_id="a",
            current_token="a:2",
            current_exposure_start_us=130,
            current_exposure_end_us=140,
            decision_sensor_us=140,
            history_tokens=(None, "a:3"),
            history_mask=(False, True),
            history_exposure_end_us=(None, 160),
            history_age_us=(None, -20),
            control="correct_past",
        )


def test_role_audit_counts_resets_and_missing_history():
    index = fixture_index()
    report = audit_temporal_support(index, {"train": ["a"], "selection": ["b"]})
    assert report["roles"]["train"]["frames"] == 5
    assert report["roles"]["train"]["history_0"] == 1
    assert report["roles"]["train"]["history_1"] == 1
    assert report["roles"]["train"]["history_2"] == 3
    assert report["roles"]["selection"]["history_0"] == 1
    assert report["roles"]["selection"]["history_1"] == 1


def test_roles_must_cover_index_without_overlap():
    index = fixture_index()
    with pytest.raises(ValueError, match="crosses"):
        audit_temporal_support(index, {"train": ["a", "b"], "selection": ["b"]})
    with pytest.raises(ValueError, match="cover exactly"):
        audit_temporal_support(index, {"train": ["a"]})


def test_history_shapes_masks_and_repeated_current_are_strict():
    with pytest.raises(ValueError, match="exactly two slots"):
        TemporalRGBSample(
            sequence_id="a",
            current_token="a:0",
            current_exposure_start_us=100,
            current_exposure_end_us=110,
            decision_sensor_us=110,
            history_tokens=(None,),  # type: ignore[arg-type]
            history_mask=(False, False),
            history_exposure_end_us=(None, None),
            history_age_us=(None, None),
            control="current_only",
        )
    with pytest.raises(ValueError, match="booleans"):
        TemporalRGBSample(
            sequence_id="a",
            current_token="a:0",
            current_exposure_start_us=100,
            current_exposure_end_us=110,
            decision_sensor_us=110,
            history_tokens=(None, None),
            history_mask=(0, False),  # type: ignore[arg-type]
            history_exposure_end_us=(None, None),
            history_age_us=(None, None),
            control="current_only",
        )
    with pytest.raises(ValueError, match="exactly two current"):
        TemporalRGBSample(
            sequence_id="a",
            current_token="a:0",
            current_exposure_start_us=100,
            current_exposure_end_us=110,
            decision_sensor_us=110,
            history_tokens=(None, "a:0"),
            history_mask=(False, True),
            history_exposure_end_us=(None, 110),
            history_age_us=(None, 0),
            control="current_repeated",
        )


def test_index_rejects_forged_current_and_history_metadata():
    index = fixture_index()
    forged_current = TemporalRGBSample(
        sequence_id="a",
        current_token="a:2",
        current_exposure_start_us=1000,
        current_exposure_end_us=1100,
        decision_sensor_us=1100,
        history_tokens=(None, None),
        history_mask=(False, False),
        history_exposure_end_us=(None, None),
        history_age_us=(None, None),
        control="current_only",
    )
    with pytest.raises(ValueError, match="current timestamps"):
        index.validate_sample(forged_current)

    forged_history = TemporalRGBSample(
        sequence_id="a",
        current_token="a:2",
        current_exposure_start_us=130,
        current_exposure_end_us=140,
        decision_sensor_us=140,
        history_tokens=(None, "a:1"),
        history_mask=(False, True),
        history_exposure_end_us=(None, 1),
        history_age_us=(None, 139),
        control="correct_past",
    )
    with pytest.raises(ValueError, match="history timestamps"):
        index.validate_sample(forged_history)


def test_index_uses_canonical_history_cutoff_not_forged_end():
    index = TemporalRGBIndex([row("a", 0, 100, 155), row("a", 1, 150, 160)])
    forged = TemporalRGBSample(
        sequence_id="a",
        current_token="a:1",
        current_exposure_start_us=150,
        current_exposure_end_us=160,
        decision_sensor_us=160,
        history_tokens=(None, "a:0"),
        history_mask=(False, True),
        history_exposure_end_us=(None, 140),
        history_age_us=(None, 20),
        control="correct_past",
    )
    with pytest.raises(ValueError, match="timestamps differ"):
        index.validate_sample(forged)


def test_persisted_audit_binds_media_split_roles_and_implementation(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    media = data / "train.parquet"
    rows = [
        {**row("a", 0, 100, 110), "split": "train"},
        {**row("b", 0, 200, 210), "split": "train"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), media)
    with media.open("rb") as stream:
        media_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "schema": "eap_detection_sequence_split_v1",
                "dataset_revision": "a" * 40,
                "media_sha256": media_hash,
                "source_split": "train",
                "assignments": {
                    "train": ["a"],
                    "selection": ["b"],
                    "calibration": ["c"],
                },
            }
        ),
        encoding="utf-8",
    )
    audit = build_audit(data_root=tmp_path, split_path=split)
    assert audit["media"]["sha256"] == media_hash
    assert audit["split"]["train_sequence_ids"] == ["a"]
    assert audit["split"]["selection_sequence_ids"] == ["b"]
    assert audit["split"]["calibration_sequence_ids_excluded"] == ["c"]
    assert audit["support"]["roles"]["train"]["frames"] == 1
    assert len(audit["implementation"]["module_sha256"]) == 64
    assert len(audit["implementation"]["entrypoint_sha256"]) == 64
