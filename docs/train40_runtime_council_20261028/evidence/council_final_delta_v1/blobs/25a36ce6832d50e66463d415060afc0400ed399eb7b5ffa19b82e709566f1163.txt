"""Causal scheduling, ROI identity and coverage tests for new replay queries."""

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from operational.evttc_transfer.inputs import _canonical_sha256
from operational.rgb_port_revision.transfer_inputs import prepare_event, raw_rgb_observations
from operational.train40_runtime.replay_inputs import generate_sensor_clock_replay
from scripts.final_train40_transfer import reject_targets


@pytest.fixture
def base(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    for index in range(16):
        obj = {"category": "car", "group": 1, "bbox": [2 + index / 4, 3, 16, 19]}
        (labels / f"{index}.json").write_text(json.dumps({"objects": [obj]}), encoding="utf-8")
    path = tmp_path / "sensor.h5"
    clock = np.arange(16, dtype=np.int64) * 100_000
    t = np.arange(1501, dtype=np.int64) * 1000
    with h5py.File(path, "w") as handle:
        handle["blackflys/left/ts"] = clock
        handle["blackflys/left/data"] = np.random.default_rng(7).integers(
            0, 256, (16, 24, 24, 3), np.uint8
        )
        prefix = "prophesee/event_cam_left"
        handle[f"{prefix}/t"] = t
        handle[f"{prefix}/x"] = np.arange(len(t), dtype=np.int32) % 24
        handle[f"{prefix}/y"] = np.arange(len(t), dtype=np.int32) % 20
        handle[f"{prefix}/p"] = np.arange(len(t), dtype=np.int8) % 2
        handle[f"{prefix}/ms_map_idx"] = np.searchsorted(t, np.arange(1502) * 1000)
        for camera in ("blackflys/left", prefix):
            handle[f"{camera}/calib/intrinsics"] = np.asarray([100, 100, 12, 12], np.float64)
            handle[f"{camera}/calib/distortion_coeffs"] = np.zeros(4, np.float64)
        handle[f"{prefix}/calib/T_bfs_to_prophesee"] = np.eye(4)
    stat = path.stat()
    rows = []
    for anchor in (600_000, 900_000):
        frame = anchor // 100_000
        rows.append(
            {
                "query_id": f"BASE:{anchor}",
                "sequence_id": "seq",
                "physical_sequence_id": "seq",
                "anchor_us": anchor,
                "raw_path": str(path),
                "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                "target_type": "car",
                "scenario_family": "CCRs",
                "speed_bucket": "low",
                "bbox_sources": [
                    {"path": str(labels / f"{index}.json")} for index in range(frame - 2, frame + 1)
                ],
                "ttc_csv_path": str(tmp_path / "must_not_read_missing_ttc.csv"),
            }
        )
    return {"schema": "frozen_fixture", "rows": rows}


def test_new_manifest_is_deterministic_distinct_and_does_not_mutate_base(base):
    original = copy.deepcopy(base)
    result = generate_sensor_clock_replay(base)
    assert result == generate_sensor_clock_replay(base)
    assert base == original
    assert result["schema"] != base["schema"]
    assert result["scheduled_updates"] == result["accepted_updates"] == 16
    assert result["excluded"] == []
    assert len({row["query_id"] for row in result["rows"]}) == 16
    assert not {row["query_id"] for row in result["rows"]}.intersection(
        row["query_id"] for row in base["rows"]
    )
    assert result["provenance"]["hardware_arrivals_measured"] is False
    for row in result["rows"]:
        reject_targets(row)
        unsigned = {key: value for key, value in row.items() if key != "metadata_sha256"}
        assert row["metadata_sha256"] == _canonical_sha256(unsigned)
        assert "ttc_csv_path" not in row
        assert row["windows_us"][-1][1] == row["anchor_us"]
        assert row["anchor_us"] - (row["windows_us"][0][0] - 300_000) == 600_000


def test_between_frame_query_keeps_real_sensor_times_and_causal_roi(base):
    result = generate_sensor_clock_replay(base, max_updates_per_sequence=2)
    first, second = result["rows"]
    assert second["anchor_us"] == 620_000
    assert second["rgb_timestamps_us"] == [400_000, 500_000, 600_000]
    assert second["roi_sensor_anchor_us"] == second["roi_available_us"] == 600_000
    assert second["roi_age_us"] == 20_000
    assert second["square_xyxy"] == first["square_xyxy"]
    for source in second["bbox_sources"]:
        assert source["timestamp_us"] == source["available_us"] <= second["anchor_us"]
    # New query rows remain executable by the untouched reference sensor adapter.
    events = prepare_event(second)["own_events"]
    rgb, reason = raw_rgb_observations(second)
    assert events.shape == (4, 3, 12, 128, 128)
    assert events.dtype == np.float32
    assert rgb and reason == ""
    assert max(item[1]["anchor_us"] for item in rgb) <= second["anchor_us"]


def test_annotation_delay_is_explicit_and_no_unavailable_box_is_selected(base):
    result = generate_sensor_clock_replay(
        base, annotation_delay_us=50_000, max_updates_per_sequence=4
    )
    first = result["rows"][0]
    assert first["rgb_timestamps_us"] == [300_000, 400_000, 500_000]
    assert first["roi_available_us"] == 550_000
    last = result["rows"][-1]
    assert last["anchor_us"] == 660_000
    assert last["roi_sensor_anchor_us"] == 600_000
    assert last["roi_available_us"] == 650_000
    for row in result["rows"]:
        assert all(source["available_us"] <= row["anchor_us"] for source in row["bbox_sources"])
    assert result["provenance"]["annotation_delay_us"] == 50_000
    assert result["provenance"]["annotation_availability_measured"] is False


def test_missing_current_annotation_excludes_without_older_triplet_fallback(base):
    labels = Path(base["rows"][0]["bbox_sources"][0]["path"]).parent
    (labels / "7.json").unlink()
    result = generate_sensor_clock_replay(base)
    assert result["accepted_updates"] == 5
    assert len(result["excluded"]) == 11
    assert {row["reason"] for row in result["excluded"]} == {"MISSING_CAUSAL_TARGET_ANNOTATION"}
    assert result["scheduled_updates"] == 16
    assert result["accepted_updates"] + len(result["excluded"]) == 16


def test_future_annotation_bytes_are_never_read(base, monkeypatch):
    original_read = Path.read_bytes

    def guarded(path):
        if path.parent.name == "labels" and int(path.stem) > 9:
            raise AssertionError("Future annotation contents were read")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = generate_sensor_clock_replay(base)
    assert result["accepted_updates"] == 16


def test_ambiguous_primary_target_is_rejected_and_attempt_cap_counts_rejections(base):
    path = Path(base["rows"][0]["bbox_sources"][-1]["path"])
    obj = {"category": "car", "group": 1, "bbox": [2, 3, 16, 19]}
    path.write_text(json.dumps({"objects": [obj, obj]}), encoding="utf-8")
    result = generate_sensor_clock_replay(base, max_updates_per_sequence=3)
    assert result["scheduled_updates"] == 3
    assert result["accepted_updates"] == 0
    assert len(result["excluded"]) == 3
    assert result["exclusion_reasons"] == {"AMBIGUOUS_OR_NONPRIMARY_SCENARIO_TARGET": 3}


def test_insufficient_event_support_preserves_scheduled_coverage(base):
    result = generate_sensor_clock_replay(base, bounds_us={"seq": (580_000, 620_000)})
    assert result["scheduled_updates"] == 3
    assert result["accepted_updates"] == 2
    assert result["excluded"][0]["reason"] == "INSUFFICIENT_FULL_600MS_EVENT_SUPPORT"


def test_changed_raw_source_stat_fails_closed(base):
    base["rows"][0]["raw_stat"]["mtime_ns"] = 0
    with pytest.raises(ValueError, match="source identity"):
        generate_sensor_clock_replay(base)


@pytest.mark.parametrize(
    "options",
    [
        {"step_us": 0},
        {"step_us": True},
        {"annotation_delay_us": -1},
        {"max_updates_per_sequence": 0},
        {"sequence_ids": ["seq", "seq"]},
        {"sequence_ids": ["absent"]},
        {"bounds_us": {"seq": (900_000, 600_000)}},
    ],
)
def test_invalid_schedule_is_rejected(base, options):
    with pytest.raises(ValueError):
        generate_sensor_clock_replay(base, **options)
