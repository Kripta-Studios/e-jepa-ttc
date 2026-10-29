from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from e_jepa_ttc.data.types import DatasetSequence
from operational.evttc_transfer import inputs


def _write_events(
    path: Path,
    *,
    duration_us: int = 4_000_000,
    event_group: str = "prophesee/event_cam_left",
) -> None:
    timestamps = np.arange(0, duration_us, 10_000, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        event = handle.create_group(event_group)
        event.create_dataset("x", data=np.full(len(timestamps), 500, dtype=np.int16))
        event.create_dataset("y", data=np.full(len(timestamps), 300, dtype=np.int16))
        event.create_dataset("t", data=timestamps)
        event.create_dataset("p", data=np.arange(len(timestamps), dtype=np.int8) % 2)
        millisecond = np.arange(0, duration_us // 1000 + 2, dtype=np.int64)
        event.create_dataset("ms_map_idx", data=np.searchsorted(timestamps, millisecond * 1000))
        event_calibration = event.create_group("calib")
        event_calibration.create_dataset("intrinsics", data=[1000.0, 1000.0, 640.0, 360.0])
        event_calibration.create_dataset("distortion_coeffs", data=np.zeros(4))
        transform = np.eye(4)
        transform[:3, 3] = [100.0, -200.0, 300.0]
        event_calibration.create_dataset("T_bfs_to_prophesee", data=transform)
        blackfly = handle.create_group("blackflys/left")
        blackfly.create_dataset("ts", data=np.arange(0, duration_us, 50_000, dtype=np.int64))
        rgb_calibration = blackfly.create_group("calib")
        rgb_calibration.create_dataset("intrinsics", data=[1000.0, 1000.0, 640.0, 360.0])
        rgb_calibration.create_dataset("distortion_coeffs", data=np.zeros(4))


def _write_labels(directory: Path, count: int = 80) -> None:
    directory.mkdir()
    for frame in range(count):
        payload = {
            "info": {"width": 1280, "height": 720},
            "objects": [{"category": "car", "bbox": [450, 250, 550, 350]}],
        }
        (directory / f"{frame:04d}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_rotation_only_projection_uses_direction_and_ignores_translation() -> None:
    angle = np.pi / 2
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [999.0, 999.0, 999.0]
    calibration = {
        "rgb_intrinsics": np.asarray([100.0, 100.0, 0.0, 0.0]),
        "rgb_distortion": np.zeros(4),
        "event_intrinsics": np.asarray([100.0, 100.0, 0.0, 0.0]),
        "event_distortion": np.zeros(4),
        "event_from_rgb": transform,
    }

    projected = inputs.project_rgb_box_rotation_only((10.0, 20.0, 30.0, 40.0), calibration)

    assert projected == pytest.approx((-40.0, 10.0, -20.0, 30.0), abs=1e-8)


def test_h8_uses_exact_trained_eight_lags() -> None:
    assert inputs.H8_LAGS_US.tolist() == [
        350_000,
        300_000,
        250_000,
        200_000,
        150_000,
        100_000,
        50_000,
        0,
    ]


def test_label_admission_ignores_background_but_never_disambiguates_targets(
    tmp_path: Path,
) -> None:
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "0000.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"category": "__background__", "bbox": [0, 0, 1200, 700]},
                    {"category": "car", "bbox": [10, 20, 30, 50], "group": 7},
                ]
            }
        ),
        encoding="utf-8",
    )
    (label_dir / "0001.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"category": "car", "bbox": [10, 20, 30, 50]},
                    {"category": "car", "bbox": [100, 200, 300, 500]},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, audit = inputs._label_rows(
        label_dir,
        np.asarray([0, 50_000], dtype=np.int64),
        target_type="car",
    )

    assert len(rows) == 1
    assert rows[0]["bbox_xyxy_rgb"] == (10.0, 20.0, 30.0, 50.0)
    assert rows[0]["category"] == "car"
    assert rows[0]["group"] == 7
    assert audit["accepted_single_compatible_object"] == 1
    assert audit["ambiguous_compatible_objects"] == 1
    assert inputs._uniform_indices(3, 32).tolist() == [0, 1, 2]


def test_causal_box_expires_after_documented_jitter_allowance() -> None:
    labels = [{"timestamp_us": 1_000_000}]

    fresh, fresh_reason = inputs._causal_box(labels, 1_101_000)
    stale, stale_reason = inputs._causal_box(labels, 1_101_001)

    assert fresh is labels[0]
    assert fresh_reason is None
    assert stale is None
    assert stale_reason == "stale"


def test_reader_is_half_open_bounded_and_uses_microseconds(tmp_path: Path) -> None:
    path = tmp_path / "events.hdf5"
    _write_events(path, duration_us=100_000)

    with inputs.EvTTCEventReader(path, chunk_events_max=3) as reader:
        chunks = list(reader.iter_window_chunks(20_000, 60_000, chunk_events=3))
        values = np.concatenate([chunk["t"] for chunk in chunks])
        assert values.tolist() == [20_000, 30_000, 40_000, 50_000]
        assert all(len(chunk["t"]) <= 3 for chunk in chunks)
        assert reader.timestamp_unit == "microseconds"
        with pytest.raises(ValueError, match="chunk_events"):
            list(reader.iter_window_chunks(0, 10_000, chunk_events=4))


def test_reader_rejects_implicit_nonleft_camera(tmp_path: Path) -> None:
    path = tmp_path / "right-events.hdf5"
    _write_events(path, duration_us=100_000, event_group="prophesee/event_cam_right")

    with pytest.raises(ValueError, match="automatic camera selection is forbidden"):
        inputs.EvTTCEventReader(path)


def test_build_manifest_is_label_free_causal_and_uniform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence_dir = tmp_path / "shared"
    sequence_dir.mkdir()
    raw_path = sequence_dir / "events.hdf5"
    _write_events(raw_path)
    _write_labels(sequence_dir / "leftlabel")
    sequences = [
        DatasetSequence(
            dataset="EvTTC",
            sequence_id=f"sequence-{index:02d}",
            local_path="ignored",
            event_hdf5="events.hdf5",
            ttc_csv="forbidden-ttc.csv",
            gt_hdf5="forbidden-gt.hdf5",
            label_dir="leftlabel",
            scenario_family="family",
            speed_bucket="medium",
            target_type="car",
            extra={"relative_parts": ["shared"]},
        )
        for index in range(32)
    ]
    monkeypatch.setattr(inputs, "read_manifest", lambda _: sequences)
    destination = tmp_path / "queries.json"

    manifest = inputs.build_manifest(tmp_path / "inventory.yaml", tmp_path, destination)

    assert manifest["sequence_count"] == 32
    assert len(manifest["rows"]) == 32 * 32
    assert destination.is_file()
    assert "ttc" not in manifest["rows"][0]
    assert "distance" not in manifest["rows"][0]
    assert len(manifest["rows_metadata_sha256"]) == 64
    assert len(manifest["rows"][0]["calibration_sha256"]) == 64
    assert len(manifest["rows"][0]["bbox_sources"][0]["sha256"]) == 64
    for row in manifest["rows"]:
        assert np.diff(np.asarray(row["windows_us"]), axis=1).ravel().tolist() == [100_000] * 3
        for endpoint, source in zip(
            (row["anchor_us"] - 200_000, row["anchor_us"] - 100_000, row["anchor_us"]),
            row["bbox_sources"],
            strict=True,
        ):
            assert source["available_us"] <= endpoint
    assert all(
        len(
            {
                row["anchor_us"]
                for row in manifest["rows"]
                if row["sequence_id"] == sequence.sequence_id
            }
        )
        == 32
        for sequence in sequences
    )


def test_prepare_produces_exact_frozen_shapes_without_eap_offset(tmp_path: Path) -> None:
    raw_path = tmp_path / "events.hdf5"
    _write_events(raw_path)
    anchor = 1_000_000
    boxes = [(450.0, 250.0, 550.0, 350.0)] * 3
    row = {
        "sequence_id": "sequence",
        "anchor_us": anchor,
        "raw_path": str(raw_path),
        "raw_stat": {
            "size_bytes": raw_path.stat().st_size,
            "mtime_ns": raw_path.stat().st_mtime_ns,
        },
        "windows_us": [
            [anchor - 300_000, anchor - 200_000],
            [anchor - 200_000, anchor - 100_000],
            [anchor - 100_000, anchor],
        ],
        "boxes_xyxy3": boxes,
        "square_xyxy": inputs.common_square_from_boxes(boxes, (0, 1, 2)),
    }

    prepared = inputs.prepare(row)

    assert prepared["own_events"].shape == (8, 3, 12, 128, 128)
    assert prepared["own_events"].dtype == np.float32
    assert prepared["garl_events"].shape == (40, 128, 128)
    assert prepared["garl_events"].dtype == np.float32
    assert prepared["valid"].tolist() == [True] * 8
    assert prepared["delta_t_s"] == pytest.approx(0.1)
    assert np.isfinite(prepared["own_events"]).all()
    assert np.isfinite(prepared["garl_events"]).all()
