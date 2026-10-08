from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from operational.evttc_transfer.inputs import EvTTCEventReader
from operational.sota_eval import fcwd_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_group(handle: h5py.File, camera: str, values: dict[str, np.ndarray]) -> None:
    group = handle.create_group(f"prophesee/event_cam_{camera}")
    for name, value in values.items():
        group.create_dataset(name, data=value)


def _hdf5(path: Path, *, frames: int = 20) -> None:
    timestamps = np.arange(0, frames * 50_000, 500, dtype=np.int64)
    values = {
        "x": np.mod(np.arange(len(timestamps)), 16).astype(np.uint16),
        "y": np.mod(np.arange(len(timestamps)), 8).astype(np.uint16),
        "t": timestamps,
        "p": np.mod(np.arange(len(timestamps)), 2).astype(np.int8),
        "ms_map_idx": np.searchsorted(
            timestamps, np.arange(0, frames * 50 + 1) * 1000, side="left"
        ).astype(np.uint64),
    }
    with h5py.File(path, "w") as handle:
        _event_group(handle, "right", values)
        _event_group(handle, "left", values)
        right = handle.create_group("blackflys/right")
        right.create_dataset("ts", data=np.arange(frames, dtype=np.int64) * 50_000)


def test_right_reader_matches_equivalent_left_layout(tmp_path: Path) -> None:
    path = tmp_path / "sensor.hdf5"
    _hdf5(path)
    with fcwd_inputs.FCWDEventReader(path) as right, EvTTCEventReader(path) as left:
        right_window = right.read_window(100_000, 200_000)
        left_window = left.read_window(100_000, 200_000)
    for name in ("x", "y", "t", "p"):
        np.testing.assert_array_equal(right_window[name], left_window[name])
    assert np.all((right_window["t"] >= 100_000) & (right_window["t"] < 200_000))


def test_build_manifest_never_opens_ttc_and_preserves_direct_event_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "FCWD"
    assets = []
    references = []
    for number in (1, 2, 3):
        input_dir = root / "inputs" / f"sequence_{number}"
        label_dir = root / "sealed_labels" / f"sequence_{number}"
        input_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        raw = input_dir / f"sequence_{number}.hdf5"
        _hdf5(raw, frames=20)
        bbox = label_dir / "bounding_box.csv"
        values = np.column_stack(
            (
                np.arange(20),
                np.arange(20) * 0.05,
                np.full(20, 1.0),
                np.full(20, 2.0),
                np.full(20, 1282.0),
                np.full(20, 722.0),
            )
        )
        np.savetxt(bbox, values, delimiter=",")
        (label_dir / "gt_ttc.csv").write_text("FORBIDDEN", encoding="ascii")
        assets.extend(
            (
                {
                    "name": f"fcwd_{number}_hdf5",
                    "path": str(raw),
                    "sha256": _sha(raw),
                },
                {
                    "name": f"fcwd_{number}_bbox",
                    "path": str(bbox),
                    "sha256": _sha(bbox),
                },
            )
        )
        references.append(
            {
                "sequence": number,
                "bbox_rows": 20,
                "bbox_sha256": _sha(bbox),
            }
        )
    asset_path = tmp_path / "assets.json"
    asset_path.write_text(
        json.dumps({"status": "COMPLETE", "targets_read": False, "assets": assets}),
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "schema": fcwd_inputs.REFERENCE_SCHEMA,
                "status": "READY_WITH_SPATIAL_CALIBRATION_LIMITATION",
                "targets_read": False,
                "official_commit": "79ff0842955304ec4f6164ec09baddc71386d225",
                "sequences": references,
            }
        ),
        encoding="utf-8",
    )
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if "gt_ttc" in str(file).lower():
            raise AssertionError("TTC target was opened")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    result = fcwd_inputs.build_manifest(root, asset_path, reference_path, tmp_path / "queries.json")
    assert result["ttc_targets_read"] is False
    assert result["sequence_count"] == 3
    assert result["query_count"] > 0
    assert all(
        row["anchor_us"] == round(row["anchor_relative_seconds"] * 1e6)
        for row in result["rows"]
    )
    first = result["rows"][0]
    assert first["boxes_xyxy3"][0] == [1.0, 2.0, 1280.0, 720.0]
    assert first["garl_full_available"] is False
    assert "gt_ttc" not in json.dumps(result["rows"] + result["source_files"]).lower()


def test_prepare_returns_frozen_shapes_and_explicit_full_rgb_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sensor.hdf5"
    _hdf5(path, frames=30)
    row = {
        "sequence_id": "FCWD1",
        "query_id": "FCWD1:0000",
        "anchor_us": 900_000,
        "anchor_relative_seconds": 0.9,
        "raw_path": str(path),
        "raw_stat": fcwd_inputs._pin_stat(path),
        "windows_us": [[600_000, 700_000], [700_000, 800_000], [800_000, 900_000]],
        "boxes_xyxy3": [[1.0, 2.0, 15.0, 8.0]] * 3,
        "square_xyxy": [1.0, 0.0, 15.0, 14.0],
        "bbox_sources": [],
        "garl_full_available": False,
        "garl_full_unavailable_reason": "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING",
    }
    row["metadata_sha256"] = fcwd_inputs._canonical_sha256(row)
    monkeypatch.setattr(
        fcwd_inputs,
        "encode_union",
        lambda *_args, **_kwargs: torch.zeros(16, 3, 12, 128, 128),  # pyright: ignore
    )
    monkeypatch.setattr(
        fcwd_inputs,
        "native_feature",
        lambda *_args, **_kwargs: torch.zeros(20, 128, 128),  # pyright: ignore
    )
    prepared = fcwd_inputs.prepare(row)
    assert prepared["own_events"].shape == (8, 3, 12, 128, 128)
    assert prepared["garl_events"].shape == (40, 128, 128)
    assert prepared["garl_full_sensor"] is None
    assert prepared["garl_full_unavailable_reason"] == (
        "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING"
    )
    assert prepared["metadata"]["targets_read"] is False
