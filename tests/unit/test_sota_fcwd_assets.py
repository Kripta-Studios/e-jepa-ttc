from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from operational.sota_eval import fcwd_assets


def test_capacity_requires_more_than_full_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        fcwd_assets.shutil, "disk_usage", lambda _path: SimpleNamespace(free=10 * 1024**3)
    )
    with pytest.raises(RuntimeError, match="more than 10 GiB"):
        fcwd_assets.check_capacity(tmp_path)


def test_hdf5_inspection_reads_only_safe_timestamp_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "sample.hdf5"
    with h5py.File(path, "w") as handle:
        events = handle.create_group("event_camera/events")
        events.create_dataset("x", data=np.asarray([1, 2], np.int16))
        events.create_dataset("t", data=np.asarray([100, 200], np.int64))
        handle.create_dataset("depth/secret", data=np.asarray([42], np.int64))
        handle.create_dataset("navigation/time", data=np.asarray([7], np.int64))
        handle.create_dataset("gt_ttc", data=np.asarray([3.0], np.float32))
    report = fcwd_assets.inspect_hdf5(path)
    paths = {item["path"] for item in report["safe_datasets"]}
    assert "event_camera/events/x" in paths
    assert "event_camera/events/t" in paths
    assert all(
        "depth" not in item and "navigation" not in item and "gt" not in item for item in paths
    )
    assert report["timestamp_endpoints"] == [
        {
            "path": "event_camera/events/t",
            "count": 2,
            "first": 100,
            "last": 200,
            "finite": True,
            "monotonic_non_decreasing": True,
        }
    ]
    assert report["targets_read"] is False


def test_acquire_keeps_labels_opaque_and_writes_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, output = tmp_path / "data", tmp_path / "out"
    monkeypatch.setattr(
        fcwd_assets,
        "ASSETS",
        (
            fcwd_assets.Asset("raw", "raw-id", "inputs/sequence_1/raw.hdf5", "sensor_hdf5", "1K"),
            fcwd_assets.Asset(
                "target",
                "target-id",
                "sealed_labels/sequence_1/gt_ttc.csv",
                "ttc_target",
                "1K",
                True,
            ),
        ),
    )
    monkeypatch.setattr(fcwd_assets, "CALIBRATION_FOLDERS", (("camera", "folder", "1K"),))
    monkeypatch.setattr(
        fcwd_assets.shutil, "disk_usage", lambda _path: SimpleNamespace(free=20 * 1024**3)
    )

    def download(asset: fcwd_assets.Asset, destination: Path) -> Path:
        path = destination / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if asset.sealed_label:
            path.write_bytes(b"SECRET_TARGET,DO_NOT_PARSE\n")
        else:
            with h5py.File(path, "w") as handle:
                handle.create_dataset("event_camera/t", data=np.asarray([1, 2], np.int64))
        return path

    def folder(name: str, _drive_id: str, destination: Path) -> list[Path]:
        path = destination / "inputs/calibration" / name / "intrinsics.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"calibration")
        return [path]

    monkeypatch.setattr(fcwd_assets, "_download_file", download)
    monkeypatch.setattr(fcwd_assets, "_download_folder", folder)
    manifest = fcwd_assets.acquire(root, output)
    assert manifest["status"] == "COMPLETE"
    target = next(item for item in manifest["assets"] if item["name"] == "target")
    assert target["contents_read"] is False
    assert manifest["labels_opened"] is False and manifest["targets_read"] is False
    compatibility = json.loads((output / "HDF5_COMPATIBILITY.json").read_text("utf-8"))
    assert compatibility["status"] == "COMPATIBLE_METADATA_ONLY"
    table = json.loads((output / "GARL_TABLE_VII_CONTRACT.json").read_text("utf-8"))
    assert table["published_ours_full"] == {"FCWD1": 5.2, "FCWD2": 6.1, "FCWD3": 5.4}


def test_download_command_is_resumable_and_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset = fcwd_assets.Asset("raw", "id", "inputs/raw.hdf5", "sensor_hdf5", "1K")
    observed: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> None:
        assert check
        observed.append(command)
        Path(command[-1]).write_bytes(b"hdf5")

    monkeypatch.setattr(fcwd_assets.subprocess, "run", run)
    result = fcwd_assets._download_file(asset, tmp_path)
    assert result.read_bytes() == b"hdf5"
    assert "--continue" in observed[0]
    assert not result.with_suffix(".hdf5.partial").exists()
