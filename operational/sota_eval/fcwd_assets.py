"""Acquire and inventory the public FCWD assets without reading target values.

The public HDF5 recordings and calibration files are model inputs. Bounding-box
and TTC CSV files are downloaded into a separate sealed directory and hashed as
opaque bytes; this module never parses them. HDF5 inspection is restricted to
camera, event, image, calibration, and timestamp metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_PAGE = "https://nail-hnu.github.io/EventAidedTTC/"
PAPER_URL = "https://arxiv.org/abs/2407.07324"
GARL_PAPER_URL = "https://arxiv.org/abs/2603.16303"
DEFAULT_ROOT = Path("E:/FCWD_dataset")
DEFAULT_OUTPUT = Path("artifacts/sota_campaign_20261008/fcwd")
DOWNLOAD_BUDGET_BYTES = 10 * 1024**3
MIN_FREE_BEFORE_DOWNLOAD_BYTES = 10 * 1024**3
BLOCK_SIZE = 8 * 1024**2


@dataclass(frozen=True)
class Asset:
    """One immutable public FCWD download."""

    name: str
    drive_id: str
    relative_path: str
    role: str
    published_size: str
    sealed_label: bool = False

    @property
    def url(self) -> str:
        return f"https://drive.google.com/file/d/{self.drive_id}/view?usp=sharing"


ASSETS = (
    Asset(
        "fcwd_1_hdf5",
        "1UP7mGo9Sm-FBe8cw1j51ej2hwlYa-E9Y",
        "inputs/sequence_1/2024-03-02-10-35-27.hdf5",
        "sensor_hdf5",
        "2.22G",
    ),
    Asset(
        "fcwd_2_hdf5",
        "1fQmlxH5JMgV6JzoxjV4XcRQ2uSkLT4fs",
        "inputs/sequence_2/2024-03-02-10-37-09.hdf5",
        "sensor_hdf5",
        "2.01G",
    ),
    Asset(
        "fcwd_3_hdf5",
        "1o0TWqkc6KCbHsEGwwxiVI-XyZeM5FUZZ",
        "inputs/sequence_3/2024-03-02-10-38-25.hdf5",
        "sensor_hdf5",
        "1.99G",
    ),
    Asset(
        "fcwd_1_bbox",
        "160p6IE2BTUJG_4Pw3NoTYKjupV_OtfZm",
        "sealed_labels/sequence_1/bounding_box.csv",
        "bbox_label",
        "6K",
        True,
    ),
    Asset(
        "fcwd_1_ttc",
        "1Nq7YwqAvua4N2FZqBFq4CLHRQKdHN0om",
        "sealed_labels/sequence_1/gt_ttc.csv",
        "ttc_target",
        "7K",
        True,
    ),
    Asset(
        "fcwd_2_bbox",
        "1bXw7DTz0symg-liO_t4EsQJQ1cSUoxe2",
        "sealed_labels/sequence_2/bounding_box.csv",
        "bbox_label",
        "7K",
        True,
    ),
    Asset(
        "fcwd_2_ttc",
        "1-MTOlgYlXSa7v7MTSdux0mw2smlUd25V",
        "sealed_labels/sequence_2/gt_ttc.csv",
        "ttc_target",
        "7K",
        True,
    ),
    Asset(
        "fcwd_3_bbox",
        "1obM4PKJyX4VPhpJ8NUJ1Q3snR1StDPxw",
        "sealed_labels/sequence_3/bounding_box.csv",
        "bbox_label",
        "6K",
        True,
    ),
    Asset(
        "fcwd_3_ttc",
        "1gH80TDTE-rLW55fkHPAzdGCPXR5yVhn1",
        "sealed_labels/sequence_3/gt_ttc.csv",
        "ttc_target",
        "7K",
        True,
    ),
)

CALIBRATION_FOLDERS = (
    ("camera_intrinsic", "1RH8MnHrv_L_Z_EQLivvwmZ4gHGLY79Lf", "112K"),
    ("lidar_to_rgb_extrinsic", "1Wpgm6pMLHus4JQ4NTEYWk29xQY8aDA6X", "1K"),
)

_SAFE_NAME_TOKENS = (
    "event",
    "camera",
    "cam",
    "image",
    "rgb",
    "frame",
    "calib",
    "intrinsic",
    "timestamp",
    "time",
)
_SAFE_FCWD_ROOTS = ("base_timestamp", "blackflys", "prophesee")
_FORBIDDEN_NAME_TOKENS = (
    "gt",
    "groundtruth",
    "ground_truth",
    "ttc",
    "depth",
    "lidar",
    "nav",
    "imu",
    "gps",
    "pose",
    "odometry",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def check_capacity(root: Path, existing_bytes: int = 0) -> dict[str, int]:
    """Require a full 10 GiB reserve and enforce the FCWD byte budget."""

    root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(root).free)
    if existing_bytes > DOWNLOAD_BUDGET_BYTES:
        raise RuntimeError("FCWD assets exceed the authorized 10 GiB budget")
    if free <= MIN_FREE_BEFORE_DOWNLOAD_BYTES:
        raise RuntimeError("FCWD requires more than 10 GiB free before download")
    return {
        "free_bytes": free,
        "existing_bytes": existing_bytes,
        "budget_bytes": DOWNLOAD_BUDGET_BYTES,
    }


def _download_file(asset: Asset, root: Path) -> Path:
    destination = root / asset.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--continue",
        asset.drive_id,
        "--output",
        str(partial),
    ]
    subprocess.run(command, check=True)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"empty FCWD download: {asset.name}")
    os.replace(partial, destination)
    return destination


def _download_folder(name: str, drive_id: str, root: Path) -> list[Path]:
    destination = root / "inputs/calibration" / name
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        f"https://drive.google.com/drive/folders/{drive_id}",
        "--output",
        str(destination),
    ]
    subprocess.run(command, check=True)
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"empty FCWD calibration folder: {name}")
    return files


def _dataset_metadata(dataset: h5py.Dataset) -> dict[str, Any]:
    chunks = None if dataset.chunks is None else list(dataset.chunks)
    return {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "chunks": chunks,
        "compression": dataset.compression,
    }


def _timestamp_metadata(dataset: h5py.Dataset, *, block_rows: int = 1_000_000) -> dict[str, Any]:
    """Validate timestamp ordering in bounded blocks."""

    previous: int | float | None = None
    monotonic = True
    finite = True
    row_count = int(dataset.shape[0])
    for start in range(0, row_count, block_rows):
        values = np.asarray(dataset[start : start + block_rows])
        if values.size == 0:
            continue
        finite = finite and bool(np.isfinite(values).all())
        monotonic = monotonic and bool((np.diff(values) >= 0).all())
        first = values[0].item()
        if previous is not None and first < previous:
            monotonic = False
        previous = values[-1].item()
    return {
        "count": row_count,
        "first": dataset[0].item(),
        "last": dataset[-1].item(),
        "finite": finite,
        "monotonic_non_decreasing": monotonic,
    }


def inspect_hdf5(path: Path) -> dict[str, Any]:
    """Inspect only safe HDF5 schema and endpoint timestamps, never GT/depth/nav."""

    datasets: list[dict[str, Any]] = []
    timestamp_endpoints: list[dict[str, Any]] = []
    skipped_roots: list[str] = []
    with h5py.File(path, "r") as handle:
        root_names = sorted(handle.keys())
        for root_name in root_names:
            lowered = root_name.lower()
            if any(token in lowered for token in _FORBIDDEN_NAME_TOKENS):
                skipped_roots.append(root_name)
                continue
            if lowered not in _SAFE_FCWD_ROOTS and not any(
                token in lowered for token in _SAFE_NAME_TOKENS
            ):
                skipped_roots.append(root_name)
                continue

            def visitor(
                name: str,
                value: h5py.Group | h5py.Dataset,
                parent: str = root_name,
            ) -> None:
                full_name = f"{parent}/{name}" if name else parent
                lowered_name = full_name.lower()
                if any(token in lowered_name for token in _FORBIDDEN_NAME_TOKENS):
                    return
                if isinstance(value, h5py.Dataset):
                    row = {"path": full_name, **_dataset_metadata(value)}
                    datasets.append(row)
                    if (
                        any(token in lowered_name for token in ("timestamp", "/t", "time"))
                        and value.ndim == 1
                        and int(value.shape[0]) > 0
                        and value.dtype.kind in "iuf"
                    ):
                        timestamp_endpoints.append(
                            {
                                "path": full_name,
                                **_timestamp_metadata(value),
                            }
                        )

            node = handle[root_name]
            if isinstance(node, h5py.Dataset):
                visitor("", node)
            elif isinstance(node, h5py.Group):
                node.visititems(visitor)
    by_path = {item["path"]: item for item in datasets}
    event_prefix = "prophesee/event_cam_right/"
    event_paths = [event_prefix + name for name in ("x", "y", "t", "p")]
    event_lengths = {path: by_path[path]["shape"][0] for path in event_paths if path in by_path}
    frame_count = by_path.get("blackflys/right/data", {}).get("shape", [None])[0]
    frame_ts_count = by_path.get("blackflys/right/ts", {}).get("shape", [None])[0]
    return {
        "file": path.name,
        "root_names": root_names,
        "safe_datasets": datasets,
        "timestamp_endpoints": timestamp_endpoints,
        "skipped_roots_unread": skipped_roots,
        "adapter_contract": {
            "event_paths": event_paths,
            "event_columns_present": len(event_lengths) == 4,
            "event_columns_equal_length": len(set(event_lengths.values())) == 1,
            "event_column_lengths": event_lengths,
            "rgb_path": "blackflys/right/data",
            "rgb_timestamp_path": "blackflys/right/ts",
            "rgb_frames_match_timestamps": frame_count == frame_ts_count,
            "event_ms_index_path": "prophesee/event_cam_right/ms_map_idx",
            "rgb_event_ms_index_path": "blackflys/right/prophesee_event_cam_right_ms_map_idx",
        },
        "targets_read": False,
        "depth_navigation_read": False,
    }


def _files(root: Path) -> Iterable[Path]:
    return (
        path for path in root.rglob("*") if path.is_file() and not path.name.endswith(".partial")
    )


def acquire(root: Path, output: Path, *, download: bool = True) -> dict[str, Any]:
    """Download, hash, and safely inspect FCWD public assets."""

    root, output = root.resolve(), output.resolve()
    existing = sum(path.stat().st_size for path in _files(root)) if root.exists() else 0
    capacity = check_capacity(root, existing)
    seal = root / "sealed_labels/SEALED_DO_NOT_READ_BEFORE_PREDICTIONS.txt"
    seal.parent.mkdir(parents=True, exist_ok=True)
    if not seal.exists():
        seal.write_text(
            "Opaque label assets. Do not parse before prediction artifacts are frozen.\n",
            encoding="utf-8",
        )
    if download:
        for asset in ASSETS:
            _download_file(asset, root)
            current = sum(path.stat().st_size for path in _files(root))
            if current > DOWNLOAD_BUDGET_BYTES:
                raise RuntimeError("FCWD downloads exceeded the authorized 10 GiB budget")
        for name, drive_id, _published_size in CALIBRATION_FOLDERS:
            _download_folder(name, drive_id, root)

    missing = [asset.name for asset in ASSETS if not (root / asset.relative_path).is_file()]
    records = []
    inspections = []
    for asset in ASSETS:
        path = root / asset.relative_path
        if not path.is_file():
            continue
        records.append(
            {
                "name": asset.name,
                "role": asset.role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "source_url": asset.url,
                "drive_id": asset.drive_id,
                "published_size": asset.published_size,
                "sealed_label": asset.sealed_label,
                "contents_read": False if asset.sealed_label else asset.role == "sensor_hdf5",
            }
        )
        if asset.role == "sensor_hdf5":
            inspections.append(inspect_hdf5(path))

    calibration = []
    missing_calibration = []
    for name, drive_id, published_size in CALIBRATION_FOLDERS:
        directory = root / "inputs/calibration" / name
        found = False
        for path in sorted(directory.rglob("*")) if directory.exists() else ():
            if path.is_file():
                found = True
                calibration.append(
                    {
                        "name": name,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "folder_url": f"https://drive.google.com/drive/folders/{drive_id}",
                        "published_size": published_size,
                    }
                )
        if not found:
            missing_calibration.append(name)

    total = sum(item["bytes"] for item in records + calibration)
    if total > DOWNLOAD_BUDGET_BYTES:
        raise RuntimeError("FCWD inventory exceeds the authorized 10 GiB budget")
    manifest = {
        "schema": "fcwd_public_assets_v1",
        "status": (
            "COMPLETE" if not missing and not missing_calibration else "DEPENDENCIES_RECORDED"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset_root": str(root),
        "project_page": PROJECT_PAGE,
        "paper": PAPER_URL,
        "capacity_preflight": capacity,
        "download_budget_bytes": DOWNLOAD_BUDGET_BYTES,
        "total_bytes": total,
        "assets": records,
        "calibration_assets": calibration,
        "missing_assets": missing,
        "missing_calibration_folders": missing_calibration,
        "labels_downloaded_as_opaque_bytes": True,
        "labels_opened": False,
        "targets_read": False,
        "depth_navigation_read": False,
        "resume_strategy": "gdown --continue into .partial followed by atomic rename",
        "source_sha256": _sha256(Path(__file__)),
        "dependencies": {
            "python": sys.version.split()[0],
            "gdown": importlib.metadata.version("gdown"),
            "h5py": h5py.__version__,
        },
    }
    compatibility = {
        "status": "COMPATIBLE_METADATA_ONLY" if inspections else "PENDING_HDF5",
        "hdf5": inspections,
        "required_adapter_work": [
            "map discovered event x/y/t/p datasets to EventBatch without copying the full stream",
            "align RGB/event timestamps using only recorded sensor timestamps",
            "freeze prediction inputs before opening sealed bbox/TTC CSV files",
            "score per-sequence relative TTC error under the published no-fine-tuning protocol",
        ],
        "architecture_or_training_changes": False,
        "prior_repository_exposure": {
            "status": "DEPENDENCY_ONLY_BEFORE_THIS_MODULE",
            "population_contract": (
                "operational/sota_eval/population.py declared FCWD separately gated"
            ),
            "existing_baseline_note": (
                "STRTTC wrapper declares its causal ROI differs from official FCWD epochs"
            ),
            "prior_download_or_adapter_found": False,
        },
    }
    table_vii = {
        "schema": "garl_paper_table_vii_fcwd_reference_v1",
        "source": GARL_PAPER_URL,
        "evaluation": "FCWD benchmark without model fine-tuning",
        "metric": "RTE_percent_lower_is_better",
        "published_ours_full": {"FCWD1": 5.2, "FCWD2": 6.1, "FCWD3": 5.4},
        "own_results": None,
        "not_a_reproduced_result": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "ASSET_MANIFEST.json", manifest)
    _atomic_json(output / "HDF5_COMPATIBILITY.json", compatibility)
    _atomic_json(output / "GARL_TABLE_VII_CONTRACT.json", table_vii)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    result = acquire(args.root, args.output, download=not args.inventory_only)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ASSETS", "DOWNLOAD_BUDGET_BYTES", "acquire", "check_capacity", "inspect_hdf5"]
