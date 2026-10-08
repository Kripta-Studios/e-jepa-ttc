"""Label-free FCWD population and frozen H8/Garl event-input adapter.

FCWD publishes right-event-space oracle boxes at Blackfly timestamps.  The
official MATLAB code consumes those boxes directly for event filtering and
only later converts retained event coordinates to MATLAB's one-based convention.
This adapter therefore preserves the published event coordinates and clips
them to 1280x720.  It does not infer a Blackfly RGB crop: the release has
temporal synchronization but no audited event-to-RGB spatial correspondence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.garl_official_preprocessing import official_square_box
from e_jepa_ttc.efficient_context.garl_input import native_feature
from e_jepa_ttc.efficient_context.mapped_union import encode_union

SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
BBOX_AVAILABILITY_MARGIN_US = 50_000
BBOX_MAX_POST_AVAILABILITY_AGE_US = 51_000
H8_LAGS_US = np.arange(7, -1, -1, dtype=np.int64) * 50_000
BASE_WINDOWS_OFFSETS_US = ((-300_000, -200_000), (-200_000, -100_000), (-100_000, 0))
READ_BYTES_MAX = 256 * 1024**2
EVENT_BYTES_ESTIMATE = 12
REFERENCE_SCHEMA = "fcwd_reference_input_contract_v1"

CONTRACT: dict[str, Any] = {
    "schema": "fcwd_label_free_population_v1",
    "sequence_ids": ["FCWD1", "FCWD2", "FCWD3"],
    "camera": {"rgb": "blackflys/right", "events": "prophesee/event_cam_right"},
    "timestamps": "relative microseconds from each HDF5 recording start",
    "windows_us": [[-300_000, -200_000], [-200_000, -100_000], [-100_000, 0]],
    "h8_lags_us": H8_LAGS_US.tolist(),
    "bbox_availability_margin_us": BBOX_AVAILABILITY_MARGIN_US,
    "bbox_max_post_availability_age_us": BBOX_MAX_POST_AVAILABILITY_AGE_US,
    "bbox_coordinates": "published event-right xyxy used directly, then clipped to sensor",
    "own_events_shape": [8, 3, 12, 128, 128],
    "garl_events_shape": [40, 128, 128],
    "garl_full_sensor": "unavailable for every query: event-to-RGB spatial mapping unpublished",
    "forbidden_reads": ["gt_ttc.csv", "livox", "depth", "navigation"],
}


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _immutable_json(path: Path, value: object) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable FCWD artifact differs: {path}")
        return
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(data)
    os.replace(pending, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _pin_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _verify_stat(path: Path, expected: Mapping[str, Any]) -> None:
    actual = path.stat()
    if actual.st_size != int(expected["size_bytes"]) or actual.st_mtime_ns != int(
        expected["mtime_ns"]
    ):
        raise ValueError(f"frozen file stat changed: {path}")


class FCWDEventReader:
    """Bounded, half-open reader pinned to the Prophesee right camera."""

    timestamp_unit = "microseconds"
    paths = {
        "x": "prophesee/event_cam_right/x",
        "y": "prophesee/event_cam_right/y",
        "t": "prophesee/event_cam_right/t",
        "p": "prophesee/event_cam_right/p",
        "ms": "prophesee/event_cam_right/ms_map_idx",
    }

    def __init__(self, path: str | Path, *, chunk_events_max: int = 250_000) -> None:
        self.path = Path(path)
        self.chunk_events_max = int(chunk_events_max)
        if not 0 < self.chunk_events_max <= 250_000:
            raise ValueError("chunk_events_max must be in [1,250000]")
        self._handle: h5py.File | None = None

    def open(self) -> FCWDEventReader:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
            missing = sorted(path for path in self.paths.values() if path not in self._handle)
            if missing:
                self.close()
                raise ValueError(f"FCWD right-event datasets missing: {missing}")
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> FCWDEventReader:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require(self) -> h5py.File:
        return self.open()._handle  # type: ignore[return-value]

    def _bounds(self, start_us: int, end_us: int) -> tuple[int, int]:
        if start_us < 0 or end_us <= start_us:
            raise ValueError("window must satisfy 0 <= start_us < end_us")
        handle = self._require()
        timestamps = handle[self.paths["t"]]
        ms_map = handle[self.paths["ms"]]
        assert isinstance(timestamps, h5py.Dataset) and isinstance(ms_map, h5py.Dataset)
        if len(ms_map) == 0:
            return 0, 0
        start_ms = min(start_us // 1000, len(ms_map) - 1)
        end_ms = min(math.ceil(end_us / 1000) + 1, len(ms_map) - 1)
        rough_first = int(ms_map[start_ms])
        rough_last = int(ms_map[end_ms]) if end_ms < len(ms_map) - 1 else len(timestamps)
        if (rough_last - rough_first) * EVENT_BYTES_ESTIMATE > READ_BYTES_MAX:
            raise InterruptedError("FCWD event window exceeds 256 MiB read limit")
        local = np.asarray(timestamps[rough_first:rough_last], dtype=np.int64)
        first, last = np.searchsorted(local, (start_us, end_us), side="left")
        return rough_first + int(first), rough_first + int(last)

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[dict[str, np.ndarray]]:
        if not 0 < chunk_events <= self.chunk_events_max:
            raise ValueError("chunk_events exceeds configured bound")
        handle = self._require()
        first, last = self._bounds(start_us, end_us)
        datasets = {
            name: cast(h5py.Dataset, handle[path])
            for name, path in self.paths.items()
            if name != "ms"
        }
        if not all(isinstance(value, h5py.Dataset) for value in datasets.values()):
            raise ValueError("FCWD right-event paths must be datasets")
        for cursor in range(first, last, chunk_events):
            stop = min(last, cursor + chunk_events)
            timestamps = np.asarray(datasets["t"][cursor:stop], dtype=np.int64)
            keep = (timestamps >= start_us) & (timestamps < end_us)
            yield {
                "x": np.asarray(datasets["x"][cursor:stop], dtype=np.int32)[keep],
                "y": np.asarray(datasets["y"][cursor:stop], dtype=np.int32)[keep],
                "t": timestamps[keep],
                "p": np.asarray(datasets["p"][cursor:stop], dtype=np.int8)[keep],
            }

    def read_window(self, start_us: int, end_us: int) -> dict[str, np.ndarray]:
        pieces = list(self.iter_window_chunks(start_us, end_us))
        dtypes = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
        return {
            name: np.concatenate([piece[name] for piece in pieces])
            if pieces
            else np.empty(0, dtype=dtype)
            for name, dtype in dtypes.items()
        }


def _asset_lookup(asset_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in asset_manifest["assets"]}


def _verify_reference(reference_path: Path) -> dict[str, Any]:
    reference = _read_json(reference_path)
    if reference.get("schema") != REFERENCE_SCHEMA or reference.get("targets_read") is not False:
        raise ValueError("verified FCWD reference-input contract required")
    if reference.get("official_commit") != "79ff0842955304ec4f6164ec09baddc71386d225":
        raise ValueError("unexpected official FCWD source revision")
    return reference


def _clip_published_box(values: np.ndarray) -> list[float]:
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("FCWD bbox must contain four finite values")
    # Official main_FCWD filters using the CSV values directly, and only then
    # adds one to retained event coordinates for later MATLAB calculations.
    x0, y0, x1, y1 = values.astype(np.float64)
    result = [
        float(np.clip(x0, 0, SENSOR_WIDTH)),
        float(np.clip(y0, 0, SENSOR_HEIGHT)),
        float(np.clip(x1, 0, SENSOR_WIDTH)),
        float(np.clip(y1, 0, SENSOR_HEIGHT)),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("clipped FCWD bbox is degenerate")
    return result


def _causal_box(rows: list[dict[str, Any]], endpoint_us: int) -> dict[str, Any] | None:
    eligible = [row for row in rows if int(row["available_us"]) <= endpoint_us]
    if not eligible:
        return None
    selected = eligible[-1]
    if endpoint_us - int(selected["available_us"]) > BBOX_MAX_POST_AVAILABILITY_AGE_US:
        return None
    return selected


def build_manifest(
    dataset_root: Path,
    asset_manifest_path: Path,
    reference_contract_path: Path,
    destination: Path,
) -> dict[str, Any]:
    """Build every causal FCWD event query without opening TTC targets."""

    dataset_root = dataset_root.resolve()
    asset_manifest_path = asset_manifest_path.resolve()
    reference_contract_path = reference_contract_path.resolve()
    assets = _read_json(asset_manifest_path)
    reference = _verify_reference(reference_contract_path)
    if assets.get("status") != "COMPLETE" or assets.get("targets_read") is not False:
        raise ValueError("complete opaque FCWD asset inventory required")
    lookup = _asset_lookup(assets)
    reference_sequences = {int(item["sequence"]): item for item in reference["sequences"]}
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, dict[str, int]] = {}
    source_files: list[dict[str, Any]] = []
    for number in (1, 2, 3):
        sequence_id = f"FCWD{number}"
        hdf_asset = lookup[f"fcwd_{number}_hdf5"]
        bbox_asset = lookup[f"fcwd_{number}_bbox"]
        raw_path = Path(hdf_asset["path"]).resolve()
        bbox_path = Path(bbox_asset["path"]).resolve()
        if raw_path.parent != dataset_root / "inputs" / f"sequence_{number}":
            raise ValueError(f"{sequence_id}: HDF5 path escapes the expected input root")
        if bbox_path.parent != dataset_root / "sealed_labels" / f"sequence_{number}":
            raise ValueError(f"{sequence_id}: bbox path escapes the permitted oracle-input root")
        if _sha256_file(bbox_path) != bbox_asset["sha256"]:
            raise ValueError(f"{sequence_id}: bbox bytes changed")
        reference_row = reference_sequences[number]
        if reference_row["bbox_sha256"] != bbox_asset["sha256"]:
            raise ValueError(f"{sequence_id}: bbox reference contract changed")
        values = np.loadtxt(bbox_path, delimiter=",", dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6 or len(values) != reference_row["bbox_rows"]:
            raise ValueError(f"{sequence_id}: bbox CSV schema changed")
        with h5py.File(raw_path, "r") as handle:
            rgb_timestamps = np.asarray(handle["blackflys/right/ts"], dtype=np.int64)
            event_timestamps = handle[FCWDEventReader.paths["t"]]
            assert isinstance(event_timestamps, h5py.Dataset)
            stream_start, stream_end = int(event_timestamps[0]), int(event_timestamps[-1]) + 1
        indices = values[:, 0].astype(np.int64)
        timestamps = np.rint(values[:, 1] * 1_000_000).astype(np.int64)
        if (
            (values[:, 0] != indices).any()
            or (indices < 0).any()
            or (indices >= len(rgb_timestamps)).any()
            or not np.array_equal(rgb_timestamps[indices], timestamps)
        ):
            raise ValueError(f"{sequence_id}: bbox/RGB timestamp identity changed")
        bbox_rows = [
            {
                "row_index": index,
                "rgb_frame_index": int(indices[index]),
                "timestamp_us": int(timestamps[index]),
                "available_us": int(timestamps[index]) + BBOX_AVAILABILITY_MARGIN_US,
                "box_xyxy": _clip_published_box(values[index, 2:6]),
            }
            for index in range(len(values))
        ]
        rejected_history = rejected_causal = 0
        selected = 0
        for anchor in timestamps:
            anchor_us = int(anchor)
            if anchor_us - 650_000 < stream_start or anchor_us > stream_end:
                rejected_history += 1
                continue
            endpoints = (anchor_us - 200_000, anchor_us - 100_000, anchor_us)
            sources = [_causal_box(bbox_rows, endpoint) for endpoint in endpoints]
            if any(source is None for source in sources):
                rejected_causal += 1
                continue
            admitted = [source for source in sources if source is not None]
            boxes = [source["box_xyxy"] for source in admitted]
            square = common_square_from_boxes(boxes, (0, 1, 2))
            row: dict[str, Any] = {
                "sequence_id": sequence_id,
                "query_id": f"{sequence_id}:{selected:04d}",
                "anchor_us": anchor_us,
                "anchor_relative_seconds": anchor_us / 1_000_000.0,
                "raw_path": str(raw_path),
                "raw_stat": _pin_stat(raw_path),
                "windows_us": [
                    [anchor_us + start, anchor_us + end]
                    for start, end in BASE_WINDOWS_OFFSETS_US
                ],
                "boxes_xyxy3": boxes,
                "square_xyxy": list(square),
                "bbox_sources": [
                    {
                        "path": str(bbox_path),
                        "sha256": bbox_asset["sha256"],
                        "row_index": source["row_index"],
                        "rgb_frame_index": source["rgb_frame_index"],
                        "timestamp_us": source["timestamp_us"],
                        "available_us": source["available_us"],
                    }
                    for source in admitted
                ],
                "garl_full_available": False,
                "garl_full_unavailable_reason": "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING",
            }
            row["metadata_sha256"] = _canonical_sha256(row)
            rows.append(row)
            selected += 1
        exclusions[sequence_id] = {
            "bbox_rows": len(values),
            "history_support": rejected_history,
            "causal_box_unavailable_or_stale": rejected_causal,
            "selected": selected,
        }
        source_files.extend(
            [
                {
                    "role": "sensor_hdf5",
                    "path": str(raw_path),
                    "sha256": hdf_asset["sha256"],
                },
                {
                    "role": "oracle_event_bbox",
                    "path": str(bbox_path),
                    "sha256": bbox_asset["sha256"],
                },
            ]
        )
    result = {
        **CONTRACT,
        "status": "FROZEN_LABEL_FREE",
        "rows": rows,
        "query_count": len(rows),
        "sequence_count": 3,
        "causal_exclusions": exclusions,
        "source_files": source_files,
        "asset_manifest_path": str(asset_manifest_path),
        "asset_manifest_sha256": _sha256_file(asset_manifest_path),
        "reference_contract_path": str(reference_contract_path),
        "reference_contract_sha256": _sha256_file(reference_contract_path),
        "bbox_values_read": True,
        "ttc_targets_read": False,
        "selection_uses_targets_or_predictions": False,
        "rows_metadata_sha256": _canonical_sha256(
            [row["metadata_sha256"] for row in rows]
        ),
    }
    _immutable_json(destination, result)
    freeze = {
        "status": "FROZEN_LABEL_FREE",
        "manifest_path": str(destination.resolve()),
        "manifest_sha256": _sha256_file(destination),
        "query_count": len(rows),
        "sequence_count": 3,
        "input_adapter_sha256": _sha256_file(Path(__file__)),
        "asset_manifest_sha256": result["asset_manifest_sha256"],
        "reference_contract_sha256": result["reference_contract_sha256"],
        "bbox_values_read": True,
        "ttc_targets_read": False,
        "gpu_used": False,
        "optimizer_updates": 0,
    }
    _immutable_json(destination.with_name("FCWD_INPUT_FREEZE.json"), freeze)
    return result


def _validate_row(row: Mapping[str, Any]) -> Path:
    metadata = dict(row)
    expected = str(metadata.pop("metadata_sha256"))
    if _canonical_sha256(metadata) != expected:
        raise ValueError("FCWD query metadata SHA-256 mismatch")
    path = Path(str(row["raw_path"]))
    _verify_stat(path, row["raw_stat"])
    return path


def prepare(row: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare frozen H8 and Garl event-only tensors; preserve full-RGB absence."""

    raw_path = _validate_row(row)
    windows = np.asarray(row["windows_us"], dtype=np.int64)
    if windows.shape != (3, 2) or not np.array_equal(
        windows[:, 1] - windows[:, 0], np.full(3, 100_000)
    ):
        raise ValueError("FCWD query requires three exact 100 ms windows")
    boxes: list[tuple[float, float, float, float]] = []
    for box in row["boxes_xyxy3"]:
        if len(box) != 4:
            raise ValueError("FCWD event box must be xyxy")
        boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
    if len(boxes) != 3:
        raise ValueError("FCWD query requires three causal event-space boxes")
    full_lags = np.concatenate((np.arange(15, 7, -1, dtype=np.int64) * 50_000, H8_LAGS_US))
    valid16 = np.concatenate((np.zeros(8, dtype=bool), np.ones(8, dtype=bool)))
    with FCWDEventReader(raw_path) as reader:
        own = encode_union(
            reader,  # type: ignore[arg-type]
            windows,
            full_lags,
            valid16,
            tuple(float(value) for value in row["square_xyxy"]),
            sequence_id=str(row["sequence_id"]),
            roi_size=128,
            event_pixel_diff=0.0,
            retained_bytes_max=1024 * 1024**2,
        )[-8:]
        garl_parts = []
        garl_squares = []
        for local_index, window_index in enumerate((1, 2)):
            raw = reader.read_window(*[int(value) for value in windows[window_index]])
            if not len(raw["t"]):
                garl_parts = []
                break
            square = official_square_box(boxes[-2:], local_index)
            garl_squares.append(list(square))
            garl_parts.append(native_feature(raw, square, size=128).numpy())
    own_events = own.numpy().astype(np.float32, copy=False)
    garl_events = (
        np.concatenate(garl_parts, axis=0).astype(np.float32, copy=False)
        if len(garl_parts) == 2
        else None
    )
    if own_events.shape != (8, 3, 12, 128, 128):
        raise ValueError("FCWD H8 tensor violates frozen shape")
    if garl_events is not None and garl_events.shape != (40, 128, 128):
        raise ValueError("FCWD Garl event-only tensor violates frozen shape")
    return {
        "own_events": own_events,
        "garl_events": garl_events,
        "garl_unavailable_reason": "EMPTY_EVENT_ENDPOINT" if garl_events is None else None,
        "garl_full_sensor": None,
        "garl_full_unavailable_reason": "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING",
        "valid": np.ones(8, dtype=bool),
        "delta_t_s": np.float32(0.1),
        "metadata": {
            "camera": "prophesee/event_cam_right",
            "coordinate_conversion": "published event-space bbox unchanged, then sensor clip",
            "garl_squares_xyxy": garl_squares,
            "full_rgb_branch_status": "BLOCKED_MISSING_INPUT_CORRESPONDENCE",
            "targets_read": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(
        args.dataset_root, args.asset_manifest, args.reference_contract, args.output
    )
    print(json.dumps({"queries": result["query_count"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONTRACT", "FCWDEventReader", "H8_LAGS_US", "build_manifest", "prepare"]
