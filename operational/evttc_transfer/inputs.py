"""Build label-free EvTTC queries and prepare frozen H8/Garl inputs.

This module intentionally separates query selection and prediction inputs from
scoring.  It never opens TTC, distance, depth, navigation, or ``gt.hdf5``
assets.  The object ROI is an oracle RGB-label ROI, made causally available one
frame (50 ms) after the labelled Blackfly-left frame.

The RGB-left box is mapped to the event-left sensor with the documented
``T_bfs_to_prophesee`` direction.  Rays are undistorted with the RGB camera
calibration, rotated by the RGB-to-event rotation, and distorted with the event
camera calibration.  Translation is deliberately ignored because applying it
would require object depth.  Consequently this is a transparent rotation-only
cross-camera approximation, not a depth-correct projection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.annotations import _bbox_from_segmentation
from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.evttc import HDF5EventLayout, discover_event_layout, read_manifest
from e_jepa_ttc.data.evttc_object_cache import _distort_radtan, _undistort_radtan
from e_jepa_ttc.data.garl_official_preprocessing import official_square_box
from e_jepa_ttc.data.types import DatasetSequence
from e_jepa_ttc.efficient_context.garl_input import native_feature
from e_jepa_ttc.efficient_context.mapped_union import encode_union

SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
QUERY_COUNT_PER_SEQUENCE = 32
BBOX_AVAILABILITY_MARGIN_US = 50_000
BBOX_MAX_POST_AVAILABILITY_AGE_US = 51_000
BASE_WINDOWS_OFFSETS_US = ((-300_000, -200_000), (-200_000, -100_000), (-100_000, 0))
H8_LAGS_US = np.arange(7, -1, -1, dtype=np.int64) * 50_000
READ_BYTES_MAX = 256 * 1024**2
EVENT_BYTES_ESTIMATE = 12

LIMITATIONS: tuple[str, ...] = (
    "historical EvTTC development32 transfer cohort; not an official blind benchmark split",
    "oracle RGB-left object boxes are prediction inputs",
    "boxes are available only after a conservative 50 ms causal margin",
    "RGB-left to event-left projection uses calibrated rotation but omits translation "
    "because depth is forbidden",
    "no TTC, distance, depth, navigation, gt.hdf5, future box interpolation, or model "
    "prediction selects queries",
)


def _resolve_sequence_path(sequence: DatasetSequence, data_root: Path) -> Path:
    parts = sequence.extra.get("relative_parts")
    if not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
        raise ValueError(f"{sequence.sequence_id}: manifest lacks relative_parts")
    return data_root.joinpath(*parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _left_event_layout(path: Path) -> HDF5EventLayout:
    layout = discover_event_layout(path)
    expected = {
        "x": "prophesee/event_cam_left/x",
        "y": "prophesee/event_cam_left/y",
        "t": "prophesee/event_cam_left/t",
        "p": "prophesee/event_cam_left/p",
        "ms_map_idx": "prophesee/event_cam_left/ms_map_idx",
    }
    if layout is None or layout.kind != "separate":
        raise ValueError("EvTTC requires separate Prophesee-left event datasets")
    actual = {name: getattr(layout, name) for name in expected}
    if actual != expected:
        raise ValueError(
            f"automatic camera selection is forbidden; expected {expected}, got {actual}"
        )
    return layout


def _calibration(handle: h5py.File) -> dict[str, np.ndarray]:
    event_transform = next(
        (
            path
            for path in (
                "prophesee/event_cam_left/calib/T_bfs_to_prophesee",
                "prophesee/event_cam_left/calib/T_to_left_bfs",
            )
            if path in handle
        ),
        "prophesee/event_cam_left/calib/T_bfs_to_prophesee",
    )
    names = {
        "rgb_intrinsics": "blackflys/left/calib/intrinsics",
        "rgb_distortion": "blackflys/left/calib/distortion_coeffs",
        "event_intrinsics": "prophesee/event_cam_left/calib/intrinsics",
        "event_distortion": "prophesee/event_cam_left/calib/distortion_coeffs",
        # EvTTC files use both names.  This protocol adopts the documented
        # Blackfly-left -> Prophesee-left direction for either dataset name;
        # the legacy name alone is not treated as proof of transform direction.
        "event_from_rgb": event_transform,
    }
    missing = sorted(path for path in names.values() if path not in handle)
    if missing:
        raise ValueError(f"left-camera calibration is incomplete: {missing}")
    values = {name: np.asarray(handle[path], dtype=np.float64) for name, path in names.items()}
    if values["event_from_rgb"].shape != (4, 4):
        raise ValueError("T_bfs_to_prophesee must be a 4x4 RGB-to-event transform")
    return values


def _calibration_sha256(calibration: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(calibration):
        value = np.ascontiguousarray(calibration[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def project_rgb_box_rotation_only(
    box_xyxy: Sequence[float], calibration: Mapping[str, np.ndarray]
) -> tuple[float, float, float, float]:
    """Map an RGB-left box to event-left using calibrated rays and rotation only."""

    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain four coordinates")
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    if not np.isfinite((x0, y0, x1, y1)).all() or x1 <= x0 or y1 <= y0:
        raise ValueError("box_xyxy must be finite and non-degenerate")
    corners = np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float64)
    rgb = np.asarray(calibration["rgb_intrinsics"], dtype=np.float64)
    event = np.asarray(calibration["event_intrinsics"], dtype=np.float64)
    normalized_distorted = np.column_stack(
        ((corners[:, 0] - rgb[2]) / rgb[0], (corners[:, 1] - rgb[3]) / rgb[1])
    )
    normalized = _undistort_radtan(normalized_distorted, calibration["rgb_distortion"])
    rays_rgb = np.column_stack((normalized, np.ones(len(normalized), dtype=np.float64)))
    rotation = np.asarray(calibration["event_from_rgb"], dtype=np.float64)[:3, :3]
    rays_event = (rotation @ rays_rgb.T).T
    if np.any(rays_event[:, 2] <= 0.0):
        raise ValueError("rotation-only projection places an RGB ray behind event-left")
    event_normalized = rays_event[:, :2] / rays_event[:, 2:3]
    event_distorted = _distort_radtan(event_normalized, calibration["event_distortion"])
    pixels = np.column_stack(
        (event_distorted[:, 0] * event[0] + event[2], event_distorted[:, 1] * event[1] + event[3])
    )
    result = (
        float(pixels[:, 0].min()),
        float(pixels[:, 1].min()),
        float(pixels[:, 0].max()),
        float(pixels[:, 1].max()),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("projected event-left box is degenerate")
    return result


class EvTTCEventReader:
    """Bounded half-open EvTTC reader with explicit microsecond/ms-map semantics."""

    timestamp_unit = "microseconds"
    millisecond_index_semantics = "ms_map_idx[floor(t_us/1000)] then exact half-open refinement"

    def __init__(self, path: str | Path, *, chunk_events_max: int = 250_000) -> None:
        self.path = Path(path)
        self.chunk_events_max = int(chunk_events_max)
        if self.chunk_events_max <= 0 or self.chunk_events_max > 250_000:
            raise ValueError("chunk_events_max must be in [1, 250000]")
        self.layout = _left_event_layout(self.path)
        self._handle: h5py.File | None = None

    def open(self) -> EvTTCEventReader:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> EvTTCEventReader:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require_handle(self) -> h5py.File:
        self.open()
        if self._handle is None:
            raise RuntimeError("failed to open EvTTC event stream")
        return self._handle

    def _bounds(self, start_us: int, end_us: int) -> tuple[int, int]:
        if start_us < 0 or end_us <= start_us:
            raise ValueError("window must satisfy 0 <= start_us < end_us")
        handle = self._require_handle()
        assert self.layout.t and self.layout.ms_map_idx
        timestamps = handle[self.layout.t]
        ms_map = handle[self.layout.ms_map_idx]
        if not isinstance(timestamps, h5py.Dataset) or not isinstance(ms_map, h5py.Dataset):
            raise ValueError("event timestamps and ms_map_idx must be HDF5 datasets")
        if len(ms_map) == 0:
            return 0, 0
        start_ms = min(max(start_us // 1000, 0), len(ms_map) - 1)
        end_ms = min(max(math.ceil(end_us / 1000) + 1, 0), len(ms_map) - 1)
        rough_first = int(ms_map[start_ms])
        rough_last = int(ms_map[end_ms]) if end_ms < len(ms_map) - 1 else len(timestamps)
        if (rough_last - rough_first) * EVENT_BYTES_ESTIMATE > READ_BYTES_MAX:
            raise InterruptedError("EvTTC window exceeds the explicit 256 MiB read limit")
        local_t = np.asarray(timestamps[rough_first:rough_last], dtype=np.int64)
        first, last = np.searchsorted(local_t, (start_us, end_us), side="left")
        return rough_first + int(first), rough_first + int(last)

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield aligned events in exactly ``[start_us, end_us)``."""

        if chunk_events <= 0 or chunk_events > self.chunk_events_max:
            raise ValueError(f"chunk_events must be in [1, {self.chunk_events_max}]")
        handle = self._require_handle()
        assert self.layout.x and self.layout.y and self.layout.t and self.layout.p
        x_dataset = handle[self.layout.x]
        y_dataset = handle[self.layout.y]
        t_dataset = handle[self.layout.t]
        p_dataset = handle[self.layout.p]
        if not all(
            isinstance(dataset, h5py.Dataset)
            for dataset in (x_dataset, y_dataset, t_dataset, p_dataset)
        ):
            raise ValueError("EvTTC x/y/t/p paths must identify datasets")
        assert isinstance(x_dataset, h5py.Dataset)
        assert isinstance(y_dataset, h5py.Dataset)
        assert isinstance(t_dataset, h5py.Dataset)
        assert isinstance(p_dataset, h5py.Dataset)
        first, last = self._bounds(start_us, end_us)
        for cursor in range(first, last, chunk_events):
            stop = min(last, cursor + chunk_events)
            timestamps = np.asarray(t_dataset[cursor:stop], dtype=np.int64)
            exact = (timestamps >= start_us) & (timestamps < end_us)
            yield {
                "x": np.asarray(x_dataset[cursor:stop], dtype=np.int32)[exact],
                "y": np.asarray(y_dataset[cursor:stop], dtype=np.int32)[exact],
                "t": timestamps[exact],
                "p": np.asarray(p_dataset[cursor:stop], dtype=np.int8)[exact],
            }

    def read_window(self, start_us: int, end_us: int) -> dict[str, np.ndarray]:
        """Read a bounded exact window; intended only for one 100 ms endpoint."""

        pieces = list(self.iter_window_chunks(start_us, end_us))
        dtypes = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
        return {
            key: np.concatenate([piece[key] for piece in pieces])
            if pieces
            else np.empty(0, dtype=dtype)
            for key, dtype in dtypes.items()
        }


def _compatible_category(category: object, target_type: str | None) -> bool:
    normalized = str(category).strip().lower()
    if normalized in {"background", "__background__", "_background_"}:
        return False
    if target_type == "car":
        return normalized == "car"
    if target_type == "pedestrian":
        return normalized in {"pedestrian", "person"}
    raise ValueError(f"unsupported EvTTC target_type for ROI admission: {target_type!r}")


def _single_object_bbox(item: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = _bbox_from_segmentation(item.get("segmentation"))
    raw_bbox = item.get("bbox")
    if bbox is None and isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
        bbox = tuple(float(value) for value in raw_bbox[:4])
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if not np.isfinite((x0, y0, x1, y1)).all() or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _label_rows(
    label_dir: Path,
    frame_timestamps: np.ndarray,
    *,
    target_type: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    audit = {
        "label_json_total": 0,
        "label_frame_index_invalid": 0,
        "no_compatible_object": 0,
        "ambiguous_compatible_objects": 0,
        "compatible_bbox_invalid": 0,
        "accepted_single_compatible_object": 0,
    }
    for path in sorted(label_dir.glob("*.json")):
        audit["label_json_total"] += 1
        try:
            frame_index = int(path.stem)
        except ValueError:
            audit["label_frame_index_invalid"] += 1
            continue
        if frame_index < 0 or frame_index >= len(frame_timestamps):
            audit["label_frame_index_invalid"] += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        objects = payload.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        compatible = [
            item
            for item in objects
            if isinstance(item, dict) and _compatible_category(item.get("category"), target_type)
        ]
        if not compatible:
            audit["no_compatible_object"] += 1
            continue
        if len(compatible) != 1:
            audit["ambiguous_compatible_objects"] += 1
            continue
        selected = compatible[0]
        bbox = _single_object_bbox(selected)
        if bbox is None:
            audit["compatible_bbox_invalid"] += 1
            continue
        audit["accepted_single_compatible_object"] += 1
        object_id = next(
            (selected[key] for key in ("object_id", "track_id", "id") if key in selected),
            None,
        )
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_us": int(frame_timestamps[frame_index]),
                "bbox_xyxy_rgb": tuple(float(value) for value in bbox),
                "category": str(selected.get("category")),
                "object_id": object_id,
                "group": selected.get("group"),
                "source": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
        )
    return sorted(rows, key=lambda row: (row["timestamp_us"], row["frame_index"])), audit


def _causal_box(
    labels: list[dict[str, Any]], endpoint_us: int
) -> tuple[dict[str, Any] | None, str | None]:
    cutoff = endpoint_us - BBOX_AVAILABILITY_MARGIN_US
    eligible = [row for row in labels if row["timestamp_us"] <= cutoff]
    if not eligible:
        return None, "missing"
    selected = eligible[-1]
    available_us = selected["timestamp_us"] + BBOX_AVAILABILITY_MARGIN_US
    if endpoint_us - available_us > BBOX_MAX_POST_AVAILABILITY_AGE_US:
        return None, "stale"
    return selected, None


def _uniform_indices(length: int, count: int) -> np.ndarray:
    selected_count = min(length, count)
    if selected_count == 0:
        return np.empty(0, dtype=np.int64)
    return np.linspace(0, length - 1, selected_count, dtype=np.int64)


def build_manifest(
    inventory_path: Path,
    data_root: Path,
    destination: Path,
    max_per_sequence: int = QUERY_COUNT_PER_SEQUENCE,
) -> dict[str, Any]:
    """Predeclare queries from metadata only, without labels used by the scorer."""

    if max_per_sequence <= 0:
        raise ValueError("max_per_sequence must be positive")
    sequences = read_manifest(inventory_path)
    if len(sequences) != 32:
        raise ValueError(
            "historical EvTTC development inventory must contain 32 sequences, "
            f"got {len(sequences)}"
        )
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, dict[str, int]] = {}
    for sequence in sequences:
        sequence_dir = _resolve_sequence_path(sequence, data_root)
        raw_path = sequence_dir / sequence.event_hdf5
        label_dir = sequence_dir / str(sequence.label_dir)
        if not raw_path.is_file() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"{sequence.sequence_id}: raw events or left-label directory missing"
            )
        layout = _left_event_layout(raw_path)
        if layout.t is None:
            raise ValueError(f"{sequence.sequence_id}: event timestamp layout missing")
        raw_stat = raw_path.stat()
        raw_pin = {"size_bytes": raw_stat.st_size, "mtime_ns": raw_stat.st_mtime_ns}
        with h5py.File(raw_path, "r") as handle:
            if "blackflys/left/ts" not in handle:
                raise ValueError(
                    f"{sequence.sequence_id}: explicit Blackfly-left timestamps missing"
                )
            frame_dataset = handle["blackflys/left/ts"]
            if not isinstance(frame_dataset, h5py.Dataset):
                raise ValueError(f"{sequence.sequence_id}: Blackfly-left timestamps invalid")
            frame_timestamps = np.asarray(frame_dataset, dtype=np.int64)
            event_timestamps = handle[layout.t]
            if not isinstance(event_timestamps, h5py.Dataset):
                raise ValueError(f"{sequence.sequence_id}: event timestamps are not a dataset")
            stream_start = int(event_timestamps[0])
            stream_end = int(event_timestamps[-1]) + 1
            calibration = _calibration(handle)
            calibration_sha256 = _calibration_sha256(calibration)
        labels, label_audit = _label_rows(
            label_dir,
            frame_timestamps,
            target_type=sequence.target_type,
        )
        feasible: list[dict[str, Any]] = []
        counts = {
            "history_support": 0,
            "causal_box_missing": 0,
            "causal_box_stale": 0,
            "projection_invalid": 0,
        }
        for anchor_us in frame_timestamps:
            anchor = int(anchor_us)
            if anchor - 650_000 < stream_start or anchor > stream_end:
                counts["history_support"] += 1
                continue
            endpoints = (anchor - 200_000, anchor - 100_000, anchor)
            resolved = [_causal_box(labels, endpoint) for endpoint in endpoints]
            sources = [source for source, _ in resolved]
            if any(source is None for source in sources):
                reasons = {reason for _, reason in resolved if reason is not None}
                if "stale" in reasons:
                    counts["causal_box_stale"] += 1
                else:
                    counts["causal_box_missing"] += 1
                continue
            try:
                boxes = [
                    project_rgb_box_rotation_only(source["bbox_xyxy_rgb"], calibration)
                    for source in sources
                    if source is not None
                ]
                own_square = common_square_from_boxes(boxes, (0, 1, 2))
            except ValueError:
                counts["projection_invalid"] += 1
                continue
            feasible.append(
                {
                    "anchor_us": anchor,
                    "boxes_xyxy3": [list(box) for box in boxes],
                    "square_xyxy": list(own_square),
                    "bbox_sources": [
                        {
                            "path": source["source"],
                            "frame_index": source["frame_index"],
                            "timestamp_us": source["timestamp_us"],
                            "available_us": source["timestamp_us"] + BBOX_AVAILABILITY_MARGIN_US,
                            "sha256": source["sha256"],
                            "category": source["category"],
                            "object_id": source["object_id"],
                            "group": source["group"],
                        }
                        for source in sources
                        if source is not None
                    ],
                }
            )
        selected = [feasible[index] for index in _uniform_indices(len(feasible), max_per_sequence)]
        exclusions[sequence.sequence_id] = {
            **label_audit,
            **counts,
            "feasible": len(feasible),
            "selected": len(selected),
            "shortfall": max(0, max_per_sequence - len(selected)),
        }
        for ordinal, selected_row in enumerate(selected):
            anchor = int(selected_row["anchor_us"])
            rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "query_id": f"{sequence.sequence_id}:{ordinal:02d}",
                    "anchor_us": anchor,
                    "raw_path": str(raw_path.resolve()),
                    "raw_stat": raw_pin,
                    "calibration_sha256": calibration_sha256,
                    "windows_us": [
                        [anchor + start, anchor + end] for start, end in BASE_WINDOWS_OFFSETS_US
                    ],
                    "boxes_xyxy3": selected_row["boxes_xyxy3"],
                    "square_xyxy": selected_row["square_xyxy"],
                    "bbox_sources": selected_row["bbox_sources"],
                    "scenario_family": sequence.scenario_family,
                    "speed_bucket": sequence.speed_bucket,
                    "target_type": sequence.target_type,
                }
            )
            rows[-1]["metadata_sha256"] = _canonical_sha256(rows[-1])
    result = {
        "protocol": "evttc_historical_dev32_frozen_transfer_rotation_only_v1",
        "status": "LABEL_FREE_MANIFEST",
        "camera": {"rgb": "blackflys/left", "events": "prophesee/event_cam_left"},
        "projection": (
            "documented Blackfly-left to Prophesee-left direction, using "
            "T_bfs_to_prophesee or legacy T_to_left_bfs; rotation-only, translation "
            "omitted without depth"
        ),
        "bbox_availability_margin_us": BBOX_AVAILABILITY_MARGIN_US,
        "bbox_max_post_availability_age_us": BBOX_MAX_POST_AVAILABILITY_AGE_US,
        "queries_per_sequence": max_per_sequence,
        "minimum_selected_per_sequence": min(
            (item["selected"] for item in exclusions.values()), default=0
        ),
        "sequence_count": len(sequences),
        "rows": rows,
        "causal_exclusions": exclusions,
        "limitations": list(LIMITATIONS),
        "forbidden_assets": ["ttc.csv", "gt.hdf5", "distance", "depth", "navigation"],
    }
    result["rows_metadata_sha256"] = _canonical_sha256([row["metadata_sha256"] for row in rows])
    from e_jepa_ttc.utils.io import write_structured

    write_structured(destination, result)
    return result


def prepare(row: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare frozen H8 and published Garl event-only inputs for one query."""

    required = {
        "sequence_id",
        "anchor_us",
        "raw_path",
        "raw_stat",
        "windows_us",
        "boxes_xyxy3",
        "square_xyxy",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"query row lacks required fields: {missing}")
    windows = np.asarray(row["windows_us"], dtype=np.int64)
    boxes: list[tuple[float, float, float, float]] = []
    for raw_box in row["boxes_xyxy3"]:
        if len(raw_box) != 4:
            raise ValueError("each boxes_xyxy3 item must have four coordinates")
        boxes.append((float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])))
    if windows.shape != (3, 2) or len(boxes) != 3:
        raise ValueError("query requires three exact 100 ms windows and three causal boxes")
    if not np.array_equal(windows[:, 1] - windows[:, 0], np.full(3, 100_000)):
        raise ValueError("all EvTTC transfer windows must be exactly 100 ms")
    raw_path = Path(row["raw_path"])
    current_stat = raw_path.stat()
    expected_stat = row["raw_stat"]
    if (
        int(expected_stat["size_bytes"]) != current_stat.st_size
        or int(expected_stat["mtime_ns"]) != current_stat.st_mtime_ns
    ):
        raise ValueError("EvTTC raw file stat changed after label-free query selection")
    unused_prefix_lags = np.arange(15, 7, -1, dtype=np.int64) * 50_000
    full_lags = np.concatenate((unused_prefix_lags, H8_LAGS_US))
    valid16 = np.concatenate((np.zeros(8, dtype=bool), np.ones(8, dtype=bool)))
    with EvTTCEventReader(raw_path) as reader:
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
        for box_index, window_index in ((1, 1), (2, 2)):
            start, end = (int(value) for value in windows[window_index])
            raw = reader.read_window(start, end)
            if len(raw["t"]) == 0:
                garl_parts = []
                break
            square = official_square_box(boxes[-2:], box_index - 1)
            garl_parts.append(native_feature(raw, square, size=128))
    own_events = own.numpy().astype(np.float32, copy=False)
    garl_events = (
        np.concatenate([part.numpy() for part in garl_parts], axis=0).astype(np.float32, copy=False)
        if len(garl_parts) == 2
        else None
    )
    if own_events.shape != (8, 3, 12, 128, 128) or (
        garl_events is not None and garl_events.shape != (40, 128, 128)
    ):
        raise ValueError("prepared model tensors violate frozen H8/Garl shapes")
    return {
        "own_events": own_events,
        "garl_events": garl_events,
        "garl_unavailable_reason": "empty_sensor_endpoint" if garl_events is None else None,
        "valid": np.ones(8, dtype=bool),
        "delta_t_s": np.float32(0.1),
        "own_square_xyxy": np.asarray(row["square_xyxy"], dtype=np.float64),
        "garl_square_xyxy2": np.asarray(
            [official_square_box(boxes[-2:], index) for index in range(2)], dtype=np.int64
        ),
    }


__all__ = [
    "BBOX_AVAILABILITY_MARGIN_US",
    "EvTTCEventReader",
    "LIMITATIONS",
    "H8_LAGS_US",
    "build_manifest",
    "prepare",
    "project_rgb_box_rotation_only",
]
