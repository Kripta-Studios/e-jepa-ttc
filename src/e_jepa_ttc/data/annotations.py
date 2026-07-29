"""Annotation parsing for local ISAT bbox/segmentation labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.data.targets import interpolate_ttc_seconds, load_ttc_csv
from e_jepa_ttc.data.types import DatasetSequence


@dataclass(frozen=True)
class LabelMeasurement:
    """One labeled object measurement aligned to a sequence timestamp."""

    sequence_id: str
    frame_index: int
    timestamp_us: int
    category: str
    bbox_xyxy: tuple[float, float, float, float]
    bbox_area: float
    bbox_scale: float
    ttc_seconds: float
    image_width: int | None = None
    image_height: int | None = None
    segmentation_xy: tuple[tuple[float, float], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize measurement for JSON/YAML outputs."""

        return asdict(self)


def _bbox_from_segmentation(segmentation: object) -> tuple[float, float, float, float] | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    points: list[tuple[float, float]] = []
    for point in segmentation:
        if isinstance(point, list) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class ParsedISATLabel:
    """Largest labeled object and source image dimensions from one ISAT file."""

    category: str
    bbox_xyxy: tuple[float, float, float, float]
    image_width: int | None
    image_height: int | None
    segmentation_xy: tuple[tuple[float, float], ...] | None


def parse_isat_label_metadata(path: str | Path) -> ParsedISATLabel | None:
    """Parse the largest object bbox and source image size from an ISAT JSON file."""

    label_path = Path(path)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    objects = data.get("objects", [])
    if not isinstance(objects, list):
        return None
    info = data.get("info", {})
    image_width = None
    image_height = None
    if isinstance(info, dict):
        raw_width = info.get("width")
        raw_height = info.get("height")
        if raw_width is not None:
            image_width = int(raw_width)
        if raw_height is not None:
            image_height = int(raw_height)

    best: (
        tuple[
            str,
            tuple[float, float, float, float],
            float,
            tuple[tuple[float, float], ...] | None,
        ]
        | None
    ) = None
    for item in objects:
        if not isinstance(item, dict):
            continue
        raw_segmentation = item.get("segmentation")
        bbox = _bbox_from_segmentation(raw_segmentation)
        segmentation = (
            tuple(
                (float(point[0]), float(point[1]))
                for point in raw_segmentation
                if isinstance(point, list) and len(point) >= 2
            )
            if isinstance(raw_segmentation, list)
            else None
        )
        if not segmentation:
            segmentation = None
        raw_bbox = item.get("bbox")
        if bbox is None and isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
            bbox = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        category = str(item.get("category", "unknown"))
        if best is None or area > best[2]:
            best = (category, bbox, area, segmentation)
    if best is None:
        return None
    return ParsedISATLabel(
        category=best[0],
        bbox_xyxy=best[1],
        image_width=image_width,
        image_height=image_height,
        segmentation_xy=best[3],
    )


def parse_isat_label(path: str | Path) -> tuple[str, tuple[float, float, float, float]] | None:
    """Parse the largest object bbox from an ISAT JSON label file."""

    parsed = parse_isat_label_metadata(path)
    if parsed is None:
        return None
    return parsed.category, parsed.bbox_xyxy


def load_label_measurements(sequence: DatasetSequence) -> list[LabelMeasurement]:
    """Load object label measurements aligned through Blackfly frame timestamps."""

    label_dir = sequence.resolve("label_dir")
    event_hdf5 = sequence.resolve("event_hdf5")
    ttc_csv = sequence.resolve("ttc_csv")
    if label_dir is None or event_hdf5 is None or ttc_csv is None:
        return []
    if not label_dir.exists():
        return []

    table = load_ttc_csv(ttc_csv)
    measurements: list[LabelMeasurement] = []
    with h5py.File(event_hdf5, "r") as h5:
        if "blackflys/left/ts" not in h5:
            return []
        frame_ts = h5["blackflys/left/ts"][:].astype(np.int64)

    for label_path in sorted(label_dir.glob("*.json")):
        try:
            frame_index = int(label_path.stem)
        except ValueError:
            continue
        if frame_index < 0 or frame_index >= len(frame_ts):
            continue
        parsed = parse_isat_label_metadata(label_path)
        if parsed is None:
            continue
        bbox = parsed.bbox_xyxy
        timestamp_us = int(frame_ts[frame_index])
        ttc_seconds = interpolate_ttc_seconds(table, timestamp_us)
        if ttc_seconds is None:
            continue
        x0, y0, x1, y1 = bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area <= 0:
            continue
        measurements.append(
            LabelMeasurement(
                sequence_id=sequence.sequence_id,
                frame_index=frame_index,
                timestamp_us=timestamp_us,
                category=parsed.category,
                bbox_xyxy=bbox,
                bbox_area=area,
                bbox_scale=float(np.sqrt(area)),
                ttc_seconds=float(ttc_seconds),
                image_width=parsed.image_width,
                image_height=parsed.image_height,
                segmentation_xy=parsed.segmentation_xy,
            )
        )
    return measurements


def load_measurements_from_manifest(path: str | Path) -> dict[str, list[LabelMeasurement]]:
    """Load label measurements for every sequence in a manifest."""

    return {
        sequence.sequence_id: load_label_measurements(sequence) for sequence in read_manifest(path)
    }
