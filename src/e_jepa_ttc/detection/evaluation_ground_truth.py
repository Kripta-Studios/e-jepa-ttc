"""Detector-independent ground truth population for detection evaluation v2."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners, validate_calibration

_CUBOID_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must contain {length} numbers")
    result = tuple(_finite(item, name) for item in value)  # type: ignore[union-attr]
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return result


@dataclass(frozen=True)
class EvaluationBox:
    """One strict box in the original eAP ego-frame geometry."""

    class_name: str
    translation_m: tuple[float, float, float]
    size_lwh_m: tuple[float, float, float]
    yaw_rad: float
    source_index: int
    ignore_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON representation."""
        return {
            "class_name": self.class_name,
            "translation_m": list(self.translation_m),
            "size_lwh_m": list(self.size_lwh_m),
            "yaw_rad": self.yaw_rad,
            "source_index": self.source_index,
            "ignore_reason": self.ignore_reason,
        }


@dataclass(frozen=True)
class EvaluationGroundTruth:
    """Complete raw denominator split into eligible and explicitly ignored objects."""

    protocol_sha256: str
    eligible: tuple[EvaluationBox, ...]
    ignored: tuple[EvaluationBox, ...]
    raw_count: int
    ignored_by_reason: tuple[tuple[str, int], ...]
    intrinsic: tuple[tuple[float, ...], ...]
    event_from_ego: tuple[tuple[float, ...], ...]
    image_wh: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        """Serialize the population, including calibration needed for prediction filtering."""
        return {
            "protocol_sha256": self.protocol_sha256,
            "eligible": [box.to_dict() for box in self.eligible],
            "ignored": [box.to_dict() for box in self.ignored],
            "raw_count": self.raw_count,
            "ignored_by_reason": dict(self.ignored_by_reason),
            "intrinsic": [list(row) for row in self.intrinsic],
            "event_from_ego": [list(row) for row in self.event_from_ego],
            "image_wh": list(self.image_wh),
        }

    @property
    def sha256(self) -> str:
        """Hash population and geometry independently of detector configuration."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def parse_box(
    row: Mapping[str, object], *, source_index: int, classes: tuple[str, ...], prediction: bool
) -> tuple[EvaluationBox, float | None]:
    """Parse either a raw annotation or a normalized prediction row."""
    if prediction:
        category = row.get("class_name")
        translation = _vector(row.get("translation_m"), 3, "translation_m")
        size = _vector(row.get("size_lwh_m"), 3, "size_lwh_m")
        yaw = _finite(row.get("yaw_rad"), "yaw_rad")
        score = _finite(row.get("score"), "score")
        if not 0 <= score <= 1:
            raise ValueError("score must be in [0,1]")
    else:
        category = row.get("category")
        raw = _vector(row.get("bbox_3d_ego"), 7, "bbox_3d_ego")
        translation = raw[:3]
        size = (raw[3], raw[5], raw[4])
        yaw = raw[6]
        score = None
    if not isinstance(category, str) or category not in classes:
        raise ValueError(f"Unknown class: {category!r}")
    if any(value <= 0 for value in size):
        raise ValueError("Box dimensions must be strictly positive")
    return (
        EvaluationBox(
            class_name=category,
            translation_m=translation,  # type: ignore[arg-type]
            size_lwh_m=size,  # type: ignore[arg-type]
            yaw_rad=yaw,
            source_index=source_index,
        ),
        score,
    )


def domain_reason(
    box: EvaluationBox,
    *,
    intrinsic: np.ndarray,
    event_from_ego: np.ndarray,
    image_wh: tuple[int, int],
    protocol: EvaluationProtocol,
) -> str | None:
    """Apply the provisional symmetric GT/prediction evaluation domain."""
    bundle = [
        *box.translation_m,
        box.size_lwh_m[0],
        box.size_lwh_m[2],
        box.size_lwh_m[1],
        box.yaw_rad,
    ]
    corners = original_bundle_corners(bundle)
    rotation = event_from_ego[:3, :3]
    offset = event_from_ego[:3, 3]
    camera_center = rotation @ np.asarray(box.translation_m) + offset
    low, high = protocol.center_depth_range_m
    if not low <= camera_center[2] < high:
        return "camera_center_depth_outside_protocol"
    camera_corners = corners @ rotation.T + offset
    clipped = [corner for corner in camera_corners if corner[2] >= low]
    for left_index, right_index in _CUBOID_EDGES:
        left, right = camera_corners[left_index], camera_corners[right_index]
        if (left[2] < low) == (right[2] < low):
            continue
        fraction = (low - left[2]) / (right[2] - left[2])
        clipped.append(left + fraction * (right - left))
    if not clipped:
        return "cuboid_behind_near_plane"
    # Project the near-plane-clipped cuboid vertices. Rectangle intersection, rather
    # than a corner-inside test, retains large boxes that cover the viewport.
    projected = np.asarray(clipped) @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    width, height = image_wh
    minimum, maximum = pixels.min(axis=0), pixels.max(axis=0)
    if maximum[0] < 0 or maximum[1] < 0 or minimum[0] >= width or minimum[1] >= height:
        return "projected_cuboid_outside_image"
    return None


def build_evaluation_ground_truth(
    annotations: Sequence[Mapping[str, object]],
    *,
    intrinsic: object,
    event_from_ego: object,
    image_wh: tuple[int, int],
    protocol: EvaluationProtocol,
) -> EvaluationGroundTruth:
    """Build one frame's population directly from raw labels, without trainer filters."""
    validate_calibration(intrinsic, event_from_ego)
    if len(image_wh) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_wh
    ):
        raise ValueError("image_wh must contain two positive integers")
    k = np.asarray(intrinsic, dtype=np.float64)
    transform = np.asarray(event_from_ego, dtype=np.float64)
    eligible: list[EvaluationBox] = []
    ignored: list[EvaluationBox] = []
    reasons: Counter[str] = Counter()
    for index, annotation in enumerate(annotations):
        box, _ = parse_box(
            annotation, source_index=index, classes=protocol.classes, prediction=False
        )
        reason = domain_reason(
            box,
            intrinsic=k,
            event_from_ego=transform,
            image_wh=image_wh,
            protocol=protocol,
        )
        if reason is None:
            eligible.append(box)
        else:
            ignored.append(
                EvaluationBox(
                    box.class_name,
                    box.translation_m,
                    box.size_lwh_m,
                    box.yaw_rad,
                    box.source_index,
                    reason,
                )
            )
            reasons[reason] += 1
    return EvaluationGroundTruth(
        protocol_sha256=protocol.sha256,
        eligible=tuple(eligible),
        ignored=tuple(ignored),
        raw_count=len(annotations),
        ignored_by_reason=tuple(sorted(reasons.items())),
        intrinsic=tuple(tuple(float(value) for value in row) for row in k),
        event_from_ego=tuple(tuple(float(value) for value in row) for row in transform),
        image_wh=image_wh,
    )
