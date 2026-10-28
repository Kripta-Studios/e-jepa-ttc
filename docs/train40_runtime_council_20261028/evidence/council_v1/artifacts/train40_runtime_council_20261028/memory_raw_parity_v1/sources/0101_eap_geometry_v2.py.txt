"""Object-centric eAP geometry targets for lateral motion and collision-path risk.

This module deliberately keeps the legacy six-dimensional eAP-Geo target intact
and defines a versioned v2 extension.  All values are dimensionless and bounded
where practical so the auxiliary head can share a Smooth-L1 objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE, EAPObjectState

EAP_GEOMETRY_V1_NAMES: tuple[str, ...] = (
    "center_x",
    "center_y",
    "width",
    "height",
    "closing_speed",
    "log_height_rate",
)

EAP_GEOMETRY_V2_NAMES: tuple[str, ...] = EAP_GEOMETRY_V1_NAMES + (
    "center_velocity_x",
    "center_velocity_y",
    "lateral_speed",
    "radiality",
    "visibility_fraction",
    "truncated_left",
    "truncated_right",
    "track_age",
    "log_area_rate",
    "abrupt_area_change",
    "corridor_center_distance",
    "corridor_overlap",
    "corridor_velocity",
    "corridor_entry_direction",
)

EAP_GEOMETRY_V2_DIM = len(EAP_GEOMETRY_V2_NAMES)


@dataclass(frozen=True)
class EAPGeometryV2:
    """Numerical target, validity mask and deterministic balance stratum."""

    values: np.ndarray
    valid: np.ndarray
    sampling_group: str


def coarse_category(category: str) -> str:
    """Map public eAP taxonomy strings to stable coarse categories."""

    value = category.lower().replace("_", ".").replace("-", ".")
    if "pedestrian" in value or "person" in value or "human" in value:
        return "pedestrian"
    if any(token in value for token in ("car", "truck", "bus", "vehicle", "van")):
        return "vehicle"
    if any(token in value for token in ("bicycle", "bike", "motorcycle", "rider")):
        return "two_wheeler"
    return "other"


def category_index(category: str) -> int:
    return {"vehicle": 0, "pedestrian": 1, "two_wheeler": 2, "other": 3}[coarse_category(category)]


def _normalized_box(state: EAPObjectState) -> tuple[float, float, float, float]:
    image_width, image_height = EAP_IMAGE_SIZE
    x0, y0, x1, y1 = state.bbox_xyxy
    return (
        x0 / image_width,
        y0 / image_height,
        x1 / image_width,
        y1 / image_height,
    )


def _finite_delta(
    current: EAPObjectState,
    previous: EAPObjectState | None,
    *,
    maximum_gap_s: float,
) -> float | None:
    if previous is None or current.track_id != previous.track_id:
        return None
    delta_s = (current.timestamp_us - previous.timestamp_us) * 1e-6
    return delta_s if 0.0 < delta_s <= maximum_gap_s else None


def geometry_v2_targets(
    current: EAPObjectState,
    previous: EAPObjectState | None,
    *,
    corridor_half_width: float = 0.18,
    maximum_gap_s: float = 0.25,
    radiality_epsilon: float = 1e-3,
    velocity_scale: float = 3.0,
    log_rate_scale: float = 5.0,
    track_age_scale_s: float = 2.0,
) -> EAPGeometryV2:
    """Build motion, visibility and image-plane collision-corridor targets.

    The corridor is an image-centred proxy because the public media adapter does
    not expose ego lane geometry.  It should be interpreted as a pretraining
    signal, not as a calibrated physical collision probability.
    """

    if not 0.0 < corridor_half_width <= 0.5:
        raise ValueError("corridor_half_width must lie in (0, 0.5].")
    if min(maximum_gap_s, radiality_epsilon, velocity_scale, log_rate_scale) <= 0:
        raise ValueError("Geometry-v2 scales must be positive.")

    x0, y0, x1, y1 = _normalized_box(current)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    closing = np.clip(-current.depth_velocity_mps / 20.0, -1.0, 1.0)

    delta_s = _finite_delta(current, previous, maximum_gap_s=maximum_gap_s)
    dcx = dcy = log_height_rate = log_area_rate = float("nan")
    if delta_s is not None and previous is not None:
        px0, py0, px1, py1 = _normalized_box(previous)
        previous_center_x = 0.5 * (px0 + px1)
        previous_center_y = 0.5 * (py0 + py1)
        previous_area = max((px1 - px0) * (py1 - py0), 1e-8)
        area = max(width * height, 1e-8)
        dcx = (center_x - previous_center_x) / delta_s
        dcy = (center_y - previous_center_y) / delta_s
        log_height_rate = (
            math.log(max(current.visible_height_px, 1e-3))
            - math.log(max(previous.visible_height_px, 1e-3))
        ) / delta_s
        log_area_rate = (math.log(area) - math.log(previous_area)) / delta_s

    lateral_speed_raw = (
        abs(dcx) + abs(dcy) if np.isfinite(dcx) and np.isfinite(dcy) else float("nan")
    )
    radiality_raw = (
        abs(log_height_rate) / (lateral_speed_raw + radiality_epsilon)
        if np.isfinite(log_height_rate) and np.isfinite(lateral_speed_raw)
        else float("nan")
    )

    visibility_fraction = float(np.clip(getattr(current, "visibility_fraction", 1.0), 0.0, 1.0))
    raw_box = getattr(current, "unclipped_bbox_xyxy", None)
    image_width, _image_height = EAP_IMAGE_SIZE
    truncated_left = float(raw_box is not None and raw_box[0] < 0.0)
    truncated_right = float(raw_box is not None and raw_box[2] > image_width - 1)
    first_seen = getattr(current, "first_seen_timestamp_us", None)
    track_age = (
        max(0.0, (current.timestamp_us - first_seen) * 1e-6)
        if first_seen is not None
        else float("nan")
    )

    abrupt_area_change = (
        float(abs(log_area_rate) > 3.0) if np.isfinite(log_area_rate) else float("nan")
    )
    corridor_left = 0.5 - corridor_half_width
    corridor_right = 0.5 + corridor_half_width
    overlap = max(0.0, min(x1, corridor_right) - max(x0, corridor_left))
    corridor_overlap = overlap / max(width, 1e-6)
    corridor_center_distance = abs(center_x - 0.5) / corridor_half_width
    corridor_velocity = (
        -math.copysign(1.0, center_x - 0.5) * dcx if np.isfinite(dcx) else float("nan")
    )
    entry_direction = (
        float(np.sign(corridor_velocity)) if np.isfinite(corridor_velocity) else float("nan")
    )

    values = np.asarray(
        [
            center_x,
            center_y,
            width,
            height,
            closing,
            np.clip(log_height_rate / log_rate_scale, -1.0, 1.0),
            np.clip(dcx / velocity_scale, -1.0, 1.0),
            np.clip(dcy / velocity_scale, -1.0, 1.0),
            np.clip(lateral_speed_raw / velocity_scale, 0.0, 1.0),
            np.tanh(radiality_raw),
            visibility_fraction,
            truncated_left,
            truncated_right,
            np.clip(track_age / track_age_scale_s, 0.0, 1.0),
            np.clip(log_area_rate / log_rate_scale, -1.0, 1.0),
            abrupt_area_change,
            np.clip(corridor_center_distance, 0.0, 2.0) / 2.0,
            np.clip(corridor_overlap, 0.0, 1.0),
            np.clip(corridor_velocity / velocity_scale, -1.0, 1.0),
            entry_direction,
        ],
        dtype=np.float32,
    )
    valid = np.isfinite(values)
    values[~valid] = 0.0

    category = coarse_category(current.category)
    motion = (
        "transverse"
        if np.isfinite(lateral_speed_raw) and lateral_speed_raw > 0.25
        else "longitudinal"
    )
    visibility = "partial" if visibility_fraction < 0.95 else "visible"
    corridor = "intersecting" if corridor_overlap >= 0.25 else "off_corridor"
    return EAPGeometryV2(
        values=values,
        valid=valid,
        sampling_group=f"{category}:{motion}:{visibility}:{corridor}",
    )
