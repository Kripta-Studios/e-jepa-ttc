"""Data-blind camera geometry for the EvTTC competition inputs.

The competition boxes do not declare their camera plane.  This module therefore
keeps coordinate normalization separate from cross-camera projection.  The only
cross-camera mapping provided here is a rotation-only ray projection: translation
is deliberately omitted because the test input supplies no object depth.  It is
an explicit approximation, not an exact stereo reprojection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Calibration:
    """Minimal event/RGB calibration with an auditable transform contract."""

    event_intrinsics: np.ndarray = field(repr=False)
    event_distortion: np.ndarray = field(repr=False)
    event_resolution: tuple[int, int]
    rgb_intrinsics: np.ndarray = field(repr=False)
    rgb_distortion: np.ndarray = field(repr=False)
    rgb_resolution: tuple[int, int]
    rotation_event_to_rgb: np.ndarray = field(repr=False)
    source_kind: str
    transform_source: str
    translation_omitted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, shape in (
            ("event_intrinsics", (4,)),
            ("rgb_intrinsics", (4,)),
            ("event_distortion", (4,)),
            ("rgb_distortion", (4,)),
            ("rotation_event_to_rgb", (3, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        for name in ("event_resolution", "rgb_resolution"):
            width, height = getattr(self, name)
            if width <= 0 or height <= 0:
                raise ValueError(f"{name} must contain positive width and height")
        rotation = self.rotation_event_to_rgb
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
            raise ValueError("rotation_event_to_rgb must be orthonormal")
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("rotation_event_to_rgb must be a proper rotation")


def _as_path(sequence_path: str | Path, filename: str) -> Path:
    path = Path(sequence_path)
    return path if path.is_file() else path / filename


def _resolution(value: np.ndarray, *, matlab_image_size: bool = False) -> tuple[int, int]:
    flattened = np.asarray(value, dtype=np.float64).reshape(-1)
    if flattened.size != 2 or not np.isfinite(flattened).all():
        raise ValueError("camera resolution must contain two finite values")
    if matlab_image_size:
        height, width = flattened
    else:
        width, height = flattened
    return int(round(width)), int(round(height))


def _intrinsics_from_matlab_k(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("MATLAB camera K must be 3x3")
    conventional = matrix.T
    return np.asarray(
        [conventional[0, 0], conventional[1, 1], conventional[0, 2], conventional[1, 2]],
        dtype=np.float64,
    )


def _distortion(radial: np.ndarray, tangential: np.ndarray) -> np.ndarray:
    radial_values = np.asarray(radial, dtype=np.float64).reshape(-1)
    tangent_values = np.asarray(tangential, dtype=np.float64).reshape(-1)
    if radial_values.size < 2 or tangent_values.size < 2:
        raise ValueError("radtan calibration requires k1, k2, p1 and p2")
    return np.asarray(
        [radial_values[0], radial_values[1], tangent_values[0], tangent_values[1]],
        dtype=np.float64,
    )


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    if angle < 1e-14:
        return np.eye(3, dtype=np.float64)
    axis = value / angle
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = np.clip((np.trace(left @ right.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    left, _, right = np.linalg.svd(np.sum(np.stack(rotations), axis=0))
    result = left @ right
    if np.linalg.det(result) < 0:
        left[:, -1] *= -1
        result = left @ right
    return result


def _slider_camera_groups(handle: h5py.File) -> tuple[h5py.Group, h5py.Group, str, str]:
    candidates: list[tuple[int, str, h5py.Group]] = []
    refs = handle["#refs#"]
    if not isinstance(refs, h5py.Group):
        raise ValueError("Slider calibration #refs# must be an HDF5 group")
    for name, item in refs.items():
        if not isinstance(item, h5py.Group):
            continue
        required = {
            "ImageSize",
            "K",
            "RadialDistortion",
            "TangentialDistortion",
            "RotationVectors",
            "TranslationVectors",
        }
        if required.issubset(item.keys()):
            width, height = _resolution(np.asarray(item["ImageSize"]), matlab_image_size=True)
            candidates.append((width * height, name, item))
    if len(candidates) != 2:
        raise ValueError("Slider calibration must contain exactly two complete camera groups")
    candidates.sort(key=lambda row: row[0])
    _, event_name, event = candidates[0]
    _, rgb_name, rgb = candidates[1]
    return event, rgb, event_name, rgb_name


def _derive_slider_rotation(
    event: h5py.Group, rgb: h5py.Group
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive a reproducible candidate rotation from robustly matched board poses.

    MATLAB cameraParameters stores board-to-camera poses.  Every possible
    event/RGB pose pairing is considered; a deterministic consensus selects
    one-to-one pairs agreeing within 1 degree and 20 mm.  Translation is only a
    diagnostic and is never used for box projection.
    """

    event_r = np.asarray(event["RotationVectors"], dtype=np.float64).T
    event_t = np.asarray(event["TranslationVectors"], dtype=np.float64).T
    rgb_r = np.asarray(rgb["RotationVectors"], dtype=np.float64).T
    rgb_t = np.asarray(rgb["TranslationVectors"], dtype=np.float64).T
    candidates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for rgb_index, (rgb_vector, rgb_translation) in enumerate(zip(rgb_r, rgb_t, strict=True)):
        rotation_rgb = _rodrigues(rgb_vector)
        for event_index, (event_vector, event_translation) in enumerate(
            zip(event_r, event_t, strict=True)
        ):
            rotation_event = _rodrigues(event_vector)
            rgb_to_event = rotation_event @ rotation_rgb.T
            translation = event_translation - rgb_to_event @ rgb_translation
            candidates.append((rgb_index, event_index, rgb_to_event, translation))

    best: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    for _, _, seed_rotation, seed_translation in candidates:
        compatible = []
        for rgb_index, event_index, rotation, translation in candidates:
            angle = _rotation_distance(rotation, seed_rotation)
            distance = float(np.linalg.norm(translation - seed_translation))
            if angle <= np.deg2rad(1.0) and distance <= 20.0:
                compatible.append(
                    (angle + distance / 1000.0, rgb_index, event_index, rotation, translation)
                )
        selected = []
        used_rgb: set[int] = set()
        used_event: set[int] = set()
        for row in sorted(compatible, key=lambda value: (value[0], value[1], value[2])):
            if row[1] in used_rgb or row[2] in used_event:
                continue
            selected.append(row)
            used_rgb.add(row[1])
            used_event.add(row[2])
        if len(selected) > len(best) or (
            len(selected) == len(best)
            and sum(row[0] for row in selected) < sum(row[0] for row in best)
        ):
            best = selected
    if len(best) < 4:
        raise ValueError("Slider pose sets do not support a stable cross-camera rotation candidate")

    rgb_to_event = _mean_rotation([row[3] for row in best])
    translations = np.stack([row[4] for row in best])
    rotation_residuals = [np.rad2deg(_rotation_distance(row[3], rgb_to_event)) for row in best]
    median_translation = np.median(translations, axis=0)
    translation_residuals = np.linalg.norm(translations - median_translation, axis=1)
    metadata = {
        "pose_pairing": "deterministic all-pairs 1deg/20mm consensus",
        "rgb_pose_count": len(rgb_r),
        "event_pose_count": len(event_r),
        "matched_pose_count": len(best),
        "matched_pose_indices": [{"rgb": int(row[1]), "event": int(row[2])} for row in best],
        "rotation_residual_deg_max": float(max(rotation_residuals)),
        "translation_candidate_mm_median": median_translation.tolist(),
        "translation_residual_mm_max": float(translation_residuals.max()),
        "translation_used_for_projection": False,
        "status": "candidate_only_no_authoritative_intercamera_extrinsic",
    }
    return rgb_to_event.T, metadata


def _load_road(path: Path, side: str) -> Calibration:
    with h5py.File(path, "r") as handle:
        event_prefix = f"prophesee/event_cam_{side}/calib"
        rgb_prefix = f"blackflys/{side}/calib"
        for prefix in (event_prefix, rgb_prefix):
            if prefix not in handle:
                raise ValueError(f"missing camera calibration group: {prefix}")
        explicit = f"{event_prefix}/T_bfs_to_prophesee"
        legacy = f"{event_prefix}/T_to_{side}_bfs"
        if explicit in handle:
            source = explicit
            rotation_event_to_rgb = np.asarray(handle[source], dtype=np.float64)[:3, :3].T
            direction = "dataset key declares Blackfly/RGB to Prophesee/event; rotation transposed"
        elif legacy in handle:
            source = legacy
            rotation_event_to_rgb = np.asarray(handle[source], dtype=np.float64)[:3, :3].T
            direction = (
                "provider data-format documentation declares camera/RGB to Prophesee/event; "
                "the legacy key name alone is not used to reverse that direction; rotation "
                "transposed. The stored matrix is also numerically continuous with the newer "
                "T_bfs_to_prophesee cohorts"
            )
        else:
            raise ValueError(f"no supported cross-camera transform in {event_prefix}")
        return Calibration(
            event_intrinsics=np.asarray(handle[f"{event_prefix}/intrinsics"]),
            event_distortion=np.asarray(handle[f"{event_prefix}/distortion_coeffs"]),
            event_resolution=_resolution(np.asarray(handle[f"{event_prefix}/resolution"])),
            rgb_intrinsics=np.asarray(handle[f"{rgb_prefix}/intrinsics"]),
            rgb_distortion=np.asarray(handle[f"{rgb_prefix}/distortion_coeffs"]),
            rgb_resolution=_resolution(np.asarray(handle[f"{rgb_prefix}/resolution"])),
            rotation_event_to_rgb=rotation_event_to_rgb,
            source_kind=f"road_{side}",
            transform_source=source,
            metadata={
                "transform_direction_interpretation": direction,
                "projection_model": "rotation_only_radtan_perimeter_envelope",
                "translation_present_but_omitted_without_depth": True,
            },
        )


def _load_slider(path: Path) -> Calibration:
    with h5py.File(path, "r") as handle:
        event, rgb, event_name, rgb_name = _slider_camera_groups(handle)
        rotation, metadata = _derive_slider_rotation(event, rgb)
        metadata.update(
            {
                "event_group": f"#refs#/{event_name}",
                "rgb_group": f"#refs#/{rgb_name}",
                "camera_role_rule": "smaller image is DVS/event; larger image is APS/RGB",
                "matlab_K_transposed_to_conventional_form": True,
                "projection_model": "rotation_only_radtan_perimeter_envelope",
            }
        )
        return Calibration(
            event_intrinsics=_intrinsics_from_matlab_k(np.asarray(event["K"])),
            event_distortion=_distortion(
                np.asarray(event["RadialDistortion"]),
                np.asarray(event["TangentialDistortion"]),
            ),
            event_resolution=_resolution(np.asarray(event["ImageSize"]), matlab_image_size=True),
            rgb_intrinsics=_intrinsics_from_matlab_k(np.asarray(rgb["K"])),
            rgb_distortion=_distortion(
                np.asarray(rgb["RadialDistortion"]),
                np.asarray(rgb["TangentialDistortion"]),
            ),
            rgb_resolution=_resolution(np.asarray(rgb["ImageSize"]), matlab_image_size=True),
            rotation_event_to_rgb=rotation,
            source_kind="slider_candidate",
            transform_source="robust consensus from MATLAB cameraParameters board poses",
            metadata=metadata,
        )


def load_calibration(sequence_path: str | Path, kind: str = "road_right") -> Calibration:
    """Load a road calibration or derive the explicitly provisional Slider candidate."""

    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {"road_right", "right"}:
        return _load_road(_as_path(sequence_path, "data.hdf5"), "right")
    if normalized in {"road_left", "left"}:
        return _load_road(_as_path(sequence_path, "data.hdf5"), "left")
    if normalized in {"slider", "slider_candidate"}:
        return _load_slider(_as_path(sequence_path, "calibParams.mat"))
    raise ValueError(f"unsupported calibration kind: {kind!r}")


def normalize_box_xyxy(
    raw_xyxy: Sequence[float],
    resolution: tuple[int, int],
    *,
    one_based: bool = False,
    inclusive_max: bool = False,
    clamp: bool = True,
) -> Box:
    """Convert an explicit box convention to zero-based half-open ``xyxy`` bounds."""

    if len(raw_xyxy) != 4:
        raise ValueError("raw_xyxy must contain four coordinates")
    values = np.asarray(raw_xyxy, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("box coordinates must be finite")
    x0, y0, x1, y1 = values.tolist()
    if one_based:
        x0 -= 1.0
        y0 -= 1.0
        x1 -= 1.0
        y1 -= 1.0
    if inclusive_max:
        x1 += 1.0
        y1 += 1.0
    width, height = resolution
    if clamp:
        x0, x1 = np.clip((x0, x1), 0.0, float(width)).tolist()
        y0, y1 = np.clip((y0, y1), 0.0, float(height)).tolist()
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box is degenerate after convention conversion and clamping")
    return float(x0), float(y0), float(x1), float(y1)


def _undistort_radtan(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    distorted = np.asarray(points, dtype=np.float64)
    estimate = distorted.copy()
    k1, k2, p1, p2 = coefficients
    for _ in range(12):
        x, y = estimate[:, 0], estimate[:, 1]
        radius_squared = x * x + y * y
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
        tangent_x = 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x**2)
        tangent_y = p1 * (radius_squared + 2.0 * y**2) + 2.0 * p2 * x * y
        if np.any(np.abs(radial) < 1e-12):
            raise ValueError("radtan inversion encountered a singular radial factor")
        estimate[:, 0] = (distorted[:, 0] - tangent_x) / radial
        estimate[:, 1] = (distorted[:, 1] - tangent_y) / radial
    return estimate


def _distort_radtan(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    x, y = values[:, 0], values[:, 1]
    k1, k2, p1, p2 = coefficients
    radius_squared = x * x + y * y
    radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
    return np.column_stack(
        (
            x * radial + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x**2),
            y * radial + p1 * (radius_squared + 2.0 * y**2) + 2.0 * p2 * x * y,
        )
    )


def _perimeter(box: Box, samples: int) -> np.ndarray:
    if samples < 2:
        raise ValueError("perimeter_samples must be at least two")
    x0, y0, x1, y1 = box
    horizontal = np.linspace(x0, x1, samples)
    vertical = np.linspace(y0, y1, samples)
    return np.unique(
        np.vstack(
            (
                np.column_stack((horizontal, np.full(samples, y0))),
                np.column_stack((horizontal, np.full(samples, y1))),
                np.column_stack((np.full(samples, x0), vertical)),
                np.column_stack((np.full(samples, x1), vertical)),
            )
        ),
        axis=0,
    )


def _project_box(
    box: Box,
    *,
    source_intrinsics: np.ndarray,
    source_distortion: np.ndarray,
    target_intrinsics: np.ndarray,
    target_distortion: np.ndarray,
    rotation_source_to_target: np.ndarray,
    target_resolution: tuple[int, int],
    perimeter_samples: int,
    clamp: bool,
) -> Box:
    pixels = _perimeter(box, perimeter_samples)
    source = source_intrinsics
    distorted = np.column_stack(
        ((pixels[:, 0] - source[2]) / source[0], (pixels[:, 1] - source[3]) / source[1])
    )
    normalized = _undistort_radtan(distorted, source_distortion)
    rays = np.column_stack((normalized, np.ones(len(normalized))))
    target_rays = (rotation_source_to_target @ rays.T).T
    if np.any(target_rays[:, 2] <= 0.0):
        raise ValueError("rotation-only projection places a sampled box ray behind the camera")
    target_normalized = target_rays[:, :2] / target_rays[:, 2:3]
    target_distorted = _distort_radtan(target_normalized, target_distortion)
    target = target_intrinsics
    projected = np.column_stack(
        (
            target_distorted[:, 0] * target[0] + target[2],
            target_distorted[:, 1] * target[1] + target[3],
        )
    )
    envelope = (
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    )
    return normalize_box_xyxy(envelope, target_resolution, clamp=clamp)


def project_event_box_to_rgb(
    box: Sequence[float],
    calibration: Calibration,
    *,
    perimeter_samples: int = 9,
    clamp: bool = True,
) -> Box:
    """Project an event-plane box to RGB using rays and rotation only.

    Translation is omitted because object depth is unavailable.  The returned
    envelope is therefore suitable as a declared approximate crop, not as an
    exact stereo box or geometric ground truth.
    """

    event_box = normalize_box_xyxy(box, calibration.event_resolution, clamp=clamp)
    return _project_box(
        event_box,
        source_intrinsics=calibration.event_intrinsics,
        source_distortion=calibration.event_distortion,
        target_intrinsics=calibration.rgb_intrinsics,
        target_distortion=calibration.rgb_distortion,
        rotation_source_to_target=calibration.rotation_event_to_rgb,
        target_resolution=calibration.rgb_resolution,
        perimeter_samples=perimeter_samples,
        clamp=clamp,
    )


def input_bbox_to_event_box(
    raw_xyxy: Sequence[float],
    calibration: Calibration,
    *,
    coordinate_plane: str = "event",
    one_based: bool = False,
    inclusive_max: bool = False,
    perimeter_samples: int = 9,
) -> Box:
    """Normalize a declared input plane to the event-camera crop contract."""

    plane = coordinate_plane.strip().lower()
    if plane == "event":
        return normalize_box_xyxy(
            raw_xyxy,
            calibration.event_resolution,
            one_based=one_based,
            inclusive_max=inclusive_max,
        )
    if plane == "rgb":
        rgb_box = normalize_box_xyxy(
            raw_xyxy,
            calibration.rgb_resolution,
            one_based=one_based,
            inclusive_max=inclusive_max,
        )
        return _project_box(
            rgb_box,
            source_intrinsics=calibration.rgb_intrinsics,
            source_distortion=calibration.rgb_distortion,
            target_intrinsics=calibration.event_intrinsics,
            target_distortion=calibration.event_distortion,
            rotation_source_to_target=calibration.rotation_event_to_rgb.T,
            target_resolution=calibration.event_resolution,
            perimeter_samples=perimeter_samples,
            clamp=True,
        )
    raise ValueError("coordinate_plane must be 'event' or 'rgb'")


__all__ = [
    "Box",
    "Calibration",
    "input_bbox_to_event_box",
    "load_calibration",
    "normalize_box_xyxy",
    "project_event_box_to_rgb",
]
