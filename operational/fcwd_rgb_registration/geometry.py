"""Published intrinsics plus an image-estimated local translation, not extrinsics.

The translation is estimated once at the query time and retained for all RGB
history frames. Independently aligning history frames would erase expansion.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

SETTINGS = {
    "event_window_us": 20_000,
    "minimum_ncc": 0.22,
    "minimum_event_pixels": 100,
    "maximum_split_disagreement_px": 5.0,
    "minimum_search_px": 60,
    "search_height_fraction": 0.20,
    "large_template_inset_fraction": 0.15,
    "blur_sigma": 1.5,
    "rgb_roi_margin_fraction": 0.25,
}


def camera_parameters(values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert MATLAB row-vector K and one-based pixels to OpenCV coordinates."""
    matrix = np.asarray(values["IntrinsicMatrix"], np.float64).T.copy()
    matrix[:2, 2] -= 1.0
    radial = np.asarray(values["RadialDistortion"], np.float64).ravel()
    tangent = np.asarray(values["TangentialDistortion"], np.float64).ravel()
    distortion = np.array([radial[0], radial[1], tangent[0], tangent[1], 0.0])
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or min(matrix[0, 0], matrix[1, 1]) <= 0
    ):
        raise ValueError("Invalid camera intrinsic matrix")
    if not np.isfinite(distortion).all():
        raise ValueError("Invalid distortion coefficients")
    return matrix, distortion


def project_points(
    points: np.ndarray, calibration: dict[str, Any], shift: tuple[float, float]
) -> np.ndarray:
    """Map raw event pixels through the local undistorted plane to raw RGB."""
    ke, de = camera_parameters(calibration["DVScameraParams"])
    kr, dr = camera_parameters(calibration["APScameraParams"])
    value = np.asarray(points, np.float64).reshape(-1, 1, 2)
    if not np.isfinite(value).all() or not np.isfinite(shift).all():
        raise ValueError("Nonfinite geometry input")
    pixels = cv2.undistortPoints(value, ke, de, P=ke).reshape(-1, 2)
    pixels += np.asarray(shift)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(ke).T
    result, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), kr, dr)
    return np.asarray(result).reshape(-1, 2)


def project_box(
    box: list[float], calibration: dict[str, Any], shift: tuple[float, float]
) -> list[float]:
    """Sample every side: distortion means extrema need not be at corners."""
    x0, y0, x1, y1 = box
    if not np.isfinite(box).all() or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid event ROI")
    x = np.linspace(x0, x1, 33)
    y = np.linspace(y0, y1, 33)
    points = np.concatenate(
        (
            np.column_stack((x, np.full(33, y0))),
            np.column_stack((x, np.full(33, y1))),
            np.column_stack((np.full(33, x0), y)),
            np.column_stack((np.full(33, x1), y)),
        )
    )
    mapped = project_points(points, calibration, shift)
    return [*mapped.min(0).tolist(), *mapped.max(0).tolist()]


def _match(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(response).all() or float(template.std()) < 1e-5:
        return 0.0, (0, 0)
    _, maximum, _, location = cv2.minMaxLoc(response)
    return float(maximum), (int(location[0]), int(location[1]))


def estimate_shift(
    rgb: np.ndarray,
    early_counts: np.ndarray,
    late_counts: np.ndarray,
    event_box: list[float],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Align on causal sensor evidence; reject weak or temporally unstable matches.

    NCC and half-window agreement are quality diagnostics, not ground-truth pixel
    error. A rejected query remains in evaluation with an explicit RGB fallback.
    """
    if rgb.shape != (1200, 1920, 3) or rgb.dtype != np.uint8:
        raise ValueError("FCWD requires the original 1920x1200 RGB image")
    if early_counts.shape != (720, 1280) or late_counts.shape != (720, 1280):
        raise ValueError("FCWD requires 1280x720 event counts")
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in (early_counts, late_counts)):
        raise ValueError("Event count maps must be finite and nonnegative")
    project_box(event_box, calibration, (0.0, 0.0))  # validate before slicing
    kr, dr = camera_parameters(calibration["APScameraParams"])
    ke, de = camera_parameters(calibration["DVScameraParams"])
    mx, my = cv2.initUndistortRectifyMap(kr, dr, None, ke, (1280, 720), cv2.CV_32FC1)
    undistorted = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
    grey = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8)).apply(grey)
    gradient = cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    gradient = cv2.GaussianBlur(np.log1p(gradient), (0, 0), SETTINGS["blur_sigma"])
    early = cv2.undistort(early_counts.astype(np.float32), ke, de)
    late = cv2.undistort(late_counts.astype(np.float32), ke, de)
    # Distortion is small, but transform the template bounds as well.
    corners = np.asarray(event_box, np.float64).reshape(2, 1, 2)
    corners = cv2.undistortPoints(corners, ke, de, P=ke).reshape(2, 2)
    a, b = np.floor(corners[0]).astype(int)
    c, d = np.ceil(corners[1]).astype(int)
    a, b, c, d = max(0, a), max(0, b), min(1280, c), min(720, d)
    height, width = d - b, c - a
    if width > 1000 or height > 650:
        inset_x, inset_y = int(width * 0.15), int(height * 0.15)
        a, b, c, d = a + inset_x, b + inset_y, c - inset_x, d - inset_y
    if c - a < 16 or d - b < 16:
        return {"accepted": False, "reason": "INSUFFICIENT_ROI_AREA"}
    support = int(np.count_nonzero((early + late)[b:d, a:c]))
    if support < SETTINGS["minimum_event_pixels"]:
        return {
            "accepted": False,
            "reason": "INSUFFICIENT_EVENT_SUPPORT",
            "support_pixels": support,
        }
    pad = max(SETTINGS["minimum_search_px"], int(height * SETTINGS["search_height_fraction"]))
    x0, y0, x1, y1 = max(0, a - pad), max(0, b - pad), min(1280, c + pad), min(720, d + pad)
    search = gradient[y0:y1, x0:x1]
    matches = []
    for counts in (early + late, early, late):
        encoded = cv2.GaussianBlur(np.log1p(counts), (0, 0), SETTINGS["blur_sigma"])
        score, location = _match(search, encoded[b:d, a:c])
        matches.append(
            {"ncc": score, "shift_px": [int(location[0] + x0 - a), int(location[1] + y0 - b)]}
        )
    disagreement = float(
        np.linalg.norm(np.subtract(matches[1]["shift_px"], matches[2]["shift_px"]))
    )
    shift = matches[0]["shift_px"]
    on_boundary = abs(shift[0]) >= pad - 1 or abs(shift[1]) >= pad - 1
    accepted = (
        min(m["ncc"] for m in matches) >= SETTINGS["minimum_ncc"]
        and disagreement <= SETTINGS["maximum_split_disagreement_px"]
        and not on_boundary
    )
    return {
        "accepted": bool(accepted),
        "reason": "" if accepted else "WEAK_OR_UNSTABLE_LOCAL_REGISTRATION",
        "shift_px": shift,
        "matches": matches,
        "half_window_disagreement_px": disagreement,
        "support_pixels": support,
        "search_boundary": bool(on_boundary),
        "template_xyxy": [int(a), int(b), int(c), int(d)],
        "model": "published_intrinsics_plus_query_local_translation",
        "metric_extrinsic_calibration": False,
    }
