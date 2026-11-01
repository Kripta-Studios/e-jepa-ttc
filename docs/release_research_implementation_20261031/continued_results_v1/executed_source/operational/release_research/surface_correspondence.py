"""Fixed local RGB correspondences for V1; no TTC labels, learned flow or detector."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def match_patch(
    source: np.ndarray, target: np.ndarray, point: tuple[int, int], radius: int
) -> tuple[tuple[int, int], float, float] | None:
    """11x11 normalized cross correlation with a distinct-peak margin in a fixed search area."""
    if source.shape != target.shape or source.ndim != 2 or radius < 1:
        raise ValueError("Expected same-frame grayscale images and a positive search radius")
    x, y = point
    half = 5
    height, width = source.shape
    if not (half <= x < width - half and half <= y < height - half):
        return None
    patch = source[y - half : y + half + 1, x - half : x + half + 1].astype(np.float64)
    patch -= patch.mean()
    energy = float(np.linalg.norm(patch))
    if energy < 1e-3:
        return None
    left, right = max(half, x - radius), min(width - half - 1, x + radius)
    top, bottom = max(half, y - radius), min(height - half - 1, y + radius)
    search = target[top - half : bottom + half + 1, left - half : right + half + 1]
    windows = sliding_window_view(search, (11, 11)).astype(np.float64)
    windows -= windows.mean(axis=(-1, -2), keepdims=True)
    denominator = np.linalg.norm(windows, axis=(-1, -2)) * energy
    scores = np.einsum("ijkl,kl->ij", windows, patch) / np.maximum(denominator, 1e-12)
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    peak = float(scores[iy, ix])
    competitors = scores.copy()
    competitors[max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = -1.0
    gap = peak - float(competitors.max())
    return (int(left + ix), int(top + iy)), peak, gap


def accepted_match(
    source: np.ndarray, target: np.ndarray, point: tuple[int, int], radius: int
) -> tuple[tuple[int, int], float] | None:
    """Fixed match quality plus one-pixel forward/backward consistency."""
    found = match_patch(source, target, point, radius)
    if found is None or found[1] < 0.5 or found[2] < 0.05:
        return None
    reverse = match_patch(target, source, found[0], radius)
    if reverse is None or np.linalg.norm(np.array(reverse[0]) - point) > 1.0:
        return None
    return found[0], found[1]


def affine_surface(source: np.ndarray, target: np.ndarray) -> dict:
    """Robust affine map from independently measured correspondences, including area change."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Expected point pairs[N,2]")
    if len(source) < 6:
        return {"valid": False, "reason": "fewer_than_six_correspondences", "count": len(source)}
    design = np.column_stack((source / 128.0, np.ones(len(source))))
    if np.linalg.matrix_rank(design) < 3:
        return {"valid": False, "reason": "collinear_points", "count": len(source)}
    weights = np.ones(len(source))
    coefficient = np.zeros((3, 2))
    for _ in range(5):
        coefficient = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]),
            target / 128.0 * np.sqrt(weights[:, None]),
            rcond=None,
        )[0]
        residual = np.linalg.norm((design @ coefficient) * 128.0 - target, axis=1)
        scale = max(0.25, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(residual, 1e-12))
    jacobian = coefficient[:2].T
    sign, logdet = np.linalg.slogdet(jacobian)
    valid = bool(sign > 0 and np.isfinite(logdet))
    return {
        "valid": valid,
        "reason": "ok" if valid else "nonpositive_determinant",
        "count": len(source),
        "log_scale_change": float(0.5 * logdet) if valid else None,
        "median_residual_pixels": float(np.median(residual)),
        "jacobian": jacobian.tolist(),
        "translation_pixels": (128.0 * coefficient[2]).tolist(),
    }


def persistent_surface(frames: np.ndarray, times_us: np.ndarray) -> dict:
    """Measure a->b, b->c and independent a->c on the same persistent local patches."""
    if frames.shape != (3, 128, 128) or times_us.shape != (3,) or np.any(np.diff(times_us) <= 0):
        raise ValueError("Require three real 128x128 grayscale frames and increasing clocks")
    if not np.isfinite(frames).all():
        raise ValueError("Nonfinite images")
    rows = []
    candidates = [(x, y) for y in range(16, 113, 16) for x in range(16, 113, 16)]
    for point in candidates:
        ab = accepted_match(frames[0], frames[1], point, 8)
        ac = accepted_match(frames[0], frames[2], point, 16)
        if ab is None or ac is None:
            continue
        bc = accepted_match(frames[1], frames[2], ab[0], 8)
        if bc is None:
            continue
        error = float(np.linalg.norm(np.array(bc[0]) - ac[0]))
        rows.append(
            {
                "a": list(point),
                "b": list(ab[0]),
                "c_composed": list(bc[0]),
                "c_direct": list(ac[0]),
                "composition_error_pixels": error,
                "minimum_ncc": min(ab[1], bc[1], ac[1]),
            }
        )
    admitted = [r for r in rows if r["composition_error_pixels"] <= 2.0]
    a = np.array([r["a"] for r in admitted], float).reshape(-1, 2)
    b = np.array([r["b"] for r in admitted], float).reshape(-1, 2)
    direct = np.array([r["c_direct"] for r in admitted], float).reshape(-1, 2)
    composed = np.array([r["c_composed"] for r in admitted], float).reshape(-1, 2)
    fits = {
        "ab": affine_surface(a, b),
        "bc": affine_surface(b, composed),
        "ac": affine_surface(a, direct),
    }
    residual = None
    q_endpoint = None
    if all(v["valid"] for v in fits.values()):
        residual = (
            fits["ac"]["log_scale_change"]
            - fits["ab"]["log_scale_change"]
            - fits["bc"]["log_scale_change"]
        )
        # r_bc is measured backward from c: s_c/s_b=1+dt*q_c.
        q_endpoint = float(
            np.expm1(fits["bc"]["log_scale_change"]) / ((times_us[2] - times_us[1]) * 1e-6)
        )
    return {
        "required_patches": len(candidates),
        "matched_three_routes": len(rows),
        "persistent_patches": len(admitted),
        "coverage": len(admitted) / len(candidates),
        "routes": fits,
        "composition_logscale_residual": residual,
        "q_at_last_rgb": q_endpoint,
        "matches": rows,
    }
