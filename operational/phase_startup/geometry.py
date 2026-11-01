"""Causal inverse-apparent-scale fitting, with explicit support and residuals."""

from __future__ import annotations

from typing import Any

import numpy as np


def _fit(size: np.ndarray, times: np.ndarray) -> tuple[float, float, float]:
    inverse = size[-1] / size
    left, right = np.triu_indices(len(times), k=1)
    slopes = (inverse[right] - inverse[left]) / (times[right] - times[left])
    slope = float(np.median(slopes))
    intercept = float(np.median(inverse - slope * times))
    residual = float(np.sqrt(np.mean((inverse - intercept - slope * times) ** 2)))
    q = -slope / intercept if intercept > 0 else 0.0
    scatter = float(1.4826 * np.median(np.abs(slopes - slope)))
    return q, residual, scatter


def startup_geometry(
    boxes: np.ndarray, timestamps_us: np.ndarray, query_us: int, *, history_us: int = 650000
) -> dict[str, Any]:
    """Fit inverse width/height/area against physical time ending at the query.

    Boxes are inference-provided ROI observations, never GT distance or TTC.
    Coordinates and absolute clock do not enter the fitted TTC. Only causal
    observations within the stated history are admitted. Median pair slopes
    reduce sensitivity to one endpoint annotation without delaying the timestamp.
    Two observations are admitted with explicit count/span/quantization quality;
    their zero fit residual is not an uncertainty estimate.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    clocks = np.asarray(timestamps_us, dtype=np.int64)
    if boxes.shape != (len(clocks), 4) or clocks.ndim != 1:
        raise ValueError("Expected matching [N,4] boxes and [N] integer clocks")
    if len(clocks) and np.any(np.diff(clocks) <= 0):
        raise ValueError("BBox clocks must be unique and strictly increasing")
    causal = (clocks <= query_us) & (clocks >= query_us - history_us)
    boxes, clocks = boxes[causal], clocks[causal]
    good = np.isfinite(boxes).all(1) & np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
    admitted_fraction = float(good.mean()) if len(good) else 0.0
    boxes, clocks = boxes[good], clocks[good]
    result: dict[str, Any] = {
        "valid": False,
        "reason": "insufficient_causal_history",
        "q": 0.0,
        "ttc": None,
        "quality": [0.0] * 20,
        "count": len(clocks),
    }
    if len(clocks):
        last_extent = boxes[-1, 2:] - boxes[-1, :2]
        result["quality"][9] = min(len(clocks), 32) / 32
        result["quality"][10] = admitted_fraction
        result["quality"][11] = float((query_us - clocks[-1]) / 1e6)
        result["quality"][12] = 1 / float(last_extent.min())
        result["quality"][14] = float(last_extent[0] / last_extent[1])
        result["quality"][19] = float(last_extent.min()) / 512
    if len(clocks) < 2 or clocks[-1] - clocks[0] < 10000:
        return result
    times = (clocks - query_us) / 1e6
    extent = boxes[:, 2:] - boxes[:, :2]
    size = np.column_stack((extent, np.sqrt(extent[:, 0] * extent[:, 1])))
    fits = np.array([_fit(size[:, i], times) for i in range(3)])
    rates, residuals, scatter = fits.T
    q = float(np.median(rates))
    span = float(times[-1] - times[0])
    short = times >= -0.25
    short_q = (
        float(np.median([_fit(size[short, i], times[short])[0] for i in range(3)]))
        if short.sum() >= 3 and np.ptp(times[short]) >= 0.05
        else q
    )
    axis_spread = float(np.ptp(rates))
    aspect = extent[:, 0] / extent[:, 1]
    center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    center_motion = float(np.linalg.norm((center[-1] - center[0]) / extent[-1]) / span)
    age = float((query_us - clocks[-1]) / 1e6)
    # Dimensionless residual-to-motion ratio and pixel quantization support are
    # reliability inputs, not thresholds learned from this query's target.
    noise = float(np.median(residuals))
    quality = [
        q,
        short_q,
        *rates.tolist(),
        axis_spread / (abs(q) + 0.01),
        noise,
        noise / (abs(q) * span + 0.001),
        span,
        min(len(clocks), 32) / 32,
        admitted_fraction,
        age,
        1 / float(np.min(extent[-1])),
        abs(short_q - q) / (abs(q) + 0.01),
        float(aspect[-1]),
        abs(float(np.log(aspect[-1] / aspect[0]))),
        center_motion,
        float(scatter[0]),
        float(scatter[1]),
        float(np.min(extent[-1])) / 512,
    ]
    result.update(
        q=q,
        quality=quality,
        axis_q=rates.tolist(),
        residual=noise,
        span_seconds=span,
        age_seconds=age,
    )
    if not np.isfinite(quality).all() or abs(q) < 1e-6:
        result["reason"] = "stationary_or_undefined_expansion"
        return result
    if age > 0.15:
        result["reason"] = "stale_endpoint"
        return result
    result.update(valid=True, reason="available", ttc=1 / q)
    return result
