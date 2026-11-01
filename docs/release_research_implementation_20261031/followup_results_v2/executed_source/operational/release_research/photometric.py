"""S0 calibrated planar log-texture fixture with threshold crossings and transport masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import Array


@dataclass(frozen=True)
class PatchSimulation:
    """Observed events plus known reference; latent intensity never enters inference."""

    time_s: Array
    log_intensity: Array
    reference_initial: Array
    events: Array  # columns: pixel index, time seconds, polarity
    covered: np.ndarray
    c_positive: float
    c_negative: float


def simulate_patch(
    xy: Array,
    log_gradient: Array,
    time_s: Array,
    q_initial: float,
    illumination_rate: float,
    c_positive: float,
    c_negative: float,
    *,
    covered: np.ndarray | None = None,
    refractory_s: float = 0.0,
) -> PatchSimulation:
    """Sample threshold crossings of a transported affine log-texture.

    Under axial constant velocity, material coordinate is x*(1-q0*t).
    Threshold crossing timestamps use linear interpolation, exact for this fixture.
    A coverage hole drops observations while the physical sensor keeps updating.
    This is a deliberately limited S0 fixture, not a claim of realistic RAW data.
    """
    if xy.ndim != 2 or xy.shape[1] != 2 or log_gradient.shape != (2,) or len(xy) == 0:
        raise ValueError("Require pixel coordinates[N,2] and fixed texture gradient[2]")
    if time_s.ndim != 1 or len(time_s) < 2 or time_s[0] != 0 or np.any(np.diff(time_s) <= 0):
        raise ValueError("Times must increase strictly from a known t=0 reference")
    parameters = [q_initial, illumination_rate, c_positive, c_negative, refractory_s]
    if not np.isfinite(parameters).all() or min(c_positive, c_negative) <= 0 or refractory_s < 0:
        raise ValueError("Invalid calibrated sensor parameters")
    if not all(np.isfinite(a).all() for a in (xy, log_gradient, time_s)):
        raise ValueError("Nonfinite photometric scene")
    if np.any(1 - q_initial * time_s <= 0):
        raise ValueError("Fixture crosses contact")
    visibility = np.ones((len(time_s) - 1, len(xy)), bool) if covered is None else covered.copy()
    if visibility.shape != (len(time_s) - 1, len(xy)) or visibility.dtype != np.bool_:
        raise ValueError("Coverage must label each interval/pixel")
    texture = xy @ log_gradient
    intensity = (1 - q_initial * time_s[:, None]) * texture + illumination_rate * time_s[:, None]
    reference = intensity[0].copy()
    last_event = np.full(len(xy), -np.inf)
    events = []
    for step, (start, end) in enumerate(zip(time_s, time_s[1:], strict=False)):
        change = intensity[step + 1] - reference
        for pixel in range(len(xy)):
            polarity = 1 if change[pixel] >= 0 else -1
            threshold = c_positive if polarity > 0 else c_negative
            crossings = int(abs(change[pixel]) / threshold + 1e-12)
            if crossings > 10000:
                raise ValueError("Excessive fixture event count; reduce time step/contrast")
            slope = (intensity[step + 1, pixel] - intensity[step, pixel]) / (end - start)
            for _ in range(crossings):
                level = reference[pixel] + polarity * threshold
                timestamp = start + (level - intensity[step, pixel]) / slope
                reference[pixel] = level
                if timestamp - last_event[pixel] < refractory_s:
                    # These intervals are censored, not silence constraints.
                    visibility[step, pixel] = False
                    continue
                last_event[pixel] = timestamp
                if visibility[step, pixel]:
                    events.append((pixel, timestamp, polarity))
    array = np.asarray(events, dtype=np.float64).reshape(-1, 3)
    if len(array):
        array = array[np.argsort(array[:, 1], kind="stable")]
    return PatchSimulation(
        time_s.copy(), intensity, intensity[0].copy(), array, visibility, c_positive, c_negative
    )


def recover_q_and_light(simulation: PatchSimulation, xy: Array, log_gradient: Array) -> dict:
    """Fit known texture/calibration using event counts and explicit endpoint tolerance.

    Discard pixels with any transport/refractory hole; absence of events is not
    interpreted as immobility. Estimate includes an unknown patch-wide light ramp.
    """
    if xy.shape != (len(simulation.reference_initial), 2) or log_gradient.shape != (2,):
        raise ValueError("Calibration and sensor pixels differ")
    complete = simulation.covered.all(axis=0)
    increment = np.zeros(len(xy))
    count = np.zeros(len(xy), int)
    for pixel, _timestamp, polarity in simulation.events:
        index = int(pixel)
        increment[index] += simulation.c_positive if polarity > 0 else -simulation.c_negative
        count[index] += 1
    valid = complete & (count > 0)
    duration = simulation.time_s[-1]
    h = np.column_stack((-xy @ log_gradient, np.ones(len(xy)))) * duration
    if valid.sum() < 2 or np.linalg.matrix_rank(h[valid]) < 2:
        return {"q": None, "light": None, "accepted": int(valid.sum()), "reason": "unidentifiable"}
    estimated = np.linalg.lstsq(h[valid], increment[valid], rcond=None)[0]
    residual = h[valid] @ estimated - increment[valid]
    return {
        "q": float(estimated[0]),
        "light": float(estimated[1]),
        "accepted": int(valid.sum()),
        "required": len(xy),
        "reason": "known_texture_development_fixture",
        "endpoint_tolerance": max(simulation.c_positive, simulation.c_negative),
        "max_absolute_residual": float(np.max(np.abs(residual))),
    }
