"""Original bounded contrast solver for endpoint translational event expansion.

Coordinates are centered and divided by the current ROI's longest dimension.
The three parameters are q=1/TTC and two free translation rates in that frame.
Only mathematical motion/contrast ideas are referenced; no external code is used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


@dataclass(frozen=True)
class SolverConfig:
    """Fixed first-prototype search and acceptance recipe, independent of labels."""

    event_budget: int = 20_000
    seed: int = 7
    canvas_size: int = 128
    canvas_half_extent: float = 1.0
    blur_sigma: float = 0.8
    starts: int = 3
    max_evaluations_per_start: int = 240
    minimum_events: int = 128
    minimum_span_seconds: float = 0.01
    minimum_spatial_eigen_ratio: float = 0.01
    minimum_contrast_gain: float = 0.02
    minimum_profile_gap: float = 0.002
    maximum_lost_fraction: float = 0.01
    minimum_abs_q: float = 0.1
    objective: str = "energy"
    reference: str = "endpoint"
    minimum_temporal_agreement: float = 0.8

    def __post_init__(self) -> None:
        if not 1 <= self.event_budget <= 20_000:
            raise ValueError("Event budget must be between 1 and 20000")
        if self.starts != 3 or self.max_evaluations_per_start != 240:
            raise ValueError("Prototype fixes three starts and 240 evaluations per start")
        if self.objective not in ("energy", "temporal_ncc"):
            raise ValueError("Expected energy or temporal_ncc objective")
        if self.reference not in ("endpoint", "midpoint"):
            raise ValueError("Expected endpoint or midpoint reference")


@dataclass(frozen=True)
class ExpansionResult:
    """A diagnostic estimate; TTC is available only when every validity gate passes."""

    q: float | None
    translation: tuple[float, float] | None
    ttc_seconds: float | None
    valid: bool
    reason: str
    elapsed_seconds: float
    diagnostics: dict[str, object] = field(default_factory=dict)


def warp_endpoint(
    points: np.ndarray, age_seconds: np.ndarray, parameters: np.ndarray | list[float]
) -> np.ndarray:
    """Warp normalized observations to the query endpoint, not a packet midpoint."""
    q, bx, by = parameters
    return points + (q * points - np.array([bx, by])) * age_seconds[:, None]


def parameters_at_reference(endpoint: np.ndarray, seconds_before_query: float) -> np.ndarray:
    """Convert endpoint q,b to a causal internal reference under constant velocity."""
    denominator = 1 + endpoint[0] * seconds_before_query
    if denominator <= 0:
        raise ValueError("Nonphysical reference conversion")
    return endpoint / denominator


def parameters_at_query(reference: np.ndarray, seconds_before_query: float) -> np.ndarray:
    """Convert internal q,b to endpoint values; TTC decreases with elapsed time."""
    denominator = 1 - reference[0] * seconds_before_query
    if denominator <= 0:
        raise ValueError("Nonphysical reference conversion")
    return reference / denominator


def select_events(events: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Deterministically sample without replacement, retaining timestamp ordering."""
    if len(events) <= budget:
        return events.copy()
    indices = np.random.default_rng(seed).choice(len(events), budget, replace=False)
    return events[np.sort(indices)].copy()


def bilinear_image(
    points: np.ndarray, polarity: np.ndarray, size: int, half_extent: float
) -> tuple[np.ndarray, float]:
    """Accumulate separate polarities on a fixed canvas and report discarded mass."""
    coordinate = (points + half_extent) * ((size - 1) / (2 * half_extent))
    inside = np.isfinite(coordinate).all(1) & (coordinate >= 0).all(1)
    inside &= (coordinate < size - 1).all(1)
    coordinate = coordinate[inside]
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower
    channel = (polarity[inside] < 0).astype(np.int64)
    flat = np.zeros(2 * size * size, dtype=np.float64)
    for dx, dy in product((0, 1), repeat=2):
        weight_x = fraction[:, 0] if dx else 1 - fraction[:, 0]
        weight_y = fraction[:, 1] if dy else 1 - fraction[:, 1]
        index = channel * size * size + (lower[:, 1] + dy) * size + lower[:, 0] + dx
        flat += np.bincount(index, weights=weight_x * weight_y, minlength=len(flat))
    return flat.reshape(2, size, size), float(1 - inside.mean()) if len(points) else 0.0


def temporal_agreement(
    points: np.ndarray,
    polarity: np.ndarray,
    early: np.ndarray,
    size: int,
    half_extent: float,
    sigma: float,
) -> tuple[float, float]:
    """Zero-mean normalized correlation between separately warped time halves.

    Per-image energy normalization removes a direct reward for concentrating mass.
    The halves retain separate polarity channels and share a fixed padded canvas.
    """
    images = []
    losses = []
    for selected in (early, ~early):
        image, lost = bilinear_image(points[selected], polarity[selected], size, half_extent)
        image = gaussian_filter(image, sigma=(0, sigma, sigma))
        image -= image.mean(axis=(1, 2), keepdims=True)
        images.append(image)
        losses.append(lost * int(selected.sum()))
    norm = np.sqrt(np.sum(images[0] ** 2) * np.sum(images[1] ** 2))
    agreement = float(np.sum(images[0] * images[1]) / norm) if norm > 0 else 0.0
    return agreement, float(sum(losses) / max(1, len(points)))


def estimate_expansion(
    events: np.ndarray,
    roi: np.ndarray | list[float],
    query_us: int,
    config: SolverConfig | None = None,
) -> ExpansionResult:
    """Estimate from [x,y,t_us,polarity] and one current ROI; reject future events.

    The caller supplies undistorted coordinates in a consistent camera frame.
    No previous box, label, depth, or sequence identifier is an input.
    """
    started = time.perf_counter()
    config = config or SolverConfig()
    events = np.asarray(events, dtype=np.float64)
    roi = np.asarray(roi, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4 or not np.isfinite(events).all():
        raise ValueError("Expected finite [N,4] event array")
    if roi.shape != (4,) or not np.isfinite(roi).all() or np.any(roi[2:] <= roi[:2]):
        raise ValueError("Expected finite positive current ROI")
    if not isinstance(query_us, (int, np.integer)):
        raise ValueError("Query clock must be integer microseconds")
    if np.any(events[:, 2] > query_us):
        raise ValueError("Input contains future events")
    if np.any(np.diff(events[:, 2]) < 0) or np.any(events[:, 2] != np.floor(events[:, 2])):
        raise ValueError("Event timestamps must be sorted integer microseconds")
    if not np.isin(events[:, 3], (-1, 1)).all():
        raise ValueError("Expected polarity in {-1,+1}")
    inside = (events[:, :2] >= roi[:2]).all(1) & (events[:, :2] <= roi[2:]).all(1)
    admitted = select_events(events[inside], config.event_budget, config.seed)
    diagnostics: dict[str, object] = {
        "input_events": len(events),
        "roi_events": int(inside.sum()),
        "used_events": len(admitted),
        "objective_evaluations": 0,
        "reference_time_us": query_us,
    }

    def abstain(reason: str) -> ExpansionResult:
        return ExpansionResult(
            None, None, None, False, reason, time.perf_counter() - started, diagnostics
        )

    if len(admitted) < config.minimum_events:
        return abstain("insufficient_events")
    age = (query_us - admitted[:, 2]) / 1e6
    diagnostics["event_span_seconds"] = float(np.ptp(age))
    if np.ptp(age) < config.minimum_span_seconds:
        return abstain("insufficient_temporal_support")
    points = (admitted[:, :2] - (roi[:2] + roi[2:]) / 2) / np.max(roi[2:] - roi[:2])
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
    ratio = float(eigenvalues[0] / max(eigenvalues[-1], 1e-15))
    diagnostics["spatial_eigen_ratio"] = ratio
    if ratio < config.minimum_spatial_eigen_ratio:
        return abstain("insufficient_spatial_support")
    polarity = admitted[:, 3]
    midpoint_age = float((age.min() + age.max()) / 2)
    early = age >= midpoint_age
    if config.objective == "temporal_ncc" and min(early.sum(), (~early).sum()) < 64:
        return abstain("insufficient_temporal_bin_support")
    reference_offset = midpoint_age if config.reference == "midpoint" else 0.0
    diagnostics["internal_reference_seconds_before_query"] = reference_offset
    diagnostics["objective"] = config.objective
    evaluations = 0

    def contrast(parameters: np.ndarray) -> tuple[float, float]:
        reference_parameters = parameters_at_reference(parameters, reference_offset)
        warped = warp_endpoint(points, age - reference_offset, reference_parameters)
        if config.objective == "temporal_ncc":
            return temporal_agreement(
                warped,
                polarity,
                early,
                config.canvas_size,
                config.canvas_half_extent,
                config.blur_sigma,
            )
        image, lost = bilinear_image(
            warped, polarity, config.canvas_size, config.canvas_half_extent
        )
        smoothed = gaussian_filter(image, sigma=(0, config.blur_sigma, config.blur_sigma))
        # A fixed denominator/canvas keeps event loss visible to the penalty.
        return float(np.sum(smoothed * smoothed) / len(points) ** 2), lost

    baseline, _ = contrast(np.zeros(3))

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        scale = 1 + parameters[0] * age
        if np.any(scale <= 0):
            return 1000.0 + float(np.maximum(0, -scale).sum())
        score, lost = contrast(parameters)
        normalization = max(baseline, 1e-15) if config.objective == "energy" else 1.0
        return -score / normalization + 100 * lost

    grid = [np.array(p) for p in product(np.linspace(-10, 10, 9), (-2.5, 0, 2.5), (-2.5, 0, 2.5))]
    ranked = sorted(((objective(p), p) for p in grid), key=lambda pair: pair[0])
    starts: list[np.ndarray] = []
    for _, point in ranked:
        if not starts or all(abs(point[0] - previous[0]) >= 2.5 for previous in starts):
            starts.append(point)
        if len(starts) == config.starts:
            break
    solutions = [
        minimize(
            objective,
            point,
            method="Powell",
            bounds=[(-10, 10), (-5, 5), (-5, 5)],
            options={"maxfev": config.max_evaluations_per_start, "xtol": 1e-4, "ftol": 1e-5},
        )
        for point in starts
    ]
    best = min(solutions, key=lambda result: result.fun)
    q, bx, by = map(float, best.x)
    score, lost = contrast(best.x)
    gain = score / max(baseline, 1e-15) - 1 if config.objective == "energy" else score - baseline
    nearby = []
    for delta in (-0.5, 0.5):
        probe = best.x.copy()
        probe[0] = np.clip(q + delta, -10, 10)
        nearby.append(objective(probe))
    profile_gap = float(min(nearby) - best.fun)
    diagnostics.update(
        objective_evaluations=evaluations,
        contrast_baseline=baseline,
        contrast_best=score,
        contrast_gain=gain,
        lost_fraction=lost,
        profile_gap_fixed_translation=profile_gap,
        minimum_warp_scale=float(np.min(1 + q * age)),
        solutions=[
            {
                "parameters": s.x.tolist(),
                "objective": float(s.fun),
                "success": bool(s.success),
                "evaluations": int(s.nfev),
                "message": str(s.message),
            }
            for s in solutions
        ],
    )
    reason = "available"
    if not best.success:
        reason = "optimizer_failure"
    elif abs(q) >= 9.99 or max(abs(bx), abs(by)) >= 4.99:
        reason = "search_boundary"
    elif np.min(1 + q * age) <= 0:
        reason = "nonpositive_warp_scale"
    elif lost > config.maximum_lost_fraction:
        reason = "lost_events"
    elif config.objective == "temporal_ncc" and score < config.minimum_temporal_agreement:
        reason = "weak_temporal_agreement"
    elif gain < config.minimum_contrast_gain:
        reason = "flat_objective"
    elif abs(q) < config.minimum_abs_q:
        reason = "unresolved_expansion"
    elif profile_gap < config.minimum_profile_gap:
        reason = "ambiguous_expansion"
    valid = reason == "available"
    return ExpansionResult(
        q,
        (bx, by),
        1 / q if valid else None,
        valid,
        reason,
        time.perf_counter() - started,
        diagnostics,
    )
