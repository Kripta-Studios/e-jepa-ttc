"""Separate float64 graph solver with the frozen CPU recipe and bounded cache."""

from __future__ import annotations

import gc
import time
from dataclasses import replace
from itertools import product

import numpy as np
import torch
from scipy.optimize import minimize

from operational.phase_event_expansion.model import (
    ExpansionResult,
    SolverConfig,
    estimate_expansion,
    select_events,
    temporal_agreement,
    warp_endpoint,
)
from operational.phase_event_expansion.torch_objective import (
    CachedTorchObjective,
    CudaGraphObjective,
)


class CudaObjectiveCache:
    """One observation shape, two graphs (B1/B81); same-shape observations reload in place."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose backend='cpu' explicitly")
        self.model: CachedTorchObjective | None = None
        self.single: CudaGraphObjective | None = None
        self.grid: CudaGraphObjective | None = None
        self.count = 0
        self.capture_count = 0
        self.replay_count = 0
        self.last_setup: dict[str, float | bool] = {}

    @property
    def cached_shapes(self) -> int:
        """At most one event shape can remain resident."""
        return int(self.model is not None)

    def close(self) -> None:
        """Release both graphs before creating a new shape, bounding retained VRAM."""
        torch.cuda.synchronize()
        self.single = None
        self.grid = None
        self.model = None
        self.count = 0
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def prepare(self, points: np.ndarray, age: np.ndarray, polarity: np.ndarray) -> None:
        """Update cached tensors, including time-bin identity, without using labels."""
        reused = self.model is not None and self.count == len(points)
        if not reused:
            self.close()
        started = time.perf_counter()
        if self.model is None:
            self.model = CachedTorchObjective(
                points, age, polarity, dtype=torch.float64, device="cuda"
            )
            self.count = len(points)
        else:
            early = age >= (age.min() + age.max()) / 2
            channels = (polarity < 0).astype(np.int64) + 2 * (~early).astype(np.int64)
            self.model.points.copy_(torch.from_numpy(np.asarray(points, np.float64)))
            self.model.age.copy_(torch.from_numpy(np.asarray(age, np.float64)))
            self.model.channel_offset.copy_(torch.from_numpy(channels * 128 * 128))
        torch.cuda.synchronize()
        preload = time.perf_counter() - started
        capture, warmup = 0.0, 0.0
        if not reused:
            self.single = CudaGraphObjective(self.model, 1)
            self.grid = CudaGraphObjective(self.model, 81)
            self.capture_count += 2
            capture = self.single.capture_seconds + self.grid.capture_seconds
            warmup = self.single.warmup_seconds + self.grid.warmup_seconds
        self.last_setup = {
            "preload_seconds": preload,
            "capture_seconds": capture,
            "capture_warmup_seconds": warmup,
            "reused_graphs": reused,
        }

    @torch.inference_mode()
    def evaluate(self, parameters: np.ndarray) -> np.ndarray:
        """Synchronous host callback; results own storage outside captured graphs."""
        parameters = np.asarray(parameters, dtype=np.float64)
        graph = (
            self.single if len(parameters) == 1 else self.grid if len(parameters) == 81 else None
        )
        if graph is None:
            raise ValueError("Prepare the cache and evaluate exactly 1 or 81 candidates")
        result = graph(torch.from_numpy(parameters)).cpu().numpy().copy()
        self.replay_count += 1
        return result


def estimate_accelerated(
    events: np.ndarray,
    roi: np.ndarray | list[float],
    query_us: int,
    config: SolverConfig | None = None,
    cache: CudaObjectiveCache | None = None,
    *,
    backend: str = "cuda",
) -> ExpansionResult:
    """Run the unchanged bounded recipe with explicit CPU or float64 CUDA execution."""
    started = time.perf_counter()
    config = config or SolverConfig(objective="temporal_ncc", reference="endpoint")
    if backend == "cpu":
        result = estimate_expansion(events, roi, query_us, config)
        return replace(
            result, diagnostics={**result.diagnostics, "execution_backend": "cpu_explicit"}
        )
    if backend != "cuda":
        raise ValueError("Expected explicit cpu or cuda backend")
    if (
        config.objective != "temporal_ncc"
        or config.reference != "endpoint"
        or config.canvas_size != 128
        or config.canvas_half_extent != 1.0
        or config.blur_sigma != 0.8
    ):
        raise ValueError("Accelerator requires frozen endpoint NCC,128 canvas,extent1,sigma.8")
    owns_cache = cache is None
    active = cache if cache is not None else CudaObjectiveCache()
    try:
        with torch.inference_mode():
            result = _estimate_cuda(events, roi, query_us, config, active)
    finally:
        if owns_cache:
            active.close()
    return replace(result, elapsed_seconds=time.perf_counter() - started)


def _estimate_cuda(
    events: np.ndarray,
    roi: np.ndarray | list[float],
    query_us: int,
    config: SolverConfig,
    cache: CudaObjectiveCache,
) -> ExpansionResult:
    """Versioned copy of CPU validation/search/gates with an explicit objective provider."""
    started = time.perf_counter()
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
        "execution_backend": "cuda_graph_float64",
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
    early = age >= (age.min() + age.max()) / 2
    if min(early.sum(), (~early).sum()) < 64:
        return abstain("insufficient_temporal_bin_support")
    diagnostics.update(internal_reference_seconds_before_query=0.0, objective="temporal_ncc")
    captures_before, replays_before = cache.capture_count, cache.replay_count
    setup_start = time.perf_counter()
    cache.prepare(points, age, polarity)
    diagnostics["cache_setup_wall_seconds"] = time.perf_counter() - setup_start
    diagnostics["cache_setup"] = cache.last_setup.copy()
    evaluations = 0

    def contrast(parameters: np.ndarray) -> tuple[float, float]:
        return temporal_agreement(
            warp_endpoint(points, age, parameters),
            polarity,
            early,
            config.canvas_size,
            config.canvas_half_extent,
            config.blur_sigma,
        )

    baseline, _ = contrast(np.zeros(3))

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return float(cache.evaluate(parameters[None])[0])

    grid = [np.array(p) for p in product(np.linspace(-10, 10, 9), (-2.5, 0, 2.5), (-2.5, 0, 2.5))]
    grid_values = cache.evaluate(np.stack(grid))
    evaluations += len(grid)
    ranked = sorted(zip(grid_values, grid, strict=True), key=lambda pair: pair[0])
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
    gain = score - baseline
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
        graph_captures=cache.capture_count - captures_before,
        graph_replays=cache.replay_count - replays_before,
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
    elif score < config.minimum_temporal_agreement:
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
