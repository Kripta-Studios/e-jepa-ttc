"""Separate candidate: retain valid GPU estimates, reproduce every abstention on CPU."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from operational.phase_event_expansion.accelerated import CudaObjectiveCache, estimate_accelerated
from operational.phase_event_expansion.model import (
    ExpansionResult,
    SolverConfig,
    estimate_expansion,
)


@dataclass(frozen=True)
class HybridResult:
    """Keep the unmodified selected result, attempted GPU result, and complete cost."""

    result: ExpansionResult
    gpu_attempt: ExpansionResult
    fallback_used: bool
    gpu_wall_seconds: float
    cpu_wall_seconds: float
    total_wall_seconds: float


def estimate_hybrid(
    events: np.ndarray,
    roi: np.ndarray | list[float],
    query_us: int,
    config: SolverConfig | None = None,
    cache: CudaObjectiveCache | None = None,
    *,
    gpu_estimator: Callable[..., ExpansionResult] = estimate_accelerated,
    cpu_estimator: Callable[..., ExpansionResult] = estimate_expansion,
) -> HybridResult:
    """Use the original CPU solver for ANY GPU-invalid result, without label access."""
    started = time.perf_counter()
    config = config or SolverConfig(objective="temporal_ncc", reference="endpoint")
    gpu = gpu_estimator(events, roi, query_us, config, cache)
    gpu_wall = time.perf_counter() - started
    fallback, result, cpu_wall = not gpu.valid, gpu, 0.0
    if fallback:
        cpu_started = time.perf_counter()
        result = cpu_estimator(events, roi, query_us, config)
        cpu_wall = time.perf_counter() - cpu_started
    return HybridResult(result, gpu, fallback, gpu_wall, cpu_wall, time.perf_counter() - started)
