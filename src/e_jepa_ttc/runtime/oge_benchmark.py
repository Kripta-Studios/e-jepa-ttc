"""Synchronized model-only and complete-online OGE latency protocols."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.models.object_geo_jepa_ttc import ObjectGeometryJEPATTC, OGEOutput


def _summary(
    durations_s: list[float],
    *,
    protocol: str,
    runtime_includes: list[str],
) -> dict[str, Any]:
    values = np.asarray(durations_s, dtype=np.float64)
    return {
        "protocol": protocol,
        "runtime_includes": runtime_includes,
        "individual_cost_time_s": values.tolist(),
        "latency_ms_mean": float(1000.0 * values.mean()),
        "latency_ms_p50": float(1000.0 * np.percentile(values, 50)),
        "latency_ms_p95": float(1000.0 * np.percentile(values, 95)),
        "latency_ms_p99": float(1000.0 * np.percentile(values, 99)),
        "windows_per_second": float(values.shape[0] / values.sum()),
    }


def benchmark_complete_online_estimator(
    prediction: Callable[[], Any],
    *,
    device: torch.device,
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, Any]:
    """Measure a callback that must include windowing through postprocessing."""

    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("Invalid latency iteration counts.")

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            prediction()
        synchronize()
        durations: list[float] = []
        for _ in range(measured_iterations):
            synchronize()
            started = time.perf_counter()
            prediction()
            synchronize()
            durations.append(time.perf_counter() - started)
    return _summary(
        durations,
        protocol="complete_online_ttc_estimation_v1",
        runtime_includes=[
            "online_window_and_voxelization",
            "host_to_device_transfer",
            "encoder",
            "queries",
            "geometry",
            "postprocessing",
        ],
    )


def benchmark_oge_model_only(
    model: ObjectGeometryJEPATTC,
    inputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark tensors already voxelized/on-device; never label this end-to-end."""

    model = model.to(device).eval()
    floating = {
        "context_events",
        "context_times_s",
        "context_boxes",
        "context_ego_actions",
    }
    batch = {
        name: value.to(
            device=device,
            dtype=torch.float32 if name in floating else torch.bool,
        )
        for name, value in inputs.items()
    }

    def invoke() -> OGEOutput:
        return model(**batch)

    result = benchmark_complete_online_estimator(
        invoke,
        device=device,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    result["protocol"] = "oge_model_only_prevoxelized_v1"
    result["runtime_includes"] = ["encoder", "queries", "geometry", "postprocessing"]
    result["explicitly_excludes"] = [
        "event_windowing",
        "voxelization",
        "host_to_device_transfer",
        "checkpoint_loading",
    ]
    result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    return result


__all__ = ["benchmark_complete_online_estimator", "benchmark_oge_model_only"]
