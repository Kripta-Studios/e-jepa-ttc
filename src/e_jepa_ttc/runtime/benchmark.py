"""Repeatable end-to-end and model-only TTC latency benchmarks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA


def benchmark_object_ttc_model(
    model: ObjectCentricEventJEPA,
    inputs: dict[str, torch.Tensor],
    *,
    device: str | torch.device,
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, Any]:
    """Measure synchronized model latency with percentile reporting."""

    if warmup_iterations < 0 or measured_iterations <= 0:
        msg = "Latency warmup must be non-negative and measured_iterations positive."
        raise ValueError(msg)
    target = torch.device(device)
    model = model.to(target).eval()
    names = (
        "context_events",
        "context_boxes",
        "context_object_mask",
        "context_sampling_boxes",
        "context_ego_actions",
        "context_ego_action_mask",
    )
    floating_inputs = {
        "context_events",
        "context_boxes",
        "context_sampling_boxes",
        "context_ego_actions",
    }
    batch = {
        name: inputs[name].to(
            target,
            dtype=torch.float32 if name in floating_inputs else torch.bool,
        )
        for name in names
    }

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def invoke() -> None:
        model.predict_ttc(
            batch["context_events"],
            batch["context_boxes"],
            batch["context_object_mask"],
            context_sampling_boxes=batch["context_sampling_boxes"],
            context_ego_actions=batch["context_ego_actions"],
            context_ego_action_mask=batch["context_ego_action_mask"],
        )

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            invoke()
        synchronize()
        durations: list[float] = []
        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target)
        for _ in range(measured_iterations):
            synchronize()
            start = time.perf_counter()
            invoke()
            synchronize()
            durations.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(durations)
    return {
        "device": str(target),
        "batch_size": int(batch["context_events"].shape[0]),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "latency_ms_mean": float(values.mean()),
        "latency_ms_std": float(values.std()),
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p90": float(np.percentile(values, 90)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "latency_ms_p99": float(np.percentile(values, 99)),
        "windows_per_second": float(1000.0 * values.shape[0] / values.sum()),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


__all__ = ["benchmark_object_ttc_model"]
