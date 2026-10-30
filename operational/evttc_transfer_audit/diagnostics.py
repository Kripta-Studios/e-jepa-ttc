"""Passive instrumentation of the exact runtime; labels never enter this module."""

# ruff: noqa: ANN401 -- torch hooks expose heterogeneous native model outputs

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from operational.evttc_competition.runtime import CompetitionRuntime, PreparedObservation


def serialize(value: Any) -> Any:
    """Serialize diagnostics, preserving nonfinite values as explicit strings."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() > 4096:
            array = tensor.float().contiguous().numpy()
            finite = array[np.isfinite(array)]
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "hash_encoding": "contiguous_float32",
                "finite_count": int(finite.size),
                "minimum": float(finite.min()) if finite.size else None,
                "maximum": float(finite.max()) if finite.size else None,
                "mean": float(finite.mean()) if finite.size else None,
            }
        return serialize(tensor.tolist())
    if isinstance(value, np.ndarray):
        return serialize(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: serialize(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def expansion_reference(previous: float, current: float, delta_seconds: float) -> dict[str, Any]:
    """Analytic constant-speed expansion fixture, with both physical endpoints."""
    if any(not math.isfinite(x) or x <= 0 for x in (previous, current, delta_seconds)):
        raise ValueError("Positive finite scales and physical delta are required")
    ratio = current / previous
    if ratio == 1:
        return {"status": "no_expansion", "current_ttc": None, "previous_ttc": None}
    current_ttc = delta_seconds / (ratio - 1)
    return {
        "status": "finite",
        "current_ttc": current_ttc,
        "previous_ttc": current_ttc + delta_seconds,
        "log_growth": math.log(ratio),
    }


def trace_query(
    runtime: CompetitionRuntime,
    events: Sequence[PreparedObservation],
    rgb: Sequence[PreparedObservation],
    query_us: int,
) -> dict[str, Any]:
    """Observe producers, accepted features, contexts and fusion without changing outputs.

    Single-threaded diagnostic use only: temporary hooks and an instance-local wrapper
    are always removed. Every producer call is retained, including rejected observations.
    """
    calls: dict[str, list[Any]] = {name: [] for name in runtime.models}
    accepted: dict[str, list[Any]] = {}
    original = runtime._prepare_values

    def prepare(observations: Sequence[PreparedObservation], prefix: str) -> Any:
        admitted, values, summary = original(observations, prefix)  # type: ignore[arg-type]
        modality = "event" if prefix == "E" else "rgb"
        mean, scale = (t.detach().cpu().numpy() for t in runtime.normalizers[modality])
        accepted[modality] = []
        for observation, (raw, experts) in zip(admitted, values, strict=True):
            accepted[modality].append(
                {
                    "anchor_us": observation.anchor_us,
                    "available_us": observation.available_us,
                    "first_sensor_ts_us": observation.first_sensor_ts_us,
                    "timestamps_us": list(observation.timestamps_us),
                    "delta_t_seconds": observation.delta_t_seconds.tolist(),
                    "input_sha256": hashlib.sha256(observation.pixels.tobytes()).hexdigest(),
                    "raw_features": serialize(raw),
                    "expert_phases": serialize(experts),
                    "standardized_features": serialize((raw - mean) / scale),
                    "normalizer_mean": serialize(mean),
                    "normalizer_scale": serialize(scale),
                }
            )
        return admitted, values, summary

    def hook_for(name: str) -> Any:
        def hook(module: torch.nn.Module, arguments: tuple[Any, ...], output: Any) -> None:
            del module
            inputs = serialize(arguments) if name == "PHASE_ONLY_EVENT_S7_TRAIN40" else None
            calls[name].append({"output": serialize(output), "final_head_inputs": inputs})

        return hook

    hooks = []
    try:
        for name, model in runtime.models.items():
            hooks.append(model.register_forward_hook(hook_for(name)))
        with patch.object(runtime, "_prepare_values", prepare):
            prediction = runtime.predict(events, rgb, query_us)
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "schema": "evttc_transfer_trace_v1",
        "query_us": query_us,
        "labels_read": False,
        "prediction": serialize(prediction),
        "accepted": accepted,
        "module_calls": calls,
        "limitations": ["Instrumentation timing is not production latency."],
    }
