"""Replay frozen context/head computation from owned, unnormalized observations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from operational.final_refit.test12 import FinalV16Train40Predictor
from scripts.final_train40_transfer import HEAD


def native_from_observations(
    models: dict,
    normalizers: dict,
    observations: dict[str, list[dict[str, Any]]],
    query_us: int,
    device: str | torch.device,
    *,
    keep: int | None = None,
) -> tuple[FinalV16Train40Predictor, dict[str, Any]]:
    """Keep final observations and let the original context builder recompute clocks.

    This resets context only: retained producer observations still include their
    original first_sensor_ts_us and must not be described as a sensor cold start.
    """
    if keep is not None and keep < 1:
        raise ValueError("Keep must be positive or None")
    native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
    native.device = torch.device(device)
    native.models, native.normalizers = models, normalizers
    native.inputs = {"event_observations": {}, "rgb_observations": {}}
    native.event_values, native.rgb_values = {}, {}
    query: dict[str, Any] = {"query_time_us": int(query_us)}
    for modality in ("event", "rgb"):
        retained = observations[modality] if keep is None else observations[modality][-keep:]
        anchors = [int(item["anchor_us"]) for item in retained]
        if len(retained) > 8 or any(b <= a for a, b in zip(anchors, anchors[1:], strict=False)):
            raise ValueError("Observation anchors must increase, with at most eight retained")
        query[f"{modality}_history"] = []
        values = native.event_values if modality == "event" else native.rgb_values
        for i, item in enumerate(retained):
            identity = f"{modality}:{i}"
            if not item["anchor_us"] <= item["available_us"] <= query_us:
                raise ValueError("Observation is not causally available")
            raw = np.asarray(item["raw17"], np.float32)
            experts = np.asarray(item["expert3"], np.float32)
            if raw.shape != (17,) or experts.shape != (3,):
                raise ValueError("Expected raw17 and expert3 per observation")
            query[f"{modality}_history"].append(identity)
            native.inputs[f"{modality}_observations"][identity] = {
                "anchor_us": int(item["anchor_us"]),
                "available_us": int(item["available_us"]),
            }
            values[identity] = raw.copy(), experts.copy()
    return native, query


@contextmanager
def capture_context(models: dict) -> Iterator[dict[str, np.ndarray]]:
    """Observe context inputs, all tensor outputs and the actual 340-dimensional head input."""
    arrays: dict[str, np.ndarray] = {}
    handles = []

    def make_hook(name: str) -> Callable:
        def hook(_module: torch.nn.Module, args: tuple, output: dict) -> None:
            for i, tensor in enumerate(args):
                if isinstance(tensor, torch.Tensor):
                    arrays[f"{name}.input_{i}"] = tensor.detach().cpu().numpy().copy()
            for key, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    arrays[f"{name}.output_{key}"] = tensor.detach().cpu().numpy().copy()

        return hook

    for name in ("E_CTX_MATCHED", "R_CTX", HEAD):
        handles.append(models[name].register_forward_hook(make_hook(name)))
    try:
        yield arrays
    finally:
        for handle in handles:
            handle.remove()


def replay(
    models: dict,
    normalizers: dict,
    observations: dict,
    query_us: int,
    device: str,
    *,
    keep: int | None = None,
) -> tuple[float | None, dict[str, np.ndarray]]:
    """Replay through the original context and phase340/head pipeline."""
    native, query = native_from_observations(
        models, normalizers, observations, query_us, device, keep=keep
    )
    if not query["event_history"]:
        return None, {}
    with torch.inference_mode(), capture_context(models) as arrays:
        prediction = native.predict(query, native.inputs)
    return prediction, arrays
