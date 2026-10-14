"""Frozen P2 descriptors and native clocks for a prospective fusion feature schema.

These functions do not select parents or admit evaluation populations. The
scientific caller must freeze those choices before creating production caches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
from operational.range_transfer.common import digest

FEATURE_NAMES = (
    *(f"spatial_mean_{i:03d}" for i in range(96)),
    *(f"spatial_max_{i:03d}" for i in range(96)),
    "native_delta_t_s",
    "log_native_delta_t_s",
    "raw_phase",
    "point_phase",
)
SCHEMA_SHA256 = digest({"name": "v18_pair_spatial_phase196_v1", "columns": FEATURE_NAMES})


def clock_pairs(
    rows: pd.DataFrame, *, modality: str, delta_t_s: np.ndarray, available: np.ndarray
) -> np.ndarray:
    """Extract each modality's native last-two timestamps and verify its actual dt.

    Event pairs use producer_times_us, the registered event clock. RGB pairs use
    producer_sensor_times_us, the real image clock. Annotation and image times
    must never be substituted for each other because their intervals are close.
    """
    columns = {"event": "producer_times_us", "rgb": "producer_sensor_times_us"}
    count = len(rows)
    if (
        modality not in columns
        or count < 1
        or delta_t_s.shape != (count,)
        or delta_t_s.dtype != np.float32
        or available.shape != (count,)
        or available.dtype != np.bool_
    ):
        raise ValueError("Aligned FP32 intervals and boolean availability required")
    if modality == "event" and not available.all():
        raise ValueError("Event fallback clock must cover every query")
    pairs = np.zeros((count, 2), np.int64)
    for index in np.flatnonzero(available):
        times = np.asarray(rows.iloc[index][columns[modality]])
        if times.ndim != 1 or times.dtype != np.int64 or len(times) < 2:
            raise ValueError("Native pair requires at least two int64 sensor timestamps")
        previous, current = int(times[-2]), int(times[-1])
        if current <= previous:
            raise ValueError("Native pair clock is not chronological")
        measured = np.float32((current - previous) / 1_000_000)
        if not np.isfinite(measured) or measured != delta_t_s[index]:
            raise ValueError("Native sensor clock differs from the model's measured interval")
        pairs[index] = [previous, current]
    return pairs


@torch.inference_mode()
def extract_pair(
    model: nn.Module, inputs: torch.Tensor, delta_t_s: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 196 declared values and signed TTC; no GT/boxes/metadata enter forward."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Feature extraction requires a private frozen eval endpoint")
    if (
        inputs.ndim != 5
        or inputs.shape[1] != 2
        or inputs.shape[2] not in (3, 12)
        or inputs.shape[0] < 1
        or delta_t_s.shape != (inputs.shape[0],)
        or inputs.dtype != torch.float32
        or delta_t_s.dtype != torch.float32
        or inputs.device != delta_t_s.device
        or not torch.isfinite(inputs).all()
        or not torch.isfinite(delta_t_s).all()
        or not (delta_t_s > 0).all()
    ):
        raise ValueError("Expected finite FP32 P2 pixels and their own positive intervals")
    result = model(inputs, delta_t_s)
    descriptor, raw = result["descriptor"], result["raw_phase"]
    if (
        descriptor.shape != (len(inputs), 194)
        or raw.shape != (len(inputs),)
        or descriptor.dtype != torch.float32
        or raw.dtype != torch.float32
        or not torch.isfinite(descriptor).all()
        or not torch.isfinite(raw).all()
        or not torch.equal(descriptor[:, 192], delta_t_s)
        or not torch.equal(descriptor[:, 193], delta_t_s.log())
    ):
        raise ValueError("Endpoint output does not satisfy the registered pair descriptor schema")
    prediction = phase_to_ttc(raw)
    features = torch.cat((descriptor, raw[:, None], ttc_to_phase(prediction)[:, None]), 1)
    return features.cpu().numpy().copy(), prediction.cpu().numpy().copy()
