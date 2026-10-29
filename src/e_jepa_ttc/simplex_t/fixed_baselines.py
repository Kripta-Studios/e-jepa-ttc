"""Label-free current-median and fixed H8 EWMA predictions from cached inputs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .cache import CachedQueries
from .controls import ewma_phase
from .phase import emitted_phase, phase_to_ttc


def predict_fixed_baselines(
    source: CachedQueries,
    *,
    expected_source_sha256: str,
    validate_prerequisites: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict[str, dict[str, np.ndarray]]:
    """Consume only raw expert phases, input times and H8 availability.

    No gather call, target, training mass, normalizer, fitted head or raw expert
    forward is used. The caller binds OLD_DEV role/time/producer lineage. A pause
    raises without returning incomplete query predictions; replay is cached CPU
    arithmetic only. All constants are registered, not chosen from scores.
    """
    validate_prerequisites()
    if source.identity_sha256 != expected_source_sha256:
        raise ValueError("fixed baseline source differs from frozen identity")
    if source.length != 8 or source.control != "NONE" or source.zero_latent:
        raise ValueError("fixed baselines require unperturbed H8 input")
    if source.population == 0:
        raise ValueError("nonempty baseline population required")
    points: dict[str, list[np.ndarray]] = {"CURRENT_MEDIAN": [], "EWMA_0P3S_H8": []}
    with torch.inference_mode():
        for start in range(0, source.population, 128):
            validate_prerequisites()
            if not resource_ok():
                raise InterruptedError("fixed baseline resource pause")
            index = source.history[start : start + 128, -8:]
            valid = index >= 0
            if (
                (index < -1).any()
                or (index >= len(source.features)).any()
                or not valid[:, -1].all()
                or (valid[:, :-1] & ~valid[:, 1:]).any()
            ):
                raise ValueError("invalid fixed baseline history index")
            safe = np.maximum(index, 0)
            current = safe[:, -1]
            anchors = source.anchor_us[safe]
            availability = source.available_us[safe]
            if (
                anchors[valid]
                > np.broadcast_to(source.anchor_us[current, None], index.shape)[valid]
            ).any() or (
                availability[valid]
                > np.broadcast_to(source.available_us[current, None], index.shape)[valid]
            ).any():
                raise ValueError("future dependency in fixed baseline context")
            lag = ((source.anchor_us[current, None] - anchors) / 1e6).astype(np.float32)
            phases = np.asarray(source.features[safe, 8:11], dtype=np.float32).copy()
            phases[~valid] = 0
            phase_tensor = torch.from_numpy(phases)
            median = emitted_phase(phase_tensor[:, -1].median(-1).values)
            smooth = ewma_phase(phase_tensor, torch.from_numpy(lag), torch.from_numpy(valid))
            points["CURRENT_MEDIAN"].append(median.numpy())
            points["EWMA_0P3S_H8"].append(smooth.numpy())
    validate_prerequisites()
    result = {}
    for name, chunks in points.items():
        phase = np.concatenate(chunks)
        result[name] = {
            "prediction_phase": phase,
            # Match prediction_frame's final TTC emission from saved FP32 phase.
            "prediction_ttc_s": phase_to_ttc(torch.from_numpy(phase.astype(np.float64))).numpy(),
        }
    return result
