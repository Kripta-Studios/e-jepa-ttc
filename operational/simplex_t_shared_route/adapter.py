"""Canonical frozen expert extraction with genuine reduced-producer dispatch."""

# Checked external Torch model payloads have dynamic diagnostics.
# ruff: noqa: ANN401
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc
from operational.simplex_t_cost_context.model import mask_features
from operational.simplex_t_post_campaign.route_policy import ALLOWED


@torch.inference_mode()
def extract(model: str, models: dict[str, Any], events: Tensor, delta: Tensor) -> np.ndarray:
    """Emit raw PHASE17 coordinates without calling an excluded producer.

    Retain canonical FP32 producer operations and float64 phase arithmetic before
    the final FP32 feature cast. PAIR consumes the sole A5 output. The declared
    current-ROI sensor mask is independent of these diagnostics or predictions.
    """
    allowed = ALLOWED[model]
    if events.dtype != torch.float32 or delta.dtype != torch.float32:
        raise ValueError("frozen FP32 inputs required")
    if events.shape[1:3] != (3, 12) or delta.shape != (len(events), 2):
        raise ValueError("frozen three-window producer shape required")
    if not bool((delta > 0).all()) or any(models[key].training for key in allowed):
        raise ValueError("positive producer deltas and frozen eval models required")
    values = np.zeros((len(events), 17), np.float64)
    values[:, 0] = events[:, -1, -2].mean((-2, -1)).cpu().numpy()
    values[:, 1] = events[:, -1, -1].mean((-2, -1)).cpu().numpy()
    a5_output = None
    for name, offset, phase_column in (("A5", 2, 8), ("C2F", 5, 9)):
        if name not in allowed:
            continue
        output = models[name](events, delta, return_dense_features=True)
        if name == "A5":
            a5_output = output
        ttc = output.ttc_mean_seconds.float().cpu().numpy()
        if not np.isfinite(ttc).all():
            raise ValueError("nonfinite permitted A5/C2F output")
        values[:, offset] = output.diagnostics["transport_flow_magnitude"][:, -1].float().cpu()
        values[:, offset + 1] = (
            torch.minimum(
                output.log_height_ratio[:, -1].abs() / 0.002,
                output.sensor_support[:, -1] / 0.0001,
            )
            .float()
            .cpu()
        )
        values[:, offset + 2] = output.ttc_log_variance.float().cpu()
        values[:, phase_column] = expert_phase_from_ttc(ttc)
    if "PAIR" in allowed:
        if a5_output is None:
            raise ValueError("PAIR requires its declared shared A5 ancestor")
        elapsed = delta[:, 1]
        support = a5_output.sensor_support
        pair_features = torch.cat(
            (
                a5_output.pair_tokens[:, -1].float(),
                torch.stack((elapsed, torch.log(elapsed + 1e-8), elapsed.reciprocal()), -1),
                support[:, -1:],
                torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
            ),
            -1,
        ).float()
        pair_ttc = models["PAIR"].predict_ttc(PairFeatureBatch(pair_features)).float().cpu().numpy()
        if np.isnan(pair_ttc).any():
            raise ValueError("NaN permitted PAIR prediction")
        values[:, 10] = expert_phase_from_ttc(pair_ttc)
    if {"A5", "PAIR"} <= allowed:
        values[:, 11] = values[:, 10] - values[:, 8]
        values[:, 14] = np.abs(values[:, 11])
    if {"C2F", "PAIR"} <= allowed:
        values[:, 12] = values[:, 10] - values[:, 9]
        values[:, 15] = np.abs(values[:, 12])
    if {"A5", "C2F"} <= allowed:
        values[:, 13] = values[:, 9] - values[:, 8]
        values[:, 16] = np.abs(values[:, 13])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite permitted features")
    return values.astype(np.float32)


def inputs(
    model: str,
    raw: np.ndarray,
    valid: np.ndarray,
    lag_us: np.ndarray,
    anchor_us: int,
    available_us: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply the original diagonal normalizer, then mask exclusions and padding."""
    length = 1 if model == "H1_SEED7" else 16 if model == "H16_SEED7" else 8
    mask = valid[-length:].copy()
    if not mask[-1] or (mask[:-1] & ~mask[1:]).any():
        raise ValueError("current and contiguous chronological sensor context required")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("verified historical normalizer required")
    x = ((raw[-length:] - mean) / scale).astype(np.float32)
    x[~mask] = 0
    anchors = anchor_us - lag_us[-length:]
    timing = np.zeros((length, 4), np.float32)
    timing[:, 0] = lag_us[-length:] / 1e6
    timing[:, 3] = (available_us - anchors) / 1e6
    timing[1:, 2] = np.where(mask[:-1], np.diff(anchors) / 1e6, 0)
    timing[~mask] = 0
    experts = raw[-1, 8:11].copy()
    allowed = ALLOWED[model]
    for i, name in enumerate(("A5", "C2F", "PAIR")):
        if name not in allowed:
            experts[i] = 0
    features = torch.from_numpy(x[None])
    if model.endswith("_C0"):
        features = mask_features(features, model)
    if model == "SET_NOTIME_C0":
        timing[:] = 0
    return (
        features,
        torch.from_numpy(timing[None]),
        torch.from_numpy(mask[None]),
        torch.from_numpy(
            experts[None],
        ),
    )
