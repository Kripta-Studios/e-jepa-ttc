"""One coherent FP32 frozen-family extraction route for current and past inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase

from .expert_phase import build_expert_features as build_router_features


@torch.inference_mode()
def extract_family(
    a5: CausalScaleTTC,
    c2f: CausalScaleTTC,
    pair: CachedPairDirectPhase,
    events: Tensor,
    delta: Tensor,
) -> dict[str, np.ndarray]:
    """Extract PHASE17 plus128 A5 tokens without supervision or any optimizer.

    Caller owns verified producer lineage, FP32 runtime, immutable batch layout
    and resource lease. Historical BF16 fields are never accepted as arguments.
    """
    if events.dtype != torch.float32 or delta.dtype != torch.float32:
        raise ValueError("coherent FP32 inputs required")
    if events.ndim != 5 or events.shape[1:3] != (3, 12):
        raise ValueError("three frozen12-channel event windows required")
    if delta.shape != (len(events), 2) or not bool((delta > 0).all()):
        raise ValueError("two positive producer time deltas required")
    if any(model.training for model in (a5, c2f, pair)):
        raise ValueError("frozen inference requires eval mode")
    frames = []
    predictions = []
    known = []
    tokens = None
    pair_features = None
    for name, model in (("a5", a5), ("c2f", c2f)):
        output = model(events, delta, return_dense_features=True)
        prediction = output.ttc_mean_seconds.float()
        predictions.append(prediction.cpu().numpy())
        known.append(output.known_mask.cpu().numpy())
        frame = {
            "token_id": np.arange(len(events)).astype(str),
            "prediction_ttc": prediction.cpu().numpy(),
            "shared_event_count_log1p": events[:, -1, -2].mean((-2, -1)).cpu().numpy(),
            "shared_event_rate_log1p": events[:, -1, -1].mean((-2, -1)).cpu().numpy(),
            f"{name}_flow": output.diagnostics["transport_flow_magnitude"][:, -1]
            .float()
            .cpu()
            .numpy(),
            f"{name}_margin": torch.minimum(
                output.log_height_ratio[:, -1].abs() / 0.002,
                output.sensor_support[:, -1] / 0.0001,
            )
            .float()
            .cpu()
            .numpy(),
            f"{name}_log_variance": output.ttc_log_variance.float().cpu().numpy(),
        }
        frames.append(pd.DataFrame(frame))
        if name == "a5":
            elapsed = delta[:, 1]
            support = output.sensor_support
            tokens = output.pair_tokens[:, -1].float()
            pair_features = torch.cat(
                (
                    tokens,
                    torch.stack((elapsed, torch.log(elapsed + 1e-8), elapsed.reciprocal()), -1),
                    support[:, -1:],
                    torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
                ),
                -1,
            ).float()
    assert tokens is not None and pair_features is not None
    native_phase: list[Tensor] = []

    def capture_phase(module: torch.nn.Module, inputs: tuple, output: Tensor) -> None:
        native_phase.append(output)

    handle = pair.register_forward_hook(capture_phase)
    try:
        pair_ttc = pair.predict_ttc(PairFeatureBatch(pair_features)).float().cpu().numpy()
    finally:
        handle.remove()
    if len(native_phase) != 1:
        raise ValueError("PAIR native phase provenance missing")
    phase = native_phase[0].float().cpu().numpy()
    if not np.isfinite(phase).all() or (np.isinf(pair_ttc) & (phase != 0)).any():
        raise ValueError("infinite PAIR TTC without an exact native zero phase")
    if not all(np.isfinite(value).all() for value in predictions):
        raise ValueError("nonfinite A5/C2F prediction")
    predictions.append(pair_ttc)
    _, features = build_router_features(frames[0], frames[1], pair_ttc)
    values = features.to_numpy(dtype=np.float32)
    latent = tokens.cpu().numpy()
    expert_ttc = np.stack(predictions, axis=1)
    if latent.shape != (len(events), 128) or values.shape != (len(events), 17):
        raise ValueError("frozen expert feature schema drift")
    if not all(np.isfinite(v).all() for v in (values, latent)) or np.isnan(expert_ttc).any():
        raise ValueError("nonfinite expert extraction; do not silently drop observations")
    return {
        "features145": np.concatenate((values, latent), axis=1),
        "expert_ttc": expert_ttc,
        "known": np.stack(known, axis=1),
        "pair_features": pair_features.cpu().numpy(),
    }
