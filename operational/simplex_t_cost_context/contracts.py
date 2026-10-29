"""Reduced frozen-producer interface; historical cached availability is retained.

This adapter demonstrates expert-call independence and exact shared A5 use.
It does not regenerate the inherited ROI/history index, certify raw availability,
or make a claim about deployment cost without measured raw-context execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .model import ALLOWED_EXPERTS, FEATURE_NAMES


@torch.inference_mode()
def extract_reduced_family(
    arm: str,
    producers: Mapping[str, nn.Module],
    events: Tensor,
    delta: Tensor,
) -> dict[str, np.ndarray]:
    """Call only allowed models; PAIR consumes the single already computed A5 output.

    Output features are raw PHASE17 with absent slots zero. Apply the verified
    diagonal historical normalizer, then mask absent slots again. Unused expert
    TTC slots are NaN to identify absence, never fabricated historical predictions.
    The caller supplies sealed allowed weights, ROI/windows and resource lease.
    """
    from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
    from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc

    if arm not in ALLOWED_EXPERTS:
        raise ValueError("registered cost/context arm required")
    allowed = ALLOWED_EXPERTS[arm]
    if set(producers) != set(allowed):
        raise ValueError("supply exactly allowed producers; excluded runtimes are unnecessary")
    if events.dtype != torch.float32 or delta.dtype != torch.float32:
        raise ValueError("coherent FP32 producer inputs required")
    if events.ndim != 5 or events.shape[1:3] != (3, 12):
        raise ValueError("expected three frozen 12-channel event windows")
    if delta.shape != (len(events), 2) or not bool((delta > 0).all()):
        raise ValueError("two positive producer deltas required")
    if any(model.training for model in producers.values()):
        raise ValueError("allowed frozen producers must be in eval mode")
    values: dict[str, np.ndarray] = {
        "shared_event_count_log1p": events[:, -1, -2].mean((-2, -1)).cpu().numpy(),
        "shared_event_rate_log1p": events[:, -1, -1].mean((-2, -1)).cpu().numpy(),
    }
    ttc = np.full((len(events), 3), np.nan, dtype=np.float32)
    known = np.zeros((len(events), 2), dtype=bool)
    a5_output: Any = None  # noqa: ANN401 -- external frozen model output
    for name, column in (("a5", 0), ("c2f", 1)):
        if name not in allowed:
            continue
        output = producers[name](events, delta, return_dense_features=True)
        prediction = output.ttc_mean_seconds.float().cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite A5/C2F prediction")
        ttc[:, column] = prediction
        known[:, column] = output.known_mask.cpu().numpy()
        values[f"{name}_flow"] = (
            output.diagnostics["transport_flow_magnitude"][:, -1].float().cpu().numpy()
        )
        values[f"{name}_margin"] = (
            torch.minimum(
                output.log_height_ratio[:, -1].abs() / 0.002,
                output.sensor_support[:, -1] / 0.0001,
            )
            .float()
            .cpu()
            .numpy()
        )
        values[f"{name}_log_variance"] = output.ttc_log_variance.float().cpu().numpy()
        values[f"{name}_benchmark_phase"] = expert_phase_from_ttc(prediction)
        if name == "a5":
            a5_output = output
    pair_features: np.ndarray | None = None
    if "pair" in allowed:
        if a5_output is None:
            raise ValueError("PAIR requires the already computed A5 output")
        elapsed = delta[:, 1]
        support = a5_output.sensor_support
        tokens = a5_output.pair_tokens[:, -1].float()
        inputs = torch.cat(
            (
                tokens,
                torch.stack((elapsed, torch.log(elapsed + 1e-8), elapsed.reciprocal()), -1),
                support[:, -1:],
                torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
            ),
            -1,
        ).float()
        native: list[Tensor] = []

        def capture(module: nn.Module, args: tuple[object, ...], phase: Tensor) -> None:
            del module, args
            native.append(phase)

        pair_model: Any = producers["pair"]  # noqa: ANN401 -- predict_ttc external API
        handle = pair_model.register_forward_hook(capture)
        try:
            prediction = pair_model.predict_ttc(PairFeatureBatch(inputs)).float().cpu().numpy()
        finally:
            handle.remove()
        if len(native) != 1 or not np.isfinite(native[0].cpu().numpy()).all():
            raise ValueError("PAIR native phase provenance missing/nonfinite")
        if (np.isinf(prediction) & (native[0].cpu().numpy() != 0)).any():
            raise ValueError("infinite PAIR prediction requires exact native zero phase")
        ttc[:, 2] = prediction
        values["pair_benchmark_phase"] = expert_phase_from_ttc(prediction)
        pair_features = inputs.cpu().numpy()
    for label, left, right in (
        ("pair_minus_a5_phase", "pair", "a5"),
        ("pair_minus_c2f_phase", "pair", "c2f"),
        ("c2f_minus_a5_phase", "c2f", "a5"),
    ):
        if left in allowed and right in allowed:
            difference = values[f"{left}_benchmark_phase"] - values[f"{right}_benchmark_phase"]
            values[label] = difference
            values[f"abs_{label}"] = np.abs(difference)
    features = np.stack(
        [values.get(name, np.zeros(len(events), dtype=np.float64)) for name in FEATURE_NAMES],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite permitted raw features; do not discard observations")
    result = {"features17": features, "expert_ttc": ttc, "known": known}
    if pair_features is not None:
        result["pair_features"] = pair_features
    return result
