"""Stage 62 model evaluation without labels entering the forward path."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.stage61_pair_feature_cache import LocalTemporalFieldBatch
from e_jepa_ttc.models.collision_clock_math import benchmark_phase_to_inverse_ttc
from e_jepa_ttc.models.local_temporal_phase_field import LocalTemporalPhaseField


@torch.no_grad()
def predict_local_field_ttc(
    model: LocalTemporalPhaseField,
    *,
    features: np.ndarray,
    valid: np.ndarray,
    a5_phase: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run frozen X2 inference and return unclipped signed scientific TTC."""

    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        batch = LocalTemporalFieldBatch(
            torch.as_tensor(features[start:stop], dtype=torch.float32, device=device),
            torch.as_tensor(valid[start:stop], dtype=torch.bool, device=device),
            torch.as_tensor(a5_phase[start:stop], dtype=torch.float32, device=device),
        )
        phase = model(batch).benchmark_phase
        outputs.append(
            torch.reciprocal(benchmark_phase_to_inverse_ttc(phase, metric_delta_t_s=0.1))
            .cpu()
            .numpy()
        )
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("X2 emitted non-finite TTC")
    return result


__all__ = ["predict_local_field_ttc"]
