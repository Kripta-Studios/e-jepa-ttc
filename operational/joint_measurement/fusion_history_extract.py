"""Label-free RGB history descriptors for a separately admitted R_CTX anchor.

The historical endpoint and normalizer must be admitted by the caller. This
adapter does not select parents or populations, read labels, or fit statistics.
R_CTX has H-supervised ancestry: a further head on H is sequential fitting.
"""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import digest

FEATURE_NAMES = (
    *(f"current_normalized_rgb_{i:02d}" for i in range(17)),
    *(f"current_native_timing_{i}" for i in range(4)),
    *(f"current_rgb_expert_phase_{i}" for i in range(3)),
    "anchor_raw_phase",
    "anchor_point_phase",
    "anchor_q10_phase",
    "anchor_q90_phase",
)
SCHEMA_SHA256 = digest({"name": "v18_rgb_history_anchor28_v1", "columns": FEATURE_NAMES})


@torch.inference_mode()
def extract_history(
    model: TemporalRefiner,
    features: torch.Tensor,
    timing: torch.Tensor,
    valid: torch.Tensor,
    experts: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the unchanged native H8 head; preserve signed FP32 anchor TTC.

    Only available RGB rows may enter. Missing rows are handled by the cache
    writer's explicit availability mask and the fusion model's event fallback.
    The descriptor retains current normalized features, native timing and expert
    phases, plus four outputs from the complete historical context. It is neither
    a new visual encoder nor an independent H prediction.
    """
    expected = TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
    if (
        not isinstance(model, TemporalRefiner)
        or model.cfg != expected
        or any(module.training for module in model.modules())
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise ValueError("RGB history extraction requires a private frozen native R_CTX head")
    if (
        features.ndim != 3
        or features.shape[0] < 1
        or not 1 <= features.shape[1] <= 8
        or features.shape[2] != 17
        or timing.shape != (*features.shape[:2], 4)
        or valid.shape != features.shape[:2]
        or experts.shape != (features.shape[0], 3)
        or valid.dtype != torch.bool
        or any(value.dtype != torch.float32 for value in (features, timing, experts))
        or any(value.device != features.device for value in (timing, valid, experts))
        or any(value.device != features.device for value in model.state_dict().values())
    ):
        raise ValueError("RGB history requires aligned native FP32 features, clocks and masks")
    # Native forward checks finite valid rows and the contiguous valid suffix.
    result = model(features, timing, valid, experts)
    phase = [result[key] for key in ("raw_location", "point_phase", "q10", "q90")]
    descriptor = torch.cat((features[:, -1], timing[:, -1], experts, torch.stack(phase, 1)), 1)
    prediction = phase_to_ttc(result["point_phase"])
    if (
        descriptor.shape != (len(features), len(FEATURE_NAMES))
        or not torch.isfinite(descriptor).all()
        or not torch.isfinite(prediction).all()
        or (prediction == 0).any()
    ):
        raise ValueError("RGB history produced invalid features or signed TTC")
    return descriptor.cpu().numpy().copy(), prediction.cpu().numpy().copy()
