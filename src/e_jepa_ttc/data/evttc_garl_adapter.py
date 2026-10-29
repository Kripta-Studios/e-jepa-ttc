"""Strict EvTTC-to-Garl input adapter for predict/score separation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from e_jepa_ttc.data.garl_input_contract import GarlTTCModelInput

FORBIDDEN_PREDICT_KEYS = frozenset(
    {"ttc", "ttc_s", "frame_ttc", "target_ttc", "gt_ttc", "future_labels"}
)


@dataclass(frozen=True)
class EvTTCGarlAdapterConfig:
    """Explicit protocol metadata used while adapting an EvTTC sample."""

    protocol_id: str = "evttc_garl_p0_zero_shot_v1"
    input_size: tuple[int, int] = (128, 128)
    endpoint_count: int = 2
    event_planes: int = 20


def reject_labels_from_predict_payload(payload: Mapping[str, object]) -> None:
    """Fail if a predict-stage payload contains TTC or future-label fields."""

    leaked = sorted(FORBIDDEN_PREDICT_KEYS.intersection(payload))
    if leaked:
        raise ValueError(f"EvTTC predict payload contains forbidden labels: {leaked}")


def make_garl_model_input(
    event_roi_endpoints: torch.Tensor,
    endpoint_timestamps_us: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor | None = None,
    full_event_context: torch.Tensor | None = None,
    rgb_endpoints: torch.Tensor | None = None,
    input_valid: torch.Tensor | None = None,
    config: EvTTCGarlAdapterConfig | None = None,
) -> GarlTTCModelInput:
    """Construct and validate model inputs; labels cannot enter this function."""

    selected = config or EvTTCGarlAdapterConfig()
    valid = (
        input_valid
        if input_valid is not None
        else torch.ones(
            event_roi_endpoints.shape[0], dtype=torch.bool, device=event_roi_endpoints.device
        )
    )
    model_input = GarlTTCModelInput(
        event_roi_endpoints=event_roi_endpoints,
        endpoint_timestamps_us=endpoint_timestamps_us,
        delta_t_s=delta_t_s,
        boxes_xyxy=boxes_xyxy,
        full_event_context=full_event_context,
        rgb_endpoints=rgb_endpoints,
        input_valid=valid,
        protocol_id=selected.protocol_id,
    )
    model_input.validate()
    return model_input


__all__ = [
    "EvTTCGarlAdapterConfig",
    "FORBIDDEN_PREDICT_KEYS",
    "make_garl_model_input",
    "reject_labels_from_predict_payload",
]
