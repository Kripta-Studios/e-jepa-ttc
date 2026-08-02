"""Typed adapter around the source-audited local Garl-TTC replica."""

from __future__ import annotations

from torch import nn

from e_jepa_ttc.data.garl_input_contract import GarlTTCModelInput
from e_jepa_ttc.models.garl_ttc_replica import GarlTTCOutput, GarlTTCReplica


class GarlReferenceAdapter(nn.Module):
    """Convert canonical endpoint inputs into the replica's source-like call."""

    def __init__(self, replica: GarlTTCReplica) -> None:
        super().__init__()
        self.replica = replica

    def forward(self, model_input: GarlTTCModelInput) -> GarlTTCOutput:
        model_input.validate()
        events = model_input.event_roi_endpoints.flatten(1, 2)
        rgb = model_input.rgb_endpoints
        return self.replica(events, model_input.delta_t_s, rgb_pair=rgb)


__all__ = ["GarlReferenceAdapter"]
