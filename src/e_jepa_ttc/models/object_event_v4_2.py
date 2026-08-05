"""Full event-only TTC screen built from the validated v4.1 encoder.

The v4.1 diagnostic showed that the corrected common-coordinate event tensor can
encode signed expansion without boxes or observable motion.  v4.2 deliberately
keeps that scientific contract and removes the weak activity branch from the
primary prediction.  Only the spatial-temporal encoded branch is trainable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventTTCV41, ObjectEventV41Config


@dataclass
class ObjectEventV42Output:
    expansion: torch.Tensor
    reverse_expansion: torch.Tensor
    raw_score: torch.Tensor
    reverse_raw_score: torch.Tensor
    reversal_consistency_error: torch.Tensor
    endpoint_embeddings: torch.Tensor
    spatial_embeddings: torch.Tensor


class ObjectEventTTCV42(ObjectEventTTCV41):
    """Encoded-only event model for the full sequence-held-out screen.

    ``activity_head`` and ``branch_scale`` are retained only to preserve a simple
    inheritance path from v4.1, but are frozen and never used in the forward
    result.  This makes the event contribution auditable and avoids the globally
    pooled activity shortcut that failed to generalise in v4.1.
    """

    def __init__(self, config: ObjectEventV41Config | None = None) -> None:
        super().__init__(config)
        self.activity_head.requires_grad_(False)
        self.branch_scale.requires_grad_(False)

    def forward(self, events: torch.Tensor) -> ObjectEventV42Output:
        _, encoded_raw, _, endpoints, spatial = self._forward_one_order(events)
        _, reverse_encoded_raw, _, _, _ = self._forward_one_order(
            torch.flip(events, dims=(1,))
        )
        maximum = self.config.max_abs_expansion
        expansion = maximum * torch.tanh(encoded_raw)
        reverse_expansion = maximum * torch.tanh(reverse_encoded_raw)
        return ObjectEventV42Output(
            expansion=expansion,
            reverse_expansion=reverse_expansion,
            raw_score=encoded_raw,
            reverse_raw_score=reverse_encoded_raw,
            reversal_consistency_error=(expansion + reverse_expansion).abs(),
            endpoint_embeddings=endpoints,
            spatial_embeddings=spatial,
        )


__all__ = ["ObjectEventTTCV42", "ObjectEventV42Output", "ObjectEventV41Config"]
