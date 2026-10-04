"""Fixed WIDE selection over the canonical normalized H16 query source."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.simplex_t.cache import CachedQueries
from e_jepa_ttc.simplex_t.training import state_digest

WIDE_SLOTS = (0, 2, 4, 6, 9, 11, 13, 15)


@dataclass
class WideSource:
    """Retain original supervision, normalization, ages and current expert phases."""

    parent: CachedQueries

    def __post_init__(self) -> None:
        """Bind selection and gap semantics to the parent's complete identity."""
        self.identity_sha256 = state_digest(
            {
                "parent": self.parent.identity_sha256,
                "slots": WIDE_SLOTS,
                "gap": "selected_valid_anchors",
                "campaign": "EFFICIENT_CONTEXT_20261004",
            }
        )

    @property
    def population(self) -> int:
        """Preserve every query including cold starts."""
        return self.parent.population

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Select fixed slots and rebuild only gaps between selected valid observations."""
        x, t, mask, experts, truth, mass = self.parent.gather(query_ids)
        if x.shape[1:] != (16, 17):
            raise ValueError("WIDE requires canonical H16 PHASE17")
        slots = torch.tensor(WIDE_SLOTS)
        x, t, mask = x[:, slots].clone(), t[:, slots].clone(), mask[:, slots].clone()
        t[:, :, 2] = 0
        t[:, 1:, 2] = torch.where(mask[:, :-1] & mask[:, 1:], t[:, :-1, 0] - t[:, 1:, 0], 0)
        # Use parent int64 anchors when available to avoid subtracting rounded ages.
        idx = self.parent.history[query_ids.numpy()][:, WIDE_SLOTS]
        anchors = self.parent.anchor_us[np.maximum(idx, 0)]
        gaps = np.diff(anchors, axis=1).astype(np.float64) / 1e6
        t[:, 1:, 2] = torch.where(
            mask[:, :-1] & mask[:, 1:], torch.from_numpy(gaps.astype(np.float32)), 0
        )
        t[~mask] = 0
        return x, t, mask, experts, truth, mass
