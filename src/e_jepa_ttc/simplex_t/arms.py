"""Resolve canonical fit names into model and cached-source controls.

This binding does not certify cache lineage or authorize training. The campaign
caller must first verify freeze, stage gates, producers and resource permission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .cache import CachedQueries
from .model import TemporalConfig
from .registry import FitSpec, registered_graph
from .training import state_digest


@dataclass(frozen=True)
class ArmBinding:
    """Exact registered architecture and input transformation for one fit."""

    pool: str
    history: int
    model: TemporalConfig
    control: str = "NONE"
    zero_latent: bool = False

    def source(self, cache: CachedQueries) -> CachedQueries:
        """Bind a prevalidated pool; do not mutate shared normalizers or histories."""
        if cache.features.shape[1] != self.model.feature_count:
            raise ValueError("arm/cache feature lineage mismatch")
        if cache.control != "NONE" or cache.zero_latent:
            raise ValueError("arm source must be the unperturbed shared cache")
        identity = state_digest(
            {
                "cache": cache.identity_sha256,
                "pool": self.pool,
                "history": self.history,
                "control": self.control,
                "zero_latent": self.zero_latent,
            }
        )
        return replace(
            cache,
            length=self.history,
            control=self.control,
            zero_latent=self.zero_latent,
            identity_sha256=identity,
        )


def resolve_arm(spec: FitSpec, frozen_graph: list[FitSpec]) -> ArmBinding:
    """Reject renamed/invented arms and fits absent from the caller's frozen graph."""
    possible = []
    for d1 in (False, True):
        possible.extend(
            registered_graph(
                d1=d1,
                density=d1,
                t3=True,
                latent=True,
                replicate_scalar=True,
                replicate_latent=True,
            )
        )
    if spec not in possible or spec not in frozen_graph:
        raise ValueError("fit is not a registered member of the frozen graph")
    family, pool, history, capacity = spec.name.split("-")
    return ArmBinding(
        pool=pool,
        history=int(history[1:]),
        model=TemporalConfig(
            feature_count=145 if family in {"LATENT", "LATENT_ZERO"} else 17,
            hidden=int(capacity[1:]),
            backbone="transformer" if family == "TRANSFORMER" else "gru",
            output_mode={"FREE": "free", "SELECTOR": "selector"}.get(family, "residual"),
        ),
        control=family if family in {"PAST_REVERSED", "REPEAT_CURRENT"} else "NONE",
        zero_latent=family == "LATENT_ZERO",
    )
