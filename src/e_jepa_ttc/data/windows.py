"""Causal temporal-window helpers backed by the canonical index contract."""

from __future__ import annotations

from dataclasses import dataclass

from e_jepa_ttc.data.eap import build_eap_temporal_windows
from e_jepa_ttc.data.index import build_temporal_index
from e_jepa_ttc.data.types import TemporalIndexEntry


@dataclass(frozen=True)
class WindowSpec:
    """Validated window parameters used by dense and event-budget samplers."""

    context_ms: int = 100
    stride_ms: int = 20
    horizons_ms: tuple[int, ...] = (25, 50, 100, 250, 500)

    def __post_init__(self) -> None:
        if min(self.context_ms, self.stride_ms) <= 0:
            raise ValueError("context_ms and stride_ms must be positive.")
        if not self.horizons_ms or min(self.horizons_ms) < 0:
            raise ValueError("horizons_ms must contain non-negative horizons.")


DEFAULT_WINDOW_SPEC = WindowSpec()


def build_dense_windows(
    manifest_path: str,
    *,
    spec: WindowSpec = DEFAULT_WINDOW_SPEC,
) -> list[TemporalIndexEntry]:
    """Build the canonical dense index using sequence-local TTC anchors."""

    return build_temporal_index(
        manifest_path=manifest_path,
        context_ms=spec.context_ms,
        stride_ms=spec.stride_ms,
        horizons_ms=spec.horizons_ms,
    )


__all__ = ["WindowSpec", "build_dense_windows", "build_eap_temporal_windows"]
