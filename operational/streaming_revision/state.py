"""Causal feature reuse with explicit spatial, temporal and lifetime limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Policy:
    """Approximation parameters; zero tolerances require identical input geometry."""

    history: int = 8
    reuse: bool = True
    time_tolerance_us: int = 2500
    min_roi_iou: float = 0.65
    max_log_scale: float = 0.35
    max_age_us: int = 500000
    max_entries: int = 32

    def __post_init__(self) -> None:
        if self.history not in (2, 4, 8):
            raise ValueError("history must be 2, 4 or 8")
        if not 0 <= self.time_tolerance_us < 50000:
            raise ValueError("temporal tolerance must be in [0,50000)")
        if not 0 <= self.min_roi_iou <= 1 or not np.isfinite(self.max_log_scale):
            raise ValueError("invalid ROI thresholds")
        if self.max_log_scale < 0 or self.max_age_us < 0 or self.max_entries < 8:
            raise ValueError("invalid retention bounds")


@dataclass
class Observation:
    """An immutable-origin feature, never relabelled as a newly computed feature."""

    anchor: int
    windows: tuple[tuple[int, int], ...]
    roi: tuple[float, ...]
    delta: float
    available: int
    computed_at: int
    feature: Tensor


def roi_distance(old: tuple[float, ...], new: tuple[float, ...]) -> tuple[float, float]:
    """Return IoU and largest absolute log side-length change."""
    a, b = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        raise ValueError("finite xyxy ROIs required")
    size_a, size_b = a[2:] - a[:2], b[2:] - b[:2]
    if (size_a <= 0).any() or (size_b <= 0).any():
        raise ValueError("positive ROI area required")
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / (size_a.prod() + size_b.prod() - intersection)), float(
        np.abs(np.log(size_b / size_a)).max()
    )


class FeatureState:
    """One explicitly identified object stream; memory is bounded by max_entries."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.entries: list[Observation] = []
        self.identity: tuple[str, str] | None = None
        self.last_anchor: int | None = None

    def reset(self) -> None:
        """Release all retained features after identity changes or rollback."""
        self.entries.clear()
        self.last_anchor = None
        self.identity = None

    def begin(self, sequence: str, track: str, anchor: int) -> bool:
        """Reset on rollback, sequence or object change; report reset explicitly."""
        reset = self.identity != (sequence, track) or (
            self.last_anchor is not None and anchor < self.last_anchor
        )
        if reset:
            self.reset()
        self.identity, self.last_anchor = (sequence, track), anchor
        self.entries = [e for e in self.entries if anchor - e.anchor <= self.policy.max_age_us]
        return reset

    def find(
        self,
        *,
        anchor: int,
        now: int,
        windows: tuple[tuple[int, int], ...],
        roi: tuple[float, ...],
        delta: float,
        used: set[int],
    ) -> Observation | None:
        """Reuse only earlier observations; current observations are always refreshed."""
        if not self.policy.reuse or anchor == now:
            return None
        candidates = []
        for item in self.entries:
            if id(item) in used or not 0 <= anchor - item.anchor <= self.policy.time_tolerance_us:
                continue
            if item.available > now or now - item.computed_at > self.policy.max_age_us:
                continue
            if abs(delta - item.delta) > self.policy.time_tolerance_us * 1e-6:
                continue
            shift = anchor - item.anchor
            # Same translated window durations; tolerance permits sensor clock jitter.
            difference = np.asarray(windows) - np.asarray(item.windows) - shift
            if np.abs(difference).max() > self.policy.time_tolerance_us:
                continue
            iou, scale = roi_distance(item.roi, roi)
            if iou < self.policy.min_roi_iou or scale > self.policy.max_log_scale:
                continue
            candidates.append(item)
        return max(candidates, key=lambda e: (e.anchor, e.computed_at), default=None)

    def commit(self, observations: list[Observation]) -> None:
        """Publish successfully computed finite features without retaining graphs."""
        if any(
            e.feature.shape != (17,) or not bool(torch.isfinite(e.feature).all())
            for e in observations
        ):
            raise ValueError("finite PHASE17 observations required")
        for item in observations:
            item.feature = item.feature.detach().clone()
        self.entries.extend(observations)
        self.entries = self.entries[-self.policy.max_entries :]
