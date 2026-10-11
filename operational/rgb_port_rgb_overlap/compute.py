"""Candidate: cache only immutable geometry in exact local correlation matching.

All differentiable operations retain their original order and dtypes. This is
not a CUDA-graph or compiled implementation. Deployment requires separate
full-model GPU parity admission; CPU tests alone do not provide that admission.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch
from torch.nn import functional

from e_jepa_ttc.models import causal_scale_ttc, local_transport
from e_jepa_ttc.models.local_transport import LocalTransportMatch, _candidate_displacements

MAX_MASK_BYTES = 64 * 1024**2
MAX_MASK_ENTRIES = 32
MaskKey = tuple[int, int, int, int, torch.device]


class CandidateValidityCache:
    """Bounded per-runtime bool masks with original contiguous B,K,H,W layout."""

    def __init__(
        self, *, max_bytes: int = MAX_MASK_BYTES, max_entries: int = MAX_MASK_ENTRIES
    ) -> None:
        if not 0 <= max_bytes <= MAX_MASK_BYTES or not 1 <= max_entries <= MAX_MASK_ENTRIES:
            raise ValueError("Geometry cache requires <=64 MiB and 1..32 entries")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries: OrderedDict[MaskKey, torch.Tensor] = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def mask(
        self, batch: int, height: int, width: int, radius: int, device: torch.device
    ) -> torch.Tensor:
        """Reuse exact stack output; build misses using the canonical operations."""
        key = (batch, height, width, radius, device)
        cached = self.entries.get(key)
        if cached is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        y_coords = torch.arange(height, device=device)
        x_coords = torch.arange(width, device=device)
        valid_maps: list[torch.Tensor] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                valid_y = (y_coords + dy >= 0) & (y_coords + dy < height)
                valid_x = (x_coords + dx >= 0) & (x_coords + dx < width)
                valid_2d = valid_y[:, None] & valid_x[None, :]
                valid_maps.append(valid_2d[None].expand(batch, -1, -1))
        mask = torch.stack(valid_maps, dim=1)
        size = mask.numel() * mask.element_size()
        if size <= self.max_bytes:
            while self.entries and (
                self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries
            ):
                _, stale = self.entries.popitem(last=False)
                self.bytes -= stale.numel() * stale.element_size()
            self.entries[key] = mask
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return mask

    def snapshot(self) -> dict[str, int]:
        """Return retained-byte and bounded-entry evidence without tensor copies."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries),
            "bytes": self.bytes,
            "peak_bytes": self.peak_bytes,
            "max_bytes": self.max_bytes,
            "max_entries": self.max_entries,
        }

    def clear(self) -> None:
        """Release references to reusable masks without mutating saved autograd inputs."""
        self.entries.clear()
        self.bytes = 0


def local_correlation_match_cached(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    return_probability: bool = False,
    cache: CandidateValidityCache,
) -> LocalTransportMatch:
    """Canonical local matching with only constant mask construction cached."""
    if previous.ndim != 4 or current.ndim != 4 or previous.shape != current.shape:
        raise ValueError("transport endpoints must share shape [B,C,H,W]")
    if radius < 1:
        raise ValueError("transport radius must be >= 1")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("transport temperature must be finite and positive")
    batch, _channels, height, width = previous.shape
    if min(height, width) <= 2 * radius:
        raise ValueError("transport radius is too large for the dense feature grid")

    previous_f = functional.normalize(previous.float(), dim=1, eps=1.0e-6)
    current_f = functional.normalize(current.float(), dim=1, eps=1.0e-6)
    padded_current = functional.pad(current_f, (radius, radius, radius, radius))
    correlations: list[torch.Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            shifted = padded_current[:, :, y0 : y0 + height, x0 : x0 + width]
            correlations.append((previous_f * shifted).sum(dim=1))
    correlation = torch.stack(correlations, dim=1)
    candidate_valid = cache.mask(batch, height, width, radius, current_f.device)
    negative_large = torch.finfo(correlation.dtype).min
    masked_correlation = correlation.masked_fill(~candidate_valid, negative_large)
    probability = torch.softmax(masked_correlation / float(temperature), dim=1)
    probability = probability * candidate_valid.to(probability.dtype)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    dx_values, dy_values = _candidate_displacements(
        radius, device=probability.device, dtype=probability.dtype
    )
    dx = (probability * dx_values[None, :, None, None]).sum(dim=1)
    dy = (probability * dy_values[None, :, None, None]).sum(dim=1)
    top2 = masked_correlation.topk(k=2, dim=1).values
    confidence_margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    valid_count = candidate_valid.sum(dim=1).clamp_min(2).to(probability.dtype)
    entropy = -(probability.clamp_min(1.0e-12).log() * probability).sum(dim=1)
    entropy = (entropy / valid_count.log()).clamp(0.0, 1.0)
    valid = candidate_valid.any(dim=1)
    return LocalTransportMatch(
        dx=dx,
        dy=dy,
        confidence_margin=confidence_margin,
        entropy=entropy,
        valid=valid,
        probability=probability if return_probability else None,
    )


@contextmanager
def installed_compute(
    *, max_bytes: int = MAX_MASK_BYTES, max_entries: int = MAX_MASK_ENTRIES
) -> Iterator[CandidateValidityCache]:
    """Temporarily patch both the defining module and CausalScaleTTC's direct alias."""
    original = local_transport.local_correlation_match
    if causal_scale_ttc.local_correlation_match is not original:
        raise ValueError("Transport alias already differs; refusing to bypass another runtime")
    cache = CandidateValidityCache(max_bytes=max_bytes, max_entries=max_entries)

    def match(*args: Any, **kwargs: Any) -> LocalTransportMatch:  # noqa: ANN401
        return local_correlation_match_cached(*args, **kwargs, cache=cache)

    try:
        with (
            patch.object(local_transport, "local_correlation_match", match),
            patch.object(causal_scale_ttc, "local_correlation_match", match),
        ):
            yield cache
    finally:
        cache.clear()
