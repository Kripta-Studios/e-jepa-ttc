"""Bounded cache of constant local-transport masks; original differentiable arithmetic."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch.nn import functional

from e_jepa_ttc.models.local_transport import LocalTransportMatch, _candidate_displacements
from operational.oof_runtime_overlay import FoldPrefetchRuntime

_CACHE: OrderedDict[tuple[object, ...], tuple[tuple[torch.Tensor, ...], int]] = OrderedDict()
_LIMIT = 64 * 1024**2
_STATS = dict(hits=0, misses=0, bytes=0, evictions=0)


def _constants(
    batch: int, height: int, width: int, radius: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    key = (batch, height, width, radius, str(device), dtype)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        _STATS["hits"] += 1
        return _CACHE[key][0]
    _STATS["misses"] += 1
    y_coords = torch.arange(height, device=device)
    x_coords = torch.arange(width, device=device)
    valid_maps = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            valid_y = (y_coords + dy >= 0) & (y_coords + dy < height)
            valid_x = (x_coords + dx >= 0) & (x_coords + dx < width)
            valid_2d = valid_y[:, None] & valid_x[None, :]
            valid_maps.append(valid_2d[None].expand(batch, -1, -1))
    values = (
        torch.stack(valid_maps, dim=1),
        *_candidate_displacements(radius, device=device, dtype=dtype),
    )
    size = sum(t.numel() * t.element_size() for t in values)
    while _CACHE and (_STATS["bytes"] + size > _LIMIT or len(_CACHE) >= 16):
        _, (_, old_size) = _CACHE.popitem(last=False)
        _STATS["bytes"] -= old_size
        _STATS["evictions"] += 1
    if size <= _LIMIT:
        _CACHE[key] = (values, size)
        _STATS["bytes"] += size
    return values


def cached_local_correlation_match(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    return_probability: bool = False,
) -> LocalTransportMatch:
    """Match ``previous[p]`` against ``current[p + delta]`` within ``radius``.

    Features are L2-normalised and correlated in float32.  Invalid candidates
    outside the feature map are masked before softmax, so there is no wraparound.
    The expected displacement is differentiable with respect to both endpoints.
    """

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

    # Memory contract: never materialise [B,C,K,H,W].  At the production
    # 32x32 grid, r=4 and batch=32x2 temporal pairs, an unfold-based
    # implementation would transiently exceed a gigabyte even for the 64-channel
    # A4 encoder and scale linearly with capacity.  A single padded endpoint plus
    # shifted *views* keeps the differentiable state to O(BCHW + BKHW), avoiding
    # both the giant unfold and 81 autograd-retained BCHW copies.
    padded_current = functional.pad(current_f, (radius, radius, radius, radius))
    candidate_valid, dx_values, dy_values = _constants(
        batch, height, width, radius, current_f.device, current_f.dtype
    )
    correlations: list[torch.Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            shifted = padded_current[:, :, y0 : y0 + height, x0 : x0 + width]
            correlations.append((previous_f * shifted).sum(dim=1))

    correlation = torch.stack(correlations, dim=1)
    negative_large = torch.finfo(correlation.dtype).min
    masked_correlation = correlation.masked_fill(~candidate_valid, negative_large)
    probability = torch.softmax(masked_correlation / float(temperature), dim=1)
    probability = probability * candidate_valid.to(probability.dtype)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

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


class CachedTransportRuntime(FoldPrefetchRuntime):
    """Patch only the producer transport entrypoint and preserve the parent prefetch."""

    def __enter__(self) -> CachedTransportRuntime:
        from e_jepa_ttc.models import causal_scale_ttc

        if self._active:
            raise RuntimeError("Transport runtime already active")
        _CACHE.clear()
        _STATS.update(hits=0, misses=0, bytes=0, evictions=0)
        self._transport_module = causal_scale_ttc
        self._original_transport = causal_scale_ttc.local_correlation_match
        super().__enter__()
        causal_scale_ttc.local_correlation_match = cached_local_correlation_match
        return self

    def snapshot(self) -> dict[str, object]:
        result = super().snapshot()
        result["constant_transport_cache"] = {**_STATS, "entries": len(_CACHE), "max_bytes": _LIMIT}
        return result

    def close(self) -> None:
        try:
            super().close()
        finally:
            if hasattr(self, "_original_transport"):
                self._transport_module.local_correlation_match = self._original_transport
            _CACHE.clear()
            _STATS["bytes"] = 0
