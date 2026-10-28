"""Experimental redundant-work removal for frozen local transport at inference time.

The original model and transport sources remain untouched. A shallow model view
binds the original forward code to private helper globals. Each forward owns a
fresh correlation/summary session; no tensor cache crosses queries or objects.

Default reverse-volume and summary reuse pass exact CPU fixtures. Optional
adjacent-pair deduplication changes batch shapes and failed complete-model CPU
bitwise equality. It is unadmitted, has strict expected-failure tests, and must
not be enabled in production before the unchanged final-prediction parity gate.
No CUDA or 554-query admission has been performed for this module.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional

from e_jepa_ttc.models import local_transport as oracle
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCOutput


def reverse_correlation_volume(forward: Tensor, radius: int) -> Tensor:
    """Derive ``C_reverse[d,p] = C_forward[-d,p+d]`` with zero padding.

    Multiplication is commutative and the original channel reduction is reused.
    This identity avoids a second normalization and channel-dot-product pass.
    """
    if forward.ndim != 4 or radius < 1 or forward.shape[1] != (2 * radius + 1) ** 2:
        raise ValueError("Expected [B,(2r+1)^2,H,W] correlation volume with r>=1")
    _, candidates, height, width = forward.shape
    padded = functional.pad(forward, (radius, radius, radius, radius))
    values = []
    for index, (dy, dx) in enumerate(
        (dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
    ):
        values.append(
            padded[
                :,
                candidates - 1 - index,
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
        )
    return torch.stack(values, dim=1)


def _correlation_volume(previous: Tensor, current: Tensor, radius: int) -> tuple[Tensor, Tensor]:
    batch, _, height, width = previous.shape
    previous_f = functional.normalize(previous.float(), dim=1, eps=1.0e-6)
    current_f = functional.normalize(current.float(), dim=1, eps=1.0e-6)
    padded = functional.pad(current_f, (radius, radius, radius, radius))
    ys = torch.arange(height, device=previous.device)
    xs = torch.arange(width, device=previous.device)
    correlations, validity = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[
                :, :, radius + dy : radius + dy + height, radius + dx : radius + dx + width
            ]
            correlations.append((previous_f * shifted).sum(dim=1))
            valid_y = (ys + dy >= 0) & (ys + dy < height)
            valid_x = (xs + dx >= 0) & (xs + dx < width)
            validity.append((valid_y[:, None] & valid_x[None, :])[None].expand(batch, -1, -1))
    return torch.stack(correlations, dim=1), torch.stack(validity, dim=1)


def _match_from_volume(
    correlation: Tensor, valid: Tensor, radius: int, temperature: float
) -> oracle.LocalTransportMatch:
    masked = correlation.masked_fill(~valid, torch.finfo(correlation.dtype).min)
    probability = torch.softmax(masked / float(temperature), dim=1)
    probability = probability * valid.to(probability.dtype)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    offsets = torch.arange(-radius, radius + 1, device=probability.device, dtype=probability.dtype)
    dy_values, dx_values = torch.meshgrid(offsets, offsets, indexing="ij")
    dx = (probability * dx_values.reshape(-1)[None, :, None, None]).sum(dim=1)
    dy = (probability * dy_values.reshape(-1)[None, :, None, None]).sum(dim=1)
    top2 = masked.topk(k=2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    count = valid.sum(dim=1).clamp_min(2).to(probability.dtype)
    entropy = -(probability.clamp_min(1.0e-12).log() * probability).sum(dim=1)
    entropy = (entropy / count.log()).clamp(0.0, 1.0)
    return oracle.LocalTransportMatch(dx, dy, margin, entropy, valid.any(dim=1), probability)


def correlation_pair(
    previous: Tensor, current: Tensor, *, radius: int, temperature: float
) -> tuple[oracle.LocalTransportMatch, oracle.LocalTransportMatch]:
    """Compute both directions while preserving the original per-channel sums."""
    if previous.ndim != 4 or current.shape != previous.shape:
        raise ValueError("Transport endpoints must share shape [B,C,H,W]")
    if radius < 1 or min(previous.shape[-2:]) <= 2 * radius:
        raise ValueError("Transport radius is invalid for this feature grid")
    if not 0.0 < temperature < float("inf"):
        raise ValueError("Transport temperature must be finite and positive")
    correlation, valid = _correlation_volume(previous, current, radius)
    reverse = reverse_correlation_volume(correlation, radius)
    return (
        _match_from_volume(correlation, valid, radius, temperature),
        _match_from_volume(reverse, valid, radius, temperature),
    )


def adjacent_pair_indices(
    objects: int, history_group_size: int, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Select H+1 unique pairs and restore 2H triplet pairs independently per object."""
    if objects < 1 or history_group_size < 1:
        raise ValueError("Positive object and history counts are required")
    local_select = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.arange(1, 2 * history_group_size, 2, device=device),
        )
    )
    groups = torch.arange(objects, device=device)
    select = (groups[:, None] * (2 * history_group_size) + local_select).flatten()
    local_restore = (
        torch.arange(history_group_size, device=device)[:, None]
        + torch.arange(2, device=device)[None, :]
    ).flatten()
    restore = (groups[:, None] * (history_group_size + 1) + local_restore).flatten()
    return select, restore


def _select_match(match: oracle.LocalTransportMatch, indices: Tensor) -> oracle.LocalTransportMatch:
    return oracle.LocalTransportMatch(
        match.dx.index_select(0, indices),
        match.dy.index_select(0, indices),
        match.confidence_margin.index_select(0, indices),
        match.entropy.index_select(0, indices),
        match.valid.index_select(0, indices),
        match.probability.index_select(0, indices) if match.probability is not None else None,
    )


def _with_probability(
    match: oracle.LocalTransportMatch, requested: bool
) -> oracle.LocalTransportMatch:
    if requested:
        return match
    return oracle.LocalTransportMatch(
        match.dx, match.dy, match.confidence_margin, match.entropy, match.valid
    )


class CorrelationSession:
    """Consume ordered forward/reverse calls within one original model forward.

    With ``history_group_size``, the caller must validate exact overlapping raw
    triplets inside each object, as the producer runtime's GPU guard does. The
    cache reuses only dense correlation fields, never smoothed foreground data.
    Deduplicated batch shapes are experimental and do not promise bitwise parity.
    """

    def __init__(self, *, history_group_size: int | None = None) -> None:
        if history_group_size is not None and history_group_size < 1:
            raise ValueError("history_group_size must be positive")
        self.history_group_size = history_group_size
        self.pending: tuple[Tensor, Tensor, int, float, oracle.LocalTransportMatch] | None = None

    def __call__(
        self,
        previous: Tensor,
        current: Tensor,
        *,
        radius: int,
        temperature: float,
        return_probability: bool = False,
    ) -> oracle.LocalTransportMatch:
        """Return one direction; reject a mismatched second call instead of stale reuse."""
        if self.pending is not None:
            first, second, old_radius, old_temperature, reverse = self.pending
            self.pending = None
            if (
                previous is not second
                or current is not first
                or radius != old_radius
                or temperature != old_temperature
            ):
                raise ValueError("Correlation session requires immediately reversed endpoint calls")
            return _with_probability(reverse, return_probability)
        a, b = previous, current
        restore = None
        if self.history_group_size is not None:
            group_pairs = 2 * self.history_group_size
            if len(previous) % group_pairs:
                raise ValueError("Pair batch does not contain complete object histories")
            select, restore = adjacent_pair_indices(
                len(previous) // group_pairs, self.history_group_size, device=previous.device
            )
            a, b = previous.index_select(0, select), current.index_select(0, select)
        forward, reverse = correlation_pair(a, b, radius=radius, temperature=temperature)
        if restore is not None:
            forward, reverse = _select_match(forward, restore), _select_match(reverse, restore)
        self.pending = previous, current, radius, temperature, reverse
        return _with_probability(forward, return_probability)


def _private_function(
    function: Callable[..., Any], replacements: dict[str, object]
) -> Callable[..., Any]:
    original = cast(FunctionType, function)
    result = FunctionType(
        original.__code__,
        {**original.__globals__, **replacements},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result.__kwdefaults__ = original.__kwdefaults__
    return result


class _SummarySession:
    """Reuse identical denominators and means without changing arithmetic order."""

    def __init__(self) -> None:
        # Strong references prevent Python id reuse while intermediates are live.
        self.denominators: dict[int, tuple[Tensor, Tensor]] = {}
        self.means: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]] = {}

    def mean(self, values: Tensor, weights: Tensor) -> Tensor:
        key = id(values), id(weights)
        if key in self.means:
            return self.means[key][2]
        weight_key = id(weights)
        if weight_key not in self.denominators:
            denominator = weights.sum(dim=(-2, -1)).clamp_min(1.0e-6)
            self.denominators[weight_key] = weights, denominator
        numerator = (values * weights).sum(dim=(-2, -1))
        result = numerator / self.denominators[weight_key][1]
        self.means[key] = values, weights, result
        return result


def reused_transport_physical_features(
    forward: oracle.LocalTransportMatch,
    reverse: oracle.LocalTransportMatch,
    *,
    foreground_weight: Tensor | None,
    radius: int,
    deterministic_sampling: bool = False,
    reuse_reverse_sample: bool = False,
) -> Tensor:
    """Apply original summaries with reused scalar reductions and reverse sampling."""
    del reuse_reverse_sample  # Exact inference reuse is always enabled in this opt-in helper.
    session = _SummarySession()
    slope = _private_function(oracle._weighted_slope, {"_weighted_mean": session.mean})
    summarise = _private_function(
        oracle._summarise_one, {"_weighted_mean": session.mean, "_weighted_slope": slope}
    )
    physical = _private_function(oracle.transport_physical_features, {"_summarise_one": summarise})
    return cast(
        Tensor,
        physical(
            forward,
            reverse,
            foreground_weight=foreground_weight,
            radius=radius,
            deterministic_sampling=deterministic_sampling,
            reuse_reverse_sample=True,
        ),
    )


@dataclass(frozen=True)
class TransportReuseOptions:
    """Explicit inference-only choices; adjacent-pair reuse needs an external guard."""

    history_group_size: int | None = None
    summaries: bool = True
    externally_guarded_overlap: bool = False


def attach_transport_reuse(
    model: CausalScaleTTC,
    *,
    history_group_size: int | None = None,
    summaries: bool = True,
    externally_guarded_overlap: bool = False,
) -> CausalScaleTTC:
    """Return an inference-only view sharing every original parameter unchanged.

    This composes after endpoint-encoder reuse. When pair deduplication is
    requested, the caller must reject nonoverlapping triplets before publishing
    outputs. The ordinary producer runtime already provides that deferred guard.
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Transport reuse accepts only frozen eval models")
    if history_group_size is not None and not externally_guarded_overlap:
        raise ValueError("Adjacent-pair reuse requires an explicit external overlap guard")
    if history_group_size is not None and history_group_size < 1:
        raise ValueError("history_group_size must be positive")
    if hasattr(model, "_train40_transport_options"):
        raise ValueError("Transport reuse is already attached")
    original_impl = model._forward_impl.__func__
    view = copy(model)
    view._modules = model._modules.copy()
    options = TransportReuseOptions(history_group_size, summaries, externally_guarded_overlap)

    def runtime_forward(
        self: CausalScaleTTC,
        inputs: Tensor,
        delta_t_s: Tensor,
        *,
        return_dense_features: bool = False,
        replay_control: object = None,
    ) -> CausalScaleTTCOutput:
        if options.history_group_size is not None and inputs.shape[1] != 3:
            raise ValueError("Adjacent-pair reuse requires native three-endpoint observations")
        matcher = CorrelationSession(history_group_size=options.history_group_size)
        replacements: dict[str, object] = {"local_correlation_match": matcher}
        if options.summaries:
            replacements["transport_physical_features"] = reused_transport_physical_features
        implementation = _private_function(original_impl, replacements)
        result = implementation(
            self,
            inputs,
            delta_t_s,
            return_dense_features=return_dense_features,
            replay_control=replay_control,
        )
        if matcher.pending is not None:
            raise RuntimeError("Original model left an unmatched transport direction")
        return cast(CausalScaleTTCOutput, result)

    object.__setattr__(view, "_forward_impl", MethodType(runtime_forward, view))
    object.__setattr__(view, "_train40_transport_options", options)
    return view
