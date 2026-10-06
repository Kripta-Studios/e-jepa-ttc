"""Reuse one exact grad-enabled relation map for its same-loss no-grad diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ParamSpec, Protocol, TypeVar

import torch

from e_jepa_ttc.distillation import dinov3_relational as relational
from e_jepa_ttc.distillation.dinov3_relational import LocalRelationMaps

P = ParamSpec("P")
R = TypeVar("R")


class TrainingModule(Protocol):
    """Mutable training-module surface installed by the runtime wrapper."""

    _loss: Any


@dataclass
class ReuseCounters:
    """Bounded runtime evidence; counters never hold tensors."""

    loss_calls: int = 0
    original_map_calls: int = 0
    reuse_hits: int = 0
    reuse_misses: int = 0
    outside_scope_calls: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "loss_calls": self.loss_calls,
            "original_map_calls": self.original_map_calls,
            "reuse_hits": self.reuse_hits,
            "reuse_misses": self.reuse_misses,
            "outside_scope_calls": self.outside_scope_calls,
        }


@dataclass
class _LossScope:
    counters: ReuseCounters
    features: torch.Tensor | None = None
    offsets: tuple[tuple[int, int], ...] | None = None
    eps: float | None = None
    maps: LocalRelationMaps | None = None


_CURRENT: ContextVar[_LossScope | None] = ContextVar("train40_relation_reuse", default=None)


def _dispatcher(
    original: Callable[..., LocalRelationMaps], counters: ReuseCounters
) -> Callable[..., LocalRelationMaps]:
    def dispatch(
        features: torch.Tensor,
        *,
        offsets: tuple[tuple[int, int], ...] = relational.A4_RELATION_OFFSETS,
        eps: float = 1.0e-6,
    ) -> LocalRelationMaps:
        scope = _CURRENT.get()
        if scope is None:
            counters.outside_scope_calls += 1
            return original(features, offsets=offsets, eps=eps)
        if torch.is_grad_enabled() and scope.maps is None:
            result = original(features, offsets=offsets, eps=eps)
            scope.features = features
            scope.offsets = offsets
            scope.eps = eps
            scope.maps = result
            counters.original_map_calls += 1
            return result
        if (
            not torch.is_grad_enabled()
            and scope.maps is not None
            and features is scope.features
            and offsets == scope.offsets
            and eps == scope.eps
        ):
            counters.reuse_hits += 1
            return LocalRelationMaps(
                values=scope.maps.values.detach(),
                valid=scope.maps.valid.detach(),
            )
        counters.reuse_misses += 1
        counters.original_map_calls += 1
        return original(features, offsets=offsets, eps=eps)

    return dispatch


def _loss_wrapper(
    original: Callable[P, R], counters: ReuseCounters
) -> Callable[P, R]:
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        counters.loss_calls += 1
        token = _CURRENT.set(_LossScope(counters=counters))
        try:
            return original(*args, **kwargs)
        finally:
            _CURRENT.reset(token)

    return wrapped


@contextmanager
def scoped_relation_reuse(
    training: TrainingModule, counters: ReuseCounters | None = None
) -> Iterator[ReuseCounters]:
    """Patch only one process scope and restore both bindings under every exit path."""
    evidence = counters or ReuseCounters()
    original_maps = relational.local_cosine_relation_maps
    original_loss = training._loss
    relational.local_cosine_relation_maps = _dispatcher(original_maps, evidence)
    training._loss = _loss_wrapper(original_loss, evidence)
    try:
        yield evidence
    finally:
        training._loss = original_loss
        relational.local_cosine_relation_maps = original_maps


def context_is_clear() -> bool:
    """Expose tensor-free cleanup state for admission tests."""
    return _CURRENT.get() is None


__all__ = ["ReuseCounters", "TrainingModule", "context_is_clear", "scoped_relation_reuse"]
