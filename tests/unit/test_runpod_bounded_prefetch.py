"""Runpod-only byte bounds layered above frozen OOF runtime modules."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from operational.runpod_migration.bounded_prefetch import (
    ByteBoundedFoldPrefetch,
    SharedByteBudget,
)
from operational.runpod_migration.runtime_prefetch import ByteBoundedFoldPrefetchRuntime


@dataclass(frozen=True)
class _Source:
    population_size: int = 8


@dataclass(frozen=True)
class _Recipe:
    effective_batch_size: int = 3


def test_two_batches_never_exceed_shared_reservation() -> None:
    cursor = {"position": 0, "order": list(range(8))}
    budget = SharedByteBudget(64)

    def build(source, ids, recipe):
        del source, recipe
        return np.zeros((len(ids), 4), dtype=np.uint8)

    with ByteBoundedFoldPrefetch(
        _Source(),
        _Recipe(),
        cursor,
        build,
        depth=2,
        byte_budget=budget,
        reservation_bytes_per_batch=24,
    ) as prefetch:
        assert budget.snapshot()["peak_reserved_bytes"] == 48
        assert prefetch.get((0, 1, 2)).shape == (3, 4)
        cursor["position"] = 3
        assert budget.snapshot()["reserved_bytes"] <= 48
    assert budget.snapshot()["reserved_bytes"] == 0


def test_actual_batch_larger_than_reservation_fails_and_releases() -> None:
    cursor = {"position": 0, "order": list(range(8))}
    budget = SharedByteBudget(32)

    def oversized(source, ids, recipe):
        del source, ids, recipe
        return np.zeros(17, dtype=np.uint8)

    with ByteBoundedFoldPrefetch(
        _Source(),
        _Recipe(),
        cursor,
        oversized,
        depth=1,
        byte_budget=budget,
        reservation_bytes_per_batch=16,
    ) as prefetch:
        with pytest.raises(MemoryError):
            prefetch.get((0, 1, 2))
    assert budget.snapshot()["reserved_bytes"] == 0


def test_runtime_rejects_combined_bound_before_patching_module() -> None:
    module = SimpleNamespace()
    with pytest.raises(ValueError, match="combined"):
        ByteBoundedFoldPrefetchRuntime(
            module,
            prefetch_depth=2,
            shared_limit_bytes=100,
            reservation_bytes_per_batch=40,
            decoder_max_pending_bytes=21,
        )
