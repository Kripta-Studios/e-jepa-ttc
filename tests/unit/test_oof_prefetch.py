from __future__ import annotations

import random
import threading
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from operational.oof_prefetch import BoundedFoldPrefetch


@dataclass(frozen=True)
class _Recipe:
    effective_batch_size: int = 3


@dataclass(frozen=True)
class _Source:
    population_size: int = 8


def _build(source: _Source, ids: tuple[int, ...], recipe: _Recipe):
    assert source.population_size == 8
    assert recipe.effective_batch_size == 3
    return {"ids": ids, "thread": threading.current_thread().name}


def _consume(prefetch, cursor, order):
    batches = []
    while cursor["position"] < len(order):
        start = cursor["position"]
        ids = tuple(order[start : start + 3])
        batches.append(prefetch.get(ids))
        cursor["position"] += len(ids)
    return batches


def test_prefetch_matches_synchronous_batches_and_last_partial() -> None:
    source, recipe = _Source(), _Recipe()
    order = torch.arange(8)
    cursor = {"order": order, "position": 0}
    with BoundedFoldPrefetch(source, recipe, cursor, _build) as prefetch:
        actual = _consume(prefetch, cursor, order.tolist())
        assert [value["ids"] for value in actual] == [(0, 1, 2), (3, 4, 5), (6, 7)]
        assert all(value["thread"].startswith("oof-fold-prefetch") for value in actual)
        snapshot = prefetch.snapshot()
        assert snapshot["max_retained_effective_batches"] <= 2
        assert snapshot["worker_threads"] == 1
        assert snapshot["consumed"] == 3


def test_replaced_order_resets_epoch_and_discards_old_queue() -> None:
    cursor = {"order": np.arange(8), "position": 0}
    with BoundedFoldPrefetch(_Source(), _Recipe(), cursor, _build) as prefetch:
        first = prefetch.get((0, 1, 2))
        assert first["ids"] == (0, 1, 2)
        cursor["position"] = 0
        cursor["order"] = np.arange(7, -1, -1)
        second = prefetch.get((7, 6, 5))
        assert second["ids"] == (7, 6, 5)
        assert prefetch.snapshot()["epoch_resets"] == 1


def test_resume_from_mid_epoch_keeps_canonical_position() -> None:
    order = [7, 0, 6, 1, 5, 2, 4, 3]
    cursor = {"order": order, "position": 3}
    with BoundedFoldPrefetch(_Source(), _Recipe(), cursor, _build) as prefetch:
        assert prefetch.get((1, 5, 2))["ids"] == (1, 5, 2)
        cursor["position"] = 6
        assert prefetch.get((4, 3))["ids"] == (4, 3)


def test_builder_error_propagates_and_prefetch_fails_closed() -> None:
    def broken(source, ids, recipe):
        del source, recipe
        if 3 in ids:
            raise OSError("fixture decode failed")
        return ids

    cursor = {"order": list(range(8)), "position": 0}
    with BoundedFoldPrefetch(_Source(), _Recipe(), cursor, broken) as prefetch:
        assert prefetch.get((0, 1, 2)) == (0, 1, 2)
        cursor["position"] = 3
        with pytest.raises(OSError, match="decode failed"):
            prefetch.get((3, 4, 5))
        with pytest.raises(RuntimeError, match="previously failed"):
            prefetch.get((3, 4, 5))


def test_mismatched_indices_rejected_without_cursor_advance() -> None:
    cursor = {"order": list(range(8)), "position": 0}
    prefetch = BoundedFoldPrefetch(_Source(), _Recipe(), cursor, _build)
    with pytest.raises(ValueError, match="saved cursor"):
        prefetch.get((1, 2, 3))
    assert cursor["position"] == 0
    assert prefetch.snapshot()["failed"] is True
    prefetch.close()
    prefetch.close()


def test_prefetch_helper_does_not_consume_python_numpy_or_torch_rng() -> None:
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    cursor = {"order": list(range(8)), "position": 0}
    with BoundedFoldPrefetch(_Source(), _Recipe(), cursor, _build) as prefetch:
        _consume(prefetch, cursor, cursor["order"])
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_in_place_order_mutation_is_rejected() -> None:
    order = list(range(8))
    cursor = {"order": order, "position": 0}
    with BoundedFoldPrefetch(_Source(), _Recipe(), cursor, _build) as prefetch:
        order[0], order[1] = order[1], order[0]
        with pytest.raises(ValueError, match="mutated in place"):
            prefetch.get((1, 0, 2))


