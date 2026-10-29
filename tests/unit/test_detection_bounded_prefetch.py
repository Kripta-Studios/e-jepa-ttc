from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest

from e_jepa_ttc.detection.bounded_prefetch import (
    BoundedOrderedPrefetch,
    PrefetchWorkerError,
)


@dataclass
class _Tracker:
    loaded: list[int]
    closed: threading.Event
    active: int = 0
    maximum_active: int = 0


class _Loader:
    def __init__(
        self,
        tracker: _Tracker,
        *,
        fail_at: int | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.tracker = tracker
        self.fail_at = fail_at
        self.delay_s = delay_s

    def __getitem__(self, index: int) -> dict[str, object]:
        self.tracker.active += 1
        self.tracker.maximum_active = max(self.tracker.maximum_active, self.tracker.active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            self.tracker.loaded.append(index)
            if index == self.fail_at:
                raise ValueError(f"bad index {index}")
            target = [index, index + 1]
            return {"index": index, "target": target}
        finally:
            self.tracker.active -= 1

    def close(self) -> None:
        self.tracker.closed.set()


def _tracker() -> _Tracker:
    return _Tracker(loaded=[], closed=threading.Event())


def test_deterministic_order_and_resume_cursor() -> None:
    tracker = _tracker()
    plan = [4, 1, 9, 2]
    with BoundedOrderedPrefetch(plan, lambda: _Loader(tracker), capacity=2) as stream:
        first = next(stream)
        second = next(stream)
        assert [first["index"], second["index"]] == [4, 1]
        assert stream.state_dict() == {"committed_cursor": 2}

    resumed_tracker = _tracker()
    with BoundedOrderedPrefetch(
        plan,
        lambda: _Loader(resumed_tracker),
        start_cursor=2,
        capacity=1,
    ) as resumed:
        assert [item["index"] for item in resumed] == [9, 2]
    assert resumed_tracker.loaded == [9, 2]


def test_prefetch_does_not_mutate_target() -> None:
    torch = cast(Any, importlib.import_module("torch"))
    target = torch.tensor([3.0, 4.0])

    class TensorLoader:
        def __getitem__(self, index: int) -> dict[str, object]:
            return {"index": index, "target": target}

        def close(self) -> None:
            return None

    before = target.clone()
    with BoundedOrderedPrefetch([3], TensorLoader, capacity=1) as stream:
        item = next(stream)
        assert item["target"] is target
    torch.testing.assert_close(target, before, rtol=0, atol=0)


def test_empty_plan_does_not_create_worker_loader() -> None:
    factory_called = threading.Event()

    def forbidden_factory() -> _Loader:
        factory_called.set()
        raise AssertionError("empty plan must not create a loader")

    with BoundedOrderedPrefetch([], forbidden_factory, capacity=1) as stream:
        assert list(stream) == []
    assert not factory_called.is_set()
    assert not stream.worker_alive


def test_worker_exception_propagates_without_committing_cursor() -> None:
    tracker = _tracker()
    with BoundedOrderedPrefetch(
        [5, 6, 7], lambda: _Loader(tracker, fail_at=6), capacity=2
    ) as stream:
        assert next(stream)["index"] == 5
        with pytest.raises(PrefetchWorkerError, match="cursor 1.*index=6") as exc:
            next(stream)
        assert isinstance(exc.value.__cause__, ValueError)
        assert stream.committed_cursor == 1
    assert tracker.closed.wait(0.5)


def test_factory_exception_propagates() -> None:
    def fail_factory() -> _Loader:
        raise OSError("cannot open dataset")

    with BoundedOrderedPrefetch([8], fail_factory, capacity=1) as stream:
        with pytest.raises(PrefetchWorkerError) as exc:
            next(stream)
        assert isinstance(exc.value.__cause__, OSError)
        assert stream.committed_cursor == 0


def test_loader_close_exception_is_propagated_before_end_of_stream() -> None:
    class CloseFailureLoader:
        def __getitem__(self, index: int) -> int:
            return index

        def close(self) -> None:
            raise OSError("close failed")

    stream = BoundedOrderedPrefetch([2], CloseFailureLoader, capacity=1)
    assert next(stream) == 2
    with pytest.raises(PrefetchWorkerError, match="cleanup failed") as exc:
        next(stream)
    assert isinstance(exc.value.__cause__, OSError)


def test_close_failure_is_not_hidden_by_primary_load_failure() -> None:
    class DoubleFailureLoader:
        def __getitem__(self, index: int) -> int:
            raise ValueError(f"load failed {index}")

        def close(self) -> None:
            raise OSError("close failed too")

    with BoundedOrderedPrefetch([2], DoubleFailureLoader, capacity=1) as stream:
        with pytest.raises(PrefetchWorkerError, match="cursor 0") as exc:
            next(stream)
        assert isinstance(exc.value.__cause__, ValueError)
        assert any("cleanup also failed" in note for note in exc.value.__notes__)


def test_index_plan_is_snapshotted() -> None:
    tracker = _tracker()
    plan = [1, 2]
    stream = BoundedOrderedPrefetch(plan, lambda: _Loader(tracker), capacity=1)
    plan[:] = [8, 9]
    with stream:
        assert [item["index"] for item in stream] == [1, 2]


def test_close_timeout_can_be_retried_after_blocking_load_returns() -> None:
    release = threading.Event()
    entered = threading.Event()
    closed = threading.Event()

    class BlockingLoader:
        def __getitem__(self, index: int) -> int:
            entered.set()
            release.wait(1.0)
            return index

        def close(self) -> None:
            closed.set()

    stream = BoundedOrderedPrefetch([1], BlockingLoader, capacity=1)
    assert entered.wait(0.5)
    with pytest.raises(RuntimeError, match="may be retried"):
        stream.close(timeout_s=0.001)
    assert stream.worker_alive
    release.set()
    stream.close(timeout_s=1.0)
    assert closed.is_set()
    assert not stream.worker_alive


def test_early_close_has_bounded_lookahead_and_no_thread_leak() -> None:
    tracker = _tracker()
    stream = BoundedOrderedPrefetch(
        list(range(100)), lambda: _Loader(tracker, delay_s=0.002), capacity=2
    )
    assert next(stream)["index"] == 0
    time.sleep(0.03)
    committed = stream.committed_cursor
    stream.close()
    assert committed == 1
    assert len(tracker.loaded) <= committed + 2
    assert stream.maximum_outstanding <= 2
    assert tracker.maximum_active <= 1
    assert tracker.closed.wait(0.5)
    assert not stream.worker_alive


@pytest.mark.parametrize("capacity", [0, 3])
def test_rejects_capacity_outside_preregistered_bound(capacity: int) -> None:
    tracker = _tracker()
    with pytest.raises(ValueError, match="capacity must be 1 or 2"):
        BoundedOrderedPrefetch([1], lambda: _Loader(tracker), capacity=capacity)


@pytest.mark.parametrize("capacity", [True, 1.0, None])
def test_rejects_non_integer_capacity_before_worker_start(capacity: object) -> None:
    factory_called = threading.Event()

    def factory() -> _Loader:
        factory_called.set()
        return _Loader(_tracker())

    with pytest.raises(TypeError, match="capacity must be an int"):
        BoundedOrderedPrefetch([1], factory, capacity=cast(Any, capacity))
    assert not factory_called.is_set()


@pytest.mark.parametrize("cursor", [-1, 3])
def test_rejects_invalid_resume_cursor(cursor: int) -> None:
    tracker = _tracker()
    plan: Sequence[int] = [1, 2]
    with pytest.raises(ValueError, match="start_cursor"):
        BoundedOrderedPrefetch(plan, lambda: _Loader(tracker), start_cursor=cursor, capacity=1)


@pytest.mark.parametrize("cursor", [False, 0.5, 1.0, None])
def test_rejects_non_integer_cursor_before_worker_start(cursor: object) -> None:
    factory_called = threading.Event()

    def factory() -> _Loader:
        factory_called.set()
        return _Loader(_tracker())

    with pytest.raises(TypeError, match="start_cursor must be an int"):
        BoundedOrderedPrefetch([1], factory, start_cursor=cast(Any, cursor), capacity=1)
    assert not factory_called.is_set()


def test_unexpected_worker_exception_is_not_silent_eof() -> None:
    class ExplodingPrefetch(BoundedOrderedPrefetch[int, dict[str, object]]):
        def _acquire_permit(self) -> bool:
            raise LookupError("unexpected worker failure")

    tracker = _tracker()
    stream = ExplodingPrefetch([1], lambda: _Loader(tracker), capacity=1)
    with stream:
        with pytest.raises(PrefetchWorkerError, match="worker failed") as exc:
            next(stream)
        assert isinstance(exc.value.__cause__, LookupError)
