from __future__ import annotations

import time
from multiprocessing.shared_memory import SharedMemory
from threading import Event, Thread
from typing import Any

import numpy as np
import pytest

from operational.train40_system.h8_shared_pool import (
    FULL_H8_DTYPE,
    FULL_H8_SHAPE,
    MAX_ARENAS,
    SharedArrayProcessPool,
)

_INITIALIZED = False


def _toy_initializer() -> None:
    global _INITIALIZED
    _INITIALIZED = True


def _toy_prepare(job: dict[str, Any]) -> np.ndarray:
    if job.get("require_initializer") and not _INITIALIZED:
        raise RuntimeError("worker initializer did not run")
    delay = float(job.get("delay", 0.0))
    if delay:
        time.sleep(delay)
    shape = tuple(job["shape"])
    return np.full(shape, float(job["value"]), dtype=np.float32)


def _toy_failure(job: dict[str, Any]) -> np.ndarray:
    raise ValueError(str(job["message"]))


def _pool(*, workers: int = 2, profile: bool = False) -> SharedArrayProcessPool[Any]:
    return SharedArrayProcessPool(
        max_workers=workers,
        initializer=_toy_initializer,
        shape=(2, 3),
        dtype=np.float32,
        profile=profile,
    )


def test_runtime_defaults_are_exact_fixed_h8_contract() -> None:
    assert FULL_H8_SHAPE == (16, 3, 12, 128, 128)
    assert FULL_H8_DTYPE == np.dtype(np.float32)
    assert np.prod(FULL_H8_SHAPE) * FULL_H8_DTYPE.itemsize == 36 * 1024**2
    assert MAX_ARENAS == 8
    with pytest.raises(ValueError, match="between 1 and 8"):
        SharedArrayProcessPool(max_workers=9, shape=(1,), dtype=np.float32)


def test_real_spawn_returns_exact_arrays_in_order_with_initializer_and_profiles() -> None:
    with _pool(workers=4, profile=True) as pool:
        futures = [
            pool.submit(
                _toy_prepare,
                {
                    "shape": (2, 3),
                    "value": value,
                    "row_token": f"row-{value}",
                    "require_initializer": True,
                },
            )
            for value in range(4)
        ]
        for value, future in enumerate(futures):
            actual = future.result().copy()
            assert np.array_equal(actual, np.full((2, 3), value, np.float32))
            assert future.row_token == f"row-{value}"
            assert future.slot in {0, 1, 2, 3}
            assert future.profile is not None
            assert future.profile["worker_prepare_seconds"] >= 0
            assert future.profile["arena_copy_seconds"] >= 0


def test_out_of_order_result_is_rejected_without_consuming_future() -> None:
    with _pool() as pool:
        first = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 1})
        second = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 2})
        with pytest.raises(RuntimeError, match="submission order"):
            second.result()
        assert np.all(first.result() == 1)
        assert np.all(second.result() == 2)


def test_arena_backpressure_releases_only_when_result_is_consumed() -> None:
    with _pool() as pool:
        first = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 1})
        second = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 2})
        entered = Event()
        returned = Event()
        holder: list[Any] = []

        def submit_third() -> None:
            entered.set()
            holder.append(
                pool.submit(
                    _toy_prepare,
                    {"shape": (2, 3), "value": 3, "delay": 0.2},
                )
            )
            returned.set()

        thread = Thread(target=submit_third)
        thread.start()
        assert entered.wait(1)
        assert not returned.wait(0.1)
        first_value = first.result().copy()
        assert returned.wait(1)
        thread.join(timeout=1)
        assert np.all(first_value == 1)
        assert np.all(second.result() == 2)
        assert np.all(holder[0].result() == 3)


def test_worker_failure_propagates_and_releases_slot_for_later_work() -> None:
    with SharedArrayProcessPool(
        max_workers=1, shape=(2, 3), dtype=np.float32
    ) as pool:
        failed = pool.submit(_toy_failure, {"message": "raw preparation failed"})
        with pytest.raises(ValueError, match="raw preparation failed"):
            failed.result()
        recovered = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 7})
        assert np.all(recovered.result() == 7)


def test_shape_mismatch_fails_closed_and_single_consumption_is_enforced() -> None:
    with _pool(workers=1) as pool:
        wrong = pool.submit(_toy_prepare, {"shape": (1, 3), "value": 1})
        with pytest.raises(ValueError, match="result contract differs"):
            wrong.result()
        valid = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 2})
        valid.result()
        with pytest.raises(RuntimeError, match="only once"):
            valid.result()


def test_shutdown_drains_then_unlinks_every_arena() -> None:
    pool = _pool()
    names = [handle.name for handle in pool._handles]
    futures = [
        pool.submit(_toy_prepare, {"shape": (2, 3), "value": value})
        for value in (1, 2)
    ]
    pool.shutdown(wait=True)
    assert all(future.done() for future in futures)
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, create=False)


def test_non_draining_shutdown_is_rejected() -> None:
    pool = _pool(workers=1)
    try:
        with pytest.raises(ValueError, match="must drain workers"):
            pool.shutdown(wait=False)
    finally:
        pool.shutdown(wait=True)


def test_snapshot_tracks_clean_metadata_only_ownership_and_profiles() -> None:
    pool = _pool(workers=2, profile=True)
    initial = pool.snapshot()
    assert initial == {
        "submitted": 0,
        "consumed": 0,
        "inflight_current": 0,
        "inflight_max": 0,
        "arena_bytes": 2 * 2 * 3 * np.dtype(np.float32).itemsize,
        "metadata_payload_only": True,
        "pool_closed": False,
        "worker_prepare_seconds": 0.0,
        "arena_copy_seconds": 0.0,
    }
    first = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 1})
    second = pool.submit(_toy_prepare, {"shape": (2, 3), "value": 2})
    pending = pool.snapshot()
    assert pending["submitted"] == 2
    assert pending["consumed"] == 0
    assert pending["inflight_current"] == 2
    assert pending["inflight_max"] == 2
    first.result()
    consumed = pool.snapshot()
    assert consumed["consumed"] == 1
    assert consumed["inflight_current"] == 1
    assert consumed["worker_prepare_seconds"] >= 0
    assert consumed["arena_copy_seconds"] >= 0
    second.result()
    pool.shutdown()
    final = pool.snapshot()
    assert final["submitted"] == final["consumed"] == 2
    assert final["inflight_current"] == 0
    assert final["pool_closed"] is True


def test_result_timeout_keeps_arena_leased_until_terminal_consumption() -> None:
    with _pool(workers=1) as pool:
        future = pool.submit(
            _toy_prepare,
            {"shape": (2, 3), "value": 5, "delay": 0.2},
        )
        with pytest.raises(TimeoutError):
            future.result(timeout=0.01)
        timed_out = pool.snapshot()
        assert timed_out["consumed"] == 0
        assert timed_out["inflight_current"] == 1
        assert np.all(future.result(timeout=2) == 5)
        completed = pool.snapshot()
        assert completed["consumed"] == 1
        assert completed["inflight_current"] == 0
