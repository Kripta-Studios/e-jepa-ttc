from __future__ import annotations

import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from operational.sota_eval import prefetch


class ImmediateExecutor:
    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.closed = False

    def submit(self, fn, /, *args):
        row = args[0]
        self.submitted.append(row)
        future: Future[Any] = Future()
        future.set_result(
            (
                {"own_events": np.asarray([row["value"]], dtype=np.float32)},
                {"pid": 1, "rss_bytes": 2, "peak_rss_bytes": 3, "prepare_ns": 4},
            )
        )
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.closed = True


def test_ordered_prefetch_is_bounded_and_preserves_order() -> None:
    rows = [{"query_id": f"q{i}", "value": i} for i in range(4)]
    executor = ImmediateExecutor()
    adapter = prefetch.OrderedPreparePrefetch(
        rows,
        executor=executor,
        max_ahead=2,
        available_bytes=lambda: 10 * prefetch.GIB,
        wait_seconds=0,
    )
    assert [row["query_id"] for row in executor.submitted] == ["q0", "q1"]
    assert adapter(rows[0])["own_events"].item() == 0
    assert [row["query_id"] for row in executor.submitted] == ["q0", "q1", "q2"]
    assert [adapter(row)["own_events"].item() for row in rows[1:]] == [1, 2, 3]


def test_ordered_prefetch_rejects_out_of_order_consumption() -> None:
    rows = [{"query_id": "a", "value": 1}, {"query_id": "b", "value": 2}]
    adapter = prefetch.OrderedPreparePrefetch(
        rows,
        executor=ImmediateExecutor(),
        available_bytes=lambda: 10 * prefetch.GIB,
    )
    with pytest.raises(ValueError, match="order"):
        adapter(rows[1])


def test_ready_result_is_not_blocked_when_headroom_drops() -> None:
    rows = [{"query_id": f"q{i}", "value": i} for i in range(3)]
    executor = ImmediateExecutor()
    calls = 0

    def available() -> int:
        nonlocal calls
        calls += 1
        return 10 * prefetch.GIB if calls <= 2 else 0

    adapter = prefetch.OrderedPreparePrefetch(
        rows,
        executor=executor,
        max_ahead=2,
        available_bytes=available,
        wait_seconds=0,
    )
    started = time.perf_counter()
    assert adapter(rows[0])["own_events"].item() == 0
    assert time.perf_counter() - started < 0.1
    assert [row["query_id"] for row in executor.submitted] == ["q0", "q1"]


def test_stop_interrupts_empty_queue_ram_wait() -> None:
    with pytest.raises(prefetch.PrefetchStoppedError, match="stop requested"):
        prefetch.OrderedPreparePrefetch(
            [{"query_id": "q", "value": 1}],
            executor=ImmediateExecutor(),
            available_bytes=lambda: 0,
            stop_requested=lambda: True,
            wait_seconds=0,
            admission_timeout_seconds=1,
        )


def test_prepared_hash_and_equality_include_dtype_and_values() -> None:
    left = {"x": np.asarray([1, 2], np.float32), "valid": None}
    right = {"valid": None, "x": np.asarray([1, 2], np.float32)}
    assert prefetch.prepared_sha256(left) == prefetch.prepared_sha256(right)
    prefetch.assert_prepared_equal(left, right)
    changed = {"x": np.asarray([1, 2], np.float64), "valid": None}
    with pytest.raises(ValueError, match="bitwise"):
        prefetch.assert_prepared_equal(left, changed)


def test_pending_rows_requires_valid_fragment_checksum(tmp_path: Path) -> None:
    rows = [{"query_id": "a"}, {"query_id": "b"}]
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    saved = predictions / "query_00000.json"
    saved.write_text("{}\n", encoding="utf-8")
    saved.with_suffix(".sha256").write_text(prefetch.digest(saved) + "\n", encoding="ascii")
    assert prefetch.pending_rows(tmp_path, rows) == [rows[1]]
