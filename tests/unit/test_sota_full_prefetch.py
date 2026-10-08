from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any

from operational.sota_eval import full_prefetch


class ImmediatePool:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.closed = False

    def submit(self, fn, /, *args):
        self.calls.append((fn, args[0]))
        future: Future[Any] = Future()
        future.set_result("routed")
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.closed = True


def test_full_executor_ignores_event_worker_and_routes_full_worker() -> None:
    pool = ImmediatePool()
    executor = full_prefetch.FullPrepareExecutor(pool=pool)

    def ignored(row):
        return row

    future = executor.submit(ignored, {"query_id": "q"})
    assert future.result() == "routed"
    assert pool.calls == [(full_prefetch._full_worker, {"query_id": "q"})]
    executor.shutdown()
    assert pool.closed


def test_backend_binding_pins_stable_and_full_sources(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(full_prefetch, "digest", lambda path: str(path))
    result = full_prefetch.backend_binding(
        manifest, workers=2, max_ahead=2, min_available_bytes=2 * full_prefetch.GIB
    )
    assert result["workers"] == 2
    assert result["scientific_binding_changed"] is False
    assert "operational\\sota_eval\\prefetch.py" in result["sources"]
    assert "operational\\evttc_rgb_transfer\\inputs.py" in result["sources"]


def test_execution_receipt_separates_worker_and_consumer_timing() -> None:
    receipt = full_prefetch._execution_receipt(
        status="DONE",
        started_utc="now",
        started=full_prefetch.time.perf_counter(),
        previous_sha256=None,
        adapter=None,
    )
    assert receipt["records"] == []
    assert "consumer wait" in receipt["timing_warning"]
    assert receipt["scientific_binding_changed"] is False
