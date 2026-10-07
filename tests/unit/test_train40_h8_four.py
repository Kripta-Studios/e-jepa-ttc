"""CPU-only tests for the bounded four-worker H8 transport route."""

# ruff: noqa: B009, B010 -- tests inspect and restore heterogeneous module bindings.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from operational.train40_system import history_resources8_transport_four as four


def _prepare_value(job: dict[str, int]) -> np.ndarray:
    return np.asarray([job["value"], os.getpid()], dtype=np.int64)


def test_real_spawn_keeps_eight_pending_with_four_workers() -> None:
    from operational.train40_system.h8_shared_pool import SharedArrayProcessPool

    with four.four_raw_workers():
        with SharedArrayProcessPool(
            max_workers=8,
            shape=(2,),
            dtype=np.int64,
        ) as pool:
            assert pool.max_workers == 8
            assert pool._arena_bytes == 8 * 2 * np.dtype(np.int64).itemsize
            assert getattr(pool._executor, "_max_workers") == 4
            first = [pool.submit(_prepare_value, {"value": value}) for value in range(8)]
            first_values = [future.result().copy() for future in first]
            second = [
                pool.submit(_prepare_value, {"value": value}) for value in range(8, 16)
            ]
            second_values = [future.result().copy() for future in second]
    assert [int(value[0]) for value in first_values + second_values] == list(range(16))
    assert 1 <= len({int(value[1]) for value in first_values + second_values}) <= 4


def test_executor_patch_restores_after_failure() -> None:
    from operational.train40_system import h8_shared_pool

    original = h8_shared_pool.ProcessPoolExecutor
    with pytest.raises(RuntimeError, match="preserved"), four.four_raw_workers():
        assert h8_shared_pool.ProcessPoolExecutor is not original
        raise RuntimeError("preserved")
    assert h8_shared_pool.ProcessPoolExecutor is original


def test_wrapper_annotates_and_restores_all_bindings(tmp_path: Path) -> None:
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_graph as transport
    from operational.train40_system import history_resources8_transport_memory as memory
    from operational.train40_system import history_resources8_transport_receiver as receiver
    from operational.train40_system import resource_monitor

    modules = (kernel, base, transport, receiver)
    originals = tuple(getattr(module, "atomic_json") for module in modules)
    seen: list[dict[str, object]] = []

    def nested(output: Path, raw_root: Path) -> None:
        del raw_root
        assert resource_monitor.TREE_RSS_LIMIT_BYTES != memory.TREE_LIMIT
        with memory.memory_allowance():
            assert resource_monitor.TREE_RSS_LIMIT_BYTES == 23_000_000_000
            assert resource_monitor.HOST_AVAILABLE_MIN_BYTES == 2 * 1024**3
        from operational.train40_system import h8_shared_pool

        assert h8_shared_pool.ProcessPoolExecutor is not original_executor
        for module in modules:
            getattr(module, "atomic_json")(output / "H8_FAST_RUNTIME.json", {"original": True})
            getattr(module, "atomic_json")(
                output / "h8_feature_fragments/query_58241.json", {"original": True}
            )

    from operational.train40_system import h8_shared_pool

    original_executor = h8_shared_pool.ProcessPoolExecutor
    with (
        patch.object(four, "verify_admission", return_value={}),
        patch.object(four, "digest", return_value="four-freeze"),
        patch.object(memory, "run", side_effect=nested),
    ):
        for module in modules:
            setattr(module, "atomic_json", lambda _path, value: seen.append(value))
        try:
            four.run(tmp_path, tmp_path)
        finally:
            for module, original in zip(modules, originals, strict=True):
                setattr(module, "atomic_json", original)
    assert len(seen) == 8
    assert all(value["original"] for value in seen)
    assert all(value["raw_worker_count"] == 4 for value in seen)
    assert all(value["arena_count"] == 8 for value in seen)
    assert all(value["tree_rss_limit_bytes"] == 23_000_000_000 for value in seen)
    assert all(value["four_worker_execution_freeze_sha256"] == "four-freeze" for value in seen)
    assert h8_shared_pool.ProcessPoolExecutor is original_executor


def test_wrapper_restores_publishers_when_memory_run_fails(tmp_path: Path) -> None:
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_graph as transport
    from operational.train40_system import history_resources8_transport_memory as memory
    from operational.train40_system import history_resources8_transport_receiver as receiver

    modules = (kernel, base, transport, receiver)
    originals = tuple(getattr(module, "atomic_json") for module in modules)
    with (
        patch.object(four, "verify_admission", return_value={}),
        patch.object(four, "digest", return_value="four-freeze"),
        patch.object(memory, "run", side_effect=RuntimeError("preserved")),
        pytest.raises(RuntimeError, match="preserved"),
    ):
        four.run(tmp_path, tmp_path)
    assert tuple(getattr(module, "atomic_json") for module in modules) == originals


def test_fresh_import_does_not_import_torch() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import operational.train40_system.history_resources8_transport_four; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)"
        ),
    ]
    result = subprocess.run(command, cwd=Path.cwd(), check=False, timeout=30)
    assert result.returncode == 0
