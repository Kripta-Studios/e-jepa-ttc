"""Ensure shared-memory RSS headroom retains all physical resource guards."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from operational.train40_system import history_resources8_transport_memory as memory
from operational.train40_system import resource_monitor


def test_spawn_entrypoint_import_keeps_monitor_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import operational.train40_system."
            "history_resources8_transport_memory; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("rss", "available", "disk", "expected"),
    [
        (16_035_979_264, 4_476_010_496, 59_970_490_368, True),
        (19_999_999_999, 2 * 1024**3, 20_000_000_000, True),
        (20_000_000_000, 9_000_000_000, 30_000_000_000, False),
        (17_000_000_000, 2 * 1024**3 - 1, 30_000_000_000, False),
        (17_000_000_000, 9_000_000_000, 19_999_999_999, False),
    ],
)
def test_real_guard_boundaries(
    tmp_path: Path, rss: int, available: int, disk: int, expected: bool
) -> None:
    monitor = resource_monitor.ResourceMonitor(tmp_path)
    monitor._started = True
    monitor._process = SimpleNamespace(is_alive=lambda: True)
    monitor._latest = {
        "status": "OK",
        "sampled_monotonic": time.monotonic(),
        "tree_rss_bytes": rss,
        "host_available_bytes": available,
        "disk_free_bytes": disk,
    }
    budget = {
        "physical_updates_upper": 147578,
        "new_recovery_upper": 91,
        "deadline_utc": None,
        "pause_markers": [],
    }
    old_limit = resource_monitor.TREE_RSS_LIMIT_BYTES
    with patch.object(resource_monitor, "_budget_state", return_value=budget):
        with memory.memory_allowance():
            allowed, telemetry = monitor.guard(tmp_path)
            assert allowed is expected, telemetry
    assert resource_monitor.TREE_RSS_LIMIT_BYTES == old_limit


def test_restore_after_failure() -> None:
    previous = resource_monitor.TREE_RSS_LIMIT_BYTES
    with pytest.raises(RuntimeError), memory.memory_allowance():
        assert resource_monitor.TREE_RSS_LIMIT_BYTES == 20_000_000_000
        raise RuntimeError("transient")
    assert resource_monitor.TREE_RSS_LIMIT_BYTES == previous


def test_reserve_cannot_be_lowered() -> None:
    with patch.object(resource_monitor, "HOST_AVAILABLE_MIN_BYTES", 1024):
        with pytest.raises(ValueError, match="physical host reserve"), memory.memory_allowance():
            pytest.fail("Must reject a changed physical reserve")


def test_controller_routes_only_h8(tmp_path: Path) -> None:
    from operational.train40_system import controller_overlap, controller_transport_memory
    from operational.train40_system import controller_transport_receiver as receiver

    old_launch = controller_overlap._launch

    def nested(*args: object) -> None:
        controller_overlap._launch(
            tmp_path,
            "h8_features",
            "operational.train40_system.history_resources8_transport_receiver",
            ["--raw-root", "E:/eAP_dataset", "--kind", "H8"],
        )

    with (
        patch.object(memory, "verify_admission", return_value={}),
        patch.object(receiver, "run", side_effect=nested),
        patch.object(controller_overlap, "_launch") as launch,
    ):
        controller_transport_memory.run(tmp_path, tmp_path, tmp_path)
        assert launch.call_args.args[2].endswith("history_resources8_transport_memory")
        assert controller_overlap._launch is launch
    assert controller_overlap._launch is old_launch


@pytest.mark.parametrize("fail", [False, True])
def test_extraction_wrapper_binds_and_restores(tmp_path: Path, fail: bool) -> None:
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_receiver as receiver

    previous = resource_monitor.TREE_RSS_LIMIT_BYTES

    def nested(output: Path, raw_root: Path) -> None:
        assert resource_monitor.TREE_RSS_LIMIT_BYTES == memory.TREE_LIMIT
        kernel.atomic_json(output / "h8_feature_fragments/query_00001.json", {"original": True})
        base.atomic_json(output / "H8_FAST_RUNTIME.json", {"original": True})
        if fail:
            raise RuntimeError("preserved")

    with (
        patch.object(memory, "verify_admission", return_value={}),
        patch.object(memory, "digest", return_value="memory-sha"),
        patch.object(receiver, "run", side_effect=nested),
        patch.object(kernel, "atomic_json") as write_kernel,
        patch.object(base, "atomic_json") as write_base,
    ):
        if fail:
            with pytest.raises(RuntimeError, match="preserved"):
                memory.run(tmp_path, tmp_path)
        else:
            memory.run(tmp_path, tmp_path)
        for writer in (write_kernel, write_base):
            assert writer.call_args.args[1]["original"]
            assert writer.call_args.args[1]["memory_execution_freeze_sha256"] == "memory-sha"
        assert kernel.atomic_json is write_kernel
        assert base.atomic_json is write_base
    assert resource_monitor.TREE_RSS_LIMIT_BYTES == previous
