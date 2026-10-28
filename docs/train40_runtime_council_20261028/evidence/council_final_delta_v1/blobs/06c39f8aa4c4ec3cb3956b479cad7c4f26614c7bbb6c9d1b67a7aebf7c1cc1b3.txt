from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from operational.train40_system import (
    controller_transport_receiver,
    history_resources8_transport_receiver,
    hourly_supervisor_transport_receiver,
)
from operational.train40_system.h8_receiver_freeze import _qa


class FakeMonitor:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.closed = False

    def snapshot(self) -> dict[str, Any]:
        return {"closed": self.closed, "continuous_receiver": {"poll_seconds": 0.5}}


def test_receiver_wrapper_binds_receipts_and_restores(tmp_path: Path) -> None:
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_graph as transport
    from operational.train40_system import resource_monitor_receiver

    rows: list[tuple[Path, dict[str, Any]]] = []

    def write(path: Path, value: dict[str, Any]) -> None:
        rows.append((path, value))

    def frozen_run(output: Path, raw_root: Path) -> None:
        monitor = base.C2FResourceMonitor(output)
        assert isinstance(monitor, FakeMonitor)
        kernel.atomic_json(output / "h8_feature_fragments/query_000001.json", {"base": True})
        base.atomic_json(output / "H8_FAST_RUNTIME.json", {"base": True})
        transport.atomic_json(output / "H8_TRANSPORT_RUNTIME.json", {"base": True})

    original_class = base.C2FResourceMonitor
    original_kernel = kernel.atomic_json
    original_base = base.atomic_json
    original_transport = transport.atomic_json
    freeze = {"transport_freeze_sha256": "transport"}
    with (
        patch.object(
            history_resources8_transport_receiver, "verify_admission", return_value=freeze
        ),
        patch.object(history_resources8_transport_receiver, "digest", return_value="receiver"),
        patch.object(history_resources8_transport_receiver, "atomic_json", side_effect=write),
        patch.object(resource_monitor_receiver, "ContinuousC2FResourceMonitor", FakeMonitor),
        patch.object(transport, "run", side_effect=frozen_run),
        patch.object(kernel, "atomic_json", side_effect=write),
        patch.object(base, "atomic_json", side_effect=write),
        patch.object(transport, "atomic_json", side_effect=write),
    ):
        expected_kernel = kernel.atomic_json
        expected_base = base.atomic_json
        expected_transport = transport.atomic_json
        history_resources8_transport_receiver.run(tmp_path, tmp_path / "raw")
        assert kernel.atomic_json is expected_kernel
        assert base.atomic_json is expected_base
        assert transport.atomic_json is expected_transport
        assert base.C2FResourceMonitor is original_class

    bound = [value for path, value in rows if path.name != "H8_RECEIVER_RUNTIME.json"]
    assert len(bound) == 3
    assert all(value["receiver_execution_freeze_sha256"] == "receiver" for value in bound)
    runtimes = [value for path, value in rows if path.name.endswith("RUNTIME.json")]
    assert all("receiver_monitor" in value for value in runtimes)
    assert kernel.atomic_json is original_kernel
    assert base.atomic_json is original_base
    assert transport.atomic_json is original_transport


def test_receiver_wrapper_restores_after_failure(tmp_path: Path) -> None:
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system import history_resources8_transport_graph as transport
    from operational.train40_system import resource_monitor_receiver

    originals = (
        base.C2FResourceMonitor,
        kernel.atomic_json,
        base.atomic_json,
        transport.atomic_json,
    )
    with (
        patch.object(
            history_resources8_transport_receiver,
            "verify_admission",
            return_value={"transport_freeze_sha256": "transport"},
        ),
        patch.object(history_resources8_transport_receiver, "digest", return_value="receiver"),
        patch.object(history_resources8_transport_receiver, "atomic_json"),
        patch.object(resource_monitor_receiver, "ContinuousC2FResourceMonitor", FakeMonitor),
        patch.object(transport, "run", side_effect=RuntimeError("failed")),
        pytest.raises(RuntimeError, match="failed"),
    ):
        history_resources8_transport_receiver.run(tmp_path, tmp_path / "raw")

    assert (
        base.C2FResourceMonitor,
        kernel.atomic_json,
        base.atomic_json,
        transport.atomic_json,
    ) == originals


def test_receiver_controller_rewrites_only_frozen_transport_child(tmp_path: Path) -> None:
    from operational.train40_system import controller_overlap, controller_transport_graph

    calls: list[tuple[Any, ...]] = []

    def frozen_run(output: Path, raw_root: Path, teacher_path: Path) -> None:
        controller_overlap._launch(
            output,
            "h8_features",
            "operational.train40_system.history_resources8_transport_graph",
            ["--raw-root", str(raw_root), "--kind", "H8"],
        )
        controller_overlap._launch(output, "other", "unchanged.module", [])

    def previous(*args: Any) -> None:
        calls.append(args)

    with (
        patch("operational.train40_system.history_resources8_transport_receiver.verify_admission"),
        patch.object(controller_transport_graph, "run", side_effect=frozen_run),
        patch.object(controller_overlap, "_launch", side_effect=previous) as original,
    ):
        controller_transport_receiver.run(tmp_path, tmp_path / "raw", tmp_path / "teacher")
        assert calls[0][2] == "operational.train40_system.history_resources8_transport_receiver"
        assert calls[1][2] == "unchanged.module"
        assert controller_overlap._launch is original


def test_receiver_name_is_adopted_by_existing_active_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from operational.train40_system import controller_overlap

    monkeypatch.setattr(
        controller_overlap,
        "_active",
        lambda marker: marker in "operational.train40_system.history_resources8_transport_receiver",
    )
    assert controller_overlap.active("operational.train40_system.history_features")


def test_receiver_supervisor_selects_new_controller() -> None:
    from operational.train40_system import hourly_supervisor

    previous = hourly_supervisor.CONTROLLER
    try:
        with patch.object(hourly_supervisor, "main") as main:
            hourly_supervisor_transport_receiver.main()
            main.assert_called_once_with()
            assert hourly_supervisor.CONTROLLER.endswith("controller_transport_receiver")
    finally:
        hourly_supervisor.CONTROLLER = previous


def test_receiver_freeze_rejects_failed_pytest(tmp_path: Path) -> None:
    (tmp_path / "H8_RECEIVER_PYTEST.txt").write_text(
        "..... [100%]\n1 failed, 5 passed\n", encoding="utf-8"
    )
    (tmp_path / "H8_RECEIVER_RUFF.txt").write_text("All checks passed!\n", encoding="utf-8")
    (tmp_path / "H8_RECEIVER_PYRIGHT.txt").write_text(
        "0 errors, 0 warnings, 0 informations\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="PYTEST"):
        _qa(tmp_path)
