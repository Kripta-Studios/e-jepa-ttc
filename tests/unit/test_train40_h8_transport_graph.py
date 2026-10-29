from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from e_jepa_ttc.models import causal_scale_ttc
from e_jepa_ttc.models.local_transport import (
    LocalTransportMatch,
    local_correlation_match,
    transport_physical_features,
)
from operational.train40_system import (
    controller_transport_graph,
    history_resources8_transport_graph,
    hourly_supervisor_transport,
)
from operational.train40_system.h8_transport_freeze import _require_qa
from operational.train40_system.h8_transport_graph import (
    H8TransportGraph,
    _CapturedCall,
    _clone_match,
    _flatten_match,
    _tensor_signature,
    _unflatten_match,
    install_h8_transport_graph,
)


def _match(*, probability: bool = False) -> LocalTransportMatch:
    base = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    return LocalTransportMatch(
        dx=base,
        dy=base + 1,
        confidence_margin=base + 2,
        entropy=base + 3,
        valid=(base % 2) == 0,
        probability=torch.stack((base, base + 1), dim=1) if probability else None,
    )


@pytest.mark.parametrize("with_probability", [False, True])
def test_match_flatten_round_trip_preserves_structure(with_probability: bool) -> None:
    original = _match(probability=with_probability)
    restored = _unflatten_match(_flatten_match(original))

    if with_probability:
        assert restored.probability is not None
    else:
        assert restored.probability is None
    for before, after in zip(_flatten_match(original), _flatten_match(restored), strict=True):
        assert before is after


def test_unflatten_rejects_invalid_leaf_count() -> None:
    with pytest.raises(ValueError, match="five or six"):
        _unflatten_match((torch.zeros(1),))


def test_clone_match_owns_all_output_leaves() -> None:
    original = _match(probability=True)
    cloned = _clone_match(original)

    for before, after in zip(_flatten_match(original), _flatten_match(cloned), strict=True):
        assert before.data_ptr() != after.data_ptr()
        assert torch.equal(before, after)

    original.dx.fill_(-1)
    assert not torch.equal(original.dx, cloned.dx)


def test_tensor_signature_distinguishes_stride() -> None:
    contiguous = torch.zeros((2, 3), dtype=torch.float32)
    transposed = torch.zeros((3, 2), dtype=torch.float32).t()

    assert contiguous.shape == transposed.shape
    assert _tensor_signature(contiguous) != _tensor_signature(transposed)


class _FakeGraph:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls = 0

    def replay(self) -> None:
        self.calls += 1
        self.output.fill_(float(self.calls))


def test_captured_call_returns_independent_outputs() -> None:
    static_input = torch.zeros(2)
    static_output = torch.zeros(2)
    graph = _FakeGraph(static_output)
    captured = _CapturedCall(graph, (static_input,), (static_output,))  # type: ignore[arg-type]

    first = captured.replay((torch.tensor([4.0, 5.0]),))[0]
    second = captured.replay((torch.tensor([6.0, 7.0]),))[0]

    assert graph.calls == 2
    assert torch.equal(first, torch.ones(2))
    assert torch.equal(second, torch.full((2,), 2.0))
    assert torch.equal(static_input, torch.tensor([6.0, 7.0]))
    assert first.data_ptr() != static_output.data_ptr()


def test_install_patches_only_imported_aliases_and_restores_on_error() -> None:
    runtime = H8TransportGraph(max_entries=2, warmup_calls=1)

    with pytest.raises(RuntimeError, match="sentinel"):
        with runtime.install():
            assert causal_scale_ttc.local_correlation_match == runtime.local_correlation_match
            assert (
                causal_scale_ttc.transport_physical_features == runtime.transport_physical_features
            )
            # Canonical definitions are never rebound.
            from e_jepa_ttc.models import local_transport

            assert local_transport.local_correlation_match is local_correlation_match
            assert local_transport.transport_physical_features is transport_physical_features
            raise RuntimeError("sentinel")

    assert causal_scale_ttc.local_correlation_match is local_correlation_match
    assert causal_scale_ttc.transport_physical_features is transport_physical_features
    assert runtime.snapshot()["installed"] is False


def test_install_rejects_preexisting_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    def replacement(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(causal_scale_ttc, "local_correlation_match", replacement)

    with pytest.raises(RuntimeError, match="already patched"):
        with H8TransportGraph().install():
            pass


def test_convenience_context_closes_after_exception() -> None:
    holder: dict[str, H8TransportGraph] = {}

    with pytest.raises(ValueError, match="stop"):
        with install_h8_transport_graph(max_entries=1, warmup_calls=1) as runtime:
            holder["runtime"] = runtime
            raise ValueError("stop")

    assert holder["runtime"].snapshot()["closed"] is True
    assert causal_scale_ttc.local_correlation_match is local_correlation_match


def test_constructor_and_close_guards() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        H8TransportGraph(max_entries=0)
    with pytest.raises(ValueError, match="warmup_calls"):
        H8TransportGraph(warmup_calls=0)

    runtime = H8TransportGraph()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        with runtime.install():
            pass


def test_close_while_installed_is_rejected() -> None:
    runtime = H8TransportGraph()
    with runtime.install():
        with pytest.raises(RuntimeError, match="installed"):
            runtime.close()


def test_module_import_does_not_touch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # The meaningful guarantee is structural: constructors and context setup do
    # not call CUDA.  Replacing current_stream would fail this CPU-only test if
    # setup regressed.
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CUDA API called during setup")

    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    runtime = H8TransportGraph()
    with runtime.install():
        assert runtime.snapshot()["cache_entries"] == 0


def test_runtime_rejects_a_different_cuda_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = SimpleNamespace(cuda_stream=17)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    runtime = H8TransportGraph()

    runtime._assert_owner(torch.device("cuda:0"))
    stream.cuda_stream = 18
    with pytest.raises(RuntimeError, match="first thread and CUDA stream"):
        runtime._assert_owner(torch.device("cuda:0"))


def test_history_wrapper_binds_receipt_and_restores_all_patches(tmp_path: Path) -> None:
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base
    from operational.train40_system.h8_bounded_union import prepare as bounded_prepare

    original_atomic = kernel.atomic_json
    original_prepare = kernel.prepare
    receipts: list[tuple[Path, dict[str, Any]]] = []
    freeze = {
        "graph_cache_entries": 8,
        "graph_warmup_calls": 3,
        "base_freeze_sha256": "base",
    }

    def durable(path: Path, value: dict[str, Any]) -> None:
        receipts.append((path, value))

    def base_run(output: Path, raw_root: Path) -> None:
        assert causal_scale_ttc.local_correlation_match != local_correlation_match
        assert kernel.atomic_json is not original_atomic
        assert kernel.prepare is bounded_prepare
        kernel.atomic_json(
            output / "h8_feature_fragments/query_000001.json",
            {"execution_freeze_sha256": "base-freeze"},
        )
        base.atomic_json(output / "H8_FAST_RUNTIME.json", {"status": "RUNNING"})

    with (
        patch.object(history_resources8_transport_graph, "verify_admission", return_value=freeze),
        patch.object(history_resources8_transport_graph, "digest", return_value="transport-freeze"),
        patch.object(history_resources8_transport_graph, "atomic_json", side_effect=durable),
        patch.object(base, "run", side_effect=base_run),
        patch.object(base, "atomic_json", side_effect=durable) as initial_base_atomic,
        patch.object(kernel, "atomic_json", side_effect=durable) as initial_atomic,
    ):
        # Preserve the exact object installed by this test as the restoration target.
        expected_atomic = kernel.atomic_json
        history_resources8_transport_graph.run(tmp_path, tmp_path / "raw")
        assert kernel.atomic_json is expected_atomic
        assert kernel.prepare is original_prepare
        assert base.atomic_json is initial_base_atomic
        assert initial_atomic.call_count == 1

    fragment = next(value for path, value in receipts if path.name.startswith("query_"))
    assert fragment["execution_freeze_sha256"] == "base-freeze"
    assert fragment["transport_execution_freeze_sha256"] == "transport-freeze"
    runtime = next(value for path, value in receipts if path.name == "H8_FAST_RUNTIME.json")
    assert runtime["transport_execution_freeze_sha256"] == "transport-freeze"
    assert runtime["transport_graph"]["installed"] is True
    final_runtime = next(
        value for path, value in receipts if path.name == "H8_TRANSPORT_RUNTIME.json"
    )
    assert final_runtime["graph"]["installed"] is False
    assert final_runtime["graph"]["closed"] is True
    assert causal_scale_ttc.local_correlation_match is local_correlation_match


def test_history_wrapper_restores_after_base_failure(tmp_path: Path) -> None:
    from operational.train40_system import history_resources8 as kernel
    from operational.train40_system import history_resources8_fast as base

    freeze = {
        "graph_cache_entries": 8,
        "graph_warmup_calls": 3,
        "base_freeze_sha256": "base",
    }
    previous = kernel.atomic_json
    previous_prepare = kernel.prepare
    with (
        patch.object(history_resources8_transport_graph, "verify_admission", return_value=freeze),
        patch.object(history_resources8_transport_graph, "digest", return_value="transport-freeze"),
        patch.object(history_resources8_transport_graph, "atomic_json"),
        patch.object(base, "run", side_effect=RuntimeError("base failed")),
        pytest.raises(RuntimeError, match="base failed"),
    ):
        history_resources8_transport_graph.run(tmp_path, tmp_path / "raw")

    assert kernel.atomic_json is previous
    assert kernel.prepare is previous_prepare
    assert causal_scale_ttc.local_correlation_match is local_correlation_match


def test_transport_controller_changes_only_h8_dispatch(tmp_path: Path) -> None:
    from operational.train40_system import controller_c2f_graph

    calls: list[tuple[Any, ...]] = []

    def existing_run(output: Path, raw_root: Path, teacher_path: Path) -> None:
        controller_c2f_graph._launch(
            output,
            "h8_features",
            "operational.train40_system.history_features",
            ["--raw-root", str(raw_root), "--kind", "H8"],
        )
        controller_c2f_graph._launch(
            output, "c2f", "operational.train40_system.engine", ["--arm", "c2f"]
        )

    def previous(*args: Any) -> None:
        calls.append(args)

    with (
        patch("operational.train40_system.history_resources8_transport_graph.verify_admission"),
        patch.object(controller_c2f_graph, "run", side_effect=existing_run),
        patch.object(controller_c2f_graph, "_launch", side_effect=previous) as prior,
        patch("operational.train40_system.controller_overlap._launch") as launcher,
    ):
        controller_transport_graph.run(tmp_path, tmp_path / "raw", tmp_path / "teacher")
        assert (
            launcher.call_args.args[2]
            == "operational.train40_system.history_resources8_transport_graph"
        )
        assert calls[0][1:3] == ("c2f", "operational.train40_system.engine")
        assert controller_c2f_graph._launch is prior


def test_existing_active_marker_adopts_transport_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from operational.train40_system import controller_overlap

    observed: list[str] = []

    def active(marker: str) -> bool:
        observed.append(marker)
        return marker in "operational.train40_system.history_resources8_transport_graph"

    monkeypatch.setattr(controller_overlap, "_active", active)
    assert controller_overlap.active("operational.train40_system.history_features")
    assert observed == [
        "operational.train40_system.history_features",
        "operational.train40_system.history_resources8",
    ]


def test_transport_supervisor_selects_new_controller() -> None:
    from operational.train40_system import hourly_supervisor

    previous = hourly_supervisor.CONTROLLER
    try:
        with patch.object(hourly_supervisor, "main") as main:
            hourly_supervisor_transport.main()
            main.assert_called_once_with()
            assert hourly_supervisor.CONTROLLER.endswith("controller_transport_graph")
    finally:
        hourly_supervisor.CONTROLLER = previous


def test_freeze_rejects_failed_pytest_even_with_100_percent_marker(tmp_path: Path) -> None:
    (tmp_path / "H8_TRANSPORT_PYTEST.txt").write_text(
        "................ [100%]\n1 failed, 18 passed\n", encoding="utf-8"
    )
    (tmp_path / "H8_TRANSPORT_RUFF.txt").write_text("All checks passed!\n", encoding="utf-8")
    (tmp_path / "H8_TRANSPORT_PYRIGHT.txt").write_text(
        "0 errors, 0 warnings, 0 informations\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="PYTEST"):
        _require_qa(tmp_path)

    (tmp_path / "H8_TRANSPORT_PYTEST.txt").write_text(
        "................ [100%]\n1 passed, 1 error in 1.00s\nERRORS\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="PYTEST"):
        _require_qa(tmp_path)

    (tmp_path / "H8_TRANSPORT_PYTEST.txt").write_text(
        "................... [100%]\n19 passed in 1.00s\n", encoding="utf-8"
    )
    receipts = _require_qa(tmp_path)
    assert set(receipts) == {
        "H8_TRANSPORT_PYTEST.txt",
        "H8_TRANSPORT_RUFF.txt",
        "H8_TRANSPORT_PYRIGHT.txt",
    }
