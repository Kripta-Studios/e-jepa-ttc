from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from operational.final_refit.runtime_probe import (
    BoundedGraphDispatcher,
    ForwardSignature,
    schedule_signature_histogram,
    validate_case,
    validate_probe,
)


def _call_tensors(batch: int = 32, frames: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(batch, frames, 3, 4, 4), torch.ones(batch, frames - 1)


def test_forward_signature_includes_stride_dtype_device_and_keyword() -> None:
    inputs, delta = _call_tensors()
    first = ForwardSignature.from_call(inputs, delta, return_dense_features=True)
    second = ForwardSignature.from_call(inputs.transpose(-1, -2), delta, return_dense_features=True)
    third = ForwardSignature.from_call(inputs, delta, return_dense_features=False)

    assert first.batch_frames == (32, 3)
    assert first.input_dtype == "torch.float32"
    assert first.input_device == "cpu"
    assert len({first.label(), second.label(), third.label()}) == 3


def test_dispatcher_routes_only_exact_registered_signature() -> None:
    calls = {"eager": 0, "graph": 0}

    def eager(inputs: torch.Tensor, delta: torch.Tensor, **_: Any) -> torch.Tensor:
        calls["eager"] += 1
        return inputs.sum() + delta.sum()

    def graph(inputs: torch.Tensor, delta: torch.Tensor, **_: Any) -> torch.Tensor:
        calls["graph"] += 1
        return inputs.sum() + delta.sum()

    inputs, delta = _call_tensors()
    signature = ForwardSignature.from_call(inputs, delta, return_dense_features=True)
    dispatcher = BoundedGraphDispatcher(eager, max_graphs=1)
    dispatcher.register(signature, graph)

    dispatcher(inputs, delta, return_dense_features=True)
    terminal_inputs, terminal_delta = _call_tensors(batch=7, frames=2)
    dispatcher(terminal_inputs, terminal_delta, return_dense_features=True)

    assert calls == {"eager": 1, "graph": 1}
    assert dispatcher.snapshot()["registered_graphs"] == 1
    assert dispatcher.snapshot()["eager_calls"] == 1
    assert dispatcher.snapshot()["graph_calls"] == 1


def test_dispatcher_cache_is_bounded_and_never_evicts() -> None:
    inputs, delta = _call_tensors()
    first = ForwardSignature.from_call(inputs, delta, return_dense_features=True)
    dispatcher = BoundedGraphDispatcher(lambda *_args, **_kwargs: None, max_graphs=1)
    dispatcher.register(first, lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register(first, lambda *_args, **_kwargs: None)
    invalid = ForwardSignature.from_call(*_call_tensors(batch=16), return_dense_features=True)
    with pytest.raises(ValueError, match="cache is full"):
        dispatcher.register(invalid, lambda *_args, **_kwargs: None)


def test_dispatcher_rejects_non_b32_t3_graph_signature() -> None:
    inputs, delta = _call_tensors(batch=31, frames=3)
    signature = ForwardSignature.from_call(inputs, delta, return_dense_features=True)
    dispatcher = BoundedGraphDispatcher(lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="B32/T3"):
        dispatcher.register(signature, lambda *_args, **_kwargs: None)


def test_schedule_histogram_preserves_logical_boundaries_without_padding() -> None:
    source = SimpleNamespace(
        population_size=65,
        batch_keys=[("P", 3)] * 32 + [("H", 3)] * 16 + [("H", 2)] * 17,
        frame_counts=[3] * 48 + [2] * 17,
    )
    result = schedule_signature_histogram(source, list(range(65)))

    assert result["updates"] == 3
    assert result["parts"] == {
        "H:B1:T2": 1,
        "H:B16:T2": 1,
        "H:B16:T3": 1,
        "P:B32:T3": 1,
    }
    assert result["graph_eligible_examples"] == 32
    assert result["total_examples"] == 65
    assert result["padding_or_reordering"] is False


def test_schedule_histogram_rejects_non_permutation() -> None:
    source = SimpleNamespace(
        population_size=3,
        batch_keys=[("P", 3)] * 3,
        frame_counts=[3] * 3,
    )
    with pytest.raises(ValueError, match="permutation"):
        schedule_signature_histogram(source, [0, 0, 2])


def _passing_case(fit_id: str, regime: str) -> dict[str, Any]:
    compared = {"passed": True}
    return {
        "fit_id": fit_id,
        "regime": regime,
        "parity": {
            "output": compared,
            "loss": compared,
            "components": compared,
            "gradients_before_clip": compared,
            "clip_norm": compared,
            "gradients_after_clip": compared,
            "model_after_step": compared,
            "optimizer_after_step": compared,
            "scheduler_after_step": compared,
            "rng_after_step_equal": True,
        },
        "warmup_complete_state_restored": True,
        "sampler_generator_preserved": True,
        "routing": {"graph_calls": 103},
        "compile": {
            "unique_graphs": 1,
            "steady_state_new_unique_graphs": 0,
            "observed_graph_launch": True,
        },
        "peak_cuda_memory_bytes": 1024,
        "throughput": {"speedup": 1.10},
    }


def test_validate_case_fails_closed_on_recompile_or_missing_speedup() -> None:
    case = _passing_case("R_A5", "full_objective")
    case["compile"]["steady_state_new_unique_graphs"] = 1
    case["throughput"]["speedup"] = 1.0

    passed, reasons = validate_case(case, min_speedup=1.03, max_peak_bytes=2048)

    assert passed is False
    assert "steady_state_recompiled" in reasons
    assert "speedup_below_threshold" in reasons


def test_validate_case_fails_closed_when_peak_memory_is_unavailable() -> None:
    case = _passing_case("R_A5", "full_objective")
    case.pop("peak_cuda_memory_bytes")

    passed, reasons = validate_case(case, min_speedup=1.03, max_peak_bytes=2048)

    assert passed is False
    assert "peak_cuda_memory_unavailable" in reasons


def test_validate_probe_requires_both_arms_and_both_regimes() -> None:
    cases = [
        _passing_case(fit, regime)
        for fit in ("R_A5", "R_C2F")
        for regime in ("foreground_warmup", "full_objective")
    ]
    admitted = validate_probe({"cases": cases}, min_speedup=1.03, max_peak_bytes=2048)
    incomplete = validate_probe({"cases": cases[:-1]}, min_speedup=1.03, max_peak_bytes=2048)

    assert admitted["status"] == "ADMITTED"
    assert admitted["graph_training_enabled"] is True
    assert incomplete["status"] == "REJECTED"
    assert incomplete["graph_training_enabled"] is False
    assert "arm_or_regime_coverage_differs" in incomplete["reasons"]


def test_parser_help_does_not_touch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from operational.final_refit.runtime_probe import parser

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA touched")),
    )
    assert "source-descriptor" in parser().format_help()


def test_main_persists_resource_failure_when_memory_error_escapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from operational.final_refit import runtime_probe

    monkeypatch.setattr(
        runtime_probe,
        "run_probe",
        lambda _args: (_ for _ in ()).throw(MemoryError("host commit exhausted")),
    )
    output = tmp_path / "probe.json"

    code = runtime_probe.main(
        [
            "--source-descriptor",
            str(tmp_path / "unused.json"),
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 3
    assert payload["status"] == "RESOURCE_FAILURE"
    assert payload["scientific_optimizer_updates"] == 0
    assert payload["technical_work"] == "UNKNOWN_NOT_RECOVERABLE"
    assert payload["decision"]["graph_training_enabled"] is False
