"""CPU-only routing checks for the CUDA-graph admission harness."""

from pathlib import Path

import torch

from operational.train40_system.graph_launch_admission import (
    BACKEND,
    ForwardDispatcher,
    _compile_forward,
)


def test_compile_uses_only_explicit_cudagraphs_backend() -> None:
    calls = []

    def forward(value, *, return_dense_features=False):
        return value, return_dense_features

    def compiler(callable_, **kwargs):
        calls.append((callable_, kwargs))
        return callable_

    compiled, elapsed = _compile_forward(forward, compiler)
    assert compiled is forward
    assert elapsed >= 0.0
    assert calls == [
        (forward, {"backend": BACKEND, "dynamic": False, "fullgraph": False})
    ]


def test_dispatcher_compiles_only_b32_and_keeps_b8_eager() -> None:
    routes = []

    def original(inputs, delta, *, return_dense_features=False):
        routes.append(("eager", inputs.shape[0], return_dense_features))
        return delta

    def compiled(inputs, delta, *, return_dense_features=False):
        routes.append(("compiled", inputs.shape[0], return_dense_features))
        return delta

    dispatcher = ForwardDispatcher(original, compiled)
    dispatcher.compiled_enabled = True
    dispatcher(torch.zeros(32, 1), torch.ones(32), return_dense_features=True)
    dispatcher(torch.zeros(8, 1), torch.ones(8), return_dense_features=True)
    assert routes == [("compiled", 32, True), ("eager", 8, True)]
    assert dispatcher.snapshot() == {
        "compiled_calls": 1,
        "eager_calls": 1,
        "eager_tail_calls": 1,
    }


def test_admission_source_has_no_optimizer_or_default_inductor() -> None:
    source = (
        Path(__file__).parents[2]
        / "operational/train40_system/graph_launch_admission.py"
    ).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert 'BACKEND = "cudagraphs"' in source
    assert 'backend=BACKEND' in source
    assert 'backend="inductor"' not in source
    assert "relation_reuse_admission" not in source
