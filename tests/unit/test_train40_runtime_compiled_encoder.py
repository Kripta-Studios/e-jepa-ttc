"""CPU checks of ownership, ABI and compiler intent; no CUDA admission claim."""

from unittest.mock import patch

import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.train40_runtime.compiled_encoder import (
    CompiledEndpointEncoder,
    clone_producer_models_for_compilation,
    encoder_compile_options,
)


def model():
    return (
        CausalScaleTTC(
            CausalScaleTTCConfig(in_channels=12, hidden_dim=8, residual_depth=1, dropout=0.0)
        )
        .eval()
        .requires_grad_(False)
    )


def test_private_wrapper_preserves_all_outputs_and_changed_inputs():
    torch.manual_seed(91)
    source = model().encoder
    proposed = CompiledEndpointEncoder(source, compile_backend=None)
    with torch.inference_mode():
        for values in (torch.randn(2, 12, 16, 16), torch.zeros(1, 12, 16, 16)):
            expected = source(values, return_dense_features=True)
            actual = proposed(values, return_dense_features=True)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
            assert proposed(values)[2] is None
    for name, value in source.state_dict().items():
        copied = proposed.core.encoder.state_dict()[name]
        assert torch.equal(value, copied)
        assert value.data_ptr() != copied.data_ptr()
    before = next(source.parameters()).clone()
    next(proposed.parameters()).zero_()
    assert torch.equal(before, next(source.parameters()))


def test_lazy_compile_requests_fixed_shape_fp32_options_without_mutating_source():
    source = model().encoder
    with patch("torch.compile", side_effect=lambda core, **kwargs: core) as compile_call:
        proposed = CompiledEndpointEncoder(source)
    args, kwargs = compile_call.call_args
    assert args[0] is proposed.core
    assert kwargs == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": encoder_compile_options(),
    }
    assert "_compiled_callable" not in proposed._modules
    assert len(list(proposed.parameters())) == len(list(source.parameters()))


def test_producer_copy_is_private_and_other_heads_are_shared_without_mutation():
    source, head = model(), torch.nn.Linear(2, 1).eval().requires_grad_(False)
    original_encoder = source.encoder
    with patch("torch.compile", side_effect=lambda core, **kwargs: core):
        result = clone_producer_models_for_compilation({"E_A5_MATCHED": source, "PAIR": head})
    assert result["PAIR"] is head
    assert source.encoder is original_encoder
    assert isinstance(result["E_A5_MATCHED"].encoder, CompiledEndpointEncoder)
    source_pointers = {x.data_ptr() for x in source.parameters()}
    assert not source_pointers.intersection(
        x.data_ptr() for x in result["E_A5_MATCHED"].parameters()
    )


@pytest.mark.parametrize("invalid", ["training", "grad", "half"])
def test_rejects_unfrozen_or_non_fp32_source(invalid):
    source = model().encoder
    if invalid == "training":
        source.train()
    elif invalid == "grad":
        source.requires_grad_(True)
    else:
        source.half()
    with pytest.raises(ValueError):
        CompiledEndpointEncoder(source, compile_backend=None)


@pytest.mark.parametrize("backends", ["", "ATEN,ATEN", "CUDNN", "TRITON,", "ATEN, TRITON"])
def test_invalid_backend_selection_is_rejected(backends):
    with pytest.raises(ValueError, match="conv_backends"):
        encoder_compile_options(backends)


def test_jit_nesting_cannot_silently_disable_compilation():
    proposed = CompiledEndpointEncoder(model().encoder, compile_backend=None)
    with (
        patch("torch.jit.is_tracing", return_value=True),
        pytest.raises(RuntimeError, match="nested"),
    ):
        proposed(torch.zeros(1, 12, 16, 16))


@pytest.mark.parametrize("changed", [float("nan"), float("inf"), 1.01])
def test_benchmark_rejects_nonfinite_and_outside_tolerance_outputs(changed):
    from scripts.benchmark_train40_encoder import tensor_parity

    expected = (torch.ones(2), torch.ones(2), torch.ones(2))
    actual = tuple(item.clone() for item in expected)
    actual[1][0] = changed
    report = tensor_parity(actual, expected)
    assert not report["passed"]
    assert report["outputs"][1]["outside_tolerance"] == 1
