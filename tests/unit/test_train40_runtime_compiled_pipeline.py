"""CPU proofs for opaque compilation and unchanged guards; no CUDA admission."""

import gc
import warnings

import numpy as np
import pytest
import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.train40_runtime.compiled_pipeline import (
    OPERATOR_NAME,
    CompiledTransportProducerRuntime,
    OpaqueCompiledTransport,
    attach_opaque_compiled_transport,
)
from operational.train40_runtime.compiled_transport import TransportPairTensorCore
from operational.train40_runtime.producers import (
    AsyncExpertBlock,
    ExpertBlockTensorCore,
    ProducerRuntime,
    validate_async_blocks,
)


def _model(modality="event", pyramid=False):
    return (
        CausalScaleTTC(
            CausalScaleTTCConfig(
                modality=modality,
                in_channels=12 if modality == "event" else 3,
                hidden_dim=8,
                residual_depth=1,
                dropout=0.0,
                transport_enabled=True,
                transport_radius=1,
                transport_fine_radius=1,
                transport_coarse_radius=1,
                transport_mode="adaptive_pyramid" if pyramid else "legacy",
                deterministic_transport_sampling=True,
            )
        )
        .eval()
        .requires_grad_(False)
    )


class _Pair(nn.Module):
    def forward(self, batch):
        return batch.features[:, 0] * 0 + 0.03


def _models(modality="R"):
    names = (
        ("R_A5", "R_C2F", "PAIR_R")
        if modality == "R"
        else ("E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED")
    )
    sensor_modality = "rgb" if modality == "R" else "event"
    return {
        names[0]: _model(sensor_modality),
        names[1]: _model(sensor_modality, True),
        names[2]: _Pair(),
    }


def test_opaque_operator_really_calls_compiler_after_jit_trace_and_changed_inputs():
    torch.manual_seed(181)
    core = TransportPairTensorCore(1, 0.02, deterministic_sampling=True)
    wrapper = OpaqueCompiledTransport(core, backend="eager")
    inputs = (
        torch.rand(2, 8, 8, 8),
        torch.rand(2, 8, 8, 8),
        torch.rand(2, 1, 8, 8),
        torch.rand(2, 1, 8, 8),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, inputs, check_trace=False)
        assert OPERATOR_NAME in str(traced.inlined_graph)
        before = wrapper.diagnostics()["dispatcher_calls"]
        for values in (inputs, tuple(x * 0 for x in inputs), tuple(x.flip(-1) for x in inputs)):
            actual, expected = traced(*values), core(*values)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert wrapper.diagnostics()["dispatcher_calls"] == before + 3
    assert wrapper.diagnostics()["nested_jit_tracing_calls"] == 0
    with torch.inference_mode():
        # This JIT-only wrapper does not claim outer Dynamo/export support.
        # Check its mutation/aliasing schema without requesting a fake kernel.
        checked = torch.library.opcheck(
            torch.ops.train40_compiled_pipeline.transport_pair.default,
            (*inputs, wrapper.handle),
            test_utils=("test_schema",),
        )
    assert checked == {"test_schema": "SUCCESS"}


@pytest.mark.parametrize("modality", ["E", "R"])
def test_compiled_expert_block_preserves_outputs_state_and_guard(modality):
    torch.manual_seed(182)
    models = _models(modality)
    state = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    proposed = {
        name: attach_opaque_compiled_transport(model, backend="eager")
        if isinstance(model, CausalScaleTTC)
        else model
        for name, model in models.items()
    }
    sensor = torch.rand(1, 3, 12 if modality == "E" else 3, 32, 32)
    delta = torch.full((1, 2), 0.1)
    core = ExpertBlockTensorCore(proposed, modality).eval()
    oracle = ExpertBlockTensorCore(models, modality).eval()
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(core, (sensor, delta), check_trace=False)
        assert OPERATOR_NAME in str(traced.inlined_graph)
        for value in (sensor, sensor.flip(-1), sensor * 0):
            actual, expected = traced(value, delta), oracle(value, delta)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
            assert all(bool(torch.isfinite(item).all()) for item in actual)
    for name, model in models.items():
        assert all(
            torch.equal(state[name][key], value) for key, value in model.state_dict().items()
        )
        assert all(
            a is b for a, b in zip(model.parameters(), proposed[name].parameters(), strict=True)
        )


@pytest.mark.parametrize("invalid", ["clock_zero", "clock_nan", "rgb_range", "sensor_nan"])
def test_opaque_transport_trace_retains_false_guard_and_rejection(invalid):
    models = {
        name: attach_opaque_compiled_transport(model, backend="eager")
        if isinstance(model, CausalScaleTTC)
        else model
        for name, model in _models().items()
    }
    sensor, delta = torch.rand(1, 3, 3, 32, 32), torch.full((1, 2), 0.1)
    core = ExpertBlockTensorCore(models, "R").eval()
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(core, (sensor, delta), check_trace=False)
        if invalid == "clock_zero":
            delta[0, 0] = 0
        elif invalid == "clock_nan":
            delta[0, 0] = float("nan")
        elif invalid == "rgb_range":
            sensor[0, 0, 0, 0, 0] = 1.01
        else:
            sensor[0, 0, 0, 0, 0] = float("nan")
        raw, experts, guard = traced(sensor, delta)
        assert not bool(guard)
        assert bool(torch.isfinite(raw).all()) and bool(torch.isfinite(experts).all())
        with pytest.raises(ValueError, match="guard failed"):
            validate_async_blocks(AsyncExpertBlock(raw, experts, guard, "R"))


def test_orphaned_trace_fails_explicitly_instead_of_using_stale_compiled_executor():
    core = TransportPairTensorCore(1, 0.02, deterministic_sampling=True)
    wrapper = OpaqueCompiledTransport(core, backend="eager")
    inputs = (
        torch.rand(2, 8, 8, 8),
        torch.rand(2, 8, 8, 8),
        torch.rand(2, 1, 8, 8),
        torch.rand(2, 1, 8, 8),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, inputs, check_trace=False)
        del wrapper
        gc.collect()
        with pytest.raises((RuntimeError, torch.jit.Error), match="owner expired"):
            traced(*inputs)


def test_private_transport_view_retains_rebound_rgb_reuse_and_overlap_guard():
    models = _models()
    proposed = {
        name: attach_opaque_compiled_transport(model, backend="eager")
        if isinstance(model, CausalScaleTTC)
        else model
        for name, model in models.items()
    }
    core = ExpertBlockTensorCore(proposed, "R", reuse_shape=(5, 3), history_group_size=5)
    oracle = ExpertBlockTensorCore(models, "R", reuse_shape=(5, 3), history_group_size=5)
    chain = torch.rand(7, 3, 32, 32)
    sensor = torch.stack([chain[index : index + 3] for index in range(5)])
    delta = torch.full((5, 2), 0.1)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(core, (sensor, delta), check_trace=False)
        actual, expected = traced(sensor, delta), oracle(sensor, delta)
        assert OPERATOR_NAME in str(traced.inlined_graph)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        sensor[1, 0, 0, 0, 0] = 0.99
        raw, experts, guard = traced(sensor, delta)
        assert not bool(guard)
        with pytest.raises(ValueError, match="guard failed"):
            validate_async_blocks(AsyncExpertBlock(raw, experts, guard, "R"))


def test_private_transport_view_keeps_original_direct_clock_guard():
    model = attach_opaque_compiled_transport(_model(), backend="eager")
    with torch.inference_mode(), pytest.raises(ValueError, match="strictly positive"):
        model(torch.rand(1, 3, 12, 32, 32), torch.zeros(1, 2))


def test_executed_roi_snapshot_preserves_original_helpers(tmp_path):
    from scripts.benchmark_train40_compiled_pipeline import snapshot_roi_helpers
    from scripts.benchmark_train40_multiobject import artificial_regions, reference_rgb

    helpers, path, binding = snapshot_roi_helpers(tmp_path)
    original = artificial_regions((4, 5, 20, 21), (3, 4, 24, 25), 20)
    proposed = helpers.artificial_regions((4, 5, 20, 21), (3, 4, 24, 25), 20)
    assert [vars(item) for item in original] == [vars(item) for item in proposed]
    frame = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    history = helpers.RawRGBHistory(
        200000, {i: frame for i in range(3)}, {i: 100000 * i for i in range(3)}, [(0, 1, 2)]
    )
    expected = reference_rgb(history, original[0])
    actual = helpers.reference_rgb(history, proposed[0])
    assert np.array_equal(actual[0][0], expected[0][0])
    assert actual[0][1] == expected[0][1]
    assert path.is_file() and len(binding["helpers"]) == 5


def test_compiler_inventory_tracks_cache_bytes_separately_from_results(tmp_path):
    from scripts.benchmark_train40_compiled_pipeline import compiler_cache_inventory

    cache = tmp_path / "triton_cache"
    cache.mkdir()
    (cache / "kernel.ptx").write_bytes(b"first")
    (tmp_path / "summary.json").write_text("result", encoding="utf-8")
    before = compiler_cache_inventory(tmp_path)
    (cache / "kernel.ptx").write_bytes(b"second")
    after = compiler_cache_inventory(tmp_path)
    assert len(before) == len(after) == 1
    assert next(iter(before.values()))["sha256"] != next(iter(after.values()))["sha256"]


def test_async_ticket_retains_compiler_owner_until_consumption(monkeypatch):
    marker = object()
    original = AsyncExpertBlock(
        torch.zeros(1, 17), torch.zeros(1, 3), torch.tensor(True), "E", (marker,)
    )
    monkeypatch.setattr(ProducerRuntime, "expert_block_async", lambda *args, **kwargs: original)
    runtime = object.__new__(CompiledTransportProducerRuntime)
    actual = runtime.expert_block_async("E", torch.empty(0), torch.empty(0), history_group_size=4)
    assert actual._lifetime == (marker, runtime)
    assert actual.raw is original.raw and actual.experts is original.experts
    assert actual.guard is original.guard
