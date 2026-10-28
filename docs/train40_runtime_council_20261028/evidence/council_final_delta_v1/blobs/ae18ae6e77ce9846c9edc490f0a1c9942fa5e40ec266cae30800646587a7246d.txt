"""Protect the graph core's source equivalence and traced-away input guards."""

import os
from types import MethodType
from unittest.mock import patch

import pytest
import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.rgb_port import transfer as engine
from operational.rgb_port.train_producers import producer_features
from operational.rgb_port_streaming.contracts import NativeObservation
from operational.train40_runtime.producers import (
    FEATURE_NAMES,
    AsyncExpertBlock,
    ExpertBlockTensorCore,
    ProducerRuntime,
    ProducerTensorCore,
    clone_producer_models_for_layout,
    overlapping_endpoints,
    reuse_endpoint_encoders,
    trace_core,
    validate_async_blocks,
    validate_inputs,
)
from scripts.benchmark_train40_producer_variants import _route, tensor_comparison


def _model(modality="event"):
    return (
        CausalScaleTTC(
            CausalScaleTTCConfig(
                modality=modality,
                in_channels=12 if modality == "event" else 3,
                hidden_dim=8,
                transport_enabled=True,
                transport_radius=1,
                dropout=0.0,
            )
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("modality,steps", [("event", 3), ("rgb", 2), ("rgb", 3)])
def test_trace_retains_original_features_for_changed_inputs(modality, steps):
    torch.manual_seed(61)
    model = _model(modality)
    shape = (1, steps, model.config.in_channels, 16, 16)
    sensor = torch.rand(shape)
    delta = torch.full((1, steps - 1), 0.1)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        traced, messages = trace_core(model, sensor, delta)
        assert messages  # Guard elimination remains auditable.
        for changed in (sensor, torch.zeros_like(sensor), sensor.flip(-1)):
            validate_inputs(model, changed, delta)
            original = producer_features(model, changed, delta)
            actual = traced(changed, delta)
            for key, value in zip(FEATURE_NAMES, actual, strict=True):
                assert torch.equal(value, original[key]), key
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())


@pytest.mark.parametrize("invalid", ["clock_zero", "clock_nan", "rgb_range", "sensor_nan"])
def test_value_guards_survive_tracing(invalid):
    model = _model("rgb")
    sensor = torch.rand(1, 3, 3, 16, 16)
    delta = torch.full((1, 2), 0.1)
    if invalid == "clock_zero":
        delta[0, 0] = 0
    elif invalid == "clock_nan":
        delta[0, 0] = float("nan")
    elif invalid == "rgb_range":
        sensor[0, 0, 0, 0, 0] = 1.01
    else:
        sensor[0, 0, 0, 0, 0] = float("nan")
    with torch.inference_mode(), pytest.raises(ValueError, match="Nonfinite"):
        validate_inputs(model, sensor, delta)


class _Pair(nn.Module):
    def forward(self, batch):
        return batch.features[:, 0] * 0 + 0.03


def test_eager_expert_block_matches_frozen_engine():
    models = {"E_A5_MATCHED": _model(), "E_C2F_MATCHED": _model(), "PAIR_E_MATCHED": _Pair()}
    sensor = torch.rand(1, 3, 12, 16, 16)
    delta = torch.full((1, 2), 0.1)
    runtime = ProducerRuntime(models, "cpu", graphs=False)
    with torch.inference_mode(), patch.object(engine, "ProducerObservation", NativeObservation):
        expected = engine._expert_block(models, "E", sensor, delta)
        actual = runtime.expert_block("E", sensor, delta)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert runtime.diagnostics() == []


def test_graph_mode_cannot_silently_fall_back_to_cpu():
    with pytest.raises(ValueError, match="CUDA device"):
        ProducerRuntime({}, "cpu", graphs=True)


@pytest.mark.parametrize("modality,steps", [("E", 3), ("R", 2), ("R", 3)])
def test_float64_block_core_matches_historical_cpu_phase_mapping(modality, steps):
    sensor_modality = "event" if modality == "E" else "rgb"
    names = (
        ("E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED")
        if modality == "E"
        else ("R_A5", "R_C2F", "PAIR_R")
    )
    models = {
        names[0]: _model(sensor_modality),
        names[1]: _model(sensor_modality),
        names[2]: _Pair(),
    }
    sensor = torch.rand(2, steps, 12 if modality == "E" else 3, 16, 16)
    delta = torch.full((2, steps - 1), 0.1)
    with torch.inference_mode():
        expected = ProducerRuntime(models, "cpu", graphs=False).expert_block(
            modality, sensor, delta
        )
        actual = ExpertBlockTensorCore(models, modality)(sensor, delta)
    assert bool(actual[2])
    assert all(torch.equal(a, b) for a, b in zip(actual[:2], expected, strict=True))


def test_tensor_core_has_only_consumed_features():
    model = _model()
    with torch.inference_mode():
        result = ProducerTensorCore(model)(torch.rand(1, 3, 12, 16, 16), torch.full((1, 2), 0.1))
    assert len(result) == len(FEATURE_NAMES)
    assert result[0].shape == (1, 128)
    assert result[-1].dtype == torch.bool


@pytest.mark.parametrize("steps", [2, 3])
def test_endpoint_reuse_preserves_state_and_restores_full_temporal_order(steps):
    torch.manual_seed(73)
    model = _model()
    original_encoder = model.encoder
    batch = 4
    chain = torch.rand(batch + steps - 1, 12, 16, 16)
    sensor = torch.stack([chain[i : i + steps] for i in range(batch)])
    delta = torch.full((batch, steps - 1), 0.1)
    assert overlapping_endpoints(sensor)
    view = reuse_endpoint_encoders(model, batch, steps)
    assert model.encoder is original_encoder
    assert model._modules is not view._modules
    assert {id(p) for p in model.parameters()} == {id(p) for p in view.parameters()}
    encoded_sizes = []
    hook = original_encoder.register_forward_pre_hook(
        lambda _module, inputs: encoded_sizes.append(len(inputs[0]))
    )
    try:
        with torch.inference_mode():
            expected = ProducerTensorCore(model)(sensor, delta)
            actual = ProducerTensorCore(view)(sensor, delta)
    finally:
        hook.remove()
    assert encoded_sizes == [batch * steps, batch + steps - 1]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    broken = sensor.clone()
    broken[-1, 0, 0, 0, 0] += 1
    assert not overlapping_endpoints(broken)
    assert not overlapping_endpoints(sensor[:1])


def test_endpoint_reuse_requires_whole_block_graph_mode():
    with pytest.raises(ValueError, match="whole expert-block"):
        ProducerRuntime({}, "cuda", graphs=True, reuse_encodings=True)


@pytest.mark.parametrize(
    "invalid", ["clock_zero", "clock_nan", "rgb_range", "sensor_nan", "overlap"]
)
def test_async_core_guard_rejects_unsafe_inputs_before_publication(invalid):
    models = {"R_A5": _model("rgb"), "R_C2F": _model("rgb"), "PAIR_R": _Pair()}
    chain = torch.rand(4, 3, 16, 16)
    sensor = torch.stack((chain[:3], chain[1:]))
    delta = torch.full((2, 2), 0.1)
    if invalid == "clock_zero":
        delta[0, 0] = 0
    elif invalid == "clock_nan":
        delta[0, 0] = float("nan")
    elif invalid == "rgb_range":
        sensor[0, 0, 0, 0, 0] = 1.01
    elif invalid == "sensor_nan":
        sensor[0, 0, 0, 0, 0] = float("nan")
    else:
        sensor[-1, 0, 0, 0, 0] = 0.9 - sensor[-1, 0, 0, 0, 0]
    with torch.inference_mode():
        raw, experts, guard = ExpertBlockTensorCore(models, "R", reuse_shape=(2, 3))(sensor, delta)
    assert not bool(guard)
    assert torch.isfinite(raw).all() and torch.isfinite(experts).all()
    ticket = AsyncExpertBlock(raw, experts, guard, "R")
    with pytest.raises(ValueError, match="guard failed"):
        validate_async_blocks(ticket)


def test_grouped_async_guard_checks_every_ticket():
    values = torch.zeros(1)
    event = AsyncExpertBlock(values, values, torch.tensor(True), "E")
    rgb = AsyncExpertBlock(values, values, torch.tensor(False), "R")
    ProducerRuntime.validate_async_blocks(event)
    with pytest.raises(ValueError, match="R"):
        ProducerRuntime.validate_async_blocks(event, rgb)
    with pytest.raises(ValueError, match="At least one"):
        validate_async_blocks()


def test_async_mode_cannot_silently_fall_back_to_eager():
    runtime = ProducerRuntime({}, "cpu", graphs=False)
    with pytest.raises(ValueError, match="whole-block"):
        runtime.expert_block_async("E", torch.zeros(1), torch.zeros(1))


def test_grouped_endpoint_reuse_keeps_each_objects_independent_history():
    torch.manual_seed(79)
    objects, histories, steps = 3, 4, 3
    chains = torch.rand(objects, histories + steps - 1, 12, 16, 16)
    sensor = torch.stack(
        [chains[obj, i : i + steps] for obj in range(objects) for i in range(histories)]
    )
    assert overlapping_endpoints(sensor, history_group_size=histories)
    assert not overlapping_endpoints(sensor)  # Cross-object boundaries are intentionally distinct.
    model = _model()
    view = reuse_endpoint_encoders(model, len(sensor), steps, history_group_size=histories)
    counts = []
    hook = model.encoder.register_forward_pre_hook(
        lambda _module, inputs: counts.append(len(inputs[0]))
    )
    delta = torch.full((len(sensor), steps - 1), 0.1)
    try:
        with torch.inference_mode():
            expected = ProducerTensorCore(model)(sensor, delta)
            actual = ProducerTensorCore(view)(sensor, delta)
    finally:
        hook.remove()
    assert counts == [objects * histories * steps, objects * (histories + steps - 1)]
    assert actual[0].shape[0] == objects * histories
    assert not torch.equal(actual[0][0], actual[0][-1])
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_grouped_overlap_guard_rejects_one_broken_object():
    chains = torch.rand(2, 4, 3, 16, 16)
    sensor = torch.stack([chains[obj, i : i + 3] for obj in range(2) for i in range(2)])
    models = {"R_A5": _model("rgb"), "R_C2F": _model("rgb"), "PAIR_R": _Pair()}
    core = ExpertBlockTensorCore(models, "R", reuse_shape=(4, 3), history_group_size=2)
    delta = torch.full((4, 2), 0.1)
    with torch.inference_mode():
        assert bool(core(sensor, delta)[2])
        sensor[3, 0, 0, 0, 0] = 0.5 * sensor[3, 0, 0, 0, 0]
        assert not bool(core(sensor, delta)[2])


@pytest.mark.parametrize("size", [0, -1, 3, True])
def test_group_size_must_partition_observation_batch(size):
    with pytest.raises(ValueError, match="history_group_size"):
        overlapping_endpoints(torch.zeros(4, 3, 12, 16, 16), history_group_size=size)


@pytest.mark.parametrize("channels_last", [False, True])
def test_layout_copy_cannot_mutate_original_values_or_strides(channels_last):
    torch.manual_seed(113)
    model, pair = _model(), _Pair()
    original = {name: (value.clone(), value.stride()) for name, value in model.state_dict().items()}
    cloned = clone_producer_models_for_layout(
        {"E_A5_MATCHED": model, "PAIR_E_MATCHED": pair}, channels_last=channels_last
    )
    proposed = cloned["E_A5_MATCHED"]
    assert cloned["PAIR_E_MATCHED"] is pair
    assert proposed is not model
    assert not (
        {p.data_ptr() for p in proposed.parameters()} & {p.data_ptr() for p in model.parameters()}
    )
    for name, value in model.state_dict().items():
        before, stride = original[name]
        assert torch.equal(value, before)
        assert value.stride() == stride
    with torch.inference_mode():
        first = next(proposed.parameters())
        first.add_(1)
    assert all(torch.equal(value, original[name][0]) for name, value in model.state_dict().items())


def test_channels_last_encoder_preserves_output_contract_and_tolerance():
    torch.manual_seed(127)
    model = _model()
    proposed = clone_producer_models_for_layout({"producer": model}, channels_last=True)["producer"]
    inputs = torch.rand(3, 12, 16, 16)
    observed = []
    handle = proposed.encoder.encoder.register_forward_pre_hook(
        lambda _module, args: observed.append(
            args[0].is_contiguous(memory_format=torch.channels_last)
        )
    )
    try:
        with torch.inference_mode():
            expected = model.encoder(inputs, return_dense_features=True)
            actual = proposed.encoder(inputs, return_dense_features=True)
    finally:
        handle.remove()
    assert observed == [True]
    for before, after in zip(expected, actual, strict=True):
        assert before is not None and after is not None and after.is_contiguous()
        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("invalid", ["training", "gradient", "precision"])
def test_layout_experiments_require_frozen_fp32_producers(invalid):
    model = _model()
    if invalid == "training":
        model.train()
    elif invalid == "gradient":
        next(model.parameters()).requires_grad_(True)
    else:
        model.double()
    with pytest.raises(ValueError, match="frozen|FP32"):
        clone_producer_models_for_layout({"producer": model}, channels_last=True)


def test_producer_probe_reports_small_drift_without_claiming_exactness():
    result = tensor_comparison(torch.tensor([1.000001, 2.0]), torch.tensor([1.0, 2.0]))
    assert result["passed"] and not result["exact"]
    assert result["different_elements"] == 1
    assert 0 < result["max_abs"] < 1e-5


def test_producer_probe_rejects_equal_nonfinite_values():
    result = tensor_comparison(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    assert not result["passed"]
    assert result["nonfinite_pairs"] == 1


def test_producer_probe_rejects_outside_tolerance():
    result = tensor_comparison(torch.tensor([1.001]), torch.tensor([1.0]))
    assert not result["passed"]
    assert result["outside_tolerance"] == 1


def test_endpoint_reuse_rebinds_instance_forward_to_its_own_encoder_view():
    model = _model()
    object.__setattr__(model, "_forward_impl", MethodType(CausalScaleTTC._forward_impl, model))
    view = reuse_endpoint_encoders(model, 4, 3)
    assert model._forward_impl.__self__ is model
    assert view._forward_impl.__self__ is view
    counts = []
    hook = model.encoder.register_forward_pre_hook(
        lambda _module, args: counts.append(len(args[0]))
    )
    chain = torch.rand(6, 12, 16, 16)
    sensor = torch.stack([chain[index : index + 3] for index in range(4)])
    try:
        with torch.inference_mode():
            ProducerTensorCore(view)(sensor, torch.full((4, 2), 0.1))
    finally:
        hook.remove()
    assert counts == [6]


def test_per_modality_reuse_override_preserves_legacy_defaults():
    legacy = ProducerRuntime({}, "cuda", graphs=True, block_graphs=True, reuse_encodings=True)
    assert legacy.reuse_by_modality == {"E": True, "R": True}
    selected = ProducerRuntime(
        {},
        "cuda",
        graphs=True,
        block_graphs=True,
        reuse_encodings=True,
        reuse_event_encodings=False,
    )
    assert selected.reuse_by_modality == {"E": False, "R": True}
    assert selected.reuse_encodings
    disabled = ProducerRuntime(
        {},
        "cuda",
        graphs=True,
        block_graphs=True,
        reuse_encodings=True,
        reuse_event_encodings=False,
        reuse_rgb_encodings=False,
    )
    assert not disabled.reuse_encodings


def test_per_modality_reuse_still_requires_whole_block_graphs():
    with pytest.raises(ValueError, match="whole expert-block"):
        ProducerRuntime({}, "cpu", graphs=False, reuse_event_encodings=True)


@pytest.mark.parametrize("value", ["false", 0, 1, {}])
def test_per_modality_reuse_requires_explicit_booleans(value):
    with pytest.raises(ValueError, match="boolean"):
        ProducerRuntime({}, "cpu", graphs=False, reuse_event_encodings=value)


def test_capacity_chunking_preserves_all_objects_and_history_order():
    calls = []

    class Runtime:
        def expert_block(self, modality, sensor, delta, *, history_group_size):
            calls.append((modality, len(sensor), history_group_size))
            return sensor.clone(), sensor.clone()

    event = torch.arange(80).reshape(80, 1)
    rgb = torch.arange(100).reshape(100, 1)
    result = _route(
        Runtime(),
        (event, event, rgb, rgb),
        parallel=False,
        streams=(None, None),
        object_chunk_size=6,
    )
    assert calls == [
        ("E", 24, 4),
        ("R", 30, 5),
        ("E", 24, 4),
        ("R", 30, 5),
        ("E", 24, 4),
        ("R", 30, 5),
        ("E", 8, 4),
        ("R", 10, 5),
    ]
    for actual, expected in zip(result, (event, event, rgb, rgb), strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.cuda_fp32
@pytest.mark.skipif(
    os.environ.get("EJEPA_TRAIN40_CUDA_TESTS") != "1",
    reason="Opt in after reserving the GPU: EJEPA_TRAIN40_CUDA_TESTS=1",
)
def test_first_changed_parallel_update_uses_independent_capture_workspaces(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    torch.manual_seed(193)
    models = {
        "E_A5_MATCHED": _model().cuda(),
        "E_C2F_MATCHED": _model().cuda(),
        "PAIR_E_MATCHED": _Pair().eval().cuda(),
        "R_A5": _model("rgb").cuda(),
        "R_C2F": _model("rgb").cuda(),
        "PAIR_R": _Pair().eval().cuda(),
    }
    runtime = ProducerRuntime(
        models,
        "cuda",
        graphs=True,
        block_graphs=True,
        reuse_encodings=True,
        reuse_event_encodings=False,
    )
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    with torch.inference_mode():
        # Deliberately retain the one-object captures before the two-object
        # captures, and prime on zeros. This reproduces the lifecycle that
        # exposed the first-real-update race in the real-weight service pilot.
        for objects in (1, 2):
            edelta = torch.full((objects * 4, 2), 0.1, device="cuda")
            rdelta = torch.full((objects * 5, 2), 0.1, device="cuda")
            zero = (
                torch.zeros(objects * 4, 3, 12, 16, 16, device="cuda"),
                edelta,
                torch.zeros(objects * 5, 3, 3, 16, 16, device="cuda"),
                rdelta,
            )
            _route(runtime, zero, parallel=False, streams=streams)
            for _ in range(4):
                event = torch.rand(objects, 6, 12, 16, 16, device="cuda")
                rgb = torch.rand(objects, 7, 3, 16, 16, device="cuda")
                actual_inputs = (
                    torch.stack(
                        [event[obj, i : i + 3] for obj in range(objects) for i in range(4)]
                    ),
                    edelta,
                    torch.stack([rgb[obj, i : i + 3] for obj in range(objects) for i in range(5)]),
                    rdelta,
                )
                # Parallel sees these new values first: a preceding serial
                # execution must not hide a stale first-response defect.
                parallel = _route(runtime, actual_inputs, parallel=True, streams=streams)
                serial = _route(runtime, actual_inputs, parallel=False, streams=streams)
                for actual, expected in zip(parallel, serial, strict=True):
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    captures = runtime.diagnostics()
    assert len(captures) == 4
    assert all(item["dedicated_capture_stream"] for item in captures)
    assert len({item["capture_stream_id"] for item in captures}) == 4
