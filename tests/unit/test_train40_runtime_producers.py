"""Protect the graph core's source equivalence and traced-away input guards."""

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
    overlapping_endpoints,
    reuse_endpoint_encoders,
    trace_core,
    validate_async_blocks,
    validate_inputs,
)


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
