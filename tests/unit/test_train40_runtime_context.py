"""Resident context parity against the frozen native predictor, without sensor IO."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.final_refit.test12 import FinalV16Train40Predictor
from operational.final_refit.training import PhaseOnlyFinal
from operational.train40_runtime.context import HEAD, ContextRuntime


@pytest.fixture
def models_and_norms():
    torch.manual_seed(90210)
    models = {
        "E_CTX_MATCHED": TemporalRefiner(TemporalConfig(hidden=160)),
        "R_CTX": TemporalRefiner(TemporalConfig(hidden=160)),
        HEAD: PhaseOnlyFinal(),
    }
    # Nonzero trained-like output heads exercise all meaningful dependencies.
    for model in models.values():
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if any(token in name for token in ("location", "output")):
                    parameter.normal_(mean=0.0, std=0.04)
        model.eval().requires_grad_(False)
    norms = {name: (torch.randn(17), torch.rand(17) + 0.25) for name in ("event", "rgb")}
    return models, norms


def observations(event_count, rgb_count):
    query_us = 8_000_011
    event_anchors = [query_us - 50_000 * index for index in range(event_count - 1, -1, -1)]
    event_raw = torch.randn(event_count, 17)
    event_experts = torch.randn(event_count, 3) * 0.03
    rgb_raw = torch.randn(rgb_count, 17)
    rgb_experts = torch.randn(rgb_count, 3) * 0.03
    metadata = []
    for index in range(rgb_count):
        anchor = query_us - (rgb_count - index - 1) * 50_000
        metadata.append(
            {
                "first_sensor_ts_us": anchor - 200_000,
                "anchor_us": anchor,
                "available_us": query_us,
            }
        )
    return event_raw, event_experts, event_anchors, rgb_raw, rgb_experts, metadata, query_us


def oracle(models, norms, arguments):
    event_raw, event_experts, anchors, rgb_raw, rgb_experts, metadata, query_us = arguments
    native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
    native.device = next(models["E_CTX_MATCHED"].parameters()).device
    native.models, native.normalizers = models, norms
    native.inputs = {"event_observations": {}, "rgb_observations": {}}
    native.event_values, native.rgb_values = {}, {}
    query = {"query_time_us": query_us, "event_history": [], "rgb_history": []}
    for index, anchor in enumerate(anchors):
        identity = f"E:{anchor}"
        query["event_history"].append(identity)
        native.inputs["event_observations"][identity] = {
            "anchor_us": anchor,
            "available_us": query_us,
        }
        native.event_values[identity] = (
            event_raw[index].detach().cpu().numpy(),
            event_experts[index].detach().cpu().numpy(),
        )
    for index, item in enumerate(metadata):
        if (
            item["available_us"] > query_us
            or not 0 <= query_us - item["first_sensor_ts_us"] <= 650000
        ):
            continue
        identity = f"R:{item['anchor_us']}"
        query["rgb_history"].append(identity)
        native.inputs["rgb_observations"][identity] = item
        native.rgb_values[identity] = (
            rgb_raw[index].detach().cpu().numpy(),
            rgb_experts[index].detach().cpu().numpy(),
        )
    query["rgb_history"] = query["rgb_history"][-8:]
    outputs = {}

    def hook(name):
        def capture(module, inputs, output):
            outputs[name] = output

        return capture

    handles = [
        models[name].register_forward_hook(hook(name)) for name in ("E_CTX_MATCHED", "R_CTX")
    ]
    try:
        prediction = native.predict(query, native.inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "predictions": {
            "PHASE_ONLY_TRAIN40_S7": prediction,
            "E_CTX_TRAIN40": float(phase_to_ttc(outputs["E_CTX_MATCHED"]["point_phase"])[0]),
            "R_CTX_TRAIN40": (
                float(phase_to_ttc(outputs["R_CTX"]["point_phase"])[0])
                if "R_CTX" in outputs
                else None
            ),
        },
        "history_lengths": {"event": len(anchors), "rgb": len(query["rgb_history"])},
        "rgb_available": bool(query["rgb_history"]),
    }


@pytest.mark.parametrize("event_count,rgb_count", [(1, 0), (4, 1), (4, 5), (8, 8), (2, 10)])
def test_resident_context_matches_frozen_predictor(models_and_norms, event_count, rgb_count):
    models, norms = models_and_norms
    arguments = observations(event_count, rgb_count)
    reference = oracle(models, norms, arguments)
    before = {
        name: {key: tensor.clone() for key, tensor in model.state_dict().items()}
        for name, model in models.items()
    }
    rng = torch.get_rng_state().clone()
    runtime = ContextRuntime(models, norms, "cpu")
    assert runtime.predict(*arguments) == reference
    assert torch.equal(rng, torch.get_rng_state())
    for name, model in models.items():
        assert all(
            torch.equal(value, before[name][key]) for key, value in model.state_dict().items()
        )
    assert not runtime.captures


def test_rgb_causal_selection_and_query_delay_match(models_and_norms):
    models, norms = models_and_norms
    arguments = list(observations(4, 8))
    metadata = arguments[5]
    metadata[0]["first_sensor_ts_us"] = arguments[6] - 650001
    metadata[-1]["available_us"] = arguments[6] + 1
    runtime = ContextRuntime(models, norms, "cpu")
    assert runtime.predict(*arguments) == oracle(models, norms, arguments)


def test_feature_tensors_stay_resident_until_final_small_transfer(models_and_norms):
    models, norms = models_and_norms
    runtime = ContextRuntime(models, norms, "cpu")
    arguments = observations(4, 5)
    original = torch.from_numpy
    arrays = []

    def record(array):
        arrays.append(array.shape)
        return original(array)

    with patch("torch.from_numpy", side_effect=record):
        result = runtime.predict(*arguments)
    assert arrays == [(4, 4), (5, 4)]
    assert result["rgb_available"]


@pytest.mark.parametrize("kind", ["feature_nan", "expert_nan", "feature_inf", "future", "reversed"])
def test_resident_context_rejects_invalid_input(models_and_norms, kind):
    models, norms = models_and_norms
    runtime = ContextRuntime(models, norms, "cpu")
    arguments = list(observations(4, 5))
    if kind == "feature_nan":
        arguments[0][0, 0] = np.nan
    elif kind == "expert_nan":
        arguments[1][-1, 0] = np.nan
    elif kind == "feature_inf":
        arguments[0][0, 0] = np.inf
    elif kind == "future":
        arguments[2][-1] = arguments[-1] + 1
    else:
        arguments[2] = list(reversed(arguments[2]))
    with pytest.raises(ValueError):
        runtime.predict(*arguments)


def test_cpu_cuda_graph_request_fails_explicitly(models_and_norms):
    models, norms = models_and_norms
    with pytest.raises(ValueError, match="CUDA device"):
        ContextRuntime(models, norms, "cpu", graphs=True)


@pytest.mark.cuda_fp32
@pytest.mark.skipif(
    os.environ.get("EJEPA_TRAIN40_CUDA_TESTS") != "1",
    reason="Opt in after reserving the GPU: EJEPA_TRAIN40_CUDA_TESTS=1",
)
def test_context_graph_replay_matches_oracle_on_changed_inputs(models_and_norms, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    models, norms = models_and_norms
    models = {key: model.to("cuda") for key, model in models.items()}
    norms = {key: (mean.cuda(), scale.cuda()) for key, (mean, scale) in norms.items()}
    runtime = ContextRuntime(models, norms, "cuda", graphs=True)
    for event_count, rgb_count in ((4, 5), (1, 1), (8, 8), (4, 0), (1, 0), (4, 5)):
        arguments = observations(event_count, rgb_count)
        arguments = tuple(
            value.cuda() if isinstance(value, torch.Tensor) else value for value in arguments
        )
        expected = oracle(models, norms, arguments)
        actual = runtime.predict(*arguments)
        assert actual["history_lengths"] == expected["history_lengths"]
        assert actual["rgb_available"] == expected["rgb_available"]
        for name, value in expected["predictions"].items():
            if value is None:
                assert actual["predictions"][name] is None
            else:
                np.testing.assert_allclose(actual["predictions"][name], value, rtol=1e-5, atol=1e-5)
    assert set(runtime.captures) == {False, True}
