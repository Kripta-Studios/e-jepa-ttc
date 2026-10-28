"""CPU integration contracts for opt-in transport reuse with unchanged defaults."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch
from torch import Tensor, nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.train40_runtime import multiobject
from operational.train40_runtime.multiobject import MultiObjectTrain40Service
from operational.train40_runtime.producers import ExpertBlockTensorCore, ProducerRuntime
from scripts import benchmark_train40_paced as runner

MODES = ("original", "reverse", "reverse-summaries")


def _model(modality: Literal["event", "rgb"] = "event", pyramid: bool = False) -> CausalScaleTTC:
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
    def forward(self, batch: Any) -> Tensor:
        return batch.features[:, 0] * 0 + 0.03


def _expert_models(modality: str) -> dict[str, nn.Module]:
    names = (
        ("R_A5", "R_C2F", "PAIR_R")
        if modality == "R"
        else ("E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED")
    )
    sensor_modality: Literal["event", "rgb"] = "rgb" if modality == "R" else "event"
    return {
        names[0]: _model(sensor_modality),
        names[1]: _model(sensor_modality, pyramid=True),
        names[2]: _Pair().eval(),
    }


def test_original_producer_default_keeps_model_mapping_and_parameter_objects() -> None:
    models = {"E_A5_MATCHED": _model(), "unrelated": nn.Identity()}
    runtime = ProducerRuntime(models, "cpu", graphs=False)
    assert runtime.transport_reuse == "original"
    assert runtime.models is models
    assert runtime.models["E_A5_MATCHED"] is models["E_A5_MATCHED"]
    assert isinstance(runtime.transport_bindings, dict)


@pytest.mark.parametrize("mode", ["reverse", "reverse-summaries"])
def test_transport_adapter_shares_frozen_parameters_without_mutating_caller(mode: str) -> None:
    model = _model()
    models: dict[str, nn.Module] = {"E_A5_MATCHED": model, "unrelated": nn.Identity().eval()}
    original_modules = dict(model.named_modules())
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    runtime = ProducerRuntime(models, "cpu", graphs=False, transport_reuse=mode)
    proposed = runtime.models["E_A5_MATCHED"]
    assert runtime.transport_reuse == mode
    assert proposed is not model and models["E_A5_MATCHED"] is model
    assert runtime.models["unrelated"] is models["unrelated"]
    assert set(runtime.transport_bindings) == {"E_A5_MATCHED"}
    assert runtime.transport_bindings["E_A5_MATCHED"]
    assert dict(model.named_modules()) == original_modules
    assert set(proposed.state_dict()) == set(original_state)
    assert all(
        torch.equal(value, proposed.state_dict()[name]) for name, value in original_state.items()
    )
    assert all(a is b for a, b in zip(model.parameters(), proposed.parameters(), strict=True))


@pytest.mark.parametrize("mode", ["reverse", "reverse-summaries"])
def test_runtime_retains_adapted_parameter_lifetime_after_caller_releases_model(mode: str) -> None:
    model = _model()
    parameter_ref = weakref.ref(next(model.parameters()))
    models = {"E_A5_MATCHED": model}
    runtime = ProducerRuntime(models, "cpu", graphs=False, transport_reuse=mode)
    del models, model
    gc.collect()
    assert parameter_ref() is not None
    assert parameter_ref() is next(runtime.models["E_A5_MATCHED"].parameters())
    with torch.inference_mode():
        output = runtime.models["E_A5_MATCHED"](
            torch.rand(1, 3, 12, 32, 32), torch.full((1, 2), 0.1)
        )
    assert output.ttc_mean_seconds.shape == (1,)
    assert bool(torch.isfinite(output.ttc_mean_seconds).all())


@pytest.mark.parametrize("mode", ["", "compiled", None, 1, True])
def test_unknown_transport_mode_is_rejected_even_without_models(mode: Any) -> None:
    with pytest.raises(ValueError):
        ProducerRuntime({}, "cpu", graphs=False, transport_reuse=mode)


@pytest.mark.parametrize("mode", ["reverse", "reverse-summaries"])
def test_opt_in_transport_rejects_unfrozen_training_producers(mode: str) -> None:
    with pytest.raises(ValueError, match="frozen"):
        ProducerRuntime(
            {"E_A5_MATCHED": _model().train()}, "cpu", graphs=False, transport_reuse=mode
        )


@pytest.mark.parametrize("mode", ["reverse", "reverse-summaries"])
@pytest.mark.parametrize("modality", ["E", "R"])
def test_transport_adapter_composes_with_original_event_and_reused_rgb_batch(
    mode: str, modality: str
) -> None:
    torch.manual_seed(211)
    models = _expert_models(modality)
    runtime = ProducerRuntime(models, "cpu", graphs=False, transport_reuse=mode)
    if modality == "R":
        chain = torch.rand(7, 3, 32, 32)
        sensor = torch.stack([chain[index : index + 3] for index in range(5)])
        batch, reuse_shape, history_group_size = 5, (5, 3), 5
    else:
        sensor = torch.rand(4, 3, 12, 32, 32)
        batch, reuse_shape, history_group_size = 4, None, None
    delta = torch.full((batch, 2), 0.1)
    proposed = ExpertBlockTensorCore(
        runtime.models, modality, reuse_shape=reuse_shape, history_group_size=history_group_size
    ).eval()
    original = ExpertBlockTensorCore(
        models, modality, reuse_shape=reuse_shape, history_group_size=history_group_size
    ).eval()
    with torch.inference_mode():
        actual, expected = proposed(sensor, delta), original(sensor, delta)
        assert actual[0].shape == expected[0].shape == (batch, 17)
        assert actual[1].shape == expected[1].shape == (batch, 3)
        assert actual[2].shape == expected[2].shape == ()
        assert bool(actual[2]) and bool(expected[2])
        for value, reference in zip(actual[:2], expected[:2], strict=True):
            torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)
        if modality == "R":
            broken = sensor.clone()
            broken[1, 0, 0, 0, 0] += 0.01
            assert not bool(proposed(broken, delta)[2])


def _fake_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiobject, "BatchedContextRuntime", lambda *args, **kwargs: object())


def _injected(mode: str) -> ProducerRuntime:
    return cast(
        ProducerRuntime,
        SimpleNamespace(
            device=torch.device("cpu"),
            models={},
            block_graphs=False,
            reuse_by_modality={"E": False, "R": True},
            transport_reuse=mode,
        ),
    )


@pytest.mark.parametrize("mode", MODES)
def test_service_forwards_transport_selection_without_changing_cpu_execution(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _fake_context(monkeypatch)
    history = cast(Any, object())
    service = MultiObjectTrain40Service(
        {},
        {},
        history,
        "cpu",
        graphs=False,
        parallel_branches=False,
        transport_reuse=mode,
        overlap_event_with_rgb_preparation=False,
    )
    try:
        assert service.transport_reuse == mode
        assert service.producers.transport_reuse == mode
        assert service.history is history
        assert not service.producers.graphs
    finally:
        service.close()


@pytest.mark.parametrize("mode", MODES)
def test_matching_injected_transport_is_retained_without_rewrapping(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _fake_context(monkeypatch)
    injected = _injected(mode)
    service = MultiObjectTrain40Service(
        {},
        {},
        cast(Any, object()),
        "cpu",
        graphs=False,
        parallel_branches=False,
        producer_runtime=injected,
        transport_reuse=mode,
        overlap_event_with_rgb_preparation=False,
    )
    try:
        assert service.producers is injected
        assert service.transport_reuse == mode
    finally:
        service.close()


@pytest.mark.parametrize(
    "requested,injected",
    [("original", "reverse"), ("reverse", "original"), ("reverse-summaries", "reverse")],
)
def test_service_rejects_injected_transport_contract_mismatch(
    requested: str, injected: str
) -> None:
    with pytest.raises(ValueError, match="transport"):
        MultiObjectTrain40Service(
            {},
            {},
            cast(Any, object()),
            "cpu",
            graphs=False,
            parallel_branches=False,
            producer_runtime=_injected(injected),
            transport_reuse=requested,
            overlap_event_with_rgb_preparation=False,
        )


def test_service_requires_explicit_injected_transport_identity() -> None:
    injected = _injected("original")
    delattr(injected, "transport_reuse")
    with pytest.raises(ValueError, match="transport"):
        MultiObjectTrain40Service(
            {},
            {},
            cast(Any, object()),
            "cpu",
            graphs=False,
            parallel_branches=False,
            producer_runtime=injected,
            overlap_event_with_rgb_preparation=False,
            transport_reuse="original",
        )


def _arguments(*extra: str) -> Any:
    return runner.parser().parse_args(
        [
            "--candidate",
            "candidate",
            "--reference",
            "reference",
            "--base-manifest",
            "base.json",
            "--manifest",
            "replay.json",
            "--output",
            "fresh",
            "--sequence",
            "sequence",
            *extra,
        ]
    )


def test_paced_default_and_cache_hash_bind_explicit_transport_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda: pytest.fail("GPU touched"))
    monkeypatch.setattr(runner.torch.backends.cudnn, "version", lambda: 90000)
    binding = {"candidate": {"weights": "unchanged"}, "sources": {"runtime": "same-source-hash"}}
    default = _arguments()
    assert default.transport_reuse == "reverse-summaries"
    assert runner.service_options(default)["transport_reuse"] == "reverse-summaries"
    baseline = runner.cache_identity(binding, default)
    assert baseline == runner.cache_identity(
        binding, _arguments("--transport-reuse", "reverse-summaries")
    )
    alternatives = []
    for mode in ("original", "reverse"):
        args = _arguments("--transport-reuse", mode)
        assert runner.service_options(args)["transport_reuse"] == mode
        identity = runner.cache_identity(binding, args)
        assert identity["model_sha256"] == baseline["model_sha256"]
        alternatives.append(identity["runtime_sha256"])
    assert len({baseline["runtime_sha256"], *alternatives}) == 3


def test_paced_cli_rejects_unregistered_transport_mode() -> None:
    with pytest.raises(SystemExit) as caught:
        _arguments("--transport-reuse", "compiled")
    assert caught.value.code == 2
