"""CPU proofs for summary attribution; no GPU speed or complete-update admission."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import Tensor

from e_jepa_ttc.models import local_transport
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.train40_runtime.compiled_transport import TransportPairTensorCore
from scripts.diagnose_train40_transport_summaries import (
    capture_transport_inputs,
    clone_preserving_strides,
    compare_transport_vectors,
    make_summary_probe,
    native_reverse_sample_operator,
)

ProbeInputs = tuple[Tensor, Tensor, Tensor, Tensor]


def _inputs(seed: int) -> ProbeInputs:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(2, 4, 8, 8, generator=generator),
        torch.randn(2, 4, 8, 8, generator=generator),
        torch.rand(2, 1, 8, 8, generator=generator),
        torch.rand(2, 1, 8, 8, generator=generator),
    )


@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("deterministic_sampling", [False, True])
def test_summary_partitions_match_original_cpu_with_changed_and_zero_inputs(
    radius: int, deterministic_sampling: bool
) -> None:
    probes = make_summary_probe(
        radius, 0.02, deterministic_sampling=deterministic_sampling, backend="eager"
    )
    assert set(probes) == {
        "original",
        "native_match",
        "compiled_summary_native_sampler",
        "native_summary_compiled_sampler",
    }
    original = TransportPairTensorCore(
        radius,
        0.02,
        deterministic_sampling=deterministic_sampling,
        reuse_reverse=False,
        reuse_samples=False,
    ).eval()
    inputs = _inputs(191)
    changed: ProbeInputs = (
        inputs[0] * 0.5,
        inputs[1].flip(-1),
        inputs[2].flip(-2),
        inputs[3] * 0.7,
    )
    zero: ProbeInputs = (
        torch.zeros_like(inputs[0]),
        torch.zeros_like(inputs[1]),
        torch.zeros_like(inputs[2]),
        torch.zeros_like(inputs[3]),
    )
    with torch.inference_mode():
        for values in (inputs, changed, zero):
            expected = original(*values)
            for name, probe in probes.items():
                actual = probe(*values)
                assert len(actual) == 2
                for actual_direction, expected_direction in zip(actual, expected, strict=True):
                    assert actual_direction.shape == (2, 18), name
                    assert torch.equal(actual_direction, expected_direction), name


@pytest.mark.parametrize("deterministic_sampling", [False, True])
def test_native_reverse_sampler_is_exact_owned_and_has_valid_schema(
    deterministic_sampling: bool,
) -> None:
    previous, current, _, _ = _inputs(192)
    operation = native_reverse_sample_operator()
    with torch.inference_mode():
        forward = local_transport.local_correlation_match(
            previous, current, radius=1, temperature=0.02
        )
        reverse = local_transport.local_correlation_match(
            current, previous, radius=1, temperature=0.02
        )
        arguments = (
            forward.dx,
            forward.dy,
            forward.valid,
            reverse.dx,
            reverse.dy,
            reverse.valid,
            deterministic_sampling,
        )
        expected = local_transport._sample_reverse_flow(
            reverse, forward, deterministic_sampling=deterministic_sampling
        )
        actual = operation(*arguments)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert actual[0].dtype == actual[1].dtype == torch.float32
        assert actual[2].dtype == torch.bool
        storages = [value.untyped_storage().data_ptr() for value in actual]
        input_storages = {
            value.untyped_storage().data_ptr() for value in arguments if isinstance(value, Tensor)
        }
        assert len(set(storages)) == 3
        assert set(storages).isdisjoint(input_storages)
        checked = torch.library.opcheck(
            operation, arguments, test_utils=("test_schema", "test_faketensor")
        )
        assert checked == {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
def test_snapshot_preserves_strided_values_without_aliasing_source(dtype: torch.dtype) -> None:
    storage = torch.arange(3 * 4 * 8).reshape(3, 4, 8).to(dtype=dtype)
    original = storage[:, ::2, 1::2].permute(2, 0, 1)
    expected = original.clone()
    snapshot = clone_preserving_strides(original)
    assert snapshot.shape == original.shape
    assert snapshot.stride() == original.stride()
    assert snapshot.dtype == original.dtype
    assert snapshot.device == original.device
    assert torch.equal(snapshot, original)
    assert snapshot.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    storage.zero_()
    assert torch.equal(snapshot, expected)


@pytest.mark.parametrize("coordinate", [0, 8, 17])
def test_transport_comparison_identifies_coordinate_before_router(coordinate: int) -> None:
    expected = (torch.ones(2, 18), torch.ones(2, 18))
    actual = tuple(value.clone() for value in expected)
    actual[0][1, coordinate] += 0.01
    report = compare_transport_vectors(actual, expected)
    assert not report["exact"] and not report["passed_1e5"]
    assert len(report["outputs"]) == 2
    forward, reverse = report["outputs"]
    names = [row["name"] for row in forward["coordinates"]]
    assert names == list(local_transport.TRANSPORT_FEATURE_NAMES)
    failed = [row for row in forward["coordinates"] if not row["passed_1e5"]]
    assert len(failed) == 1
    assert failed[0]["name"] == local_transport.TRANSPORT_FEATURE_NAMES[coordinate]
    assert failed[0]["max_abs"] == pytest.approx(0.01, abs=1e-7)
    assert not failed[0]["exact"]
    assert all(row["exact"] for row in reverse["coordinates"])


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_transport_comparison_does_not_admit_nonfinite_coordinates(nonfinite: float) -> None:
    expected = (torch.ones(1, 18), torch.ones(1, 18))
    actual = tuple(value.clone() for value in expected)
    actual[1][0, 8] = nonfinite
    report = compare_transport_vectors(actual, expected)
    assert not report["passed_1e5"] and not report["exact"]
    assert not report["outputs"][1]["coordinates"][8]["passed_1e5"]


def test_transport_comparison_rejects_shape_change_before_coordinate_metrics() -> None:
    expected = (torch.ones(2, 18), torch.ones(2, 18))
    actual = (torch.ones(1, 18), torch.ones(2, 18))
    report = compare_transport_vectors(actual, expected)
    assert not report["passed_1e5"] and not report["exact"]
    assert report["outputs"][0]["reason"] == "shape"


def _model(pyramid: bool) -> CausalScaleTTC:
    return (
        CausalScaleTTC(
            CausalScaleTTCConfig(
                hidden_dim=8,
                geometry_dim=8,
                residual_depth=1,
                dropout=0.0,
                transport_enabled=True,
                transport_radius=1,
                transport_fine_radius=1,
                transport_coarse_radius=1,
                transport_temperature=0.02,
                transport_mode="adaptive_pyramid" if pyramid else "legacy",
                deterministic_transport_sampling=True,
            )
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("pyramid", [False, True])
def test_capture_records_real_core_inputs_and_leaves_model_untouched(pyramid: bool) -> None:
    torch.manual_seed(193)
    model = _model(pyramid)
    original_parameters = tuple(model.parameters())
    original_modules = dict(model.named_modules())
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    original_source = inspect.getsource(CausalScaleTTC._forward_impl)
    sensor = torch.rand(2, 3, 12, 32, 32)
    delta = torch.full((2, 2), 0.1)
    with torch.inference_mode():
        expected = model(sensor, delta, return_dense_features=True)
        captures = capture_transport_inputs(model, sensor, delta)
        assert [item["scale"] for item in captures] == (["fine", "coarse"] if pyramid else ["fine"])
        for capture in captures:
            previous, current, foreground_previous, foreground_current = capture["inputs"]
            assert previous.shape == current.shape
            assert previous.shape[0] == 4
            assert foreground_previous.shape == foreground_current.shape
            assert foreground_previous.shape == (4, 1, previous.shape[2], previous.shape[3])
            assert capture["radius"] == 1
            assert capture["temperature"] == 0.02
            assert capture["deterministic_sampling"] is True
            core = TransportPairTensorCore(1, 0.02, deterministic_sampling=True)
            recorded_forward, _ = core(*capture["inputs"])
            if pyramid:
                reference = torch.stack(
                    [
                        expected.diagnostics[f"transport_{capture['scale']}_{name}"]
                        for name in local_transport.TRANSPORT_FEATURE_NAMES
                    ],
                    dim=-1,
                ).reshape(4, 18)
            else:
                assert expected.transport_raw_features is not None
                reference = expected.transport_raw_features.reshape(4, 18)
            assert torch.equal(recorded_forward, reference)
    assert dict(model.named_modules()) == original_modules
    assert inspect.getsource(CausalScaleTTC._forward_impl) == original_source
    assert all(
        current is original
        for current, original in zip(model.parameters(), original_parameters, strict=True)
    )
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in original_state.items()
    )
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
