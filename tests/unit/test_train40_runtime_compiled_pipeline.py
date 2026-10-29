"""CPU proofs for opaque compilation and unchanged guards; no CUDA admission."""

import gc
import inspect
import textwrap
import warnings
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.train40_runtime.compiled_pipeline import (
    OPERATOR_NAME,
    CompiledTransportProducerRuntime,
    NativeLogTransportPairTensorCore,
    NativeMatchTransportPairTensorCore,
    OpaqueCompiledTransport,
    _rewritten_native_log_match,
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


@pytest.mark.parametrize("radius", [1, 4])
def test_native_entropy_log_core_is_exact_and_compiles_as_one_eager_graph(radius):
    torch.manual_seed(183)
    original = TransportPairTensorCore(radius, 0.02, deterministic_sampling=True)
    core = NativeLogTransportPairTensorCore(original)
    compiled = OpaqueCompiledTransport(core, backend="eager")
    inputs = (
        torch.randn(2, 8, 12, 12),
        torch.randn(2, 8, 12, 12),
        torch.rand(2, 1, 12, 12),
        torch.rand(2, 1, 12, 12),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(compiled, inputs, check_trace=False)
        for values in (inputs, tuple(value * 0 for value in inputs)):
            expected, actual = original(*values), traced(*values)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    diagnostics = compiled.diagnostics()
    assert diagnostics["native_log_binding"]["native_log_replacements"] == 2
    assert diagnostics["native_log_dispatcher_calls_process"] > 0


def test_native_entropy_log_rewrite_fails_closed_when_audited_statements_change():
    from e_jepa_ttc.models.local_transport import local_correlation_match

    source = textwrap.dedent(inspect.getsource(local_correlation_match))
    changed = source.replace("valid_count.log()", "torch.log(valid_count)")
    with pytest.raises(ValueError, match="two exact replacements"):
        _rewritten_native_log_match(changed)


def test_native_entropy_log_custom_operator_schema_and_fake_metadata():
    with torch.inference_mode():
        for values in (torch.rand(3, 4) + 0.01, torch.rand(3, 4).T + 0.01):
            checked = torch.library.opcheck(
                torch.ops.train40_compiled_pipeline.native_log.default,
                (values,),
                test_utils=("test_schema", "test_faketensor"),
            )
            assert all(value == "SUCCESS" for value in checked.values())


@pytest.mark.parametrize("radius", [1, 4])
def test_native_match_core_preserves_oracle_and_eager_compile_jit(radius):
    torch.manual_seed(184)
    original = TransportPairTensorCore(radius, 0.02, deterministic_sampling=True)
    native = NativeMatchTransportPairTensorCore(original)
    wrapper = OpaqueCompiledTransport(native, backend="eager")
    values = (
        torch.randn(2, 8, 12, 12),
        torch.randn(2, 8, 12, 12),
        torch.rand(2, 1, 12, 12),
        torch.rand(2, 1, 12, 12),
    )
    with torch.inference_mode():
        traced = cast(torch.jit.ScriptModule, torch.jit.trace(wrapper, values, check_trace=False))
        for inputs in (values, tuple(value * 0 for value in values)):
            expected, actual = original(*inputs), traced(*inputs)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    diagnostics = wrapper.diagnostics()
    assert (
        diagnostics["native_match_binding"]["boundary"] == "ORIGINAL_ATEN_LOCAL_CORRELATION_MATCH"
    )
    assert len(diagnostics["native_match_binding"]["original_match_sha256"]) == 64
    assert diagnostics["native_match_dispatcher_calls_process"] > 0
    assert diagnostics["native_log_binding"] is None


def test_native_match_custom_operator_schema_fake_metadata_and_original_guards():
    from e_jepa_ttc.models.local_transport import local_correlation_match

    operation = torch.ops.train40_compiled_pipeline.native_match.default
    with torch.inference_mode():
        for previous in (torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8).transpose(-1, -2)):
            current = previous.flip(-1)
            checked = torch.library.opcheck(
                operation,
                (previous, current, 1, 0.02),
                test_utils=("test_schema", "test_faketensor"),
            )
            assert all(value == "SUCCESS" for value in checked.values())
            original = local_correlation_match(previous, current, radius=1, temperature=0.02)
            actual = operation(previous, current, 1, 0.02)
            assert actual[-1].dtype == torch.bool
            assert all(
                torch.equal(value, getattr(original, name))
                for value, name in zip(
                    actual, ("dx", "dy", "confidence_margin", "entropy", "valid"), strict=True
                )
            )
        with pytest.raises(ValueError, match="radius is too large"):
            operation(previous, current, 4, 0.02)
        with pytest.raises(ValueError, match="temperature must be finite"):
            operation(previous, current, 1, 0.0)
        with pytest.raises(ValueError, match="FP32"):
            operation(previous.double(), current.double(), 1, 0.02)


def test_native_transport_boundaries_are_exclusive_and_do_not_change_parameters():
    model = _model()
    with pytest.raises(ValueError, match="only one native"):
        attach_opaque_compiled_transport(model, native_log=True, native_match=True)
    with pytest.raises(ValueError, match="only one native"):
        CompiledTransportProducerRuntime({}, "cpu", native_log=True, native_match=True)
    proposed = attach_opaque_compiled_transport(model, backend="eager", native_match=True)
    assert all(a is b for a, b in zip(model.parameters(), proposed.parameters(), strict=True))
    assert all(
        torch.equal(value, proposed.state_dict()[name])
        for name, value in model.state_dict().items()
    )


def _replay_prediction(value, object_id="roi_00"):
    return {
        "query_id": object_id,
        "predictions": {
            "PHASE_ONLY_TRAIN40_S7": value,
            "E_CTX_TRAIN40": value,
            "R_CTX_TRAIN40": value,
        },
        "history_lengths": {"event": 4, "rgb": 5},
        "rgb_available": True,
        "rgb_unavailable_reason": "",
    }


def _explicit_replay_schedule():
    return [
        ("original", "initial_call", 0),
        ("original", "warmup", 0),
        ("compiled", "initial_call", 0),
        ("compiled", "warmup", 0),
        ("original", "measured", 0),
        ("compiled", "measured", 0),
        ("compiled", "measured", 1),
        ("original", "measured", 1),
    ]


@pytest.mark.parametrize("failed_stage", ["initial_call", "warmup", "measured"])
def test_replay_parity_rejects_early_failure_even_when_final_replay_is_correct(failed_stage):
    from operational.train40_runtime.measurement import prediction_parity
    from scripts.benchmark_train40_compiled_pipeline import (
        capture_replay_parity,
        replay_parity_coverage,
    )

    shared = torch.ones(1)
    expected = [_replay_prediction(1.0)]
    receipts = []
    for name, stage, iteration in _explicit_replay_schedule():
        shared.fill_(
            2.0 if name == "compiled" and stage == failed_stage and iteration == 0 else 1.0
        )

        def predict(owned):
            # Simulate graph-owned storage being overwritten after the snapshot.
            shared.fill_(1.0)
            assert owned[0][0].data_ptr() != shared.data_ptr()
            assert owned[1][0] == (None, None)
            return [_replay_prediction(float(owned[0][0][0]))]

        receipts.append(
            capture_replay_parity(
                ((shared, shared), [(None, None)]),
                variant=name,
                stage=stage,
                iteration=iteration,
                predict=predict,
                compare_ttc=lambda actual: prediction_parity(expected, actual, ["roi_00"]),
                compare_features=lambda owned: {"passed": True},
            )
        )
    assert receipts[-1]["parity"]["passed"]
    assert receipts[-2]["parity"]["passed"]
    coverage = replay_parity_coverage(receipts, ["original", "compiled"], warmup=1, repeats=2)
    assert coverage["schedule_exact"]
    assert coverage["observed_calls"] == coverage["expected_calls"] == 8
    assert not coverage["passed"] and not coverage["all_ttc_passed"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "reordered"])
def test_replay_parity_requires_exact_warmup_and_repeat_coverage(damage):
    from scripts.benchmark_train40_compiled_pipeline import replay_parity_coverage

    receipts = [
        {
            "variant": name,
            "stage": stage,
            "iteration": iteration,
            "parity": {"passed": True},
            "feature_parity": {"passed": True},
        }
        for name, stage, iteration in _explicit_replay_schedule()
    ]
    assert replay_parity_coverage(receipts, ["original", "compiled"], warmup=1, repeats=2)["passed"]
    if damage == "missing":
        del receipts[0]
    elif damage == "duplicate":
        receipts.append(receipts[-1])
    else:
        receipts[0], receipts[1] = receipts[1], receipts[0]
    assert not replay_parity_coverage(receipts, ["original", "compiled"], warmup=1, repeats=2)[
        "passed"
    ]


def test_replay_parity_checks_first_call_again_after_object_shape_transition():
    from operational.train40_runtime.measurement import prediction_parity
    from scripts.benchmark_train40_compiled_pipeline import (
        capture_replay_parity,
        replay_parity_coverage,
    )

    gates = []
    for objects in (1, 2):
        ids = [f"roi_{index:02}" for index in range(objects)]
        expected = [_replay_prediction(1.0, object_id) for object_id in ids]
        receipts = []
        for name, stage, iteration in _explicit_replay_schedule():
            values = torch.ones(objects)
            if objects == 2 and name == "compiled" and stage == "initial_call":
                values[-1] = 2.0
            receipts.append(
                capture_replay_parity(
                    ((values, values), []),
                    variant=name,
                    stage=stage,
                    iteration=iteration,
                    predict=lambda owned, ids=ids: [
                        _replay_prediction(float(value), object_id)
                        for value, object_id in zip(owned[0][0], ids, strict=True)
                    ],
                    compare_ttc=lambda actual, expected=expected, ids=ids: prediction_parity(
                        expected, actual, ids
                    ),
                    compare_features=lambda owned: {"passed": True},
                )
            )
        gates.append(
            replay_parity_coverage(receipts, ["original", "compiled"], warmup=1, repeats=2)
        )
    assert gates[0]["passed"]
    assert gates[1]["schedule_exact"] and not gates[1]["passed"]
    assert not gates[1]["first_calls_passed"]
