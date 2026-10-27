"""Object-isolated context parity and input rejection on CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.final_refit.training import PhaseOnlyFinal
from operational.train40_runtime.batched_context import BatchedContextRuntime, ContextObservation
from operational.train40_runtime.context import HEAD, ContextRuntime


@pytest.fixture
def bundle():
    torch.manual_seed(2718)
    models = {
        "E_CTX_MATCHED": TemporalRefiner(TemporalConfig(hidden=160)),
        "R_CTX": TemporalRefiner(TemporalConfig(hidden=160)),
        HEAD: PhaseOnlyFinal(),
    }
    for model in models.values():
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "location" in name or "output" in name:
                    parameter.normal_(0, 0.01)
        model.eval().requires_grad_(False)
    norms = {name: (torch.randn(17), torch.rand(17) + 0.5) for name in ("event", "rgb")}
    return models, norms


def observation(identity: str, rgb_count: int = 5) -> ContextObservation:
    query = 2_000_003
    return ContextObservation(
        identity,
        torch.randn(4, 17),
        torch.rand(4, 3) * 0.05 + 0.02,
        [query - lag for lag in (300_000, 200_000, 100_000, 0)],
        torch.randn(rgb_count, 17),
        torch.rand(rgb_count, 3) * 0.05 + 0.02,
        [
            {
                "first_sensor_ts_us": query - (6 - i) * 100_000,
                "anchor_us": query - (4 - i) * 100_000,
                "available_us": query,
            }
            for i in range(rgb_count)
        ],
        query,
    )


@pytest.mark.parametrize("count", [1, 2, 5, 20])
def test_batch_context_matches_independent_objects(bundle, count):
    models, norms = bundle
    observations = [observation(str(i), 0 if i % 3 == 0 else 5) for i in range(count)]
    runtime = BatchedContextRuntime(models, norms, "cpu")
    reference = ContextRuntime(models, norms, "cpu")
    before = {
        key: {name: value.clone() for name, value in model.state_dict().items()}
        for key, model in models.items()
    }
    actual = runtime.predict_many(observations)
    for item, output in zip(observations, actual, strict=True):
        expected = reference.predict(
            item.event_raw,
            item.event_experts,
            item.event_anchor_us,
            item.rgb_raw,
            item.rgb_experts,
            item.rgb_metadata,
            item.query_us,
        )
        assert output["object_id"] == item.object_id
        assert output["history_lengths"] == expected["history_lengths"]
        assert output["rgb_available"] == expected["rgb_available"]
        for name, value in expected["predictions"].items():
            if value is None:
                assert output["predictions"][name] is None
            else:
                np.testing.assert_allclose(output["predictions"][name], value, rtol=1e-5, atol=1e-5)
    for key, model in models.items():
        assert all(
            torch.equal(before[key][name], value) for name, value in model.state_dict().items()
        )


def test_causal_selection_does_not_mix_objects(bundle):
    models, norms = bundle
    a, b = observation("a"), observation("b")
    a.rgb_metadata[0]["first_sensor_ts_us"] = a.query_us - 650_001
    b.rgb_metadata[-1]["available_us"] = b.query_us + 1
    runtime = BatchedContextRuntime(models, norms, "cpu")
    out = runtime.predict_many([a, b])
    assert [row["history_lengths"]["rgb"] for row in out] == [4, 4]
    swapped = runtime.predict_many([b, a])
    for original, changed in zip(out, reversed(swapped), strict=True):
        for name, value in original["predictions"].items():
            np.testing.assert_allclose(changed["predictions"][name], value, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["duplicate", "empty", "nan", "future", "missing_rgb"])
def test_invalid_batch_context_rejected(bundle, kind):
    models, norms = bundle
    item = observation("one")
    inputs = [item]
    if kind == "duplicate":
        inputs.append(item)
    elif kind == "empty":
        inputs = []
    elif kind == "nan":
        item.event_raw[0, 0] = float("nan")
    elif kind == "future":
        item.event_anchor_us[-1] = item.query_us + 1
    else:
        item = ContextObservation(
            item.object_id,
            item.event_raw,
            item.event_experts,
            item.event_anchor_us,
            None,
            None,
            item.rgb_metadata,
            item.query_us,
        )
        inputs = [item]
    with pytest.raises(ValueError):
        BatchedContextRuntime(models, norms, "cpu").predict_many(inputs)


def test_cpu_graphs_rejected(bundle):
    models, norms = bundle
    with pytest.raises(ValueError, match="CUDA"):
        BatchedContextRuntime(models, norms, "cpu", graphs=True)
