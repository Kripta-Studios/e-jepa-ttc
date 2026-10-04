"""Inference-only contracts, using fixtures rather than scientific eAP metrics."""

# ruff: noqa: ANN001, ANN201, ANN204
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.expert_features import extract_family
from operational.simplex_t_shared_route.adapter import extract, inputs


class Expert(torch.nn.Module):
    def __init__(self, ttc):
        super().__init__()
        self.ttc = ttc
        self.calls = 0

    def forward(self, events, delta, *, return_dense_features):
        self.calls += 1
        n = len(events)
        return SimpleNamespace(
            ttc_mean_seconds=torch.full((n,), self.ttc),
            ttc_log_variance=torch.full((n,), -1.0),
            known_mask=torch.ones(n, dtype=torch.bool),
            diagnostics={"transport_flow_magnitude": torch.full((n, 3), 0.3)},
            log_height_ratio=torch.full((n, 3), 0.02),
            sensor_support=torch.full((n, 3), 0.01),
            pair_tokens=torch.full((n, 3, 128), 0.15),
        )


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return torch.full((len(batch.features),), 0.02)

    def predict_ttc(self, batch):
        return 0.1 / (-torch.expm1(-self(batch)))


def fixture():
    return {"A5": Expert(4.0).eval(), "C2F": Expert(6.0).eval(), "PAIR": Pair().eval()}, (
        torch.ones(16, 3, 12, 4, 4),
        torch.full((16, 2), 0.1),
    )


def test_full_matches_canonical_extractor():
    models, (events, delta) = fixture()
    canonical = extract_family(models["A5"], models["C2F"], models["PAIR"], events, delta)
    assert np.array_equal(
        extract("FULL_C0", models, events, delta), canonical["features145"][:, :17]
    )


@pytest.mark.parametrize(
    "arm,names",
    [
        ("A5_ONLY_C0", {"A5"}),
        ("C2F_ONLY_C0", {"C2F"}),
        ("A5_PAIR_C0", {"A5", "PAIR"}),
    ],
)
def test_excluded_producers_never_called(arm, names):
    models, (events, delta) = fixture()
    extract(arm, models, events, delta)
    assert {name for name, model in models.items() if model.calls} == names
    assert all(model.calls <= 1 for model in models.values())


@pytest.mark.parametrize("length,arm", [(1, "H1_SEED7"), (8, "FULL_C0"), (16, "H16_SEED7")])
def test_time_and_padding(length, arm):
    valid = np.zeros(16, bool)
    valid[-1] = True
    x, times, mask, experts = inputs(
        arm,
        np.ones((16, 17), np.float32),
        valid,
        np.arange(15, -1, -1) * 50000,
        1000000,
        1005000,
        np.zeros(17),
        np.ones(17),
    )
    assert x.shape == (1, length, 17)
    assert bool(mask[0, -1]) and int(mask.sum()) == 1
    assert np.allclose(times[0, -1].numpy(), [0, 0, 0, 0.005])
    assert not bool(x[0, :-1].any())
    assert torch.isfinite(experts).all()


def test_excluded_values_do_not_affect_input_or_anchor():
    raw = np.ones((16, 17), np.float32)
    arguments = (
        np.ones(16, bool),
        np.arange(15, -1, -1) * 50000,
        1000000,
        1005000,
        np.zeros(17),
        np.ones(17),
    )
    a = inputs("C2F_ONLY_C0", raw, *arguments)
    raw[:, [2, 3, 4, 8, 10, 11, 12, 13, 14, 15, 16]] = 1000
    b = inputs("C2F_ONLY_C0", raw, *arguments)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
