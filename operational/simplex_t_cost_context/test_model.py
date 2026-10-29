"""Focused synthetic contracts only; no optimizer steps or eAP result claims."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from torch import nn

from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import emitted_phase, phase_to_ttc, pinball

from .contracts import extract_reduced_family
from .model import (
    ARMS,
    FEATURE_NAMES,
    MaskedSource,
    anchor_from_allowed,
    build_model,
    mask_features,
    training_loss,
)

if TYPE_CHECKING:
    from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    return (
        torch.randn(3, 8, 17),
        torch.rand(3, 8, 4),
        torch.ones(3, 8, dtype=torch.bool),
        torch.tensor([[0.10, 0.20, 0.15], [-0.02, 0.05, 0.01], [0.02, 0.03, 0.04]]),
    )


def activate_heads(model: nn.Module) -> None:
    """Invariance checks must exercise learned hidden features, not zero heads."""
    with torch.no_grad():
        cast(nn.Linear, model.location).weight.normal_(0, 0.1)
        if hasattr(model, "widths"):
            cast(nn.Linear, model.widths).weight.normal_(0, 0.1)
        else:
            cast(nn.Linear, model.width).weight.normal_(0, 0.1)


def test_name_schema_and_parameter_counts() -> None:
    from e_jepa_ttc.models.three_expert_router import PHASE17_FEATURES

    assert FEATURE_NAMES == PHASE17_FEATURES
    for arm in ARMS:
        model = build_model(arm)
        expected = 289765 if arm.startswith("SET_") else 313765
        assert sum(p.numel() for p in model.parameters()) == expected
    x = torch.randn(2, 8, 17)
    reverse = tuple(reversed(FEATURE_NAMES))
    expected = mask_features(x, "A5_PAIR_C0")
    actual = mask_features(x.flip(-1), "A5_PAIR_C0", reverse).flip(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_full_gru_is_exact_canonical_forward() -> None:
    model = build_model("FULL_C0")
    original = TemporalRefiner(TemporalConfig(feature_count=17, hidden=160))
    activate_heads(model)
    original.load_state_dict(model.state_dict())
    for name, value in original(*batch()).items():
        torch.testing.assert_close(model(*batch())[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("arm", ARMS[1:4])
def test_excluded_expert_cannot_change_features_anchor_or_prediction(arm: str) -> None:
    model = build_model(arm)
    activate_heads(model)
    x, times, valid, experts = batch()
    original = model(x, times, valid, experts)
    keep = mask_features(torch.ones_like(x), arm).bool()
    changed = torch.where(keep, x, torch.full_like(x, float("nan")))
    e = experts.clone()
    excluded = {"A5_ONLY_C0": (1, 2), "C2F_ONLY_C0": (0, 2), "A5_PAIR_C0": (1,)}[arm]
    e[:, excluded] = float("nan")
    torch.testing.assert_close(anchor_from_allowed(e, arm), anchor_from_allowed(experts, arm))
    for name, value in original.items():
        torch.testing.assert_close(model(changed, times, valid, e)[name], value, rtol=0, atol=0)


def test_a5_pair_anchor_is_phase_midpoint() -> None:
    experts = torch.tensor([[0.1, 100.0, 0.3], [-0.2, -99.0, 0.1]])
    torch.testing.assert_close(
        anchor_from_allowed(experts, "A5_PAIR_C0"), (experts[:, 0] + experts[:, 2]) / 2
    )
    assert not torch.equal(anchor_from_allowed(experts, "A5_PAIR_C0"), experts[:, 0])


@pytest.mark.parametrize("arm", ARMS[4:])
def test_set_permutation_and_padding_contract(arm: str) -> None:
    model = build_model(arm)
    activate_heads(model)
    x, times, valid, experts = batch()
    valid[0, :3] = False
    x[0, :3] = float("nan")
    times[0, :3] = float("nan")
    original = model(x, times, valid, experts)
    permutation = torch.tensor([4, 1, 6, 0, 5, 2, 3, 7])
    changed = model(x[:, permutation], times[:, permutation], valid[:, permutation], experts)
    for name in ("point_phase", "q10", "q90", "raw_residual"):
        torch.testing.assert_close(changed[name], original[name], rtol=2e-6, atol=2e-7)
    if arm == "SET_NOTIME_C0":
        changed = model(x, torch.full_like(times, float("nan")), valid, experts)
        for name in ("point_phase", "q10", "q90", "raw_residual"):
            torch.testing.assert_close(changed[name], original[name], rtol=0, atol=0)
    valid[:, :-1] = False
    x[:, :-1] = float("nan")
    times[:, :-1] = float("nan")
    empty = model(x, times, valid, experts)
    assert all(torch.isfinite(value).all() for value in empty.values())
    valid[:, -1] = False
    with pytest.raises(ValueError, match="current"):
        model(x, times, valid, experts)


@pytest.mark.parametrize("arm", ARMS)
def test_canonical_emission_loss_and_finite_gradients(arm: str) -> None:
    model = build_model(arm)
    x, times, valid, experts = batch()
    output = model(x, times, valid, experts)
    torch.testing.assert_close(
        output["point_phase"], emitted_phase(output["raw_location"]), rtol=0, atol=0
    )
    assert torch.isfinite(phase_to_ttc(output["point_phase"])).all()
    truth = torch.tensor([0.2, -0.03, 0.05])
    mass = torch.tensor([0.01, 0.09, 0.1])  # Deliberately does not sum to 1 in batch.
    population = 100
    loss = training_loss(output, truth, torch.full_like(experts, float("nan")), mass, population)
    manual = (
        population
        * mass
        * (
            (output["point_phase"] - truth).abs() / 0.03
            + 0.1 * (pinball(output["q10"], truth, 0.1) + pinball(output["q90"], truth, 0.9)) / 0.03
        )
    ).mean()
    torch.testing.assert_close(loss, manual, rtol=0, atol=0)
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("cost.", "costs.")):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_source_masks_after_diagonal_normalization_and_preserves_mass() -> None:
    features = np.arange(16 * 17, dtype=np.float32).reshape(16, 17)
    mean = np.arange(17, dtype=np.float64) + 30
    scale = np.arange(17, dtype=np.float64) + 1
    history = np.tile(np.arange(8, dtype=np.int64), (2, 1))
    parent = CachedQueries(
        features,
        np.arange(16, dtype=np.int64) * 100,
        np.arange(16, dtype=np.int64) * 100 + 10,
        history,
        np.array([0.1, 0.2]),
        np.array([0.2, 0.8]),
        Normalizer(mean, scale, "1" * 64),
        "2" * 64,
    )
    source = MaskedSource(parent, "A5_ONLY_C0")
    ids = torch.tensor([1, 0])
    baseline = parent.gather(ids)
    actual = source.gather(ids)
    torch.testing.assert_close(actual[0], mask_features(baseline[0], "A5_ONLY_C0"), rtol=0, atol=0)
    for i in (1, 2, 4, 5):
        torch.testing.assert_close(actual[i], baseline[i], rtol=0, atol=0)
    assert source.population == parent.population
    assert source.identity_sha256 != parent.identity_sha256
    assert torch.equal(actual[3][:, 1:], torch.zeros_like(actual[3][:, 1:]))
    assert parent.features is features and np.array_equal(parent.normalizer.mean, mean)


class FakeExpert(nn.Module):
    """Minimal producer fixture, never a trained eAP expert."""

    def __init__(self, prediction: float) -> None:
        super().__init__()
        self.prediction = prediction
        self.calls = 0

    def forward(self, events: torch.Tensor, delta: torch.Tensor, **kwargs: object) -> object:
        del delta, kwargs
        self.calls += 1
        b = len(events)
        return SimpleNamespace(
            ttc_mean_seconds=torch.full((b,), self.prediction),
            known_mask=torch.ones(b, dtype=torch.bool),
            diagnostics={"transport_flow_magnitude": torch.ones(b, 2)},
            log_height_ratio=torch.full((b, 2), 0.01),
            sensor_support=torch.full((b, 2), 0.01),
            ttc_log_variance=torch.full((b,), -1.0),
            pair_tokens=torch.ones(b, 2, 128),
        )


class FakePair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full((len(features),), 0.1)

    def predict_ttc(self, batch: PairFeatureBatch) -> torch.Tensor:
        # Hook captures the original native phase, as in the actual producer.
        return phase_to_ttc(self(batch.features))


def test_reduced_interface_calls_only_allowed_and_reuses_a5() -> None:
    a5, c2f, pair = FakeExpert(3.0).eval(), FakeExpert(4.0).eval(), FakePair().eval()
    events, delta = torch.ones(2, 3, 12, 4, 4), torch.ones(2, 2)
    only = extract_reduced_family("C2F_ONLY_C0", {"c2f": c2f}, events, delta)
    assert (a5.calls, c2f.calls, pair.calls) == (0, 1, 0)
    assert np.isnan(only["expert_ttc"][:, [0, 2]]).all()
    assert np.all(only["features17"][:, [2, 3, 4, 8, 10, 11, 12, 13, 14, 15, 16]] == 0)
    reduced = extract_reduced_family("A5_PAIR_C0", {"a5": a5, "pair": pair}, events, delta)
    assert (a5.calls, c2f.calls, pair.calls) == (1, 1, 1)
    assert reduced["pair_features"].shape == (2, 133)
    with pytest.raises(ValueError, match="exactly allowed"):
        extract_reduced_family("A5_ONLY_C0", {"a5": a5, "c2f": c2f}, events, delta)
