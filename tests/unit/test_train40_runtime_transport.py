"""Exact CPU parity for inference-only transport redundancy removal."""

from dataclasses import fields

import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.models.local_transport import local_correlation_match, transport_physical_features
from operational.train40_runtime.producers import (
    TRANSPORT_REUSE_MODES,
    transport_reuse_contract,
)
from operational.train40_runtime.transport import (
    CorrelationSession,
    adjacent_pair_indices,
    attach_transport_reuse,
    correlation_pair,
    reused_transport_physical_features,
    reverse_correlation_volume,
)


@pytest.mark.parametrize("radius,size", [(1, 16), (2, 8), (4, 16)])
@pytest.mark.parametrize("kind", ["random", "zero", "constant", "shift"])
def test_shared_reverse_matches_every_original_field(radius, size, kind):
    torch.manual_seed(9)
    a = torch.randn(2, 8, size, size)
    b = torch.randn_like(a)
    if kind == "zero":
        a.zero_()
        b.zero_()
    elif kind == "constant":
        a.fill_(1.0)
        b.fill_(0.2)
    elif kind == "shift":
        b = torch.roll(a, (1, -1), (-2, -1))
    for actual, left, right in zip(
        correlation_pair(a, b, radius=radius, temperature=0.02), (a, b), (b, a), strict=True
    ):
        reference = local_correlation_match(
            left, right, radius=radius, temperature=0.02, return_probability=True
        )
        for field in fields(reference):
            assert torch.equal(getattr(actual, field.name), getattr(reference, field.name))


@pytest.mark.parametrize("deterministic", [False, True])
def test_summary_reuse_exact(deterministic):
    torch.manual_seed(10)
    a, b = torch.randn(3, 8, 16, 16), torch.randn(3, 8, 16, 16)
    forward, reverse = correlation_pair(a, b, radius=1, temperature=0.02)
    weight = torch.sigmoid(torch.randn(3, 1, 16, 16))
    expected = transport_physical_features(
        forward, reverse, foreground_weight=weight, radius=1, deterministic_sampling=deterministic
    )
    actual = reused_transport_physical_features(
        forward, reverse, foreground_weight=weight, radius=1, deterministic_sampling=deterministic
    )
    assert torch.equal(actual, expected)


def test_pair_indices_never_cross_object_boundaries():
    select, restore = adjacent_pair_indices(2, 4, device=torch.device("cpu"))
    assert select.tolist() == [0, 1, 3, 5, 7, 8, 9, 11, 13, 15]
    assert restore.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9]


def test_session_rejects_wrong_direction_and_restores_probability_choice():
    a, b = torch.randn(2, 8, 8, 8), torch.randn(2, 8, 8, 8)
    session = CorrelationSession()
    assert session(a, b, radius=1, temperature=0.02).probability is None
    assert (
        session(b, a, radius=1, temperature=0.02, return_probability=True).probability is not None
    )
    session(a, b, radius=1, temperature=0.02)
    with pytest.raises(ValueError, match="immediately reversed"):
        session(a, b, radius=1, temperature=0.02)


@pytest.mark.parametrize("pyramid", [False, True])
@pytest.mark.parametrize("deduplicate", [False, True])
def test_model_view_preserves_all_outputs_and_frozen_state(pyramid, deduplicate):
    torch.manual_seed(11)
    model = (
        CausalScaleTTC(
            CausalScaleTTCConfig(
                hidden_dim=8,
                geometry_dim=8,
                residual_depth=1,
                dropout=0.0,
                foreground_temporal_smoothing=0.15,
                foreground_temporal_smoothing_mode="causal_left",
                temporal_inverse_ttc_blend=0.75,
                transport_enabled=True,
                transport_radius=1,
                transport_mode="adaptive_pyramid" if pyramid else "legacy",
                transport_coarse_radius=1,
                deterministic_transport_sampling=True,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    original_modules = dict(model.named_modules())
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    chain = torch.rand(2, 4, 12, 32, 32)
    sensor = torch.stack([chain[:, i : i + 3] for i in range(2)], dim=1).flatten(0, 1)
    delta = torch.full((4, 2), 0.1)
    view = attach_transport_reuse(
        model, history_group_size=2 if deduplicate else None, externally_guarded_overlap=deduplicate
    )
    with torch.inference_mode():
        expected = model(sensor, delta, return_dense_features=True)
        actual = view(sensor, delta, return_dense_features=True)
    assert dict(model.named_modules()) == original_modules
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in original_state.items()
    )
    assert all(a is b for a, b in zip(model.parameters(), view.parameters(), strict=True))

    differences = []

    def compare_tensor(a, b, name):
        assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, name
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        if deduplicate:
            # The unadmitted candidate may change CPU reduction rounding when
            # it reduces the pair batch. Bound every field before reporting an
            # exact-parity failure; never turn a large error into an xfail.
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5, msg=name)
            if not torch.equal(a, b):
                differences.append(name)
        else:
            assert torch.equal(a, b), name

    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, torch.Tensor):
            compare_tensor(a, b, field.name)
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), field.name
            for key in a:
                compare_tensor(a[key], b[key], f"{field.name}.{key}")
        else:
            assert a == b
    if differences:
        # Some CPU backends are exact on this fixture. That is a passing
        # diagnostic, not production admission. All state, structure and 1e-5
        # checks above remain hard failures on every backend.
        pytest.xfail("UNADMITTED adjacent-pair CPU bitwise differences: " + ", ".join(differences))


@pytest.mark.parametrize("mode", TRANSPORT_REUSE_MODES)
def test_registered_production_modes_exclude_adjacent_pair_deduplication(mode):
    contract = transport_reuse_contract(mode)
    assert contract["history_group_size"] is None
    assert contract["adjacent_pair_deduplication"] is False


def test_adjacent_pair_deduplication_is_not_a_registered_production_mode():
    with pytest.raises(ValueError, match="Transport reuse must be"):
        transport_reuse_contract("adjacent-pair-deduplication")


def test_runtime_guard_is_explicit_and_training_rejected():
    model = CausalScaleTTC(CausalScaleTTCConfig())
    with pytest.raises(ValueError, match="frozen eval"):
        attach_transport_reuse(model)
    model.eval().requires_grad_(False)
    with pytest.raises(ValueError, match="external overlap"):
        attach_transport_reuse(model, history_group_size=4)
    with pytest.raises(ValueError):
        reverse_correlation_volume(torch.zeros(1, 8, 16, 16), 1)
