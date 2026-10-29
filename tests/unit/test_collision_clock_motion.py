from __future__ import annotations

import torch

import e_jepa_ttc.models.collision_clock_motion as motion
from e_jepa_ttc.models.local_transport import TRANSPORT_FEATURE_NAMES

EXPECTED = (
    "translation_x",
    "translation_y",
    "divergence_x",
    "divergence_y",
    "divergence_isotropic",
    "flow_magnitude",
    "confidence_margin",
    "entropy",
    "cycle_error",
)


def test_global_transport_schema_is_exact_and_uses_no_foreground_weight(monkeypatch) -> None:
    calls: list[torch.Tensor | None] = []
    reverse_requires_grad: list[tuple[bool, bool]] = []
    matcher_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = motion.transport_physical_features
    original_matcher = motion.local_correlation_match

    def spy(forward, reverse, *, foreground_weight, **kwargs):
        calls.append(foreground_weight)
        reverse_requires_grad.append((reverse.dx.requires_grad, reverse.dy.requires_grad))
        return original(
            forward,
            reverse,
            foreground_weight=foreground_weight,
            **kwargs,
        )

    def matcher_spy(previous, current, **kwargs):
        matcher_calls.append((previous, current))
        return original_matcher(previous, current, **kwargs)

    monkeypatch.setattr(motion, "transport_physical_features", spy)
    monkeypatch.setattr(motion, "local_correlation_match", matcher_spy)
    dense = torch.randn(2, 8, 6, 6, requires_grad=True)
    output = motion.height_free_global_transport_features(
        dense,
        dense.roll(1, dims=-1),
        radius=1,
        temperature=0.02,
    )
    assert motion.GLOBAL_TRANSPORT_FEATURE_NAMES == EXPECTED
    assert motion.GLOBAL_TRANSPORT_FEATURE_NAMES == TRANSPORT_FEATURE_NAMES[:9]
    assert output.features.shape == (2, 9)
    assert calls == [None]
    assert reverse_requires_grad == [(False, False)]
    assert len(matcher_calls) == 2
    assert matcher_calls[0][0] is dense
    assert matcher_calls[1][1] is dense


def test_global_transport_is_finite_deterministic_and_differentiable() -> None:
    previous = torch.randn(1, 8, 6, 6, requires_grad=True)
    current = torch.randn(1, 8, 6, 6, requires_grad=True)
    first = motion.height_free_global_transport_features(
        previous,
        current,
        radius=1,
        temperature=0.02,
    ).features
    second = motion.height_free_global_transport_features(
        previous,
        current,
        radius=1,
        temperature=0.02,
    ).features
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(first).all())
    first.sum().backward()
    assert previous.grad is not None and bool(torch.isfinite(previous.grad).all())
    assert current.grad is not None and bool(torch.isfinite(current.grad).all())
