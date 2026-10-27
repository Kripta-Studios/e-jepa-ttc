from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_transport() -> ModuleType:
    source = (
        Path(__file__).resolve().parents[2]
        / "src/e_jepa_ttc/models/local_transport.py"
    )
    name = "runtime_v3_local_transport"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load staged transport module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


transport = _load_transport()


def _leaf(shape: tuple[int, ...], *, offset: float = 0.0) -> torch.Tensor:
    value = torch.randn(shape, dtype=torch.float64) * 0.15 + offset
    return value.requires_grad_(True)


def _matches(
    *, batch: int = 2, height: int = 5, width: int = 6, all_valid: bool = True
) -> tuple[object, object, list[torch.Tensor]]:
    forward_dx = _leaf((batch, height, width))
    forward_dy = _leaf((batch, height, width))
    forward_confidence = _leaf((batch, height, width), offset=0.2)
    forward_entropy = _leaf((batch, height, width), offset=0.5)
    reverse_dx = _leaf((batch, height, width))
    reverse_dy = _leaf((batch, height, width))
    reverse_confidence = _leaf((batch, height, width), offset=0.2)
    reverse_entropy = _leaf((batch, height, width), offset=0.5)
    valid = torch.full((batch, height, width), all_valid, dtype=torch.bool)
    forward = transport.LocalTransportMatch(
        dx=forward_dx,
        dy=forward_dy,
        confidence_margin=forward_confidence,
        entropy=forward_entropy,
        valid=valid,
    )
    reverse = transport.LocalTransportMatch(
        dx=reverse_dx,
        dy=reverse_dy,
        confidence_margin=reverse_confidence,
        entropy=reverse_entropy,
        valid=valid.clone(),
    )
    leaves = [
        forward_dx,
        forward_dy,
        forward_confidence,
        forward_entropy,
        reverse_dx,
        reverse_dy,
        reverse_confidence,
        reverse_entropy,
    ]
    return forward, reverse, leaves


def _output_and_gradients(
    *, reuse_reverse_sample: bool
) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    torch.manual_seed(17)
    forward, reverse, leaves = _matches()
    foreground = torch.rand((2, 5, 6), dtype=torch.float64, requires_grad=True)
    output = transport.transport_physical_features(
        forward,
        reverse,
        foreground_weight=foreground,
        radius=2,
        deterministic_sampling=True,
        reuse_reverse_sample=reuse_reverse_sample,
    )
    weights = torch.linspace(0.5, 1.5, output.numel(), dtype=output.dtype).reshape_as(output)
    (output * weights).sum().backward()
    gradients = [
        None if leaf.grad is None else leaf.grad.detach().clone()
        for leaf in [*leaves, foreground]
    ]
    return output.detach(), gradients


def test_reuse_preserves_forward_and_all_nine_leaf_gradients() -> None:
    historical, historical_gradients = _output_and_gradients(reuse_reverse_sample=False)
    reused, reused_gradients = _output_and_gradients(reuse_reverse_sample=True)

    torch.testing.assert_close(reused, historical, rtol=0.0, atol=0.0)
    assert len(reused_gradients) == 9
    for reused_gradient, historical_gradient in zip(
        reused_gradients, historical_gradients, strict=True
    ):
        if reused_gradient is None or historical_gradient is None:
            assert reused_gradient is historical_gradient is None
            continue
        assert torch.isfinite(reused_gradient).all()
        torch.testing.assert_close(
            reused_gradient, historical_gradient, rtol=1.0e-12, atol=1.0e-12
        )


def test_reuse_calls_reverse_sampler_once_and_default_calls_it_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward, reverse, _ = _matches(batch=1)
    original = transport._sample_reverse_flow
    calls = 0

    def counted(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(transport, "_sample_reverse_flow", counted)
    transport.transport_physical_features(
        forward, reverse, foreground_weight=None, radius=2
    )
    assert calls == 2
    calls = 0
    transport.transport_physical_features(
        forward,
        reverse,
        foreground_weight=None,
        radius=2,
        reuse_reverse_sample=True,
    )
    assert calls == 1


def test_omitted_flag_is_bit_exact_with_explicit_false() -> None:
    torch.manual_seed(23)
    forward, reverse, _ = _matches(batch=1)
    foreground = torch.rand((1, 5, 6), dtype=torch.float64)
    omitted = transport.transport_physical_features(
        forward, reverse, foreground_weight=foreground, radius=2
    )
    explicit = transport.transport_physical_features(
        forward,
        reverse,
        foreground_weight=foreground,
        radius=2,
        reuse_reverse_sample=False,
    )
    assert torch.equal(omitted, explicit)


def test_foreground_three_and_four_dimensional_forms_match() -> None:
    torch.manual_seed(29)
    forward, reverse, _ = _matches(batch=1)
    foreground = torch.rand((1, 5, 6), dtype=torch.float64)
    three_dimensional = transport.transport_physical_features(
        forward,
        reverse,
        foreground_weight=foreground,
        radius=2,
        reuse_reverse_sample=True,
    )
    four_dimensional = transport.transport_physical_features(
        forward,
        reverse,
        foreground_weight=foreground[:, None],
        radius=2,
        reuse_reverse_sample=True,
    )
    assert torch.equal(three_dimensional, four_dimensional)


def test_invalid_boundaries_and_all_invalid_fields_remain_finite() -> None:
    forward, reverse, _ = _matches(batch=1)
    with torch.no_grad():
        forward.dx[:, :, -1] = 20.0
        forward.dy[:, -1, :] = 20.0
        reverse.valid[:, 0, :] = False
        reverse.valid[:, :, 0] = False
    near_boundary = transport.transport_physical_features(
        forward,
        reverse,
        foreground_weight=None,
        radius=2,
        deterministic_sampling=True,
        reuse_reverse_sample=True,
    )
    assert torch.isfinite(near_boundary).all()

    invalid_forward, invalid_reverse, _ = _matches(batch=1, all_valid=False)
    all_invalid = transport.transport_physical_features(
        invalid_forward,
        invalid_reverse,
        foreground_weight=None,
        radius=2,
        reuse_reverse_sample=True,
    )
    assert torch.equal(all_invalid, torch.zeros_like(all_invalid))


def test_transport_rejects_field_and_foreground_shape_errors() -> None:
    forward, reverse, _ = _matches(batch=1)
    bad_reverse, _, _ = _matches(batch=1, height=4, width=6)
    with pytest.raises(ValueError, match="share shape"):
        transport.transport_physical_features(
            forward,
            bad_reverse,
            foreground_weight=None,
            radius=2,
            reuse_reverse_sample=True,
        )
    with pytest.raises(ValueError, match="foreground_weight"):
        transport.transport_physical_features(
            forward,
            reverse,
            foreground_weight=torch.ones((1, 2, 5, 6), dtype=torch.float64),
            radius=2,
            reuse_reverse_sample=True,
        )
