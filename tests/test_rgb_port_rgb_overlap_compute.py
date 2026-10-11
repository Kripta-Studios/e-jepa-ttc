"""Immutable geometry reuse must preserve outputs and every input gradient exactly."""

from dataclasses import fields

import pytest
import torch

from e_jepa_ttc.models import causal_scale_ttc, local_transport
from operational.rgb_port_rgb_overlap.compute import (
    CandidateValidityCache,
    installed_compute,
    local_correlation_match_cached,
)
from operational.rgb_port_rgb_overlap.compute_benchmark import snapshot


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["singleton_strided", "scalar", "empty"])
def test_snapshot_dense_bytes_for_strided_singleton_scalar_empty(dtype, kind):
    base = torch.arange(54, dtype=dtype).reshape(3, 18)[:, 0]
    if kind == "singleton_strided":
        actual = base[1:2]
        reference = torch.tensor([18], dtype=dtype)
        assert actual.stride() == (18,)
        assert actual.contiguous().stride() == (18,)
    elif kind == "scalar":
        actual = base[1]
        reference = torch.tensor(18, dtype=dtype)
    else:
        actual = base[:0]
        reference = torch.tensor([], dtype=dtype)
    rng = torch.get_rng_state().clone()
    original_stride = actual.stride()
    assert snapshot(actual) == snapshot(reference)
    assert actual.stride() == original_stride
    assert torch.equal(actual, reference)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,radius", [(1, 1), (3, 2), (4, 1)])
def test_cached_masks_preserve_all_outputs_gradients_and_rng(dtype, batch, radius):
    generator = torch.Generator().manual_seed(91)
    left = torch.randn(batch, 3, 7, 8, generator=generator).to(dtype)
    right = torch.randn(batch, 3, 7, 8, generator=generator).to(dtype)
    cache = CandidateValidityCache()
    before = torch.get_rng_state().clone()
    reference = None
    for variant in ("original", "cold", "hit"):
        previous = left.clone().requires_grad_()
        current = right.clone().requires_grad_()
        kwargs = dict(radius=radius, temperature=0.1, return_probability=True)
        if variant == "original":
            result = local_transport.local_correlation_match(previous, current, **kwargs)
        else:
            result = local_correlation_match_cached(previous, current, **kwargs, cache=cache)
        objective = sum(
            getattr(result, name).square().sum()
            for name in ("dx", "dy", "confidence_margin", "entropy", "probability")
        )
        objective.backward()
        values = {
            field.name: getattr(result, field.name).detach().clone() for field in fields(result)
        }
        actual = (values, previous.grad.clone(), current.grad.clone())
        if reference is None:
            reference = actual
        else:
            for name, value in values.items():
                assert value.dtype == reference[0][name].dtype
                assert value.stride() == reference[0][name].stride()
                assert torch.equal(value, reference[0][name]), name
            assert torch.equal(actual[1], reference[1])
            assert torch.equal(actual[2], reference[2])
    assert cache.hits == cache.misses == 1
    assert torch.equal(torch.get_rng_state(), before)
    assert all(
        not value.requires_grad and value.grad_fn is None for value in cache.entries.values()
    )


def test_cache_bound_eviction_and_returned_masks_remain_unchanged():
    cache = CandidateValidityCache(max_bytes=1500, max_entries=2)
    device = torch.device("cpu")
    first = cache.mask(1, 7, 8, 1, device)
    retained = first.clone()
    assert first.is_contiguous()
    for batch in (2, 3, 1, 2):
        cache.mask(batch, 7, 8, 1, device)
        assert cache.bytes <= 1500
        assert len(cache.entries) <= 2
    assert torch.equal(first, retained)
    cache.clear()
    assert cache.bytes == len(cache.entries) == 0
    assert torch.equal(first, retained)


def test_context_patches_both_aliases_and_restores_on_error():
    original = local_transport.local_correlation_match
    with pytest.raises(RuntimeError, match="test"):
        with installed_compute() as cache:
            assert (
                causal_scale_ttc.local_correlation_match is local_transport.local_correlation_match
            )
            assert causal_scale_ttc.local_correlation_match is not original
            sample = torch.zeros(1, 2, 7, 8)
            local_transport.local_correlation_match(sample, sample, radius=1, temperature=0.1)
            assert cache.misses == 1
            raise RuntimeError("test")
    assert causal_scale_ttc.local_correlation_match is original
    assert local_transport.local_correlation_match is original
    assert cache.bytes == 0


@pytest.mark.parametrize("radius,temperature", [(0, 0.1), (4, 0.1), (1, 0), (1, float("nan"))])
def test_candidate_preserves_original_validation_errors(radius, temperature):
    sample = torch.zeros(1, 2, 7, 8)
    with pytest.raises(ValueError) as reference:
        local_transport.local_correlation_match(
            sample, sample, radius=radius, temperature=temperature
        )
    with pytest.raises(ValueError) as actual:
        local_correlation_match_cached(
            sample, sample, radius=radius, temperature=temperature, cache=CandidateValidityCache()
        )
    assert str(reference.value) == str(actual.value)
