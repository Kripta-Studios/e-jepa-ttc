"""Constant caching must preserve forward and backward, including boundary masks."""

import pytest
import torch

from e_jepa_ttc.models.local_transport import local_correlation_match
from operational import oof_transport_cache as cache


@pytest.fixture(autouse=True)
def clean_cache():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cache._CACHE.clear()
    cache._STATS.update(hits=0, misses=0, bytes=0, evictions=0)
    yield
    cache._CACHE.clear()
    cache._STATS["bytes"] = 0
    torch.set_num_threads(threads)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("radius", [2, 4])
def test_forward_backward_exact_cold_and_warm(device, radius):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(71)
    previous = torch.randn(2, 4, 12, 13, device=device)
    current = torch.randn_like(previous)
    outputs = []
    for function in (
        local_correlation_match,
        cache.cached_local_correlation_match,
        cache.cached_local_correlation_match,
    ):
        p = previous.clone().requires_grad_()
        c = current.clone().requires_grad_()
        result = function(p, c, radius=radius, temperature=0.1, return_probability=True)
        values = [
            getattr(result, key)
            for key in ("dx", "dy", "confidence_margin", "entropy", "valid", "probability")
        ]
        (
            result.dx.sum()
            + result.dy.sum()
            + result.entropy.sum()
            + result.confidence_margin.sum()
        ).backward()
        outputs.append((values, p.grad, c.grad))
    for values, pgrad, cgrad in outputs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(values, outputs[0][0], strict=True))
        assert torch.equal(pgrad, outputs[0][1])
        assert torch.equal(cgrad, outputs[0][2])
    assert cache._STATS["hits"] == 1
    assert all(not t.requires_grad for values, _ in cache._CACHE.values() for t in values)


def test_cache_bound_and_eviction(monkeypatch):
    monkeypatch.setattr(cache, "_LIMIT", 3000)
    for width in (10, 11, 12):
        cache._constants(1, 10, width, 1, torch.device("cpu"), torch.float32)
    assert cache._STATS["bytes"] <= 3000
    assert cache._STATS["evictions"] >= 1
    assert len(cache._CACHE) <= 2


def test_invalid_shape_still_rejected():
    with pytest.raises(ValueError, match="share shape"):
        cache.cached_local_correlation_match(
            torch.zeros(2, 3), torch.zeros(2, 3), radius=1, temperature=0.1
        )
