"""Canonical arm/source wiring, without scientific or technical optimizer updates."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.arms import resolve_arm
from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer
from e_jepa_ttc.simplex_t.registry import FitSpec, registered_graph


def graph(d1=True):
    return registered_graph(
        d1=d1,
        density=d1,
        t3=True,
        latent=True,
        replicate_scalar=True,
        replicate_latent=True,
    )


def test_all_registered_arms_resolve_with_exact_dimensions():
    for d1 in (False, True):
        fits = graph(d1)
        for spec in fits:
            binding = resolve_arm(spec, fits)
            assert binding.history in {1, 4, 8, 16}
            assert binding.model.feature_count == (145 if spec.name.startswith("LATENT") else 17)
            assert binding.model.output_mode == (
                "selector"
                if spec.name.startswith("SELECTOR")
                else "free"
                if spec.name.startswith("FREE")
                else "residual"
            )
            assert (binding.model.backbone == "transformer") == spec.name.startswith("TRANSFORMER")


@pytest.mark.parametrize(
    "change",
    [
        {"name": "TPR-D0-H8-C128"},
        {"updates": 10},
        {"seed": 42},
        {"fold": 3},
        {"stage": "T4"},
    ],
)
def test_unregistered_fit_rejected_even_in_supplied_graph(change):
    spec = replace(FitSpec("T2", "TPR-D0-H8-C160", 0), **change)
    with pytest.raises(ValueError, match="registered"):
        resolve_arm(spec, [spec])


def test_unavailable_registered_arm_rejected():
    with pytest.raises(ValueError, match="frozen graph"):
        resolve_arm(FitSpec("T2", "TPR-D1-H8-C160", 0), graph(False))


def test_control_binding_preserves_current_timing_and_shared_cache():
    features = np.arange(8 * 17, dtype=np.float32).reshape(8, 17)
    cache = CachedQueries(
        features=features,
        anchor_us=np.arange(8, dtype=np.int64) * 10000,
        available_us=np.arange(8, dtype=np.int64) * 10000 + 1000,
        history=np.arange(8, dtype=np.int64)[None],
        target_phase=np.array([0.1]),
        mass=np.array([1.0]),
        normalizer=Normalizer(np.zeros(17), np.ones(17), "a" * 64),
        identity_sha256="b" * 64,
    )
    spec = FitSpec("T2", "REPEAT_CURRENT-D1-H8-C160", 0)
    binding = resolve_arm(spec, graph())
    controlled = binding.source(cache)
    ids = torch.tensor([0])
    original = cache.gather(ids)
    result = controlled.gather(ids)
    assert controlled.normalizer is cache.normalizer
    assert controlled.identity_sha256 != cache.identity_sha256
    assert cache.control == "NONE"
    assert torch.equal(result[0], original[0][:, -1:].expand_as(result[0]))
    for index in range(1, 6):
        assert torch.equal(result[index], original[index])
    with pytest.raises(ValueError, match="unperturbed"):
        binding.source(controlled)
