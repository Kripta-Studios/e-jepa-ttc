"""Deduplicated, global-mass cache tests with explicit held-out normalizer checks."""

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.cache import CachedQueries, fit_normalizer, training_mass


def make_cache(length=8, control="NONE", latent=False, zero=False):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 145 if latent else 17)).astype(np.float32)
    history = np.array([[-1, -1, -1, 0, 1, 2, 3, 4], list(range(12, 20))])
    times = np.arange(20, dtype=np.int64) * 100000 + 10**16
    normalizer = fit_normalizer(features, history, np.ones(20, dtype=bool))
    return CachedQueries(
        features,
        times,
        times + 1000,
        history,
        np.array([0.1, 0.2]),
        np.array([0.5, 0.5]),
        normalizer,
        "a" * 64,
        length,
        control,
        zero,
    )


def test_normalizer_deduplicates_observations():
    features = np.arange(34, dtype=np.float32).reshape(2, 17)
    normalizer = fit_normalizer(features, np.array([0, 0, 0, 1]), np.ones(2, dtype=bool))
    np.testing.assert_array_equal(normalizer.mean, features.mean(0))
    with pytest.raises(ValueError, match="non-TRAIN"):
        fit_normalizer(features, np.array([0, 1]), np.array([True, False]))


def test_present_buckets_do_not_change_sequence_mass():
    mass = training_mass(np.array([1.0, 2.0, -3.0]), np.array(["a", "a", "b"]))
    np.testing.assert_allclose(mass, [0.25, 0.25, 0.5])


def test_h1_resets_gap_and_population():
    cache = make_cache(length=1)
    x, timing, valid, experts, target, mass = cache.gather(torch.tensor([0, 1]))
    assert x.shape == (2, 1, 17) and cache.population == 2
    assert (timing[:, :, 2] == 0).all()
    assert valid.all() and len(experts) == len(target) == len(mass) == 2


@pytest.mark.parametrize("control", ["PAST_REVERSED", "REPEAT_CURRENT"])
def test_controls_preserve_actual_time_and_mask(control):
    original = make_cache().gather(torch.tensor([0, 1]))
    changed = make_cache(control=control).gather(torch.tensor([0, 1]))
    torch.testing.assert_close(original[0][:, -1], changed[0][:, -1])
    for first, second in zip(original[1:], changed[1:], strict=True):
        torch.testing.assert_close(first, second)


def test_latent_zero_after_standardization():
    batch = make_cache(latent=True, zero=True).gather(torch.tensor([0, 1]))
    assert batch[0].shape[-1] == 145 and (batch[0][:, :, 17:] == 0).all()


def test_future_dependency_rejected():
    cache = make_cache()
    cache.available_us[18] = cache.available_us[19] + 1
    with pytest.raises(ValueError, match="future"):
        cache.gather(torch.tensor([1]))
