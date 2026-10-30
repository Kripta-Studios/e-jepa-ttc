import numpy as np

from operational.phase_campaign.perturbations import perturb_events


def test_raw_perturbation_is_deterministic_causal_and_does_not_mutate_cache():
    raw = {
        "x": np.arange(100) % 10,
        "y": np.arange(100) % 7,
        "t": np.arange(100, dtype=np.int64) * 10,
        "p": np.arange(100) % 2,
    }
    saved = {k: v.copy() for k, v in raw.items()}
    first = perturb_events(raw, "jitter_200", 0, 1000, 10, 7)
    second = perturb_events(raw, "jitter_200", 0, 1000, 10, 7)
    assert len(first["t"]) > 0 and np.all(np.diff(first["t"]) >= 0)
    assert np.all((first["t"] >= 0) & (first["t"] < 1000))
    for key in raw:
        np.testing.assert_array_equal(raw[key], saved[key])
        np.testing.assert_array_equal(first[key], second[key])


def test_polarity_drop_preserves_native_binary_convention():
    raw = {"x": np.arange(4), "y": np.zeros(4), "t": np.arange(4), "p": np.array([0, 1, 0, 1])}
    negative = perturb_events(raw, "drop_positive", 0, 4, 4, 2)
    np.testing.assert_array_equal(negative["p"], [0, 0])
    np.testing.assert_array_equal(negative["x"], [0, 2])
