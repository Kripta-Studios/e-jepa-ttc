"""Input-only thinning preserves diversity and is independent of input ordering."""

import numpy as np

from e_jepa_ttc.simplex_t.density_selection import select_time_density


def test_time_density_is_permutation_invariant():
    seq = np.repeat(["a", "b"], 20)
    ts = np.tile(np.arange(20), 2)
    tokens = np.asarray([f"token{i:03d}" for i in range(40)])
    rows = select_time_density(seq, ts, tokens, per_sequence=5)
    permutation = np.random.default_rng(7).permutation(40)
    other = select_time_density(
        seq[permutation], ts[permutation], tokens[permutation], per_sequence=5
    )
    np.testing.assert_array_equal(tokens[rows], tokens[permutation][other])
    np.testing.assert_array_equal(ts[rows], [0, 4, 9, 14, 19] * 2)


def test_small_sequences_are_retained():
    seq = np.asarray(["a", "b", "b"])
    ts = np.asarray([1, 2, 3])
    tokens = np.asarray(["a", "b", "c"])
    assert select_time_density(seq, ts, tokens, per_sequence=512).tolist() == [0, 1, 2]
