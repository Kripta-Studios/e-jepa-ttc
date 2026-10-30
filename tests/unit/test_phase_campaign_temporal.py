import numpy as np


def test_temporal_transport_is_exact_at_constant_closing_speed():
    from operational.phase_campaign.temporal import causal_transport_filter

    timestamps = np.arange(10) * 100000
    ttc = 5 - timestamps / 1e6
    actual = causal_transport_filter(ttc, timestamps)
    np.testing.assert_allclose(actual, ttc, rtol=0, atol=1e-12)


def test_temporal_filter_preserves_missingness_sign_changes_and_causality():
    from operational.phase_campaign.temporal import causal_transport_filter

    ttc = np.array([5.0, 4.0, np.nan, -2.0, 1.0])
    timestamps = np.arange(5) * 100000
    actual = causal_transport_filter(ttc, timestamps)
    assert np.isnan(actual[2])
    assert actual[3] == -2
    assert actual[4] == 1
    changed = ttc.copy()
    changed[-1] = 50
    np.testing.assert_equal(causal_transport_filter(changed, timestamps)[:-1], actual[:-1])
