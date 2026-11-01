import numpy as np

from operational.phase_campaign.review_controls import endpoint_result, shared_scores


def test_paired_baseline_excludes_exactly_the_comparator_missing_rows():
    target = np.array([1.0, 2.0, 3.0])
    result = shared_scores(
        target, {"F0": np.array([1.0, 99.0, 4.0]), "bbox": np.array([1.0, np.nan, 3.0])}
    )
    assert result["F0"]["population"] == 2
    assert result["F0"]["mae_seconds"] == 0.5
    assert result["bbox"]["mae_seconds"] == 0


def test_invalid_endpoint_estimates_have_explicit_reasons():
    for q in (0.0, -1.0, np.nan, np.inf):
        value, reason = endpoint_result(q)
        assert np.isnan(value)
        assert reason == "nonpositive_or_nonfinite_endpoint_inverse_ttc"
    assert endpoint_result(0.5) == (2.0, "ok")
