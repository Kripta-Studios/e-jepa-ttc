from __future__ import annotations

import numpy as np

import scripts.analyze_causal_scale_eap_failure as analysis


def test_relationship_reports_exact_linear_signal() -> None:
    reference = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    observed = 2.0 * reference + 1.0

    result = analysis._relationship(reference, observed)

    assert result["pearson"] == 1.0
    assert result["slope"] == 2.0
    assert result["count"] == 3


def test_relationship_preserves_nonfinite_exclusion() -> None:
    result = analysis._relationship(
        np.asarray([0.0, 1.0, np.nan]),
        np.asarray([0.0, 1.0, 2.0]),
    )

    assert result["count"] == 2
    assert result["mae"] == 0.0
