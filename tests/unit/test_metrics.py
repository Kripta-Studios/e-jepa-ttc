import numpy as np

from e_jepa_ttc.evaluation.metrics import regression_metrics


def test_regression_metrics_include_relative_error_percentage() -> None:
    metrics = regression_metrics(
        np.array([2.0, 4.0], dtype=np.float64),
        np.array([1.0, 6.0], dtype=np.float64),
    )

    assert metrics["mean_abs_relative_error_pct"] == 50.0
    assert metrics["median_abs_relative_error_pct"] == 50.0
