"""Check seed aggregation preserves loss semantics and whole-group uncertainty."""

import copy

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement.fusion_replication_evaluate import loss_bootstrap, mean_metrics
from operational.range_transfer.analyze import metrics


def frame(scale):
    truth = np.tile([2.0, 4.0, 8.0, -2.0], 2)
    return pd.DataFrame(
        {
            "sample_token": [str(i) for i in range(8)],
            "group_id": ["a"] * 4 + ["b"] * 4,
            "target_ttc": truth,
            "prediction_ttc": truth * scale,
        }
    )


def score(value):
    return metrics(
        value.target_ttc.to_numpy(), value.prediction_ttc.to_numpy(), value.group_id.to_numpy()
    )


def test_identical_seed_losses_have_zero_cluster_interval():
    a = frame(1.1)
    result = loss_bootstrap([a], [a, a, a])
    for value in result["intervals"].values():
        np.testing.assert_allclose(value["ci95_if_all_defined"], [0.0, 0.0], atol=1e-12, rtol=0)


def test_seed_loss_mean_does_not_average_opposing_ttc_signs():
    reference = frame(1.0)
    candidates = [frame(1.2), frame(-1.2), frame(1.5)]
    expected = np.mean([score(value)["overall_MiD"] for value in candidates])
    interval = loss_bootstrap([reference], candidates)["intervals"]["overall"][
        "ci95_if_all_defined"
    ]
    np.testing.assert_allclose(interval, [expected, expected], rtol=1e-12)
    wrong = frame((1.2 - 1.2 + 1.5) / 3)
    assert not np.isclose(expected, score(wrong)["overall_MiD"])


def test_gate_metrics_are_explicit_means_without_seed7_diagnostics():
    values = [score(frame(scale)) for scale in (1.1, 1.2, 1.3)]
    original = copy.deepcopy(values)
    result = mean_metrics(values)
    assert values == original
    assert result["overall_MiD"] == pytest.approx(np.mean([v["overall_MiD"] for v in values]))
    assert result["diagnostics"]["mae_s"] == pytest.approx(
        np.mean([v["diagnostics"]["mae_s"] for v in values])
    )
    assert "rmse_s" not in result["diagnostics"]


def test_bootstrap_rejects_reordered_or_missing_queries():
    a = frame(1.1)
    with pytest.raises(ValueError, match="cohorts differ"):
        loss_bootstrap([a], [a.iloc[::-1]])
    with pytest.raises(ValueError, match="cohorts differ"):
        loss_bootstrap([a], [a.iloc[:-1]])


def test_invalid_seed_is_never_silently_dropped():
    a = frame(1.1)
    bad = a.copy()
    bad.loc[0, "prediction_ttc"] = np.nan
    with pytest.raises(ValueError, match="cannot discard"):
        loss_bootstrap([a], [a, bad, a])
    with pytest.raises(ValueError, match="complete signed-domain"):
        mean_metrics([score(a), score(bad)])
