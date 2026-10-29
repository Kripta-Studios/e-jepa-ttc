"""Latency intervals preserve family dependence and reject incomplete pairs."""

import pandas as pd
import pytest

from operational.ttc_revision.latency_analysis import paired_latency


def test_family_cluster_and_pair_completeness(tmp_path):
    rows = [
        {
            "mode": "trial",
            "scenario_family": family,
            "query_id": family,
            "iteration": iteration,
            "system": system,
            "warmup": False,
            "e2e_ms": duration,
        }
        for family, old in (("a", 100), ("b", 200))
        for iteration in range(5)
        for system, duration in (("h8_legacy_three", old), ("h8_fast_three", old / 2))
    ]
    table = pd.DataFrame(rows)
    result = paired_latency(table, tmp_path).iloc[0]
    assert result["families"] == 2
    assert result["pairs"] == 10
    assert result["difference_ms"] == -75
    assert result["reduction_ci_low_percent"] == 50
    assert result["reduction_ci_high_percent"] == 50
    assert result["difference_ci_low_ms"] == -100
    assert result["difference_ci_high_ms"] == -50
    with pytest.raises(ValueError, match="matching baseline"):
        paired_latency(table.iloc[:-1], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        paired_latency(pd.concat([table, table.iloc[:1]]), tmp_path)
