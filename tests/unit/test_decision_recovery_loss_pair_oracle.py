import numpy as np
import pandas as pd
import pytest

from operational.decision_recovery import common, diagnostics, loss_pair_oracle


def inputs():
    target = np.array([-2.0, -4.0, 1.0, 2.0, 4.0, 8.0])
    rows = pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in range(6)],
            "group_id": "g",
            "sequence_id": "s",
            "target_ttc": target,
            "role": "P",
            "query_time_us": np.arange(6),
        }
    )
    tables = {}
    for name, prediction in {
        "L0": [-2.0, 3.0, 1.1, 2.1, 4.1, 1.0],
        "LW": [-2.0, -3.0, 1.5, 2.0, 5.0, 60.0],
    }.items():
        tables[name] = rows.copy()
        tables[name]["prediction_ttc"] = prediction
        tables[name]["raw_phase"] = -np.log1p(-0.1 / np.array(prediction))
    return tables, rows, tables["L0"].copy()


def test_same_vector_scored_for_every_metric_and_tie_l0():
    tables, rows, baseline = inputs()
    result, panel = loss_pair_oracle.compute(tables, rows, baseline)
    assert panel.loc[0, "selected_expert"] == "L0"
    assert panel.loc[5, "selected_expert"] == "LW"
    assert panel.loc[5, "prediction_ttc"] == 60  # MiD wins while absolute error worsens.
    assert result["metrics"]["ORACLE"] == common.score(rows, panel.prediction_ttc.to_numpy())
    assert result["comparisons"]["L0"]["rescues_and_harms"]["all"]["absolute_error_harmed"] >= 1
    assert result["comparisons"]["L0"]["rescues_and_harms"]["all"]["MiD_harmed"] == 0
    assert result["comparisons"]["LW"]["rescues_and_harms"]["all"]["MiD_harmed"] == 0
    assert panel.query_time_us.equals(rows.query_time_us)
    assert result["prediction_vector_sha256_float64_le"] == diagnostics._array_hash(
        panel.prediction_ttc.to_numpy()
    )
    assert result["selected_counts"]["L0"] + result["selected_counts"]["LW"] == len(rows)
    assert not result["candidate"] and not result["H_opened"]


@pytest.mark.parametrize("case", ["H", "target", "reorder"])
def test_rejects_h_or_mismatched_targets_and_order(case):
    tables, rows, baseline = inputs()
    if case == "H":
        for frame in [*tables.values(), rows, baseline]:
            frame["role"] = "H"
    elif case == "target":
        baseline.loc[0, "target_ttc"] = -2.01
    else:
        tables["LW"] = tables["LW"].iloc[::-1]
    with pytest.raises(ValueError):
        loss_pair_oracle.compute(tables, rows, baseline)
