"""Paired system comparisons preserve every query and its original target."""

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement.system_compare import require_cohort


@pytest.mark.parametrize("fault", ["missing", "duplicate", "order", "group", "target"])
def test_system_comparison_rejects_changed_cohort(fault):
    rows = pd.DataFrame(
        {"sample_token": ["a", "b"], "group_id": ["g", "h"], "target_ttc": [0.7, -3.0]}
    )
    frame = rows.assign(prediction_ttc=[0.8, -2.0])
    if fault == "missing":
        frame = frame.iloc[:1]
    elif fault == "duplicate":
        frame.loc[1, "sample_token"] = "a"
    elif fault == "order":
        frame = frame.iloc[::-1]
    elif fault == "group":
        frame.loc[1, "group_id"] = "g"
    else:
        frame.loc[1, "target_ttc"] = -3.1
    with pytest.raises(ValueError, match="query exactly once|cohort differs"):
        require_cohort(frame, rows)


def test_invalid_predictions_are_preserved_for_coverage_gate():
    rows = pd.DataFrame(
        {"sample_token": ["a", "b"], "group_id": ["g", "h"], "target_ttc": [0.7, -3.0]}
    )
    frame = rows.assign(prediction_ttc=[np.nan, 0.0])
    require_cohort(frame, rows)
    assert len(frame) == 2 and np.isnan(frame.prediction_ttc.iloc[0])
