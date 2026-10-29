"""Independent formulas cover invalid rows and missing bands, not only good predictions."""

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement.verify_predictions import recalculate
from operational.range_transfer.analyze import metrics


@pytest.mark.parametrize("case", ["valid", "invalid", "missing_band"])
def test_independent_formula_matches_declared_metric_scope(case):
    target = np.tile([0.7, 4.0, 8.0, -3.0], 3)
    prediction = target * 1.1
    groups = np.repeat(["a", "b", "c"], 4)
    if case == "invalid":
        prediction[:4] = [0.0, np.nan, -0.05, -0.1]
    if case == "missing_band":
        keep = target > 0
        target, prediction, groups = target[keep], prediction[keep], groups[keep]
    frame = pd.DataFrame({"target_ttc": target, "prediction_ttc": prediction, "group_id": groups})
    produced = metrics(target, prediction, groups)
    for name, expected in recalculate(frame).items():
        value = produced
        for key in name.split("."):
            value = value[key]
        if expected is None:
            assert value is None
        else:
            np.testing.assert_allclose(value, expected, atol=1e-9, rtol=1e-10)
