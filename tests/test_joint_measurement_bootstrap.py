"""Small exact metric checks independent of learned predictions."""

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement.bootstrap import compare


def test_group_bootstrap_known_phase_shift(tmp_path):
    target = np.tile([0.7, 4.0, 8.0, -3.0], 3)
    phase = -np.log1p(-0.1 / target)
    reference = pd.DataFrame(
        {
            "sample_token": [str(i) for i in range(12)],
            "target_ttc": target,
            "prediction_ttc": target,
            "group_id": np.repeat(["a", "b", "c"], 4),
        }
    )
    candidate = reference.copy()
    candidate["prediction_ttc"] = -0.1 / np.expm1(-(phase + 0.001))
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    reference.to_parquet(a)
    candidate.to_parquet(b)
    result = compare(a, b, tmp_path / "result")
    for value in result["intervals"].values():
        assert value["undefined_replicates"] == 0
        np.testing.assert_allclose(value["ci95_if_all_defined"], [10, 10], atol=1e-9)
    candidate.loc[0, "group_id"] = "misaligned"
    candidate.to_parquet(b)
    with pytest.raises(ValueError, match="identical ordered cohorts"):
        compare(a, b, tmp_path / "bad")
