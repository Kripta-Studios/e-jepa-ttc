"""Raw cost requests remain label blind and balanced across acquisition groups."""

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement.raw_cost import indices_by_group


def test_query_selection_ignores_labels_and_preserves_group_counts():
    frame = pd.DataFrame(
        {
            "sample_token": [f"token-{i}" for i in range(36)],
            "group_id": np.repeat(["a", "b", "c"], 12),
            "target_ttc": np.linspace(-8, 10, 36),
        }
    )
    chosen = indices_by_group(frame)
    frame["target_ttc"] = np.nan
    frame["prediction_ttc"] = 1000000
    assert chosen == indices_by_group(frame)
    assert len(chosen) == len(set(chosen)) == 24
    assert frame.iloc[chosen].group_id.value_counts().to_dict() == {"a": 8, "b": 8, "c": 8}


def test_query_selection_rejects_inadequate_group_support():
    frame = pd.DataFrame({"sample_token": [str(i) for i in range(7)], "group_id": ["a"] * 7})
    with pytest.raises(ValueError, match="fewer than eight"):
        indices_by_group(frame)
