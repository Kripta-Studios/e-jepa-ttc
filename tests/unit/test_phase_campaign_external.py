import numpy as np
import pytest


def test_external_join_requires_identity_and_clock_not_nearest_score():
    from operational.phase_campaign.external import join_external_targets

    boxes = [{"frame_index": 1, "anchor_us": 40081}, {"frame_index": 2, "anchor_us": 80163}]
    labels = np.array([[1, 0.0400819778, 2.0, 0.5, 4.0], [2, 0.0801639556, 1.98, 0.5, 3.96]])
    np.testing.assert_allclose(join_external_targets(boxes, labels), [4.0, 3.96])
    labels[0, 1] += 0.04
    with pytest.raises(ValueError, match="clock"):
        join_external_targets(boxes, labels)
