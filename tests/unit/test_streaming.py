from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig
from e_jepa_ttc.runtime.streaming import StreamingTTCEstimator


def _estimator() -> StreamingTTCEstimator:
    model = ObjectCentricEventJEPA(
        ObjectJEPAConfig(
            in_channels=4,
            embedding_dim=16,
            feature_dim=16,
            predictor_depth=1,
            predictor_heads=4,
            dropout=0.0,
        )
    )
    return StreamingTTCEstimator(
        model,
        width=8,
        height=6,
        event_bins=2,
        event_window_ms=100,
        history_steps=3,
        device="cpu",
    )


def test_streaming_prediction_uses_bounded_causal_history() -> None:
    estimator = _estimator()
    timestamps = np.arange(0, 300_000, 1_000, dtype=np.int64)
    estimator.push_events(
        timestamps % 8,
        timestamps % 6,
        timestamps,
        np.where(np.arange(timestamps.size) % 2, 1, -1),
    )
    box = np.asarray([0.25, 0.25, 0.75, 0.75], dtype=np.float32)
    for endpoint in (100_000, 200_000, 300_000):
        estimator.push_observation(endpoint, box)

    assert estimator.ready(300_000)
    result = estimator.predict(300_000)

    assert result.event_count == 300
    assert result.risk_state in {"SAFE", "WATCH", "WARNING", "CRITICAL", "UNKNOWN"}
    assert result.preprocessing_ms >= 0
    assert result.inference_ms >= 0
    assert estimator._t_us.size <= timestamps.size


def test_streaming_rejects_timestamp_rollback_and_reset_recovers() -> None:
    estimator = _estimator()
    estimator.push_events(
        np.asarray([1]),
        np.asarray([1]),
        np.asarray([100]),
        np.asarray([1]),
    )
    with pytest.raises(ValueError, match="rollback"):
        estimator.push_events(
            np.asarray([1]),
            np.asarray([1]),
            np.asarray([99]),
            np.asarray([1]),
        )
    estimator.reset()
    estimator.push_events(
        np.asarray([1]),
        np.asarray([1]),
        np.asarray([99]),
        np.asarray([1]),
    )

