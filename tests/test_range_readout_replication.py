"""Replica aggregation preserves query losses and missing-band semantics."""

from io import StringIO

import numpy as np
import pandas as pd

from operational.range_readout.replication import aggregate


def frames(negative: bool = True) -> list[pd.DataFrame]:
    target = np.tile([0.5, 4.0, 8.0, -2.0 if negative else 2.0], 2)
    base = pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in range(8)],
            "group_id": ["g0"] * 4 + ["g1"] * 4,
            "target_ttc": target,
        }
    )
    return [base.assign(prediction_ttc=target * scale) for scale in (0.5, 1.0, 1.5)]


def test_replica_loss_is_not_loss_of_mean_prediction():
    values = frames()
    summary, table = aggregate(values)
    assert np.allclose(np.stack([v.prediction_ttc for v in values]).mean(0), values[0].target_ttc)
    assert summary["mean_MiD"] > 0
    assert summary["diagnostics"]["mae_s"] > 0
    assert summary["predictions_ensembled"] is False
    assert (table.mean_MiD_loss > 0).all()


def test_missing_negative_band_does_not_receive_overall():
    summary, _ = aggregate(frames(negative=False))
    assert summary["bins"]["negative"]["count"] == 0
    assert summary["overall_MiD"] is None


def test_replica_losses_survive_native_float32_csv_round_trip():
    original = frames()
    for value in original:
        value["prediction_ttc"] = (value.prediction_ttc.to_numpy() * 1.012345678).astype(np.float32)
    restored = [
        pd.read_csv(
            StringIO(value.to_csv(index=False)),
            converters={"target_ttc": float, "prediction_ttc": np.float32},
        )
        for value in original
    ]
    before, loss_before = aggregate(original)
    after, loss_after = aggregate(restored)
    assert before == after
    pd.testing.assert_frame_equal(loss_before, loss_after)
