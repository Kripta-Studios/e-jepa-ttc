"""Diagnostic populations must keep unlabeled inputs and use GT only for errors."""

import json

import numpy as np
import pandas as pd

from operational.efficient_context.common import digest
from operational.ttc_revision.diagnose import diagnose


def test_diagnostic_denominators_and_timing_conventions(tmp_path):
    train = np.zeros((2, 17), np.float32)
    train[:, 8:11] = -np.log1p(-0.1 / 2)
    history = np.full((2, 8), -1, np.int64)
    history[:, -1] = (0, 1)
    cache = tmp_path / "train.npz"
    np.savez(
        cache,
        features=train,
        history=history,
        anchor_us=np.array([1000000, 2000000]),
        available_us=np.array([1001000, 2002000]),
    )
    for cohort in ("dev32", "fcwd"):
        fragments = tmp_path / f"replay_{cohort}/fragments"
        fragments.mkdir(parents=True)
        hashes = {}
        for index in range(3):
            features = np.tile(train[0], (8, 1))
            features[:, 0] = 10 if index else 0
            path = fragments / f"{index:06d}.npz"
            np.savez(
                path,
                query_id=np.asarray(f"q{index}"),
                features=features,
                canonical=np.full(3, 0.999),
                compact=np.full(3, 1.001 if index == 0 else 0.999),
                direct_gpu=np.full(3, 2.0),
                direct_compact_gpu=np.full(3, 2.0),
            )
            hashes[path.name] = digest(path)
        (fragments.parent / "REPLAY_COMPLETE.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "fragments": hashes,
                }
            )
        )
        (tmp_path / cohort).mkdir()
        pd.DataFrame(
            {
                "query_id": ["q0", "q1", "q2"],
                "truth_ttc_seconds": [1.0, 4.0, np.nan],
                "scenario_family": ["a", "b", "c"],
                "H8_median3": [2.0, 2.0, 2.0],
                "Direct_median3": [1.0, 3.0, 2.0],
                "a5_transport": [0.0, 0.0, 0.0],
            }
        ).to_csv(tmp_path / cohort / "ALL_PREDICTIONS.csv", index=False)
    diagnose(tmp_path, cache)
    scored = pd.read_csv(tmp_path / "UPSTREAM_METRICS.csv")
    row = scored[
        (scored.population == "dev32") & (scored.method == "H8_median3") & (scored.cohort == "all")
    ].iloc[0]
    assert row["n"] == 2
    assert row["mae"] == 1.5
    shift = pd.read_csv(tmp_path / "FEATURE_SHIFT.csv")
    row = shift[(shift.population == "dev32") & (shift.feature == "log_roi_count")].iloc[0]
    assert row["n_train"] == 2 and row["n_transfer"] == 3
    assert np.isclose(row["outside_train_envelope_fraction"], 2 / 3)
    joined = pd.read_csv(tmp_path / "dev32/UPSTREAM_DIAGNOSTICS.csv")
    assert len(joined) == 3 and "a5_transport" in joined
    impact = pd.read_csv(tmp_path / "NUMERICAL_METRIC_IMPACT.csv")
    row = impact[(impact.population == "dev32") & (impact.method == "H8_median3")].iloc[0]
    assert row["n"] == 2
    assert row["alarm_decision_changes"] == 1 and row["urgent_miss_changes"] == 1
    assert row["sign_changes"] == 0
    assert abs(row["mae_change_compact_minus_canonical"]) < 1e-12
    timing = json.loads((tmp_path / "TIMING_DIAGNOSTICS.json").read_text())["channels"]
    assert (
        timing["channel1_train_minus_transfer_convention"]["fraction_above_one_microsecond_abs"]
        == 0
    )
    assert (
        timing["channel3_train_minus_transfer_convention"]["quantiles_seconds"]["median"] == 0.0015
    )
