"""Synthetic tests only: routing, fold exclusion and persisted output contracts."""

import json

import numpy as np
import pandas as pd
import pytest

from operational.decision_recovery.router import (
    FEATURES,
    development_gate,
    evaluate_pair,
    features,
    fit,
    predict,
    training_weights,
)


def population():
    target = np.tile([0.8, 2.0, 4.0, 5.0, 7.0, 8.0, -2.0, -4.0], 4)
    groups = np.repeat(["g0", "g1", "g2", "g3"], 8)
    base = target * np.repeat([1.1, 1.3, 1.5, 1.7], 8)
    specialist = target * np.tile([1.01, 1.02, 1.03, 1.04, 1.6, 1.7, 1.8, 1.9], 4)
    frame = pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in range(len(target))],
            "group_id": groups,
            "sequence_id": groups,
            "target_ttc": target,
            "prediction_ttc": base,
            "raw_phase": -np.log1p(-0.1 / base),
            "role": "H",
        }
    )
    special = frame.copy()
    special["prediction_ttc"] = specialist
    special["raw_phase"] = -np.log1p(-0.1 / specialist)
    return {"NATIVE": frame, "A5": frame.copy(), "FULL": special}


def test_exact_feature_schema_and_target_independence():
    tables = population()
    a, b = tables["NATIVE"].prediction_ttc.to_numpy(), tables["FULL"].prediction_ttc.to_numpy()
    x = features(a, b)
    assert x.shape == (32, 8)
    assert len(FEATURES) == 8
    np.testing.assert_array_equal(x[:, 2], x[:, 1] - x[:, 0])
    np.testing.assert_array_equal(x[:, 3], abs(x[:, 2]))
    for frame in tables.values():
        frame["target_ttc"] = 999.0
        frame["group_id"] = "changed"
        frame["sequence_id"] = "changed"
    np.testing.assert_array_equal(x, features(a, b))


@pytest.mark.parametrize("bad", [0.0, 0.05, 0.1, np.nan, np.inf, -np.nextafter(0.0, 1.0)])
def test_invalid_phase_domain_rejected(bad):
    with pytest.raises(ValueError):
        features(np.array([bad]), np.array([2.0]))


def test_group_band_weights_are_global_and_handle_unequal_cells():
    target = np.array([1, 1, 4, 7, -2, 1, 4, 4, 7, -2], dtype=float)
    group = np.array(["a"] * 5 + ["b"] * 5)
    w = training_weights(target, group)
    assert w.sum() == pytest.approx(1)
    assert w[:2].sum() == pytest.approx(0.25)
    assert w[5] == pytest.approx(0.25)
    assert w[2] == pytest.approx(0.15)
    assert w[6:8].sum() == pytest.approx(0.15)
    assert w[target < 0].sum() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="missing band"):
        training_weights(target[target > 0], group[target > 0])


def test_whole_expert_selection_tie_and_json_replay():
    a, b = np.array([2.0, -4.0, 3.0]), np.array([-3.0, 5.0, 2.0])
    x, w = features(a, b), np.ones(3) / 3
    model = fit(x, np.zeros(3), w)
    np.testing.assert_array_equal(predict(model, a, b), a)
    model["intercept"] = -1
    np.testing.assert_array_equal(predict(model, a, b), b)
    restored = json.loads(json.dumps(model, allow_nan=False))
    np.testing.assert_array_equal(predict(restored, a, b), b)
    restored["threshold"] = 0.01
    with pytest.raises(ValueError, match="contract"):
        predict(restored, a, b)


def test_missing_group_band_weights_match_bandwise_scorer():
    from operational.range_transfer.analyze import metrics

    target = np.array([1, 4, 7, -2, 1, 4, 7], dtype=float)
    groups = np.array(["a"] * 4 + ["b"] * 3)
    prediction = target * np.array([1.1, 1.2, 1.3, 2.0, 1.4, 1.5, 1.6])
    weights = training_weights(target, groups)
    assert weights[3] == pytest.approx(0.1)
    assert weights[:4].sum() == pytest.approx(0.55)
    assert weights[4:].sum() == pytest.approx(0.45)
    loss = 10000 * abs(np.log1p(-0.1 / target) - np.log1p(-0.1 / prediction))
    assert weights @ loss == pytest.approx(metrics(target, prediction, groups)["overall_MiD"])


def test_development_gate_separate_sign_band_and_improvement_checks():
    import copy

    reference = {
        "overall_MiD": 100.0,
        "complete_prediction_and_mid_coverage": True,
        "diagnostics": {"sign_error_rate": 0.05},
        "bins": {
            name: {"group_macro_mid": 100.0} for name in ("crucial", "small", "large", "negative")
        },
    }
    candidate = copy.deepcopy(reference)
    candidate["overall_MiD"] = 97.0
    assert development_gate(reference, candidate)["eligible"]
    candidate["overall_MiD"] = 101.0
    candidate["bins"]["crucial"]["group_macro_mid"] = 95.0
    assert development_gate(reference, candidate)["eligible"]
    candidate["bins"]["negative"]["group_macro_mid"] = 110.1
    assert not development_gate(reference, candidate)["eligible"]
    candidate["bins"]["negative"]["group_macro_mid"] = 100
    candidate["diagnostics"]["sign_error_rate"] = 0.056
    assert not development_gate(reference, candidate)["eligible"]


def test_ridge_intercept_unpenalized_and_weighted_normalizer():
    x = np.arange(80, dtype=float).reshape(10, 8)
    w = np.arange(1, 11, dtype=float)
    w /= w.sum()
    model = fit(x, np.full(10, 4.0), w)
    np.testing.assert_allclose(model["mean"], w @ x)
    np.testing.assert_allclose(model["coefficient"], 0, atol=1e-10)
    assert model["intercept"] == pytest.approx(4.0)


def test_logo_excludes_holdout_labels_features_and_normalizers(tmp_path, monkeypatch):
    from operational.decision_recovery import common

    monkeypatch.setattr(common, "RUN", tmp_path)
    tables = population()
    original = evaluate_pair(tables, "NATIVE", "FULL", output_dir=tmp_path / "first")
    altered = {name: frame.copy() for name, frame in tables.items()}
    for frame in altered.values():
        heldout = frame.group_id == "g0"
        frame.loc[heldout, "target_ttc"] *= 0.9
        frame.loc[heldout, "prediction_ttc"] *= 2
    changed = evaluate_pair(altered, "NATIVE", "FULL", output_dir=tmp_path / "changed")
    for kind in ("R0", "SHUFFLED"):
        before = json.loads((tmp_path / "first" / f"fold_000_{kind}.json").read_text())
        after = json.loads((tmp_path / "changed" / f"fold_000_{kind}.json").read_text())
        assert before == after
        assert "g0" not in before["training_groups"]
        train = tables["NATIVE"].group_id != "g0"
        expected = training_weights(
            tables["NATIVE"].target_ttc[train].to_numpy(),
            tables["NATIVE"].group_id[train].to_numpy(),
        ) @ features(
            tables["NATIVE"].prediction_ttc[train].to_numpy(),
            tables["FULL"].prediction_ttc[train].to_numpy(),
        )
        np.testing.assert_allclose(before["mean"], expected)
        emitted = pd.read_parquet(tmp_path / "first" / f"H_{kind}_OOF.parquet")
        assert (
            (emitted.prediction_ttc == tables["NATIVE"].prediction_ttc)
            | (emitted.prediction_ttc == tables["FULL"].prediction_ttc)
        ).all()
        np.testing.assert_array_equal(
            emitted.raw_phase.to_numpy(),
            np.where(
                emitted.router_selected_special,
                tables["FULL"].raw_phase,
                tables["NATIVE"].raw_phase,
            ),
        )
    assert not original["system_promotion"] and not changed["official_confirmation"]
    assert set(original["results"]["R0"]["gates"]) == {"NATIVE", "FULL", "A5"}
    assert original["results"]["R0"]["bootstrap"]["A5"]["replicates"] == 2000
    final = json.loads((tmp_path / "first" / "ALL_H_DEVELOPMENT_MODEL.json").read_text())
    assert final["H_in_sample"] and not final["system_promotion"]


def test_population_and_role_mismatch_fail_closed(tmp_path):
    tables = population()
    tables["FULL"].loc[0, "group_id"] = "wrong"
    with pytest.raises(ValueError, match="Population mismatch"):
        evaluate_pair(tables, "NATIVE", "FULL", output_dir=tmp_path)
    tables = population()
    for frame in tables.values():
        frame["role"] = "P"
    with pytest.raises(ValueError, match="H"):
        evaluate_pair(tables, "NATIVE", "FULL", output_dir=tmp_path)
