"""Synthetic R1 contract tests; no real H labels, extraction, or GPU operations."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from operational.decision_recovery import common, router
from operational.decision_recovery import feature_router as r1


def population():
    target = np.tile([0.8, 2.0, 4.0, 5.0, 7.0, 8.0, -2.0, -4.0], 4)
    groups = np.repeat(["g0", "g1", "g2", "g3"], 8)
    tables = {}
    for name, multiplier in (("NATIVE", 1.2), ("FULL", 1.1), ("A5", 1.3)):
        prediction = target * multiplier
        tables[name] = pd.DataFrame(
            {
                "sample_token": [f"q{i}" for i in range(len(target))],
                "group_id": groups,
                "sequence_id": groups,
                "target_ttc": target,
                "prediction_ttc": prediction,
                "raw_phase": router._phase(prediction),
                "role": "H",
            }
        )
    return tables


def saved_r0(tables):
    groups = tables["NATIVE"].group_id.to_numpy()
    unique = np.unique(groups)
    return {
        "folds": [
            {
                "held_out_group": str(g),
                "training_groups": unique[unique != g].tolist(),
                "train_rows": int((groups != g).sum()),
                "test_rows": int((groups == g).sum()),
            }
            for g in unique
        ]
    }


def test_fixed_config_and_feature_schema():
    config = common.read_json(r1.CONFIG)
    assert config["feature_names"] == list(r1.FEATURES)
    assert len(r1.FEATURES) == len(set(r1.FEATURES)) == 202
    assert config["base"] == "NATIVE" and config["special"] == "FULL"
    assert config["feature_selection_from_H_probe_results"] is False
    assert config["bootstrap_replicates"] == 2000
    assert config["ridge"] == 0.01 and config["threshold"] == 0


def test_features_have_no_target_or_id_api_and_preserve_r0():
    a, b = np.array([1.0, -2.0]), np.array([2.0, -3.0])
    d = np.arange(388, dtype=np.float32).reshape(2, 194)
    x = r1.features(a, b, d)
    np.testing.assert_array_equal(x[:, :8], router.features(a, b))
    np.testing.assert_array_equal(x[:, 8:], d)
    with pytest.raises(TypeError):
        r1.features(a, b, d, target=np.ones(2))
    for bad in (d[:, :-1], np.full_like(d, np.nan), np.full(d.shape, "group_id")):
        with pytest.raises(ValueError):
            r1.features(a, b, bad)


def test_ridge_exact_r0_extension_with_zero_descriptors_and_ties():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 8))
    w = rng.uniform(0.1, 1, 48)
    w /= w.sum()
    y = rng.normal(size=48)
    old = router.fit(x, y, w)
    extended = np.column_stack((x, np.zeros((48, 194))))
    new = r1.fit(extended, y, w)
    for field in ("mean", "scale", "coefficient"):
        np.testing.assert_allclose(new[field][:8], old[field], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new["coefficient"][8:], 0, atol=0)
    assert new["intercept"] == old["intercept"]
    tied = r1.fit(extended, np.zeros(48), w)
    assert not r1.choose_special(tied, extended).any()
    tied["intercept"] = -1
    assert r1.choose_special(json.loads(json.dumps(tied)), extended).all()
    tied["threshold"] = 0.01
    with pytest.raises(ValueError, match="contract"):
        r1.choose_special(tied, extended)


def test_model_validation_and_fixed_global_weights():
    x = np.zeros((4, 202))
    for weights in (np.ones(4), np.array([0.5, 0.5, 0, 0])):
        with pytest.raises(ValueError, match="weights"):
            r1.fit(x, np.zeros(4), weights)
    with pytest.raises(ValueError, match="shapes"):
        r1.fit(x[:, :-1], np.zeros(4), np.ones(4) / 4)
    model = r1.fit(x, np.zeros(4), np.ones(4) / 4)
    model["scale"][0] = 0
    with pytest.raises(ValueError, match="scale"):
        r1.choose_special(model, x)


def test_saved_r0_fold_and_identity_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path)
    tables = population()
    saved = saved_r0(tables)
    saved["folds"][0]["train_rows"] -= 1
    with pytest.raises(ValueError, match="folds"):
        r1.evaluate(
            tables, np.zeros((32, 194)), saved, tables["NATIVE"], output_dir=tmp_path / "bad"
        )
    wrong = tables["NATIVE"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="Population mismatch"):
        r1.evaluate(
            tables, np.zeros((32, 194)), saved_r0(tables), wrong, output_dir=tmp_path / "bad"
        )
    with pytest.raises(ValueError, match="V19 artifacts"):
        r1.evaluate(
            tables,
            np.zeros((32, 194)),
            saved_r0(tables),
            tables["NATIVE"],
            output_dir=tmp_path.parent / "outside",
        )


def test_logo_fold_normalizer_exclusion_control_and_whole_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path)
    tables = population()
    d = np.random.default_rng(8).normal(size=(32, 194)).astype(np.float32)
    first = r1.evaluate(
        tables, d, saved_r0(tables), tables["NATIVE"], output_dir=tmp_path / "first"
    )
    changed = {k: v.copy() for k, v in tables.items()}
    for frame in changed.values():
        frame.loc[frame.group_id == "g0", "target_ttc"] *= 0.9
    altered = d.copy()
    altered[:8] *= 20
    r1.evaluate(
        changed, altered, saved_r0(changed), changed["NATIVE"], output_dir=tmp_path / "second"
    )
    for kind in ("R1", "SHUFFLED_R1"):
        m1 = common.read_json(tmp_path / "first" / f"fold_000_{kind}.json")
        m2 = common.read_json(tmp_path / "second" / f"fold_000_{kind}.json")
        assert m1 == m2
        assert "g0" not in m1["training_groups"]
        train = tables["NATIVE"].group_id.to_numpy() != "g0"
        a_all = tables["NATIVE"].prediction_ttc.to_numpy()
        b_all = tables["FULL"].prediction_ttc.to_numpy()
        target = tables["NATIVE"].target_ttc.to_numpy()
        costs = (
            100
            * (
                abs(router._phase(b_all) - router._phase(target))
                - abs(router._phase(a_all) - router._phase(target))
            )[train]
        )
        if kind == "SHUFFLED_R1":
            costs = costs[np.random.default_rng(7).permutation(train.sum())]
        expected = r1.fit(
            r1.features(a_all, b_all, d)[train],
            costs,
            router.training_weights(target[train], tables["NATIVE"].group_id.to_numpy()[train]),
        )
        for field in ("mean", "scale", "coefficient", "intercept"):
            assert m1[field] == expected[field]
        pred = pd.read_parquet(tmp_path / "first" / f"H_{kind}_OOF.parquet")
        a, b = tables["NATIVE"], tables["FULL"]
        np.testing.assert_array_equal(
            pred.prediction_ttc,
            np.where(pred.router_selected_special, b.prediction_ttc, a.prediction_ttc),
        )
        np.testing.assert_array_equal(
            pred.raw_phase, np.where(pred.router_selected_special, b.raw_phase, a.raw_phase)
        )
        assert first["results"][kind]["bootstrap"]["R0"]["replicates"] == 2000
        assert first["results"][kind]["diagnostics"]["all"]["negative_rows"] == 8
    assert "SHUFFLED_R1" in first["results"]["R1"]["bootstrap"]
    assert set(first["references"]) == {"A5", "NATIVE", "FULL", "R0"}
    assert "beats_saved_R0_aggregate" in first["advancement"]["checks"]
    assert not first["system_promotion"] and not first["official_SOTA"]
    assert common.read_json(tmp_path / "first/ALL_H_DEVELOPMENT_MODEL.json")["H_in_sample"]


def test_run_requires_feature_admission_before_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path)
    monkeypatch.setattr(r1, "DIRECTORY", tmp_path / "r1")
    monkeypatch.setattr(
        r1, "verify_registration", lambda: {"expected_H_feature_binding": "expected"}
    )
    monkeypatch.setattr(common, "load_population", lambda *a: population())
    calls = []

    def reject(role, binding, rows):
        calls.append((role, binding, len(rows)))
        raise ValueError("Incomplete feature receipt")

    monkeypatch.setattr(r1.sign_probe, "_admit_complete_features", reject)
    monkeypatch.setattr(r1, "fit", lambda *a: pytest.fail("fit before feature admission"))
    with pytest.raises(ValueError, match="Incomplete"):
        r1.run()
    assert calls == [("H", "expected", 32)]
    assert not (r1.DIRECTORY / "FIT_STARTED.json").exists()


def test_interrupted_fit_refuses_silent_refit(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path)
    monkeypatch.setattr(r1, "DIRECTORY", tmp_path)
    monkeypatch.setattr(r1, "verify_registration", lambda: {})
    common.write_json(tmp_path / "FIT_STARTED.json", {})
    with pytest.raises(ValueError, match="Interrupted"):
        r1.run()


def test_frozen_verifier_rejects_mutated_source(tmp_path, monkeypatch):
    monkeypatch.setattr(r1, "REGISTRATION", tmp_path / "REGISTRATION.json")
    monkeypatch.setattr(r1.registration, "verify", lambda: {})
    monkeypatch.setattr(r1, "_sources", lambda: {"source": "changed"})
    monkeypatch.setattr(r1, "_config", lambda: {})
    monkeypatch.setattr(r1, "_recovery_binding", lambda probe: "repair_hash")
    monkeypatch.setattr(r1.sign_probe, "_binding", lambda *a: "binding")
    monkeypatch.setattr(common, "sha", lambda p: "hash")
    frozen = {
        "schema": "v19_feature_router_r1_registration_v1",
        "status": "REGISTERED_BEFORE_R1_H_FIT",
        "sources": {"source": "original"},
    }
    probe = {"populations": {"H": {}}}
    monkeypatch.setattr(
        common, "read_json", lambda p: copy.deepcopy(frozen if p == r1.REGISTRATION else probe)
    )
    with pytest.raises(ValueError, match="evidence changed"):
        r1.verify_registration()


def test_serialization_repair_requires_admission_and_binds_hash(monkeypatch):
    calls = []
    monkeypatch.setattr(r1.sign_probe, "verify_recovery_amendment", lambda p: calls.append(p))
    monkeypatch.setattr(common, "sha", lambda p: "technical_repair_sha")
    probe = {"original_science": "unchanged"}
    assert r1._recovery_binding(probe) == "technical_repair_sha"
    assert calls == [probe]

    def reject(probe):
        raise ValueError("repair changed")

    monkeypatch.setattr(r1.sign_probe, "verify_recovery_amendment", reject)
    with pytest.raises(ValueError, match="repair changed"):
        r1._recovery_binding(probe)
