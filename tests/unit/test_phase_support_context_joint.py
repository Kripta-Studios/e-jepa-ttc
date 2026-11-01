"""Joint support/context training learns support only from admitted training histories."""

import numpy as np
import pytest
import torch

from operational.phase_native_support.joint_training import (
    fit_preprocessing,
    prepare_fold,
    select_candidate,
)
from operational.phase_native_support.model import NativeSupportRouter
from operational.phase_robust.model import fit_normalization


def fixture():
    clean = {
        "token": np.array(["a", "b", "c", "d"]),
        "group": np.array(["a", "b", "c", "d"]),
        "family": np.array(["A", "A", "B", "C"]),
        "native": np.ones((4, 3), np.float32),
        "geometry": np.ones(4, np.float32),
        "quality": np.ones((4, 20), np.float32),
        "target": np.array([1, 3, 5, 1000], np.float32),
        "support": np.ones((4, 10), np.float32),
    }
    indices = np.array([0, 0, 1, 2, 2, 2, 3])
    aug = {k: v[indices].copy() for k, v in clean.items()}
    aug["keep"] = np.array([1, 2, 1, 1, 2, 3, 1])
    aug["support"][:, 4] = np.array([0.125, 0.25, 0.125, 0.125, 0.25, 0.375, 1000])
    aug["support"][-1, 0] = 1000
    aug["native"] *= 20
    return clean, aug, indices


@pytest.mark.parametrize("mode", ["F2", "Fcap"])
def test_union_support_norm_and_mask_exclude_heldout_preserve_clean_base(mode):
    clean, aug, indices = fixture()
    fold = prepare_fold(clean, aug, indices, "C")
    model = NativeSupportRouter(mode)
    fit_preprocessing(model, clean, aug, fold["clean_mask"], fold["aug_mask"])
    base = model.base_inputs(
        *(torch.from_numpy(clean[k][:3]) for k in ("native", "geometry", "quality"))
    )
    center, scale = fit_normalization(base.numpy())
    np.testing.assert_array_equal(model.center[:33], center)
    np.testing.assert_array_equal(model.scale[:33], scale)
    if mode == "F2":
        union = np.concatenate((clean["support"][:3], aug["support"][:-1]))
        center, scale = fit_normalization(union)
        np.testing.assert_array_equal(model.center[33:], center)
        np.testing.assert_array_equal(model.scale[33:], scale)
        assert model.support_mask.tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_weights_balance_family_query_and_admitted_variants():
    clean, aug, indices = fixture()
    fold = prepare_fold(clean, aug, indices, "C")
    np.testing.assert_allclose(fold["clean_mass"], [0.25, 0.25, 0.5])
    np.testing.assert_allclose(fold["aug_mass"], [0.125, 0.125, 0.25, 1 / 6, 1 / 6, 1 / 6])
    aug["native"][0] = np.nan
    aug["geometry"][0] = np.nan
    fold = prepare_fold(clean, aug, indices, "C")
    np.testing.assert_allclose(fold["aug_mass"], [0.25, 0.25, 1 / 6, 1 / 6, 1 / 6])
    assert not fold["aug_mask"][-1] and not fold["aug_mask"][0]


def test_fold_rejects_swapped_identity_and_group_leakage():
    clean, aug, indices = fixture()
    aug["group"][0] = "d"
    with pytest.raises(ValueError, match="mapping"):
        prepare_fold(clean, aug, indices, "C")
    clean, aug, indices = fixture()
    clean["group"][0] = "d"
    with pytest.raises(ValueError, match="leakage"):
        prepare_fold(clean, aug, indices, "C")


def test_only_iq_selects_and_tie_is_f2n():
    def report(x):
        return {
            d: {
                "coverage": 1.0,
                "macro_record_mae_seconds": x,
                "macro_record_relative_error_pct": x,
            }
            for d in ("all", "road", "slider")
        }

    reports = {
        "IQ": report(2),
        "IQn": report(0.01),
        "F2": report(0.01),
        "Fcap": report(0.01),
        "F2n": report(1),
        "Fcapn": report(1),
    }
    result = select_candidate(reports)
    assert result["selected"] == "F2n" and not result["support_dominates_capacity"]
    reports["F2n"]["road"]["macro_record_mae_seconds"] = 3
    assert select_candidate(reports)["selected"] == "Fcapn"


def test_informational_reference_rejects_hash_identity_and_seed_mean(tmp_path):
    import json

    from operational.evttc_transfer_audit.common import file_sha
    from scripts.train_phase_support_context_joint import load_informational_reference

    population = tmp_path / "population.npz"
    population.write_bytes(b"population")
    clean = {"token": np.array(["a", "b"]), "group": np.array(["A", "B"])}
    values = {**clean, "F2": np.ones(2, np.float32)}
    values.update({f"F2_seed{s}": values["F2"].copy() for s in (7, 13, 21)})

    def seal():
        np.savez(tmp_path / "OOF_PREDICTIONS.npz", **values)
        config = {
            "inputs": {str(population): file_sha(population)},
            "seeds": [7, 13, 21],
            "epochs": 300,
        }
        (tmp_path / "CONFIG.json").write_text(json.dumps(config))
        selection = {
            "config_sha256": file_sha(tmp_path / "CONFIG.json"),
            "oof_sha256": file_sha(tmp_path / "OOF_PREDICTIONS.npz"),
        }
        (tmp_path / "SELECTION.json").write_text(json.dumps(selection))
        (tmp_path / "VERIFICATION.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "fits_verified": 48,
                    "selection_sha256": file_sha(tmp_path / "SELECTION.json"),
                }
            )
        )

    seal()
    assert set(load_informational_reference(tmp_path, clean, ("F2",), population)) == {
        "F2",
        "F2_seed7",
        "F2_seed13",
        "F2_seed21",
    }
    values["group"] = np.array(["B", "A"])
    seal()
    with pytest.raises(ValueError, match="identities"):
        load_informational_reference(tmp_path, clean, ("F2",), population)
    values["group"] = clean["group"]
    values["F2_seed7"][0] = 2
    seal()
    with pytest.raises(ValueError, match="arithmetic"):
        load_informational_reference(tmp_path, clean, ("F2",), population)
    population.write_bytes(b"changed")
    with pytest.raises(ValueError, match="seal or population"):
        load_informational_reference(tmp_path, clean, ("F2",), population)
