"""Migration keeps identifier values and rejects wrong-path or changed caches."""

import json

import numpy as np
import pytest

from scripts.runpod_final_cache_compat import install, unicode_supervision


def test_supervision_strings_do_not_require_pickle(tmp_path):
    target = np.asarray([0.1, -0.2], np.float32)
    values = {
        "sample_token": np.asarray(["token-a", "token-b"], dtype=object),
        "group_id": np.asarray(["g0", "g1"], dtype=object),
        "sequence_id": np.asarray(["seq0", "seq1"], dtype=object),
        "role": np.asarray(["TRAIN", "TRAIN"]),
        "target_phase": target,
    }
    converted = unicode_supervision(values)
    assert converted["target_phase"] is target
    assert values["sample_token"].dtype.kind == "O"
    path = tmp_path / "cache.npz"
    np.savez_compressed(path, **converted)
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            assert np.array_equal(archive[name], values[name])
            assert archive[name].dtype.kind != "O"


def test_rejects_nonstring_identifiers():
    values = {
        name: np.asarray(["valid"], dtype=object)
        for name in ("sample_token", "group_id", "sequence_id", "role")
    }
    values["sample_token"] = np.asarray([None], dtype=object)
    with pytest.raises(ValueError, match="not a string"):
        unicode_supervision(values)


@pytest.fixture
def relocated_cache(tmp_path, monkeypatch):
    from operational.final_refit import features, scientific_oof_features, training
    from operational.final_refit import scientific_oof_final as final

    original_validate = features.validate_cache_receipt
    original_training = training.validate_cache_receipt
    original_science = scientific_oof_features.validate_cache_receipt
    original_native = final._native_supervision
    path = tmp_path / "cache.npz"
    features._seal_cache(
        path,
        {
            "sample_token": np.asarray(["t0", "t1"]),
            "group_id": np.asarray(["g0", "g1"]),
            "features": np.asarray([[1.0, 2.0], [3.0, 4.0]], np.float32),
        },
        schema="fixture_cache_v1",
        parents={"parent": "a" * 64},
        native_train40_rows_sha256="b" * 64,
    )
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    receipt["output"] = "C:\\frozen\\cache.npz"
    receipt.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = features._canonical_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt))

    class Mapping:
        def translate(self, value):
            return str(path) if value == "C:\\frozen\\cache.npz" else str(value)

    install(Mapping())
    yield path, features
    monkeypatch.setattr(features, "validate_cache_receipt", original_validate)
    monkeypatch.setattr(training, "validate_cache_receipt", original_training)
    monkeypatch.setattr(scientific_oof_features, "validate_cache_receipt", original_science)
    monkeypatch.setattr(final, "_native_supervision", original_native)


def test_path_only_relocation_retains_original_receipt(relocated_cache):
    path, features = relocated_cache
    before = path.with_suffix(".json").read_bytes()
    receipt = features.validate_cache_receipt(path, "fixture_cache_v1")
    assert receipt["output"] == "C:\\frozen\\cache.npz"
    assert path.with_suffix(".json").read_bytes() == before


def test_changed_bytes_are_rejected_after_relocation(relocated_cache):
    path, features = relocated_cache
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        features.validate_cache_receipt(path, "fixture_cache_v1")


def test_wrong_expected_role_is_rejected(relocated_cache):
    path, features = relocated_cache
    with pytest.raises(ValueError, match="changed"):
        features.validate_cache_receipt(path, "fixture_cache_v1", expected_role="TEST")
