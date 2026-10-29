"""Exercise complete-shaped synthetic source manifests without scientific fits."""

import json

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t import context_sources


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    compiled, index, dedup = (tmp_path / name for name in ("compiled", "index", "dedup"))
    for directory in (compiled, index, dedup):
        directory.mkdir()
    tokens = np.array([f"q{i}" for i in range(8192)])
    assignments = np.zeros((3, 8192), np.int16)
    assignments[0, 4096:] = 3
    np.savez(index / "query_context_index.npz", tokens=tokens, producer_family=assignments)
    history = np.full((8192, 16), -1, np.int64)
    history[:, -1] = np.arange(8192)
    np.savez(dedup / "outer0.npz", history=history)
    features = np.ones((8192, 145), np.float32)
    features[4096:] = 1000
    arrays = {
        "features145": features,
        "expert_ttc": np.ones((8192, 3), np.float32),
        "known": np.ones((8192, 2), bool),
        "anchor_us": np.arange(8192, dtype=np.int64),
        "available_us": np.arange(8192, dtype=np.int64),
    }
    hashes = {}
    for name, values in arrays.items():
        path = compiled / f"{name}.npy"
        np.save(path, values)
        hashes[name] = compute_file_hash(str(path))
    manifest = {
        "status": "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE",
        "queries": 8192,
        "outer": 0,
        "arrays": hashes,
        "dedup_sha256": compute_file_hash(str(dedup / "outer0.npz")),
        "index_sha256": compute_file_hash(str(index / "query_context_index.npz")),
    }
    manifest_path = compiled / "COMPILED.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def historical_table(root, outer, role, **kwargs):
        assert outer == 0
        assert kwargs["ancestry_sha256"] == "fixture-ancestry"
        assert kwargs["allowed_sequences"] == {"train", "dev"}
        selection = slice(0, 4096) if role == "inner_oof" else slice(4096, 8192)
        return {
            "metadata": pd.DataFrame(
                {
                    "sample_token": tokens[selection],
                    "inner_fold": np.zeros(4096, np.int64),
                    "target_ttc": np.ones(4096),
                    "sequence_id": ["train" if role == "inner_oof" else "dev"] * 4096,
                }
            ),
            "arrays": {"target_phase": np.zeros(4096, np.float32)},
            "reference": role,
        }

    monkeypatch.setattr(context_sources, "load_current_inputs", historical_table)
    return compiled, index, dedup, manifest, manifest_path


def load_fixture(fixture, feature_count=17):
    compiled, index, dedup, _, manifest_path = fixture
    return context_sources.load_context_sources(
        compiled,
        index,
        dedup,
        compiled.parent / "historical",
        compiled_manifest_sha256=compute_file_hash(str(manifest_path)),
        ancestry_sha256="fixture-ancestry",
        allowed_sequences={"train", "dev"},
        feature_count=feature_count,
    )


@pytest.mark.parametrize("feature_count", [17, 145])
def test_sources_share_train_only_normalizer(source_fixture, feature_count):
    sources = load_fixture(source_fixture, feature_count)
    train, dev = sources["inner_oof"], sources["outer_dev"]
    assert train.normalizer is dev.normalizer
    np.testing.assert_array_equal(train.normalizer.mean, np.ones(feature_count))
    np.testing.assert_array_equal(train.normalizer.scale, np.ones(feature_count))
    assert train.population == dev.population == 4096
    assert train.identity_sha256 != dev.identity_sha256
    assert not set(train.history[:, -1]) & set(dev.history[:, -1])


def test_sources_reject_changed_array_bytes(source_fixture):
    compiled = source_fixture[0]
    np.save(compiled / "expert_ttc.npy", np.zeros((8192, 3), np.float32))
    with pytest.raises(ValueError, match="array bytes changed"):
        load_fixture(source_fixture)


@pytest.mark.parametrize("corruption", ["shared_observation", "wrong_producer", "partial"])
def test_sources_reject_invalid_role_or_completion(source_fixture, corruption):
    _, index, dedup, manifest, manifest_path = source_fixture
    if corruption == "shared_observation":
        with np.load(dedup / "outer0.npz") as archive:
            history = archive["history"].copy()
        history[4096, -1] = 0
        np.savez(dedup / "outer0.npz", history=history)
        manifest["dedup_sha256"] = compute_file_hash(str(dedup / "outer0.npz"))
        message = "share a consumed observation"
    elif corruption == "wrong_producer":
        path = index / "query_context_index.npz"
        with np.load(path) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["producer_family"][0, 0] = 3
        np.savez(path, **arrays)
        manifest["index_sha256"] = compute_file_hash(str(path))
        message = "producer differs"
    else:
        manifest["queries"] = 8191
        message = "complete original D0 fold required"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_fixture(source_fixture)
