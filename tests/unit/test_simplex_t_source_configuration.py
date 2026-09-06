"""Pinned source configuration cannot grant itself new roots or TRAIN groups."""

import json

import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.source_configuration import sources_from_configuration


def fixture(tmp_path):
    directory = {"root": "work", "relative_path": "."}
    file = {"root": "work", "relative_path": "file.bin"}
    (tmp_path / "file.bin").write_bytes(b"not loaded as a dataset")
    binding = {
        "compiled": directory,
        "compiled_sha256": "a" * 64,
        "index_manifest": file,
        "index_manifest_sha256": "a" * 64,
        "dedup": file,
        "pool": file,
        "pool_sha256": "a" * 64,
        "metadata": file,
        "metadata_sha256": "a" * 64,
        "labels": file,
        "labels_sha256": "a" * 64,
        "cache_identity_sha256": "a" * 64,
    }
    config = {
        "schema": "simplex_t_campaign_sources_v1",
        "original_sequences": ["old"],
        "expansion_sequences": ["extra"],
        "original": {
            "index_root": directory,
            "dedup_root": directory,
            "historical_root": directory,
            "ancestry_sha256": "a" * 64,
            "folds": {str(f): {"path": directory, "sha256": "a" * 64} for f in range(3)},
        },
        "expansion": {str(f): binding for f in range(3)},
        "dense": {str(f): binding for f in range(3)},
        "matched": {
            "pool": file,
            "pool_sha256": "a" * 64,
            "original_index_manifest_sha256": "a" * 64,
        },
    }
    graph = registered_graph(
        d1=True,
        density=True,
        t3=True,
        latent=True,
        replicate_scalar=True,
        replicate_latent=True,
    )
    return config, dict(
        roots={"work": tmp_path},
        graph=graph,
        allowed_original_sequences={"old"},
        allowed_expansion_sequences={"extra"},
    )


def test_all_registered_pools_construct_without_loading_features(tmp_path):
    config, args = fixture(tmp_path)
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    sources = sources_from_configuration(path, expected_sha256=compute_file_hash(str(path)), **args)
    assert len(sources.graph) == 84
    assert (
        set(sources.folds) == set(sources.expansion_folds) == set(sources.dense_folds) == {0, 1, 2}
    )
    assert sources.matched is not None
    assert sources.expansion_folds[0].labels == tmp_path / "file.bin"
    assert sources._sources == {}


@pytest.mark.parametrize("failure", ["role", "root", "traversal", "field", "fold", "hash", "kind"])
def test_configuration_refuses_unapproved_or_incomplete_bindings(tmp_path, failure):
    config, args = fixture(tmp_path)
    if failure == "role":
        config["original_sequences"] = ["protected"]
    elif failure == "root":
        config["original"]["index_root"] = {"root": "foreign", "relative_path": "."}
    elif failure == "traversal":
        config["original"]["index_root"] = {"root": "work", "relative_path": "../outside"}
    elif failure == "field":
        config["authorize_holdout"] = True
    elif failure == "fold":
        config["dense"].pop("2")
    elif failure == "kind":
        config["original"]["index_root"] = {"root": "work", "relative_path": "file.bin"}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    digest = "0" * 64 if failure == "hash" else compute_file_hash(str(path))
    with pytest.raises(ValueError):
        sources_from_configuration(path, expected_sha256=digest, **args)
