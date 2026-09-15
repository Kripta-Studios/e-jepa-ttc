"""Real file loader integration with synthetic, independently pinned TRAIN files."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t.dense_file_sources import DenseBinding, load_dense_inputs
from e_jepa_ttc.simplex_t.expansion_sources import (
    ExpansionBinding,
    load_expansion_inputs,
    load_expansion_source,
)
from e_jepa_ttc.simplex_t.pools import expansion_producer


def fixture(tmp_path):
    def save(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def digest(path):
        return compute_file_hash(str(path))

    family = expansion_producer("extra", ("0" * 64, "1" * 64, "2" * 64))
    valid = np.zeros((3, 16), bool)
    valid[:, -1] = True
    assignments = np.full((3, 3), -1, np.int64)
    assignments[0, [0, 2]] = int(family[0])
    index = tmp_path / "query_context_index.npz"
    np.savez(
        index,
        tokens=np.array(["a", "unused", "b"]),
        sequences=np.array(["extra"] * 3),
        producer_family=assignments,
        valid=valid,
        anchor_us=np.array([100, 200, 300], dtype=np.int64),
    )
    manifest = save(
        "INDEX_MANIFEST.json",
        {
            "index_sha256": digest(index),
            "families": [
                {"outer_fold": 0, "role": f"inner{i}", "family_sha256": str(i) * 64}
                for i in range(3)
            ],
        },
    )
    history = np.full((3, 16), -1, np.int64)
    history[[0, 2], -1] = [0, 1]
    dedup = tmp_path / "dedup.npz"
    np.savez(dedup, history=history)
    metadata = tmp_path / "metadata.parquet"
    frame = pd.DataFrame(
        {
            "sample_token": ["a", "unused", "b"],
            "sequence_id": ["extra"] * 3,
            "timestamp_us": [100, 200, 300],
        }
    )
    frame.to_parquet(metadata, index=False)
    labels = tmp_path / "labels.parquet"
    frame.assign(ttc=[1.0, 2.0, 3.0]).to_parquet(labels, index=False)
    pool = save(
        "pool.json",
        {
            "expansion_metadata_sha256": digest(metadata),
            "folds": {
                "0": {
                    "additional_train_tokens": ["b", "a"],
                    "original_train_tokens": ["old"],
                    "D0_train_queries": 1,
                    "D1_train_queries": 3,
                    "additional_groups": ["extra"],
                    "sequence_family_sha256": {"extra": family},
                }
            },
        },
    )
    arrays = {
        "features145": np.arange(290, dtype=np.float32).reshape(2, 145),
        "expert_ttc": np.ones((2, 3), np.float32),
        "known": np.ones((2, 2), bool),
        "anchor_us": np.array([100, 300], np.int64),
        "available_us": np.array([110, 310], np.int64),
    }
    pins = {}
    for name, values in arrays.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        pins[name] = digest(path)
    compiled = save(
        "COMPILED.json",
        {
            "pool": "D1",
            "outer": 0,
            "status": "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE",
            "cache_identity_sha256": "a" * 64,
            "index_sha256": digest(index),
            "dedup_sha256": digest(dedup),
            "selected_query_ids": [0, 2],
            "queries": 2,
            "indexed_queries": 3,
            "observations": 2,
            "arrays": pins,
        },
    )
    return ExpansionBinding(
        tmp_path,
        digest(compiled),
        manifest,
        digest(manifest),
        dedup,
        pool,
        digest(pool),
        metadata,
        digest(metadata),
        labels,
        digest(labels),
        "a" * 64,
    )


@pytest.mark.parametrize("features", [17, 145])
def test_file_loader_attaches_targets_in_pool_order(tmp_path, features):
    binding = fixture(tmp_path)
    source, sequences = load_expansion_source(
        binding, outer=0, feature_count=features, allowed_sequences={"extra"}
    )
    assert source.population == 2
    assert source.history[:, -1].tolist() == [1, 0]
    assert sequences.tolist() == ["extra", "extra"]
    assert source.features.shape == (2, features)
    np.testing.assert_allclose(source.normalizer.mean, source.features.mean(0))
    assert source.target_phase[0] < source.target_phase[1]


def test_density_loader_normalizes_only_compact_selected_rows(tmp_path, monkeypatch):
    from e_jepa_ttc.simplex_t import expansion_sources

    binding = fixture(tmp_path)
    path = tmp_path / "COMPILED.json"
    compiled = json.loads(path.read_text())
    compiled.update(
        query_selection={"path": "fixture-only"}, selected_query_ids=[2], queries=1, observations=1
    )
    for name in compiled["arrays"]:
        array_path = tmp_path / f"{name}.npy"
        values = np.load(array_path)[1:2].copy()
        np.save(array_path, values)
        compiled["arrays"][name] = compute_file_hash(str(array_path))
    mapping = tmp_path / "source_observation_ids.npy"
    np.save(mapping, np.array([1], dtype=np.int64))
    compiled["source_observation_ids_sha256"] = compute_file_hash(str(mapping))
    path.write_text(json.dumps(compiled))
    binding = replace(binding, compiled_sha256=compute_file_hash(str(path)))
    monkeypatch.setattr(
        expansion_sources, "bound_selection_rows", lambda *args: np.array([2], dtype=np.int64)
    )
    result = load_expansion_inputs(binding, outer=0, feature_count=17, allowed_sequences={"extra"})
    assert result.tokens.tolist() == ["b"]
    assert result.source.population == 1
    assert result.source.history[:, -1].tolist() == [0]
    np.testing.assert_array_equal(result.source.normalizer.mean, result.source.features[0])


def test_expansion_details_preserve_original_targets_for_matched_weights(tmp_path):
    binding = fixture(tmp_path)
    result = load_expansion_inputs(binding, outer=0, feature_count=17, allowed_sequences={"extra"})
    assert result.tokens.tolist() == ["b", "a"]
    assert result.target_ttc.tolist() == [3.0, 1.0]
    legacy, sequences = load_expansion_source(
        binding, outer=0, feature_count=17, allowed_sequences={"extra"}
    )
    assert legacy.identity_sha256 == result.source.identity_sha256
    np.testing.assert_array_equal(sequences, result.sequences)


def test_pool_eligibility_metadata_is_distinct_from_garl_target_reference(tmp_path):
    binding = fixture(tmp_path)
    pool = json.loads(binding.pool.read_text(encoding="utf-8"))
    pool["expansion_metadata_sha256"] = "e" * 64
    binding.pool.write_text(json.dumps(pool), encoding="utf-8")
    binding = replace(binding, pool_sha256=compute_file_hash(str(binding.pool)))
    result = load_expansion_inputs(binding, outer=0, feature_count=17, allowed_sequences={"extra"})
    assert result.tokens.tolist() == ["b", "a"]
    assert result.target_ttc.tolist() == [3.0, 1.0]


@pytest.mark.parametrize("change", ["cache", "metadata", "family", "closed_group"])
def test_bad_binding_rejected_before_label_read(tmp_path, monkeypatch, change):
    from e_jepa_ttc.simplex_t import expansion_sources

    binding = fixture(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("labels read before source identity validation")

    monkeypatch.setattr(expansion_sources, "load_expansion_targets", forbidden)
    groups = {"extra"}
    if change == "cache":
        binding = replace(binding, compiled_sha256="0" * 64)
    elif change == "metadata":
        binding = replace(binding, metadata_sha256="0" * 64)
    elif change == "family":
        binding = replace(binding, cache_identity_sha256="b" * 64)
    else:
        groups = {"protected"}
    with pytest.raises(ValueError):
        load_expansion_source(binding, outer=0, feature_count=17, allowed_sequences=groups)


def dense_fixture(tmp_path):
    binding = fixture(tmp_path)
    original = json.loads(binding.pool.read_text(encoding="utf-8"))["folds"]["0"]
    binding.pool.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "outer": 0,
                        "nominal_common_count": 2,
                        "pools": {
                            "DENSE_OLD": {
                                "tokens": ["b", "a"],
                                "sequences": ["extra"],
                                "sequence_family_sha256": original["sequence_family_sha256"],
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    path = binding.compiled / "COMPILED.json"
    compiled = json.loads(path.read_text(encoding="utf-8"))
    compiled["pool"] = "DENSE_OLD"
    path.write_text(json.dumps(compiled), encoding="utf-8")
    return DenseBinding(
        **{
            **vars(binding),
            "pool_sha256": compute_file_hash(str(binding.pool)),
            "compiled_sha256": compute_file_hash(str(path)),
        }
    )


def test_dense_file_loader_keeps_registered_order_and_original_ttc(tmp_path):
    binding = dense_fixture(tmp_path)
    result = load_dense_inputs(binding, outer=0, allowed_sequences={"extra"})
    assert result.tokens.tolist() == ["b", "a"]
    assert result.sequences.tolist() == ["extra", "extra"]
    assert result.target_ttc.tolist() == [3.0, 1.0]
    assert result.source.history[:, -1].tolist() == [1, 0]
    assert result.source.features.shape == (2, 17)
    np.testing.assert_allclose(result.source.normalizer.mean, result.source.features.mean(0))


@pytest.mark.parametrize("failure", ["cache", "role", "schema", "family"])
def test_invalid_dense_files_refused_before_label_access(tmp_path, monkeypatch, failure):
    from e_jepa_ttc.simplex_t import dense_file_sources

    binding = dense_fixture(tmp_path)
    groups = {"extra"}
    if failure == "cache":
        binding = replace(binding, cache_identity_sha256="b" * 64)
    elif failure == "role":
        groups = {"closed"}
    elif failure == "schema":
        path = binding.compiled / "COMPILED.json"
        compiled = json.loads(path.read_text(encoding="utf-8"))
        compiled["observations"] = 3
        path.write_text(json.dumps(compiled), encoding="utf-8")
        binding = replace(binding, compiled_sha256=compute_file_hash(str(path)))
    else:
        plan = json.loads(binding.pool.read_text(encoding="utf-8"))
        plan["folds"][0]["pools"]["DENSE_OLD"]["sequence_family_sha256"]["extra"] = "f" * 64
        binding.pool.write_text(json.dumps(plan), encoding="utf-8")
        binding = replace(binding, pool_sha256=compute_file_hash(str(binding.pool)))

    def forbidden(*args, **kwargs):
        pytest.fail("labels read before dense source validation")

    monkeypatch.setattr(dense_file_sources, "load_selected_train_targets", forbidden)
    with pytest.raises(ValueError):
        load_dense_inputs(binding, outer=0, allowed_sequences=groups)
