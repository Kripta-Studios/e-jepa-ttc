"""Queue/source wiring tests; no expert inference or optimizer updates."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t import campaign_sources
from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer
from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources, CompiledFold
from e_jepa_ttc.simplex_t.expansion_sources import ExpansionBinding
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph


@pytest.fixture
def wired(tmp_path, monkeypatch):
    graph = registered_graph(
        d1=False,
        density=False,
        t3=True,
        latent=True,
        replicate_scalar=True,
        replicate_latent=True,
    )
    folds = {}
    for fold in range(3):
        folder = tmp_path / str(fold)
        folder.mkdir()
        manifest = folder / "COMPILED.json"
        manifest.write_text(json.dumps({"outer": fold}), encoding="utf-8")
        folds[fold] = CompiledFold(folder, compute_file_hash(str(manifest)))
    calls = []

    def load(path, *args, feature_count, **kwargs):
        calls.append((int(path.name), feature_count))
        result = {}
        for role in ("inner_oof", "outer_dev"):
            result[role] = CachedQueries(
                np.ones((16, feature_count), np.float32),
                np.arange(16, dtype=np.int64),
                np.arange(16, dtype=np.int64),
                np.arange(16, dtype=np.int64).reshape(1, 16),
                np.zeros(1, np.float32),
                np.ones(1),
                Normalizer(np.zeros(feature_count), np.ones(feature_count), "ids"),
                f"fixture:{path.name}:{feature_count}:{role}",
            )
        return result

    monkeypatch.setattr(campaign_sources, "load_context_sources", load)
    adapter = CampaignSources(
        graph,
        folds,
        index_root=tmp_path,
        dedup_root=tmp_path,
        historical_root=tmp_path,
        ancestry_sha256="fixture",
        allowed_sequences={"original"},
    )
    return adapter, graph, calls


def test_every_arm_gets_role_specific_identity_and_releases_cache(wired):
    adapter, graph, calls = wired
    identities = adapter.identities()
    assert set(identities) == {fit_key(spec) for spec in graph}
    assert all(row["inner_oof"] != row["outer_dev"] for row in identities.values())
    assert len(calls) == 6  # Three folds, scalar and latent; not one load per arm.
    assert adapter._sources == {}
    for spec in graph:
        source = adapter.train(spec)
        assert source.identity_sha256 == identities[fit_key(spec)]["inner_oof"]
        assert source.length == int(spec.name.split("-")[2][1:])
        assert source.zero_latent == spec.name.startswith("LATENT_ZERO-")
        expected = spec.name.split("-")[0]
        assert source.control == (
            expected if expected in {"PAST_REVERSED", "REPEAT_CURRENT"} else "NONE"
        )


def test_rejects_wrong_role_and_rechecks_cached_manifest(wired):
    adapter, graph, _ = wired
    with pytest.raises(ValueError, match="roles are authorized"):
        adapter.source(graph[0], "confirmation")
    adapter.train(graph[0])
    (adapter.folds[0].path / "COMPILED.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        adapter.train(graph[0])


def test_wrong_outer_fold_cannot_be_rebound_with_matching_hash(wired):
    adapter, graph, _ = wired
    adapter.folds[0] = adapter.folds[1]
    with pytest.raises(ValueError, match="outer fold differs"):
        adapter.train(graph[0])


def test_unavailable_pool_and_partial_folds_fail_before_loading(wired):
    adapter, _, calls = wired
    graph = registered_graph(
        d1=True,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    kwargs = dict(
        index_root=adapter.index_root,
        dedup_root=adapter.dedup_root,
        historical_root=adapter.historical_root,
        ancestry_sha256="fixture",
        allowed_sequences={"original"},
    )
    with pytest.raises(ValueError, match="pool lacks"):
        CampaignSources(graph, adapter.folds, **kwargs)
    with pytest.raises(ValueError, match="all three"):
        CampaignSources(adapter.graph, {0: adapter.folds[0]}, **kwargs)
    assert calls == []


def test_d1_dispatch_merges_and_never_reuses_d0_source(wired, monkeypatch):
    original, _, _ = wired
    graph = registered_graph(
        d1=True,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    pins = {}
    for fold, pin in original.folds.items():
        pins[fold] = ExpansionBinding(
            pin.path,
            pin.sha256,
            pin.path,
            "x",
            pin.path,
            pin.path,
            "x",
            pin.path,
            "x",
            pin.path,
            "x",
            "x",
        )
    expansion_calls = []

    def expansion(pin, **kwargs):
        expansion_calls.append(kwargs["outer"])
        return None, np.array(["extra"])

    monkeypatch.setattr(campaign_sources, "load_expansion_source", expansion)
    monkeypatch.setattr(
        campaign_sources,
        "load_current_inputs",
        lambda *args, **kwargs: {"metadata": pd.DataFrame({"sequence_id": ["original"]})},
    )

    def merge(sources, expansion, **kwargs):
        assert kwargs["original_train_sequences"].tolist() == ["original"]
        assert kwargs["expansion_train_sequences"].tolist() == ["extra"]
        return {
            role: replace(
                source,
                features=np.full_like(source.features, 9),
                identity_sha256="merged:" + source.identity_sha256,
            )
            for role, source in sources.items()
        }

    monkeypatch.setattr(campaign_sources, "merge_validated_sources", merge)
    adapter = CampaignSources(
        graph,
        original.folds,
        index_root=original.index_root,
        dedup_root=original.dedup_root,
        historical_root=original.historical_root,
        ancestry_sha256="fixture",
        allowed_sequences={"original"},
        expansion_folds=pins,
        expansion_sequences={"extra"},
    )
    d0 = next(spec for spec in graph if spec.fold == 0 and "-D0-" in spec.name)
    d1 = next(spec for spec in graph if spec.fold == 0 and "-D1-" in spec.name)
    assert (adapter.train(d0).features == 1).all()
    assert (adapter.train(d1).features == 9).all()
    assert (adapter.source(d1, "outer_dev").features == 9).all()
    assert expansion_calls == [0]
    assert (adapter.train(d0).features == 1).all()
    assert (adapter.train(d1).features == 9).all()
    assert expansion_calls == [0, 0]
    (pins[0].compiled / "COMPILED.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        adapter.train(d1)


def test_dense_dispatch_uses_file_loader_and_preserves_pool_separation(wired, monkeypatch):
    from types import SimpleNamespace

    from e_jepa_ttc.simplex_t.dense_file_sources import DenseBinding

    original, _, _ = wired
    graph = [
        s
        for s in registered_graph(
            d1=True,
            density=True,
            t3=False,
            latent=False,
            replicate_scalar=False,
            replicate_latent=False,
        )
        if "-D0-" in s.name or "-DENSE_OLD-" in s.name
    ]
    pins = {
        fold: DenseBinding(
            pin.path,
            pin.sha256,
            pin.path,
            "x",
            pin.path,
            pin.path,
            "x",
            pin.path,
            "x",
            pin.path,
            "x",
            "x",
        )
        for fold, pin in original.folds.items()
    }
    for fold, pin in list(pins.items()):
        folder = pin.compiled / "dense_fixture"
        folder.mkdir()
        (folder / "COMPILED.json").write_text(
            (pin.compiled / "COMPILED.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        pins[fold] = replace(pin, compiled=folder)
    calls = []

    def load(pin, **kwargs):
        calls.append(kwargs["outer"])
        return SimpleNamespace(
            source=None,
            tokens=np.array(["q", "new"]),
            sequences=np.array(["original", "original"]),
            target_ttc=np.array([1.0, 2.0]),
        )

    def compose(sources, dense, **kwargs):
        assert kwargs["original_tokens"].tolist() == ["q"]
        assert kwargs["dev_sequences"].tolist() == ["dev"]
        assert kwargs["dense_target_ttc"].tolist() == [1.0, 2.0]
        return {
            role: replace(
                source,
                features=np.full_like(source.features, 8),
                identity_sha256="dense:" + source.identity_sha256,
            )
            for role, source in sources.items()
        }

    monkeypatch.setattr(campaign_sources, "load_dense_inputs", load)
    monkeypatch.setattr(campaign_sources, "dense_source_pair", compose)
    monkeypatch.setattr(
        campaign_sources,
        "load_current_inputs",
        lambda root, fold, role, **kw: {
            "metadata": pd.DataFrame(
                {
                    "sample_token": ["q"],
                    "sequence_id": ["original" if role == "inner_oof" else "dev"],
                }
            )
        },
    )
    adapter = CampaignSources(
        graph,
        original.folds,
        index_root=original.index_root,
        dedup_root=original.dedup_root,
        historical_root=original.historical_root,
        ancestry_sha256="fixture",
        allowed_sequences={"original", "dev"},
        dense_folds=pins,
    )
    d0 = next(s for s in graph if s.fold == 0 and "-D0-" in s.name)
    dense = next(s for s in graph if s.fold == 0 and "-DENSE_OLD-" in s.name)
    assert (adapter.train(d0).features == 1).all()
    assert (adapter.train(dense).features == 8).all()
    assert (adapter.source(dense, "outer_dev").features == 8).all()
    assert calls == [0]
    (pins[0].compiled / "COMPILED.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="dense compiled manifest changed"):
        adapter.train(dense)


def test_diverse_dispatch_binds_original_targets_and_producers(wired, monkeypatch):
    from types import SimpleNamespace

    from e_jepa_ttc.simplex_t.campaign_sources import MatchedBinding

    original, _, _ = wired
    graph = [
        s
        for s in registered_graph(
            d1=True,
            density=True,
            t3=False,
            latent=False,
            replicate_scalar=False,
            replicate_latent=False,
        )
        if "-D0-" in s.name or "-DIVERSE_MATCHED-" in s.name
    ]
    manifest = original.index_root / "INDEX_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "families": [
                    {"outer_fold": outer, "role": f"inner{inner}", "family_sha256": "a" * 64}
                    for outer in range(3)
                    for inner in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )
    pool = original.index_root / "matched.json"
    pool.write_text("{}", encoding="utf-8")
    matched = MatchedBinding(pool, compute_file_hash(str(pool)), compute_file_hash(str(manifest)))
    pins = {}
    for fold, pin in original.folds.items():
        extra_pool = pin.path / "expansion_pool.json"
        extra_pool.write_text(
            json.dumps({"folds": {str(fold): {"sequence_family_sha256": {"extra": "b" * 64}}}}),
            encoding="utf-8",
        )
        pins[fold] = ExpansionBinding(
            pin.path,
            pin.sha256,
            pin.path,
            "x",
            pin.path,
            extra_pool,
            compute_file_hash(str(extra_pool)),
            pin.path,
            "x",
            pin.path,
            "x",
            "x",
        )
    calls = []

    def view(sources, **kwargs):
        calls.append(kwargs)
        assert kwargs["train_tokens"].tolist() == ["old", "extraq"]
        assert kwargs["train_target_ttc"].tolist() == [1.0, 2.0]
        assert kwargs["sequence_families"] == {"original": "a" * 64, "extra": "b" * 64}
        return {
            role: replace(
                source,
                features=np.full_like(source.features, 9),
                identity_sha256="diverse:" + source.identity_sha256,
            )
            for role, source in sources.items()
        }

    monkeypatch.setattr(
        campaign_sources,
        "load_expansion_inputs",
        lambda *a, **k: SimpleNamespace(
            source=None,
            tokens=np.array(["extraq"]),
            sequences=np.array(["extra"]),
            target_ttc=np.array([2.0]),
        ),
    )
    monkeypatch.setattr(
        campaign_sources,
        "load_current_inputs",
        lambda *a, **k: {
            "metadata": pd.DataFrame(
                {
                    "sample_token": ["old"],
                    "sequence_id": ["original"],
                    "inner_fold": [0],
                    "target_ttc": [1.0],
                }
            )
        },
    )
    monkeypatch.setattr(
        campaign_sources, "merge_validated_sources", lambda sources, *a, **k: sources
    )
    monkeypatch.setattr(campaign_sources, "diverse_source_view", view)
    adapter = CampaignSources(
        graph,
        original.folds,
        index_root=original.index_root,
        dedup_root=original.dedup_root,
        historical_root=original.historical_root,
        ancestry_sha256="fixture",
        allowed_sequences={"original"},
        expansion_folds=pins,
        expansion_sequences={"extra"},
        matched=matched,
    )
    d0 = next(s for s in graph if s.fold == 0 and "-D0-" in s.name)
    diverse = next(s for s in graph if s.fold == 0 and "-DIVERSE_MATCHED-" in s.name)
    assert (adapter.train(d0).features == 1).all()
    assert (adapter.train(diverse).features == 9).all()
    assert (adapter.source(diverse, "outer_dev").features == 9).all()
    assert len(calls) == 1
    pool.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="matched pool or original producer manifest changed"):
        adapter.train(diverse)
