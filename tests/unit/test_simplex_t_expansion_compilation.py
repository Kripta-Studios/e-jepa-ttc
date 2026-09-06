"""Production compiler integration on synthetic blocks; no expert or optimizer work."""

import json

import numpy as np
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t import compiled_context
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc


def fixture(tmp_path, monkeypatch, *, pool="D1"):
    cache, index, dedup = (tmp_path / name for name in ("cache", "index", "dedup"))
    for path in (cache, index, dedup):
        path.mkdir()
    monkeypatch.setattr(compiled_context, "admitted", lambda paths: {"has_headroom": True})
    active = [0, 2] if pool == "D1" else [0, 1, 2]
    families = np.full((3, 3), -1, np.int64)
    families[0, active] = 0
    valid = np.zeros((3, 16), bool)
    valid[:, -1] = True
    anchors = np.array([100, 200, 300], np.int64)
    index_file = index / "query_context_index.npz"
    np.savez(
        index_file,
        producer_family=families,
        valid=valid,
        anchor_us=anchors,
        lag_us=np.arange(15, -1, -1, dtype=np.int64),
        roi_available_us=anchors + 10,
    )
    (cache / "IDENTITY.json").write_text(
        json.dumps({"pool": pool, "index_sha256": compute_file_hash(str(index_file))}),
        encoding="utf-8",
    )
    history = np.full((3, 16), -1, np.int64)
    for observation, query in enumerate(active):
        history[query, -1] = observation
        points = np.ones((1, 3), np.float32)
        features = np.zeros((1, 145), np.float32)
        features[:, 0] = query + 1
        features[:, 8:11] = expert_phase_from_ttc(points)
        path = cache / f"family00_query{query:05d}.npz"
        np.savez(
            path,
            features145=features,
            expert_ttc=points,
            known=np.ones((1, 2), bool),
            pair_features=np.zeros((1, 133), np.float32),
            observation_ids=np.array([observation], np.int64),
            anchor_us=anchors[query : query + 1],
            available_us=anchors[query : query + 1] + 10,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"query": query, "family": 0, "sha256": compute_file_hash(str(path))}),
            encoding="utf-8",
        )
    dedup_file = dedup / "outer0.npz"
    np.savez(dedup_file, history=history, keys=np.array([f"key{i}" for i in active]))
    (dedup / "DEDUP_MANIFEST.json").write_text(
        json.dumps(
            {"outputs": [{"path": dedup_file.name, "sha256": compute_file_hash(str(dedup_file))}]}
        ),
        encoding="utf-8",
    )
    return cache, index, dedup, tmp_path / "compiled"


def test_inactive_middle_query_preserves_real_query_ids(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    compiled_context.compile_fold(*args, outer=0, pool="D1")
    manifest = json.loads((args[-1] / "COMPILED.json").read_text(encoding="utf-8"))
    assert manifest["selected_query_ids"] == [0, 2]
    assert manifest["queries"] == manifest["observations"] == 2
    assert manifest["indexed_queries"] == 3 and manifest["inactive_queries"] == 1
    assert np.load(args[-1] / "features145.npy")[:, 0].tolist() == [1, 3]
    assert np.load(args[-1] / "anchor_us.npy").tolist() == [100, 300]
    for name, digest in manifest["arrays"].items():
        assert compute_file_hash(str(args[-1] / f"{name}.npy")) == digest


def test_d0_retains_manifest_schema(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, pool="D0")
    compiled_context.compile_fold(*args, outer=0)
    manifest = json.loads((args[-1] / "COMPILED.json").read_text(encoding="utf-8"))
    assert manifest["queries"] == 3
    assert "pool" not in manifest and "selected_query_ids" not in manifest


@pytest.mark.parametrize("corruption", ["identity", "family", "mask", "missing"])
def test_invalid_d1_is_rejected_before_allocating_outputs(tmp_path, monkeypatch, corruption):
    args = fixture(tmp_path, monkeypatch)
    cache, index, dedup, output = args
    identity_path = cache / "IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if corruption == "identity":
        identity["pool"] = "D0"
    elif corruption == "missing":
        # Rename fixture data only; preserve the unreceipted source for diagnosis.
        (cache / "family00_query00002.json").rename(cache / "missing_receipt.json")
    else:
        path = index / "query_context_index.npz"
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if corruption == "family":
            arrays["producer_family"][0, 2] = 3  # D1 TRAIN cannot use outer-final.
        else:
            arrays["valid"][2, -1] = False
        np.savez(path, **arrays)
        identity["index_sha256"] = compute_file_hash(str(path))
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError):
        compiled_context.compile_fold(*args, outer=0, pool="D1")
    assert not output.exists()
