"""Production compiler integration on synthetic blocks; no expert or optimizer work."""

import json

import numpy as np
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t import compiled_context
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc
from e_jepa_ttc.simplex_t.reuse_catalog import D0ReuseCatalog


def fixture(tmp_path, monkeypatch, *, pool="D1"):
    cache, index, dedup = (tmp_path / name for name in ("cache", "index", "dedup"))
    for path in (cache, index, dedup):
        path.mkdir()
    monkeypatch.setattr(
        compiled_context,
        "admitted",
        lambda paths: {"has_headroom": True, "written_volume_free_bytes": [80_000_000_000]},
    )
    active = [0, 1, 2] if pool == "D0" else [0, 2]
    families = np.full((3, 3), -1, np.int64)
    families[0, active] = 0
    valid = np.zeros((3, 16), bool)
    valid[:, -1] = True
    anchors = np.array([100, 200, 300], np.int64)
    index_file = index / "query_context_index.npz"
    np.savez(
        index_file,
        tokens=np.array(["q0", "q1", "q2"]),
        producer_family=families,
        valid=valid,
        anchor_us=anchors,
        lag_us=np.arange(15, -1, -1, dtype=np.int64),
        roi_available_us=anchors + 10,
    )
    (cache / "IDENTITY.json").write_text(
        json.dumps(
            {
                "pool": pool,
                "index_sha256": compute_file_hash(str(index_file)),
                **{
                    key: "a" * 64
                    for key in (
                        "preprocessing_sha256",
                        "extractor_sha256",
                        "expert_phase_sha256",
                        "voxel_sha256",
                        "union_reader_sha256",
                    )
                },
                "torch": "fixture",
                "batch_size": 16,
                "layout": "fixture",
                "precision": "FP32",
                "tf32": False,
            }
        ),
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
    np.savez(dedup_file, history=history, keys=np.array([f"{i:064x}" for i in active]))
    (dedup / "DEDUP_MANIFEST.json").write_text(
        json.dumps(
            {"outputs": [{"path": dedup_file.name, "sha256": compute_file_hash(str(dedup_file))}]}
        ),
        encoding="utf-8",
    )
    return cache, index, dedup, tmp_path / "compiled"


@pytest.mark.parametrize("pool", ["D1", "DENSE_OLD"])
def test_inactive_middle_query_preserves_real_query_ids(tmp_path, monkeypatch, pool):
    args = fixture(tmp_path, monkeypatch, pool=pool)
    compiled_context.compile_fold(*args, outer=0, pool=pool, other_reserved_bytes=0)
    manifest = json.loads((args[-1] / "COMPILED.json").read_text(encoding="utf-8"))
    assert manifest["selected_query_ids"] == [0, 2]
    assert manifest["pool"] == pool
    assert manifest["queries"] == manifest["observations"] == 2
    assert manifest["indexed_queries"] == 3 and manifest["inactive_queries"] == 1
    assert np.load(args[-1] / "features145.npy")[:, 0].tolist() == [1, 3]
    assert np.load(args[-1] / "anchor_us.npy").tolist() == [100, 300]
    for name, digest in manifest["arrays"].items():
        assert compute_file_hash(str(args[-1] / f"{name}.npy")) == digest


def test_d0_retains_manifest_schema(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, pool="D0")
    compiled_context.compile_fold(*args, outer=0, other_reserved_bytes=0)
    manifest = json.loads((args[-1] / "COMPILED.json").read_text(encoding="utf-8"))
    assert manifest["queries"] == 3
    assert "pool" not in manifest and "selected_query_ids" not in manifest


def test_density_compiler_compacts_only_selected_observations(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    identity_path = args[0] / "IDENTITY.json"
    identity = json.loads(identity_path.read_text())
    identity["query_selection"] = {"path": "fixture-only", "sha256": "a" * 64}
    identity_path.write_text(json.dumps(identity))
    monkeypatch.setattr(
        compiled_context, "bound_selection_rows", lambda *args: np.array([2], dtype=np.int64)
    )
    compiled_context.compile_fold(*args, outer=0, pool="D1", other_reserved_bytes=0)
    manifest = json.loads((args[-1] / "COMPILED.json").read_text())
    assert manifest["selected_query_ids"] == [2]
    assert manifest["queries"] == manifest["observations"] == 1
    assert np.load(args[-1] / "source_observation_ids.npy").tolist() == [1]
    assert np.load(args[-1] / "features145.npy")[:, 0].tolist() == [3]


@pytest.mark.parametrize("corruption", ["identity", "family", "mask", "missing"])
@pytest.mark.parametrize("pool", ["D1", "DENSE_OLD"])
def test_invalid_d1_is_rejected_before_allocating_outputs(tmp_path, monkeypatch, corruption, pool):
    args = fixture(tmp_path, monkeypatch, pool=pool)
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
        compiled_context.compile_fold(*args, outer=0, pool=pool, other_reserved_bytes=0)
    assert not output.exists()


@pytest.mark.parametrize("failure", [None, "missing_new", "changed_old", "recipe"])
def test_mixed_dense_compilation_reuses_pinned_d0_without_copying_blocks(
    tmp_path, monkeypatch, failure
):
    source_root, dense_root = tmp_path / "source", tmp_path / "dense"
    source_root.mkdir()
    dense_root.mkdir()
    source = fixture(source_root, monkeypatch, pool="D0")
    compiled_context.compile_fold(*source, outer=0, other_reserved_bytes=0)
    catalog = D0ReuseCatalog(
        compiled=source[-1],
        compiled_sha256=compute_file_hash(str(source[-1] / "COMPILED.json")),
        cache=source[0],
        index_root=source[1],
        dedup=source[2] / "outer0.npz",
        outer=0,
    )
    dense = fixture(dense_root, monkeypatch, pool="DENSE_OLD")
    path = dense[1] / "query_context_index.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["tokens"] = np.array(["q0", "unused", "new-query"])
    np.savez(path, **arrays)
    identity_path = dense[0] / "IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["index_sha256"] = compute_file_hash(str(path))
    if failure == "recipe":
        identity["batch_size"] = 32
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    # Only temporary fixture files: the shared query deliberately has no dense copy.
    for suffix in (".json", ".npz"):
        (dense[0] / f"family00_query00000{suffix}").unlink()
    if failure == "missing_new":
        (dense[0] / "family00_query00002.json").unlink()
    elif failure == "changed_old":
        (source[0] / "family00_query00000.npz").write_bytes(b"changed")
    if failure:
        with pytest.raises(ValueError):
            compiled_context.compile_fold(
                *dense, outer=0, pool="DENSE_OLD", reuse=catalog, other_reserved_bytes=0
            )
        assert not (dense[-1] / "COMPILED.json").exists()
    else:
        compiled_context.compile_fold(
            *dense, outer=0, pool="DENSE_OLD", reuse=catalog, other_reserved_bytes=0
        )
        manifest = json.loads((dense[-1] / "COMPILED.json").read_text(encoding="utf-8"))
        assert manifest["queries"] == 2
        assert set(manifest["D0_reuse"]["blocks"]) == {"0"}
        assert np.load(dense[-1] / "features145.npy")[:, 0].tolist() == [1, 3]


@pytest.mark.parametrize("reservation", [None, -1, True])
def test_compile_requires_explicit_reservations(tmp_path, monkeypatch, reservation):
    args = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="reservations"):
        compiled_context.compile_fold(*args, outer=0, pool="D1", other_reserved_bytes=reservation)
    assert not args[-1].exists()


def test_compile_reserves_own_arrays_before_allocation(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    # Enough for the first metadata check, not for arrays plus their headers.
    monkeypatch.setattr(
        compiled_context,
        "admitted",
        lambda paths: {
            "has_headroom": True,
            "written_volume_free_bytes": [20_000_000_000 + 1_048_576],
        },
    )
    with pytest.raises(RuntimeError, match="RESOURCE_PAUSE"):
        compiled_context.compile_fold(*args, outer=0, pool="D1", other_reserved_bytes=0)
    assert not args[-1].exists()


def test_compile_midstream_pause_never_seals_partial_arrays(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    stored = []
    original_store = compiled_context.store_observations
    original_cadence = compiled_context.ResourceCadence
    monkeypatch.setattr(
        compiled_context,
        "ResourceCadence",
        lambda probe, **kw: original_cadence(probe, **kw, clock=lambda: float(len(stored))),
    )

    def store(*values):
        original_store(*values)
        stored.append(True)

    monkeypatch.setattr(compiled_context, "store_observations", store)
    monkeypatch.setattr(
        compiled_context,
        "admitted",
        lambda paths: {
            "has_headroom": not stored,
            "written_volume_free_bytes": [80_000_000_000],
        },
    )
    with pytest.raises(RuntimeError, match="RESOURCE_PAUSE"):
        compiled_context.compile_fold(*args, outer=0, pool="D1", other_reserved_bytes=0)
    assert len(stored) == 1
    assert (args[-1] / "features145.npy").exists()
    assert not (args[-1] / "COMPILED.json").exists()


def test_compile_other_reservations_are_not_free_space(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="RESOURCE_PAUSE"):
        compiled_context.compile_fold(
            *args, outer=0, pool="D1", other_reserved_bytes=60_000_000_000
        )
    assert not args[-1].exists()
