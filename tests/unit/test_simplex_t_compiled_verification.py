"""Compiled transport seals are read-only and cannot hide partial/corrupt arrays."""

import hashlib
import json

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.compiled_verification import verify_compiled_fold


def fixture(root):
    output = root / "compiled"
    output.mkdir()
    for name in ("IDENTITY.json", "query_context_index.npz", "outer0.npz"):
        (root / name).write_bytes(b"fixture-only-pinned-input")
    fields = {
        "features145": np.zeros((2, 145), np.float32),
        "expert_ttc": np.zeros((2, 3), np.float32),
        "known": np.zeros((2, 2), bool),
        "anchor_us": np.zeros(2, np.int64),
        "available_us": np.zeros(2, np.int64),
    }
    for name, array in fields.items():
        np.save(output / f"{name}.npy", array)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "simplex_t_compiled_query_context_v1",
        "status": "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE",
        "outer": 0,
        "queries": 8192,
        "observations": 2,
        "optimizer_updates": 0,
        "cache_identity_sha256": digest(root / "IDENTITY.json"),
        "index_sha256": digest(root / "query_context_index.npz"),
        "dedup_sha256": digest(root / "outer0.npz"),
        "arrays": {name: digest(output / f"{name}.npy") for name in fields},
    }
    (output / "COMPILED.json").write_text(json.dumps(manifest))
    return output, manifest


def verify(root, output, **kwargs):
    return verify_compiled_fold(
        output, root, root, root, 0, pool="D0", resource_ok=kwargs.get("resource_ok", lambda: True)
    )


def test_complete_is_read_only(tmp_path):
    output, _ = fixture(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert verify(tmp_path, output)
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_missing_seal_is_incomplete(tmp_path):
    before = set(tmp_path.iterdir())
    assert not verify(tmp_path, tmp_path / "missing")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("field", ["outer", "optimizer_updates", "queries", "observations"])
def test_bool_is_not_integer_metadata(tmp_path, field):
    output, manifest = fixture(tmp_path)
    manifest[field] = False
    (output / "COMPILED.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(tmp_path, output)


def test_changed_array_rejected(tmp_path):
    output, _ = fixture(tmp_path)
    (output / "known.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding changed"):
        verify(tmp_path, output)


def test_matching_hash_wrong_shape_rejected(tmp_path):
    output, manifest = fixture(tmp_path)
    path = output / "known.npy"
    np.save(path, np.zeros((2, 3), bool))
    manifest["arrays"]["known"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "COMPILED.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema mismatch"):
        verify(tmp_path, output)


def test_resource_denial_does_not_write(tmp_path):
    before = set(tmp_path.iterdir())
    with pytest.raises(InterruptedError, match="RESOURCE_PAUSE"):
        verify(tmp_path, tmp_path / "missing", resource_ok=lambda: False)
    assert set(tmp_path.iterdir()) == before
