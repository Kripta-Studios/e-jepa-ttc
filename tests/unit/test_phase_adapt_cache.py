import numpy as np
import pytest

from operational.phase_adapt_cache import resume_binding, save_npz


def test_migration_preserves_bytes_and_rejects_changed_scientific_source(tmp_path):
    from operational.evttc_transfer_audit.common import canonical_sha, file_sha
    from operational.phase_adapt_cache import atomic_receipt, migrate_shards

    root = tmp_path / "root"
    root.mkdir()
    (root / "model.py").write_text("original", encoding="utf-8")
    old, new = tmp_path / "old", tmp_path / "new"
    base = {
        "candidate": {"id": "same"},
        "manifest_sha256": "manifest",
        "role": "refit",
        "sources": {"model.py": file_sha(root / "model.py")},
    }
    before, after = dict(base, policy="v1"), dict(base, policy="v2")
    row = {"query_id": "q"}
    for directory, binding in ((old, before), (new, after)):
        atomic_receipt(directory / "BINDING.json", binding)
        atomic_receipt(directory / "POPULATION.json", {"rows": [row]})
    old_key = canonical_sha({"row": row, "binding": before})
    source = old / "shards" / (old_key + ".npz")
    save_npz(source, {"features": np.array([1], dtype=np.float32)})
    atomic_receipt(
        source.with_suffix(".json"),
        {"key": old_key, "query_id": "q", "feature_sha256": file_sha(source)},
    )
    report = migrate_shards(old, new, root)
    assert len(report["shards"]) == 1
    assert (
        new / "shards" / (report["shards"][0]["new_key"] + ".npz")
    ).read_bytes() == source.read_bytes()
    (root / "model.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="scientific source"):
        migrate_shards(old, new, root)


def test_resume_allows_unrelated_additions_but_rejects_changed_executed_source():
    old = {"role": "refit", "sources": {"a.py": "original"}}
    added = {"role": "refit", "sources": {"a.py": "original", "unrelated.py": "new"}}
    assert resume_binding(old, added) == old
    with pytest.raises(ValueError, match="source"):
        resume_binding(old, {"role": "refit", "sources": {"a.py": "changed"}})


def test_orphan_npz_is_recovered_only_after_exact_recomputation(tmp_path):
    path = tmp_path / "shard.npz"
    np.savez_compressed(path, features=np.array([1.0], dtype=np.float32))
    before = path.read_bytes()
    save_npz(path, {"features": np.array([1.0], dtype=np.float32)})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="different"):
        save_npz(path, {"features": np.array([2.0], dtype=np.float32)})
