"""Input-only history transport tests, with no model or label decoding."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t.history_bundle import HistoryPoolPins, history_bundle_members


@pytest.mark.parametrize("fault", ["none", "payload", "lineage", "fold", "pause"])
def test_history_inventory(tmp_path, fault):
    def save(path, payload):
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    pools = {}
    for pool in ("D0", "D1", "DENSE_OLD"):
        root = tmp_path / pool
        root.mkdir()
        data_hash = save(root / "query_context_index.npz", b"synthetic input-only index")
        index = root / "INDEX_MANIFEST.json"
        index_hash = save(index, json.dumps(dict(index_sha256=data_hash)).encode())
        rows = [
            dict(path=f"outer{fold}.npz", sha256=save(root / f"outer{fold}.npz", b"fixture"))
            for fold in range(3)
        ]
        linked = "b" * 64 if fault == "lineage" else index_hash
        state = dict(
            outputs=rows, input_manifest_sha256=linked, identity=dict(index_manifest_sha256=linked)
        )
        if fault == "fold":
            rows.pop()
        dedup = root / "DEDUP_MANIFEST.json"
        dedup_hash = save(dedup, json.dumps(state).encode())
        pools[pool] = HistoryPoolPins(index, index_hash, dedup, dedup_hash)
        if fault == "payload":
            (root / "outer0.npz").write_bytes(b"modified")
    kwargs = dict(
        work_root=tmp_path, validate_authority=lambda: None, resource_ok=lambda: fault != "pause"
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            history_bundle_members(pools, **kwargs)
        return
    members = history_bundle_members(pools, **kwargs)
    assert len(members) == 18
    for member in members.values():
        assert hashlib.sha256(member.path.read_bytes()).hexdigest() == member.sha256
