"""Pinned dedup loading and inactive-fold masks using synthetic metadata."""

import json

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.expanded_history_loader import ExpandedHistoryLoader


@pytest.mark.parametrize("pool", ["D1", "DENSE_OLD"])
@pytest.mark.parametrize("failure", ["", "inactive", "bytes", "index"])
def test_pinned_history_loader(tmp_path, pool, failure):
    valid = np.zeros((2, 16), dtype=bool)
    valid[:, -1] = True
    families = np.array([[0, 1], [4, 5], [-1, 9]])
    records = []
    for outer in range(3):
        history = np.full((2, 16), -1, dtype=np.int64)
        history[:, -1] = [0, 1]
        if outer == 2 and failure != "inactive":
            history[0, -1] = -1
        path = tmp_path / f"outer{outer}.npz"
        np.savez(path, history=history, keys=np.array(["a", "b"]))
        records.append({"path": path.name, "sha256": sha256(path)})
    manifest = {
        "status": "D1_CONTENT_INDEX_READY_NOT_FEATURE_CACHE_OR_REPLAY_AUTHORIZATION"
        if pool == "D1"
        else "DENSE_CONTENT_INDEX_READY_PENDING_TIME_ACK_AND_FEATURE_REPLAY",
        "identity": {"index_manifest_sha256": "a" * 64},
        "input_manifest_sha256": "a" * 64,
        "outputs": records,
    }
    path = tmp_path / "DEDUP_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    if failure == "index":
        with pytest.raises(ValueError):
            ExpandedHistoryLoader(
                tmp_path,
                manifest_sha256=sha256(path),
                index_manifest_sha256="b" * 64,
                pool=pool,
                producer_family=families,
                valid=valid,
            )
        return
    loader = ExpandedHistoryLoader(
        tmp_path,
        manifest_sha256=sha256(path),
        index_manifest_sha256="a" * 64,
        pool=pool,
        producer_family=families,
        valid=valid,
    )
    first = loader(0)
    assert loader(0) is first
    assert not first.flags.writeable
    if failure == "bytes":
        (tmp_path / "outer0.npz").write_bytes(b"changed cached file")
        with pytest.raises(ValueError):
            loader(0)
    elif failure == "inactive":
        with pytest.raises(ValueError):
            loader(2)
    else:
        assert (loader(2) >= 0).sum() == 1
