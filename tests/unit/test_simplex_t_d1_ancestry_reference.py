"""Actual prepared-index ancestry indirection, without feature or target loading."""

import json

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.expanded_replay_plan import verify_d1_index_ancestry


@pytest.mark.parametrize("fault", ["none", "bytes", "ancestry", "status", "conflict", "missing"])
def test_pinned_ancestry_receipt(tmp_path, fault):
    ancestry = tmp_path / "ancestry.json"
    ancestry.write_text("synthetic ancestry")
    receipt = tmp_path / "EXPANSION_ANCESTRY_RECHECK.json"
    ref = {"path": str(ancestry), "sha256": sha256(ancestry)}
    record = {
        "status": "D1_FROZEN_INNER_ANCESTRY_RECHECKED_NO_EXPERT_REFITS",
        "authoritative_ancestry": dict(ref),
    }
    if fault == "status":
        record["status"] = "unverified"
    if fault == "ancestry":
        record["authoritative_ancestry"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(record))
    manifest = {"sources": {str(receipt): sha256(receipt)}}
    if fault == "bytes":
        receipt.write_text("changed")
    if fault == "conflict":
        manifest["ancestry"] = dict(ref, sha256="0" * 64)
    if fault == "missing":
        manifest = {}
    if fault == "none":
        verify_d1_index_ancestry(manifest, ancestry, sha256(ancestry))
    else:
        with pytest.raises(ValueError):
            verify_d1_index_ancestry(manifest, ancestry, sha256(ancestry))
