"""Targeted tests of metadata admission; no model, optimizer or event payload."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from operational.simplex_t_route_readiness.inspect import (
    blockers,
    file_presence,
    metadata,
    selected_queries,
    verify_small_checkpoints,
)


def sample() -> tuple[list[dict[str, Any]], list[tuple[str, dict, dict]]]:
    queries = [{"sample_token": f"q{i}", "sequence_id": "TRAIN_SEQ"} for i in range(64)]
    manifest = {"families": [{"role": f"inner{i}", "experts": {"A5": "sha"}} for i in range(3)]}
    arrays = {
        "tokens": np.array([q["sample_token"] for q in queries]),
        "sequences": np.array(["TRAIN_SEQ"] * 64),
        "producer_family": np.zeros((3, 64), dtype=np.int64),
        "valid": np.ones((64, 16), dtype=bool),
    }
    return queries, [("D1", manifest, arrays)]


def test_fixed_train_selection() -> None:
    queries, pools = sample()
    result = selected_queries(queries, pools)
    assert len(result) == 64
    assert all(q["family_id"] == 0 and q["valid_history_slots"] == 16 for q in result)


@pytest.mark.parametrize("failure", ["duplicate", "absent", "dev", "sequence", "current"])
def test_reject_selection_drift(failure: str) -> None:
    queries, pools = sample()
    arrays = pools[0][2]
    if failure == "duplicate":
        queries[1] = queries[0]
    elif failure == "absent":
        queries[0]["sample_token"] = "absent"
    elif failure == "dev":
        pools[0][1]["families"].append({"role": "outer_dev", "experts": {"A5": "sha"}})
        arrays["producer_family"][0, 0] = 3
    elif failure == "sequence":
        queries[0]["sequence_id"] = "SEALED"
    else:
        arrays["valid"][0, -1] = False
    with pytest.raises(ValueError):
        selected_queries(queries, pools)


def test_ambiguous_token_not_resolved_by_preference() -> None:
    queries, pools = sample()
    with pytest.raises(ValueError, match="ambiguous"):
        selected_queries(queries, pools + pools)


def test_idle_gpu_does_not_grant_slot() -> None:
    assert blockers(None, {"status": "OBSERVED", "live_python_gpu_processes": []}, True) == [
        "MISSING_EXPLICIT_EXCLUSIVE_GPU_HEAVY_IO_SLOT",
    ]


def test_slot_still_requires_live_validation_and_no_foreign_job() -> None:
    result = blockers(
        {"owner": "x"}, {"status": "OBSERVED", "live_python_gpu_processes": [{"pid": 42}]}, True
    )
    assert result == [
        "SLOT_REQUIRES_OWNER_SCOPE_EXPIRY_AND_LIVE_VALIDATION",
        "OTHER_LIVE_PYTHON_GPU_JOB",
    ]


def test_unknown_gpu_and_missing_sources_are_explicit() -> None:
    result = blockers(None, {"status": "UNKNOWN"}, False)
    assert "GPU_OBSERVATION_UNAVAILABLE" in result
    assert "REFERENCED_SOURCE_FILE_MISSING" in result


def test_bound_metadata_tampering_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bound.json"
    path.write_bytes(b'{"status":"old"}')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata(path, sha)["status"] == "old"
    path.write_bytes(b'{"status":"new"}')
    with pytest.raises(ValueError, match="changed"):
        metadata(path, sha)


def test_metadata_bound_before_read(tmp_path: Path) -> None:
    path = tmp_path / "too_big.json"
    path.write_bytes(b" " * (1_048_576 + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        metadata(path)


def test_weights_checked_against_registered_bytes(tmp_path: Path) -> None:
    path = tmp_path / "weight.pt"
    path.write_bytes(b"frozen weights")
    row = {
        **file_presence(path),
        "registered_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    verify_small_checkpoints([row])
    assert row["payload_hash_reverified"]
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="weights changed"):
        verify_small_checkpoints([row])


def test_heavy_weights_denied_before_payload_access(tmp_path: Path) -> None:
    row = {"path": str(tmp_path / "never_open.pt"), "bytes": 32 * 1024**2 + 1, "exists": True}
    with pytest.raises(ValueError, match="light-I/O cap"):
        verify_small_checkpoints([row])
