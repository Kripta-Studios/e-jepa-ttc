"""T0 safeguards do not need real data, GPU, or fitted experts."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import (
    interface_status,
    parquet_header,
    resource_headroom,
    verify_handoff,
    write_new_json,
)


def test_absolute_resource_floors():
    gib = 1024**3
    assert resource_headroom(
        available_ram=8 * gib, process_tree_rss=4 * gib, written_volume_free=[40_000_000_000]
    )["has_headroom"]
    result = resource_headroom(
        available_ram=8 * gib - 1,
        process_tree_rss=4 * gib + 1,
        written_volume_free=[40_000_000_000 - 1],
    )
    assert len(result["reasons"]) == 3


def test_missing_authority_cannot_enable_fitting(tmp_path):
    result = interface_status({"shared_coordination": str(tmp_path)})
    assert result["execution_status"] == "WAITING_SHARED_ROLE_MANIFEST"
    assert not result["scientific_run_enabled"]


def test_ack_presence_is_not_semantic_authority(tmp_path):
    (tmp_path / "SIMPLEX_T_STAGE70_ACK.json").write_text("{}")
    result = interface_status(
        {"shared_coordination": str(tmp_path), "role_manifest": "x", "time_charter": "y"}
    )
    assert result["execution_status"] == "INTERFACE_REVIEW_REQUIRED"
    assert not result["authoritative_roles_adopted"]
    assert not result["scientific_run_enabled"]


def test_ttc_perturbation_does_not_change_projected_identity_metadata(tmp_path):
    path = tmp_path / "train.parquet"
    counts = []
    for targets in ([1.0, 2.0], [-999.0, float("nan")]):
        pq.write_table(pa.table({"sequence_id": ["s1", "s2"], "ttc": targets}), path)
        result = parquet_header(path, identity_projection=True)
        counts.append(result["sequence_row_counts"])
        assert result["projected_columns"] == ["sequence_id"]
        assert not result["target_values_read"]
    assert counts[0] == counts[1]


def test_protected_payload_not_opened(tmp_path):
    with pytest.raises(ValueError, match="TRAIN"):
        parquet_header(tmp_path / "test.parquet", identity_projection=True)


def test_footer_only_does_not_read_target_columns(tmp_path):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"ttc": [1.0]}), path)
    result = parquet_header(path, identity_projection=False)
    assert result["rows"] == 1 and result["projected_columns"] == []


def test_evidence_refuses_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    write_new_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"first": False})
    assert json.loads(path.read_text())["first"]


def test_checksum_and_traversal(tmp_path):
    content = b"payload"
    (tmp_path / "a").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(f"{digest}  a\n")
    assert verify_handoff(tmp_path) == {"a": digest}
    (tmp_path / "a").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_handoff(tmp_path)
    sums.write_text(f"{digest}  ../outside\n")
    with pytest.raises(ValueError, match="invalid"):
        verify_handoff(tmp_path)
