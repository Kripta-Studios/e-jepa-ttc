"""ACK validation and absolute shared-volume reservations, no training."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack


def test_shared_disk_after_reservations():
    assert shared_write_admission(93_000_000_000, 73_000_000_000)
    assert not shared_write_admission(93_000_000_000, 73_000_000_001)
    assert not shared_write_admission(165_000_000_000, None)
    assert shared_write_admission(93_000_000_000, 0)


def test_ack_revalidates_every_bound_interface(tmp_path):
    metadata = tmp_path / "bound.json"
    metadata.write_text("{}")
    record = dict(
        path=str(metadata),
        sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        authoritative=True,
        read_only_import_authorized=True,
    )
    payload = dict(
        request_id="SIMPLEX_T_2026-09-06_T0",
        request=record,
        interfaces=dict(role_manifest=record, time_charter=record),
        producers=dict(authoritative_historical_manifest=record),
        resources=dict(resource_amendment=record),
    )
    path = tmp_path / "ack.json"
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verified_ack(path, digest)["request_id"] == payload["request_id"]
    metadata.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        verified_ack(path, digest)
