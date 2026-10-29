"""Check the second authorization chain, exact local cutoff, and fail-closed expiry."""

import hashlib
import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from scripts import runpod_deadline_extension_0100_20261027 as extension


@pytest.fixture
def bound(tmp_path, monkeypatch):
    authority = tmp_path / "authority.json"
    authority.write_text("{}")
    hashes = {str(authority): hashlib.sha256(authority.read_bytes()).hexdigest()}
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "schema": "ejepa_explicit_deadline_extension_v1",
                "authorized": True,
                "pod_id": "lreiivpf18vxu7",
                "previous_deadline_utc": "2026-10-27T14:09:00+00:00",
                "extended_deadline_utc": "2026-10-27T20:09:00+00:00",
                "historical_authority_sha256": hashes,
            }
        )
    )
    value = {
        "schema": "ejepa_explicit_deadline_extension_v2",
        "authorized": True,
        "pod_id": "lreiivpf18vxu7",
        "original_deadline_utc": "2026-10-27T14:09:00+00:00",
        "previous_effective_deadline_utc": "2026-10-27T20:09:00+00:00",
        "extended_deadline_utc": "2026-10-28T00:00:00+00:00",
        "historical_authority_sha256": hashes,
        "supersedes_receipt": str(parent),
        "supersedes_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    }
    receipt = tmp_path / "extension.json"
    receipt.write_text(json.dumps(value))
    monkeypatch.setenv("EJEPA_DEADLINE_EXTENSION", str(receipt))
    monkeypatch.setenv(
        "EJEPA_DEADLINE_EXTENSION_SHA256", hashlib.sha256(receipt.read_bytes()).hexdigest()
    )

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 27, 22, tzinfo=UTC)

    monkeypatch.setattr(extension, "datetime", Clock)
    return authority, parent, receipt


def test_exact_cutoff_after_previous_expired(bound):
    deadline = extension.extension_deadline(datetime(2026, 10, 27, 14, 9, tzinfo=UTC))
    assert deadline.astimezone(ZoneInfo("Europe/Madrid")).isoformat() == "2026-10-28T01:00:00+01:00"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_modified_authority_parent_or_receipt_rejected(bound, index):
    bound[index].write_text('{"modified":true}')
    with pytest.raises(ValueError):
        extension.extension_deadline(datetime(2026, 10, 27, 14, 9, tzinfo=UTC))


def test_wrong_original_rejected(bound):
    with pytest.raises(ValueError):
        extension.extension_deadline(datetime(2026, 10, 27, 15, 9, tzinfo=UTC))


def test_expired_extension_rejected(bound, monkeypatch):
    class Expired(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 28, 0, tzinfo=UTC)

    monkeypatch.setattr(extension, "datetime", Expired)
    with pytest.raises(ValueError):
        extension.extension_deadline(datetime(2026, 10, 27, 14, 9, tzinfo=UTC))
