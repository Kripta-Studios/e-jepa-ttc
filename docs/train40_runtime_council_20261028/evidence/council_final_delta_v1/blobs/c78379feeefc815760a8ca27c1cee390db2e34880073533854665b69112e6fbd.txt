"""Only complete, unchanged payloads get durable authoritative SHA-256 receipts."""

import json

import pytest

from operational.efficient_context.common import digest
from operational.train40_system.verify_downloads import verify


def test_verifier_reuses_receipt_and_rejects_changed_data(tmp_path):
    media = tmp_path / "events.h5"
    media.write_bytes(b"complete TRAIN fixture")
    receipt = tmp_path / "receipt.json"
    expected = {"bytes": media.stat().st_size, "sha256": digest(media)}
    assert verify((media, receipt, expected))
    assert json.loads(receipt.read_text())["status"] == "VERIFIED"
    assert verify((media, receipt, expected))
    media.write_bytes(b"changed! TRAIN fixture")
    with pytest.raises(ValueError, match="changed"):
        verify((media, receipt, expected))


def test_incomplete_or_corrupt_download_gets_no_valid_receipt(tmp_path):
    media = tmp_path / "rgb.tar"
    media.write_bytes(b"partial")
    receipt = tmp_path / "receipt.json"
    assert not verify((media, receipt, {"bytes": 8, "sha256": "wrong"}))
    assert not receipt.exists()
    with pytest.raises(ValueError, match="SHA-256"):
        verify((media, receipt, {"bytes": 7, "sha256": "wrong"}))
    assert not receipt.exists()
    assert receipt.with_suffix(".failure.json").is_file()
