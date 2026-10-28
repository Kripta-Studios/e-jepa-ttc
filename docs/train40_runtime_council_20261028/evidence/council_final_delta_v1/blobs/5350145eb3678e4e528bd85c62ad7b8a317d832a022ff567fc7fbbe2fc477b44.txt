"""Archive verification and physical accounting must detect tampering and incomplete endpoints."""

import json
import zipfile

import pytest

from operational.train40_system.delivery import accounting, write_bundle
from operational.train40_system.durable_io import atomic_json


def test_partial_accounting_retains_pending_and_recovery(tmp_path):
    atomic_json(tmp_path / "AUTHORIZATION.json", {"previous_physical_execution_upper": 40748})
    atomic_json(tmp_path / "TECHNICAL_ACCOUNTING.json", {"synthetic_optimizer_updates": 35})
    atomic_json(
        tmp_path / "fits/a5_seed7/UPDATE_JOURNAL.json",
        {
            "committed_updates": 123,
            "durable_updates": 100,
            "pending_update_upper": 1,
            "recovery_upper": 9,
        },
    )
    result = accounting(tmp_path)
    assert result["physical_upper"] == 40748 + 35 + 123 + 1 + 9
    assert result["fits"]["c2f_seed7"]["status"] == "NOT_STARTED"
    assert result["new_fixed_scientific_endpoint"] == 114204


def test_unregistered_fit_update_count_cannot_be_reported_complete(tmp_path):
    atomic_json(tmp_path / "AUTHORIZATION.json", {"previous_physical_execution_upper": 40748})
    atomic_json(tmp_path / "TECHNICAL_ACCOUNTING.json", {"synthetic_optimizer_updates": 35})
    atomic_json(
        tmp_path / "fits/a5_seed7/UPDATE_JOURNAL.json",
        {
            "committed_updates": 49933,
            "durable_updates": 49933,
            "pending_update_upper": 0,
            "recovery_upper": 0,
        },
    )
    with pytest.raises(ValueError, match="Unregistered"):
        accounting(tmp_path)


def test_essential_archive_has_independently_verified_manifest(tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(bytes(range(256)) * 20)
    destination = tmp_path / "bundle.zip"
    result = write_bundle({"outputs/weights.bin": source}, destination)
    assert result["status"] == "PASSED"
    assert result["verified_members"] == 1
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["files"][0]["bytes"] == source.stat().st_size
        assert archive.read("outputs/weights.bin") == source.read_bytes()
