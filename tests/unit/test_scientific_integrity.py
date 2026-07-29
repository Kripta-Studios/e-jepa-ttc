"""Integrity regression tests for protocol and migration safeguards."""

from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_recovery_protocol_contains_physical_manifest_hash() -> None:
    protocol = yaml.safe_load(
        (_repo_root() / "configs" / "recovery_v3_protocol.yaml").read_text(encoding="utf-8")
    )
    manifest_hash = protocol["requirements"]["dataset"]["manifest_hash"]
    assert len(manifest_hash) == 64
    assert all(character in "0123456789abcdef" for character in manifest_hash)
    assert protocol["claim_level"] == "diagnostic"
    assert protocol["test_status"] == "reused_test_diagnostic"
