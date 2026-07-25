"""Shared pytest fixtures that preserve production provenance checks."""

import json

import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact


@pytest.fixture(autouse=True)
def isolated_frozen_protocol(tmp_path, monkeypatch) -> None:
    """Point tests and their subprocesses at a signed isolated protocol."""

    protocol_path = tmp_path / "frozen_protocol.json"
    protocol = sign_artifact(
        {
            "artifact_type": "frozen_protocol_v3",
            "protocol_version": "3.0",
            "protocol_sha256": "a" * 64,
            "code_commit": "b" * 40,
            "claim_level": "test_fixture",
        }
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setenv("E_JEPA_TTC_FROZEN_PROTOCOL", str(protocol_path))
