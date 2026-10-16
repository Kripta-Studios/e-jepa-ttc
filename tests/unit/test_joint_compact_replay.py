"""A compact bundle must disclose unavailable evidence rather than claim replay."""

import json

import pytest

from operational.joint_measurement.replay import omitted_payloads


def test_declared_missing_checkpoint_is_reported(tmp_path):
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(
        json.dumps({"context": {"omitted_files": [{"path": "verification/state.pt"}]}})
    )
    path = tmp_path / "campaign/verification/state.pt"
    assert omitted_payloads(tmp_path, [path]) == ["verification/state.pt"]


def test_undeclared_missing_checkpoint_fails(tmp_path):
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps({"context": {}}))
    with pytest.raises(ValueError, match="Missing undeclared"):
        omitted_payloads(tmp_path, [tmp_path / "campaign/state.pt"])


def test_present_payload_is_not_excluded(tmp_path):
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps({"context": {}}))
    (tmp_path / "campaign").mkdir()
    path = tmp_path / "campaign/state.pt"
    path.write_bytes(b"retained")
    assert omitted_payloads(tmp_path, [path]) == []
