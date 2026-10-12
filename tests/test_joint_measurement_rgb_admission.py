"""Scientific stage boundaries must hold before any RGB optimizer is constructed."""

import pytest

from operational.joint_measurement import rgb_train


def test_incomplete_primary_prevents_rgb_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(rgb_train, "REGISTRATION", tmp_path / "REGISTRATION.json")

    def incomplete():
        raise ValueError("Primary endpoint not complete")

    monkeypatch.setattr(rgb_train, "freeze_primary", incomplete)
    with pytest.raises(ValueError, match="endpoint not complete"):
        rgb_train.register("CORR", "Pending evidence")
    assert not rgb_train.REGISTRATION.exists()


def test_missing_final_comparison_prevents_rgb_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(rgb_train, "RUN", tmp_path)
    monkeypatch.setattr(rgb_train, "REGISTRATION", tmp_path / "REGISTRATION.json")
    monkeypatch.setattr(rgb_train, "freeze_primary", lambda: {})
    with pytest.raises(ValueError, match="Complete primary P/H"):
        rgb_train.register("CORR", "Pending comparison")
    assert not rgb_train.REGISTRATION.exists()


def test_unregistered_source_change_blocks_rgb_fit(tmp_path, monkeypatch):
    path = tmp_path / "REGISTRATION.json"
    path.write_text('{"sources": {"trainer": "old"}}')
    monkeypatch.setattr(rgb_train, "REGISTRATION", path)
    monkeypatch.setattr(rgb_train, "source_binding", lambda: {"trainer": "changed"})
    with pytest.raises(ValueError, match="source differs"):
        rgb_train.fit("RGB-PAIR-CORR-FT-S7")
