"""Transient Windows reader locks must not abort otherwise valid scientific state."""

import json

import pytest

from operational.train40_system import durable_io


def test_transient_reader_lock_preserves_atomic_publication(tmp_path, monkeypatch):
    destination = tmp_path / "progress.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    original = durable_io.os.replace
    attempts = []

    def flaky(source, target):
        attempts.append(source)
        if len(attempts) == 1:
            assert json.loads(destination.read_text()) == {"old": True}
            error = PermissionError("Windows reader lock")
            error.winerror = 5
            raise error
        original(source, target)

    monkeypatch.setattr(durable_io.os, "replace", flaky)
    durable_io.atomic_json(destination, {"new": True})
    assert len(attempts) == 2
    assert json.loads(destination.read_text()) == {"new": True}


def test_nonsharing_failure_retains_pending_and_original(tmp_path, monkeypatch):
    destination = tmp_path / "progress.json"
    destination.write_text('{"old": true}', encoding="utf-8")

    def permanent(source, target):
        raise PermissionError("Not a Windows sharing lock")

    monkeypatch.setattr(durable_io.os, "replace", permanent)
    with pytest.raises(PermissionError):
        durable_io.atomic_json(destination, {"new": True})
    assert json.loads(destination.read_text()) == {"old": True}
    assert len(list(tmp_path.glob("*.pending"))) == 1
