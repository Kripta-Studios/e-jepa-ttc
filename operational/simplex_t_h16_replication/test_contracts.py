"""Operational guardrail/ownership tests; never construct or update an optimizer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import common


def test_only_authorized_six_ids() -> None:
    rows = common.ids()
    assert len(rows) == len({r["key"] for r in rows}) == 6
    assert {(r["seed"], r["fold"]) for r in rows} == {
        (seed, fold) for seed in (13, 23) for fold in range(3)
    }
    assert sum(r["updates"] for r in rows) == 15000
    assert all("TPR-D1-H16-C160" in r["key"] for r in rows)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("available", 2 * 1024**3 - 1),
        ("tree_rss", 4 * 1024**3 + 1),
        ("commit_headroom", 2 * 1024**3 - 1),
        ("free_disk", 10_000_000_000 + common.RESERVATION - 1),
    ],
)
def test_guardrails_fail_closed(monkeypatch, tmp_path: Path, field: str, bad: int) -> None:
    m = dict(
        available=2 * 1024**3,
        tree_rss=4 * 1024**3,
        commit_headroom=2 * 1024**3,
        free_disk=10_000_000_000 + common.RESERVATION,
    )
    monkeypatch.setattr(common, "OUT", tmp_path)
    monkeypatch.setattr(common, "memory", lambda: m)
    probe = common.Resources()
    assert probe()
    m[field] = bad
    assert not probe()
    assert common.record(tmp_path / "RESOURCES.json")["reasons"]


def test_hash_rejects_same_stat_changed_input(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"first")
    pin = common.inventory([path])[0]
    path.write_bytes(b"other")
    os.utime(path, ns=(path.stat().st_atime_ns, pin["mtime_ns"]))
    common.validate_pins(dict(input_inventory=[pin]), full=False)
    with pytest.raises(ValueError, match="hash changed"):
        common.validate_pins(dict(input_inventory=[pin]), full=True)


def test_live_writer_cannot_be_replaced(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    with common.Lease():
        pin = common.digest(tmp_path / "WRITER.lock")
        with pytest.raises(RuntimeError, match="live campaign owner"):
            with common.Lease():
                raise AssertionError("second writer admitted")
        assert common.digest(tmp_path / "WRITER.lock") == pin
    assert not (tmp_path / "WRITER.lock").exists()


def test_publication_conflict_preserves_old_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    common.publish_json(path, dict(fit="authorized", updates=100))
    pin = common.digest(path)
    common.publish_json(path, dict(fit="authorized", updates=100))
    with pytest.raises(ValueError, match="publication changed"):
        common.publish_json(path, dict(fit="different", updates=100))
    assert common.digest(path) == pin


def test_changed_protocol_cannot_resume(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    path = tmp_path / "PROTOCOL.json"
    common.publish_json(path, dict(updates=15000))
    (tmp_path / "PROTOCOL.sha256").write_text(common.digest(path) + "  PROTOCOL.json\n")
    assert common.protocol()[0]["updates"] == 15000
    common.atomic_json(path, dict(updates=15001))
    with pytest.raises(ValueError, match="protocol changed"):
        common.protocol()
