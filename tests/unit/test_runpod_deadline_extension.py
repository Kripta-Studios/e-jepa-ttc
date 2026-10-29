"""A time extension must not loosen lineage or bypass expiry."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts.runpod_deadline_extension_20261027 import extension_deadline, rewritten_function


def bind(tmp_path, monkeypatch, old):
    authority = tmp_path / "authority.json"
    authority.write_text('{"immutable": true}')
    receipt = tmp_path / "extension.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "ejepa_explicit_deadline_extension_v1",
                "pod_id": "lreiivpf18vxu7",
                "authorized": True,
                "additional_hours": 6,
                "previous_deadline_utc": old.isoformat(),
                "extended_deadline_utc": (old + timedelta(hours=6)).isoformat(),
                "historical_authority_sha256": {
                    str(authority): hashlib.sha256(authority.read_bytes()).hexdigest()
                },
            }
        )
    )
    monkeypatch.setenv("EJEPA_DEADLINE_EXTENSION", str(receipt))
    monkeypatch.setenv(
        "EJEPA_DEADLINE_EXTENSION_SHA256", hashlib.sha256(receipt.read_bytes()).hexdigest()
    )
    return authority, receipt


def test_extends_expired_original_but_only_by_six_hours(tmp_path, monkeypatch):
    old = datetime.now(UTC) - timedelta(hours=1)
    bind(tmp_path, monkeypatch, old)
    assert extension_deadline(old) == old + timedelta(hours=6)
    with pytest.raises(ValueError):
        extension_deadline(old + timedelta(seconds=1))


@pytest.mark.parametrize("target", ["authority", "receipt"])
def test_tampering_rejected(tmp_path, monkeypatch, target):
    old = datetime.now(UTC)
    paths = dict(zip(("authority", "receipt"), bind(tmp_path, monkeypatch, old), strict=True))
    paths[target].write_text("{}")
    with pytest.raises(ValueError):
        extension_deadline(old)


def test_extended_expiry_enforced(tmp_path, monkeypatch):
    old = datetime.now(UTC) - timedelta(hours=7)
    bind(tmp_path, monkeypatch, old)
    with pytest.raises(ValueError):
        extension_deadline(old)


def test_rewrite_preserves_other_integrity_checks(tmp_path):
    path = tmp_path / "module.py"
    path.write_text(
        "def check(deadline, valid):\n"
        '    if not valid: raise ValueError("integrity")\n'
        "    return deadline > 10\n"
    )
    module = SimpleNamespace(__file__=str(path))
    changed = rewritten_function(module, "check", {"deadline > 10": "deadline > 20"})
    assert changed(21, True) and not changed(11, True)
    with pytest.raises(ValueError, match="integrity"):
        changed(21, False)
    with pytest.raises(ValueError, match="Expected deadline"):
        rewritten_function(module, "check", {"deadline > 11": "True"})
