"""Transport fixture pins only, never data referenced inside an authority manifest."""

import hashlib
import json
from pathlib import Path

import pytest

from e_jepa_ttc.simplex_t import provenance_bundle as module


@pytest.mark.parametrize("fault", ["none", "payload", "freeze", "pause", "authority", "changed"])
def test_pinned_provenance_and_binary_references(tmp_path, monkeypatch, fault):
    metadata = tmp_path / "roles.json"
    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"must not be read")
    metadata.write_text(json.dumps({"protected_reference": str(protected)}))
    weight = tmp_path / "old_expert.pt"
    weight.write_bytes(b"reference only")
    freeze = tmp_path / "freeze.json"
    record = {
        "files": [
            dict(
                category=category,
                root="work",
                relative_path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for category, path in (("roles", metadata), ("producers", weight))
        ]
    }
    freeze.write_text(json.dumps(record))
    freeze_hash = hashlib.sha256(freeze.read_bytes()).hexdigest()
    calls = 0

    def verify(*args, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["validate_prerequisites"]()
        if fault == "changed" and calls > 1:
            return dict(record, changed=True)
        return record

    # Semantic freeze validation is independently tested; byte transport is real.
    monkeypatch.setattr(module, "read_scientific_freeze", verify)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path in {protected, weight}:
            pytest.fail("transport traversed a reference-only payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    if fault == "payload":
        metadata.write_text("{}")
    elif fault == "freeze":
        freeze.write_text("{}")

    def authority():
        if fault == "authority":
            raise ValueError("role authority missing")

    kwargs = dict(
        freeze_sha256=freeze_hash,
        roots={"work": tmp_path},
        validate_scientific_authority=authority,
        resource_ok=lambda: fault != "pause",
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            module.provenance_bundle_members(freeze, **kwargs)
        return
    members = module.provenance_bundle_members(freeze, **kwargs)
    assert set(members) == {"provenance/SCIENTIFIC_FREEZE.json", "provenance/work/roles.json"}
    assert calls == 2
    assert all(m.bytes == m.path.stat().st_size for m in members.values())
