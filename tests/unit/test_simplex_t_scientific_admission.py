"""Composition admission tests; no training, expert inference or real datasets."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import scientific_admission as module


@pytest.mark.parametrize("mode", ["ok", "expanded", "pin", "qa", "changed", "resource"])
def test_admission_composition(tmp_path: Path, monkeypatch, mode: str):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    ack_path = save("SIMPLEX_T_STAGE70_ACK.json", {})
    ancestry_path = save("ancestry.json", {})
    time_path = save("time.json", {})
    ancestry = {"path": str(ancestry_path), "sha256": sha256(ancestry_path)}
    ack = {
        "producers": {"authoritative_historical_manifest": ancestry},
        "interfaces": {"time_charter": {"path": str(time_path), "sha256": sha256(time_path)}},
    }
    local = save("paths.json", {"worktree": str(tmp_path), "shared_coordination": str(tmp_path)})
    config = save("source.json", {"expansion": {}} if mode == "expanded" else {})
    profile = save("profile.json", {})
    index = save("INDEX_MANIFEST.json", {"ancestry": ancestry})
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    index_hash = "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
    monkeypatch.setattr(module, "sha256", lambda p: index_hash if p == index else sha256(p))
    monkeypatch.setattr(module, "verified_ack", lambda *args: ack)
    flags = dict(
        d1=False, density=False, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )
    record = {
        "files": [],
        "source_contract": {
            "configuration_sha256": sha256(config),
            "authority_sha256": ack_hash,
            "availability": flags,
        },
    }
    for path, digest, category in (
        (local, sha256(local), "config"),
        (config, sha256(config), "config"),
        (profile, sha256(profile), "qa"),
        (ack_path, ack_hash, "roles"),
        (ancestry_path, sha256(ancestry_path), "producers"),
        (time_path, sha256(time_path), "time"),
        (index, index_hash, "data"),
    ):
        record["files"].append(
            dict(root="work", relative_path=path.name, sha256=digest, category=category)
        )
    if mode == "pin":
        record["files"][0]["sha256"] = "0" * 64
    calls = []
    sources = SimpleNamespace(
        index_root=tmp_path,
        graph=module.registered_graph(**flags),
        release=lambda: calls.append("release"),
    )

    def open_sources(*args):
        calls.append("open")
        return sources, {"ack_sha256": ack_hash}

    def verify(*args, **kwargs):
        calls.append("qa")
        assert all(
            kwargs[key] is True
            for key in (
                "require_h16",
                "require_unit_qa",
                "require_static_qa",
                "require_types",
                "require_powershell",
            )
        )
        if mode == "qa":
            raise ValueError("real evidence missing")
        if mode == "changed":
            profile.write_text("changed", encoding="utf-8")
        return {
            "local_paths_sha256": sha256(local),
            "ancestry": {"ancestry_sha256": ancestry["sha256"]},
        }

    monkeypatch.setattr(module, "open_acknowledged_source_configuration", open_sources)
    monkeypatch.setattr(module, "verify_component_profile", verify)
    kwargs = dict(
        roots={"work": tmp_path},
        local_paths=local,
        source_configuration=config,
        source_configuration_sha256=sha256(config),
        evidence_profile=profile,
        evidence_profile_sha256=sha256(profile),
        resource_ok=lambda: mode != "resource",
    )
    if mode == "ok":
        result = module.validate_scientific_admission(record, **kwargs)
        assert result["optimizer_updates_executed"] == 0
        assert result["holdout_authorized"] is False
        assert calls == ["open", "qa", "release"]
    else:
        with pytest.raises((ValueError, InterruptedError)):
            module.validate_scientific_admission(record, **kwargs)
        if mode in {"expanded", "pin", "resource"}:
            assert not calls
        else:
            assert calls == ["open", "qa", "release"]
