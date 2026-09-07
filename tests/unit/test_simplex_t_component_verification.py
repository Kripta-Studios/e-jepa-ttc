"""Composition/rejection fixtures; numerical component verifiers have own tests."""

import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import component_verification as module


def inputs(tmp_path: Path, monkeypatch, *, h16: bool = True):
    local = tmp_path / "local.json"
    local.write_text(
        json.dumps({"worktree": str(tmp_path), "shared_coordination": str(tmp_path)}),
        encoding="utf-8",
    )
    for name in (
        "historical.json",
        "coherent.json",
        "src/e_jepa_ttc/simplex_t/training.py",
        "scripts/probe_simplex_t_context_resume.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    profile = {
        "schema": "simplex_t_component_evidence_profile_v1",
        "ack_sha256": "a" * 64,
        "historical_replay": {"path": "historical.json", "sha256": "b" * 64},
        "coherent_replay": {"path": "coherent.json", "sha256": "c" * 64},
        "resume": {
            "root": ".",
            "pins": {},
            "source_sha256": "d" * 64,
            "compiled_sha256": "e" * 64,
        },
    }
    if h16:
        profile["h16_replay"] = {"root": ".", "report_sha256": "f" * 64}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    calls = []

    def verifier(name):
        def verify(*args, **kwargs):
            calls.append(name)
            if name == "h16":
                kwargs["validate_execution_identity"]({"synthetic": True})
            return {"fixture": name}

        return verify

    for name, function in (
        ("ancestry", "verify_acknowledged_producers"),
        ("replay", "verify_coherent_replay"),
        ("resume", "verify_real_cpu_resume"),
        ("h16", "verify_h16_replay"),
        ("identity", "verify_h16_execution_identity"),
    ):
        monkeypatch.setattr(module, function, verifier(name))
    return local, path, calls


@pytest.mark.parametrize("h16,required", [(False, False), (False, True), (True, True)])
def test_required_h16_is_not_diagnostic_default(tmp_path, monkeypatch, h16, required):
    local, profile, calls = inputs(tmp_path, monkeypatch, h16=h16)
    if required and not h16:
        with pytest.raises(ValueError, match="WAITING_H16"):
            module.verify_component_profile(
                local, profile, sha256(profile), resource_ok=lambda: True, require_h16=required
            )
        assert calls == []
        return
    result = module.verify_component_profile(
        local, profile, sha256(profile), resource_ok=lambda: True, require_h16=required
    )
    assert calls == ["ancestry", "replay", "resume"] + (["h16", "identity"] if h16 else [])
    assert result["optimizer_updates_executed"] == 0
    assert result["scientific_admission"] is False


@pytest.mark.parametrize("failure", ["profile_hash", "profile_mutation", "resource", "path"])
def test_rejection_propagates_before_next_component(tmp_path, monkeypatch, failure):
    local, profile, calls = inputs(tmp_path, monkeypatch)
    digest = sha256(profile)
    if failure == "path":
        record = json.loads(profile.read_text(encoding="utf-8"))
        record["historical_replay"]["path"] = "../forbidden.json"
        profile.write_text(json.dumps(record), encoding="utf-8")
        digest = sha256(profile)
    if failure == "profile_hash":
        digest = "0" * 64
    if failure == "profile_mutation":

        def mutate(*args, **kwargs):
            calls.append("ancestry")
            profile.write_text("{}", encoding="utf-8")
            return {}

        monkeypatch.setattr(module, "verify_acknowledged_producers", mutate)
    with pytest.raises(InterruptedError if failure == "resource" else ValueError):
        module.verify_component_profile(
            local,
            profile,
            digest,
            resource_ok=lambda: failure != "resource",
            require_h16=True,
        )
    assert calls == ([] if failure in {"profile_hash", "resource"} else ["ancestry"])


def test_numerical_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    local, profile, calls = inputs(tmp_path, monkeypatch)

    def reject(*args, **kwargs):
        raise ValueError("actual H16 mismatch")

    monkeypatch.setattr(module, "verify_h16_replay", reject)
    with pytest.raises(ValueError, match="actual H16 mismatch"):
        module.verify_component_profile(
            local, profile, sha256(profile), resource_ok=lambda: True, require_h16=True
        )
    assert calls == ["ancestry", "replay", "resume"]


def test_missing_unit_evidence_rejected_before_heavy_reads(tmp_path, monkeypatch):
    local, profile, calls = inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="WAITING_CURRENT_UNIT_QA"):
        module.verify_component_profile(
            local,
            profile,
            sha256(profile),
            resource_ok=lambda: True,
            require_h16=True,
            require_unit_qa=True,
        )
    assert not calls


def test_missing_static_evidence_rejected_before_heavy_reads(tmp_path, monkeypatch):
    local, profile, calls = inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="WAITING_CURRENT_STATIC_QA"):
        module.verify_component_profile(
            local,
            profile,
            sha256(profile),
            resource_ok=lambda: True,
            require_h16=True,
            require_static_qa=True,
        )
    assert not calls


def test_static_evidence_reader_is_required(tmp_path, monkeypatch):
    local, profile, calls = inputs(tmp_path, monkeypatch)
    record = json.loads(profile.read_text(encoding="utf-8"))
    record["static_qa"] = {"root": ".", "pins": {}}
    profile.write_text(json.dumps(record), encoding="utf-8")

    def reject(*args, **kwargs):
        raise ValueError("stale real static evidence")

    monkeypatch.setattr(module, "verify_ruff_comparison", reject)
    with pytest.raises(ValueError, match="stale real static evidence"):
        module.verify_component_profile(
            local,
            profile,
            sha256(profile),
            resource_ok=lambda: True,
            require_h16=True,
            require_static_qa=True,
        )


@pytest.mark.parametrize("reject", [False, True])
def test_unit_profile_is_actually_verified(tmp_path, monkeypatch, reject):
    local, profile, calls = inputs(tmp_path, monkeypatch)
    record = json.loads(profile.read_text(encoding="utf-8"))
    record["unit_qa"] = {"root": ".", "pins": {}, "technical_ledger_sha256": "a" * 64}
    profile.write_text(json.dumps(record), encoding="utf-8")

    def verify(*args, **kwargs):
        calls.append("unit_qa")
        assert kwargs["technical_ledger_sha256"] == "a" * 64
        if reject:
            raise ValueError("unit source changed")
        return {"fixture": "unit_qa"}

    monkeypatch.setattr(module, "verify_companion_unit_qa", verify)
    if reject:
        with pytest.raises(ValueError, match="unit source changed"):
            module.verify_component_profile(
                local,
                profile,
                sha256(profile),
                resource_ok=lambda: True,
                require_h16=True,
                require_unit_qa=True,
            )
    else:
        result = module.verify_component_profile(
            local,
            profile,
            sha256(profile),
            resource_ok=lambda: True,
            require_h16=True,
            require_unit_qa=True,
        )
        assert result["unit_qa"] == {"fixture": "unit_qa"}
    assert calls[-1] == "unit_qa"
