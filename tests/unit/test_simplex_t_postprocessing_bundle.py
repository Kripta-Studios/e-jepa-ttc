"""Pinned postprocessing-to-ZIP integration using real fixture bytes."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t import postprocessing_bundle as module
from e_jepa_ttc.simplex_t.bundle_creation import BundleMember, create_verified_bundle
from e_jepa_ttc.simplex_t.campaign_accounting import AccountingPins
from e_jepa_ttc.simplex_t.postprocessing_bundle import postprocessing_bundle_members
from e_jepa_ttc.simplex_t.postprocessing_inventory import inventory_postprocessing


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "added",
        "removed",
        "changed",
        "manifest",
        "freeze",
        "pause",
        "phase_omitted",
        "phase_path",
        "phase_hash",
        "phase_bytes",
        "phase_extra",
        "history_omitted",
        "history_path",
        "accounting_missing",
        "accounting_hash",
        "accounting_changed",
        "accounting_unverified",
        "accounting_actual",
    ],
)
def test_publication_to_transport(tmp_path, monkeypatch, fault):
    output = tmp_path / "owned"
    output.mkdir()
    data = output / "weights.npz"
    data.write_bytes(b"synthetic fixture weights")
    accounting = {"fixture": True, "optimizer_updates_executed": 0}
    receipt = output / "CAMPAIGN_ACCOUNTING.json"
    receipt.write_text(json.dumps(accounting), encoding="utf-8")
    accounting_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
    accounting_pins = AccountingPins(
        evidence, evidence_hash, evidence, evidence_hash, evidence, evidence_hash
    )

    def verify_accounting(**kwargs):
        assert kwargs["journal"] == evidence
        if fault == "accounting_actual":
            raise ValueError("actual journal not settled")
        return accounting

    monkeypatch.setattr(module, "verify_campaign_accounting", verify_accounting)
    monkeypatch.setattr(module, "technical_bundle_members", lambda *a, **kw: {})

    def provenance(freeze, **kwargs):
        assert freeze == tmp_path / "freeze.json"
        assert kwargs["roots"]["work"] == tmp_path
        return {"provenance/fixture.json": BundleMember(evidence, evidence_hash, 2)}

    monkeypatch.setattr(module, "provenance_bundle_members", provenance)
    if fault == "accounting_missing":
        receipt.unlink()
    elif fault == "accounting_changed":
        receipt.write_text('{"fixture": false}', encoding="utf-8")
        accounting_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    prediction = tmp_path / "predictions.parquet"
    prediction.write_bytes(b"fixture prediction bytes; graph verifier tested separately")
    member = BundleMember(
        prediction, hashlib.sha256(prediction.read_bytes()).hexdigest(), prediction.stat().st_size
    )
    name = "publications/T2/fixture/predictions.parquet"
    monkeypatch.setattr(module, "phase_bundle_members", lambda *args, **kwargs: {name: member})
    phase_pins = {
        name: dict(work_relative_path=prediction.name, sha256=member.sha256, bytes=member.bytes)
    }
    history_name = "history/D0/query_context_index.npz"
    monkeypatch.setattr(
        module, "history_bundle_members", lambda *args, **kwargs: {history_name: member}
    )
    history_pins = {
        history_name: dict(
            work_relative_path=prediction.name, sha256=member.sha256, bytes=member.bytes
        )
    }
    if fault == "history_omitted":
        history_pins = {}
    elif fault == "history_path":
        history_pins[history_name]["work_relative_path"] = "../not_authorized.npz"
    if fault == "phase_omitted":
        phase_pins = {}
    elif fault == "phase_path":
        phase_pins[name]["work_relative_path"] = "../protected/do_not_open.parquet"
    elif fault == "phase_hash":
        phase_pins[name]["sha256"] = "b" * 64
    elif fault == "phase_bytes":
        phase_pins[name]["bytes"] = 0
    elif fault == "phase_extra":
        phase_pins["publications/T2/extra"] = phase_pins[name]
    document = {
        "schema": "simplex_t_campaign_postprocessing_v1",
        "status": "SEALED_ANALYSES_WEIGHTS_INTERFACE_ASSEMBLED_NOT_TRANSPORT_DELIVERY",
        "freeze_sha256": "a" * 64,
        "output_inventory": inventory_postprocessing(output, resource_ok=lambda: True),
        "phase_payload_inventory": phase_pins,
        "history_payload_inventory": history_pins,
        "campaign_accounting_sha256": "f" * 64 if fault == "accounting_hash" else accounting_hash,
        "optimizer_work_accounting_verified": fault != "accounting_unverified",
    }
    manifest = output / "POSTPROCESSING.json"
    payload = json.dumps(document).encode()
    manifest.write_bytes(payload)
    if fault == "added":
        (output / "extra.json").write_text("{}")
    elif fault == "removed":
        data.unlink()
    elif fault == "changed":
        data.write_bytes(b"changed payload")
    elif fault == "manifest":
        manifest.write_text("{}")

    def bind():
        return postprocessing_bundle_members(
            manifest,
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
            freeze=tmp_path / "freeze.json",
            freeze_sha256=("b" if fault == "freeze" else "a") * 64,
            roots={"work": tmp_path},
            work_root=tmp_path,
            phases={},
            history_pools={},
            accounting_pins=accounting_pins,
            verify_completed_graph=lambda: {},
            validate_scientific_authority=lambda: None,
            resource_ok=lambda: fault != "pause",
        )

    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            bind()
        return
    members = bind()

    def verify():
        assert bind() == members

    result = create_verified_bundle(
        tmp_path / "fixture.zip",
        members,
        validate_inventory_authority=verify,
        resource_ok=lambda: True,
    )
    assert result["members"] == 9
    assert result["optimizer_updates"] == 0
