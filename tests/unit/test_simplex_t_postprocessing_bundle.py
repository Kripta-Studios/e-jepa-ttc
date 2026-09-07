"""Pinned postprocessing-to-ZIP integration using real fixture bytes."""
import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t import postprocessing_bundle as module
from e_jepa_ttc.simplex_t.bundle_creation import BundleMember, create_verified_bundle
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
    ],
)
def test_publication_to_transport(tmp_path, monkeypatch, fault):
    output = tmp_path / "owned"
    output.mkdir()
    data = output / "weights.npz"
    data.write_bytes(b"synthetic fixture weights")
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
            freeze_sha256=("b" if fault == "freeze" else "a") * 64,
            work_root=tmp_path,
            phases={},
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
    assert result["members"] == 3
    assert result["optimizer_updates"] == 0
