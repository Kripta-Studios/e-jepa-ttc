"""Pinned postprocessing-to-ZIP integration using real fixture bytes."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t.bundle_creation import create_verified_bundle
from e_jepa_ttc.simplex_t.postprocessing_bundle import postprocessing_bundle_members
from e_jepa_ttc.simplex_t.postprocessing_inventory import inventory_postprocessing


@pytest.mark.parametrize(
    "fault", ["none", "added", "removed", "changed", "manifest", "freeze", "pause"]
)
def test_publication_to_transport(tmp_path, fault):
    output = tmp_path / "owned"
    output.mkdir()
    data = output / "weights.npz"
    data.write_bytes(b"synthetic fixture weights")
    document = {
        "schema": "simplex_t_campaign_postprocessing_v1",
        "status": "SEALED_ANALYSES_WEIGHTS_INTERFACE_ASSEMBLED_NOT_TRANSPORT_DELIVERY",
        "freeze_sha256": "a" * 64,
        "output_inventory": inventory_postprocessing(output, resource_ok=lambda: True),
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
    assert result["members"] == 2
    assert result["optimizer_updates"] == 0
