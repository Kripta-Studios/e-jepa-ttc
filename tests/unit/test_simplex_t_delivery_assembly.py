"""Real ZIP composition with stubbed scientific authority, never scientific completion."""

import hashlib
import json
import zipfile

import pytest

from e_jepa_ttc.simplex_t import delivery_assembly as assembly
from e_jepa_ttc.simplex_t.bundle_creation import BundleMember


@pytest.mark.parametrize("fault", ["none", "payload", "authority", "resource"])
def test_delivery_transport_preserves_documents_and_checks_inputs(tmp_path, monkeypatch, fault):
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    pin = BundleMember(source, hashlib.sha256(source.read_bytes()).hexdigest(), 2)
    original = {
        f"postprocessing/{name}.json": pin
        for name in (
            "SCIENTIFIC_GRAPH_COVERAGE",
            "CAMPAIGN_ACCOUNTING",
        )
    }
    calls = []

    def bind():
        calls.append(True)
        if fault == "authority" and len(calls) > 1:
            return {}
        return original

    monkeypatch.setattr(
        assembly,
        "render_delivery_documents",
        lambda *a, **kw: (
            "# Fixture only\n",
            {"holdout_authorized": False},
        ),
    )
    if fault == "payload":
        source.write_text("changed")
    output = tmp_path / "artifacts/delivery"
    arguments = dict(
        work_root=tmp_path,
        analysis_commit="a" * 40,
        bind_verified_members=bind,
        resource_ok=lambda: fault != "resource",
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            assembly.assemble_delivery(output, **arguments)
        assert not list(output.glob("*.zip"))
        assert not (output / "DELIVERY.json").exists()
        return
    result = assembly.assemble_delivery(output, **arguments)
    archive = output / "E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_aaaaaaaaaaaa.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["bundle"]["sha256"]
    assert result["campaign_complete"] is False
    assert len(calls) == 3
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) == 5
        inventory = json.loads(bundle.read("CONTENT_MANIFEST.json"))
        assert set(inventory["members"]) == set(bundle.namelist()) - {"CONTENT_MANIFEST.json"}
        for name, item in inventory["members"].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == item["sha256"]
    assert archive.with_suffix(".zip.sha256").read_text().startswith(result["bundle"]["sha256"])
    with pytest.raises(ValueError, match="new companion"):
        assembly.assemble_delivery(output, **arguments)
